# MIT License
#
# Copyright (c) 2026 sgl-project
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .ir import ContextLayout


@dataclass(frozen=True)
class ContextDecodeLayout:
    """Immutable final prompt view; generated KV stays in the native raw row."""

    raw_indices: np.ndarray
    positions: np.ndarray
    device_indices: torch.Tensor
    device_positions: torch.Tensor
    prompt_length: int
    next_position: int

    @classmethod
    def from_layout(cls, layout: ContextLayout, device) -> ContextDecodeLayout:
        raw = np.flatnonzero(layout.keep_mask.numpy()).astype(np.int32)
        positions = layout.positions.numpy()[raw].copy()
        if np.any(np.diff(positions.astype(np.int64)) <= 0):
            raise ValueError("Active decode positions must be strictly increasing")
        packed = torch.from_numpy(np.concatenate((raw, positions))).to(
            device=device, non_blocking=True
        )
        # An empty active prompt still needs a non-null registry marker.
        if packed.numel() == 0:
            packed = torch.zeros(1, dtype=torch.int32, device=device)
        return cls(
            raw,
            positions,
            packed[: len(raw)] if len(raw) else packed,
            packed[len(raw) :] if len(raw) else packed,
            len(layout.positions),
            layout.next_position,
        )

    def swa_raw_floor(self, computed_raw: int, window: int) -> int:
        """First possibly read raw slot at the last completed query position.

        This conservative floor also protects an overlapped previous decode.
        Before final prefill, stage-specific occurrences own their own lifetime.
        """
        if computed_raw < self.prompt_length:
            return 0
        generated = computed_raw - self.prompt_length
        lower = self.next_position + generated - 1 - window
        index = int(np.searchsorted(self.positions, lower))
        if index < len(self.raw_indices):
            return int(self.raw_indices[index])
        return self.prompt_length + max(0, lower - self.next_position)


@dataclass(frozen=True)
class OccurrenceWindow:
    """Lazy attention/KV plan for one post-match raw query window.

    Every ``(raw token, RoPE position)`` pair owns a distinct occurrence.  The
    scheduler later assigns one page from the ordinary KV page allocator to
    each required occurrence; K and V share a slot. The initial supported page size is one token; allocations remain under
    the native allocator ownership contract.
    """

    occurrence_raw_tokens: torch.Tensor
    occurrence_positions: torch.Tensor
    birth_occurrences: torch.Tensor
    terminal_occurrences: torch.Tensor
    segment_query_starts: torch.Tensor
    segment_query_ends: torch.Tensor
    segment_key_offsets: torch.Tensor
    segment_key_occurrences: torch.Tensor

    @property
    def occurrence_count(self) -> int:
        return len(self.occurrence_raw_tokens)

    @property
    def segment_count(self) -> int:
        return len(self.segment_query_starts)


def compile_occurrence_window(
    layout: ContextLayout,
    full_token_visible_until: torch.Tensor,
    terminal_positions: torch.Tensor,
    *,
    query_start: int,
    query_end: int,
) -> OccurrenceWindow:
    """Compile the same ordered plan using request-local native CPU arrays.

    All inputs are read-only views. Working arrays belong to this invocation;
    there is no mutable global scratch or trusted-validation flag. In particular
    the final materialization still covers *all* raw tokens, including dropped
    tokens needed by the final-position Radix cache.
    """
    tensors = (
        layout.birth_positions,
        layout.birth_stages,
        layout.transition_offsets,
        layout.transition_raw_tokens,
        layout.transition_old_positions,
        layout.transition_new_positions,
        full_token_visible_until,
        terminal_positions,
    )
    if any(
        t.device.type != "cpu" or t.dtype != torch.int32 or t.ndim != 1 for t in tensors
    ):
        raise ValueError(
            "Compact occurrence inputs must be one-dimensional CPU int32 tensors."
        )
    birth, stages, offsets, changed_raw, old, new, expiry, terminal = (
        t.numpy() for t in tensors
    )
    n = len(birth)
    if n < 1 or len(stages) != n:
        raise ValueError(
            "Compact occurrence birth metadata must cover a nonempty prompt."
        )
    if len(expiry) != n or len(terminal) != n:
        raise ValueError(
            "Occurrence visibility and terminal positions must cover the prompt."
        )
    if not 0 <= query_start < query_end <= n:
        raise ValueError("Occurrence query window is outside the raw prompt.")
    if len(offsets) < 1:
        raise ValueError("Occurrence transition offsets require an initial zero.")
    if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(
            "Occurrence transition offsets must start at zero and be monotonic."
        )
    if not offsets[-1] == len(changed_raw) == len(old) == len(new):
        raise ValueError(
            "Occurrence transition offsets do not cover the transition arrays."
        )
    if np.any(birth < 0) or np.any(new < 0):
        raise ValueError("Occurrence positions must be non-negative.")
    stage_count = len(offsets) - 1
    if np.any(stages < 0) or np.any(stages > stage_count):
        raise ValueError("Occurrence birth stages are outside the Reposition program.")
    if np.any(stages[1:] < stages[:-1]):
        raise ValueError("Occurrence birth stages must preserve raw-token order.")
    raw = np.arange(n, dtype=np.int32)
    if np.any(expiry <= raw):
        raise ValueError("A token cannot become invisible before it has been computed.")
    bounds = np.searchsorted(stages, np.arange(stage_count + 2))
    current_ids = raw.copy()
    current_pos = birth.copy()
    materialized_pos = birth.copy()
    raw_parts, pos_parts = [raw], [birth]
    next_id = n
    starts, ends, keys, key_offsets = [], [], [], [0]

    def materialize(indices):
        nonlocal next_id
        stale = indices[materialized_pos[indices] != current_pos[indices]]
        count = len(stale)
        if not count:
            return
        if next_id + count > np.iinfo(np.int32).max:
            raise ValueError("Occurrence IDs exceed int32 capacity.")
        raw_parts.append(stale)
        pos_parts.append(current_pos[stale])
        current_ids[stale] = np.arange(next_id, next_id + count, dtype=np.int32)
        materialized_pos[stale] = current_pos[stale]
        next_id += count

    covered = query_start
    for stage in range(stage_count + 1):
        if stage:
            begin, end = int(offsets[stage - 1]), int(offsets[stage])
            ids = changed_raw[begin:end]
            if np.any(ids < 0) or np.any(ids >= n):
                raise ValueError(
                    "Reposition transition references an invalid raw token."
                )
            if np.any(ids[1:] <= ids[:-1]) and len(np.unique(ids)) != len(ids):
                raise ValueError(
                    "One Reposition stage cannot transition a raw token twice."
                )
            if not np.array_equal(current_pos[ids], old[begin:end]):
                raise ValueError(
                    "Reposition transition old positions do not match current state."
                )
            current_pos[ids] = new[begin:end]
        local_start = max(query_start, int(bounds[stage]))
        local_end = min(query_end, int(bounds[stage + 1]))
        if local_start >= local_end:
            continue
        if local_start != covered:
            raise RuntimeError(
                "Occurrence stages do not cover the requested query window."
            )
        values = expiry[:local_end]
        cuts = [
            local_start,
            *np.unique(values[(values > local_start) & (values < local_end)]),
            local_end,
        ]
        for start, end in pairwise(cuts):
            active = raw[:start][expiry[:start] > start]
            materialize(active)
            selected = np.concatenate((current_ids[active], raw[start:end]))
            if not len(selected) or selected[-1] != end - 1:
                raise RuntimeError(
                    "Occurrence segment does not end at its final query token."
                )
            starts.append(start)
            ends.append(end)
            keys.append(selected)
            key_offsets.append(key_offsets[-1] + len(selected))
        covered = local_end
    if covered != query_end:
        raise RuntimeError("Occurrence stages do not cover the requested query window.")
    if not np.array_equal(current_pos, terminal):
        raise ValueError(
            "Compact occurrence transitions disagree with terminal Radix positions."
        )
    materialize(raw)
    if key_offsets[-1] > np.iinfo(np.int32).max:
        raise ValueError("Occurrence segment offsets exceed int32 capacity.")
    all_raw = np.concatenate(raw_parts)
    if not np.array_equal(all_raw[current_ids], raw):
        raise RuntimeError("Terminal occurrences do not cover the raw stream in order.")
    return OccurrenceWindow(
        occurrence_raw_tokens=torch.from_numpy(all_raw),
        occurrence_positions=torch.from_numpy(np.concatenate(pos_parts)),
        birth_occurrences=torch.from_numpy(raw),
        terminal_occurrences=torch.from_numpy(current_ids),
        segment_query_starts=torch.from_numpy(np.asarray(starts, dtype=np.int32)),
        segment_query_ends=torch.from_numpy(np.asarray(ends, dtype=np.int32)),
        segment_key_offsets=torch.from_numpy(np.asarray(key_offsets, dtype=np.int32)),
        segment_key_occurrences=torch.from_numpy(np.concatenate(keys)),
    )


@dataclass(frozen=True)
class OccurrenceMaterialization:
    """CPU ownership decision, made before asking the native allocator for KV.

    Source rows address canonical slots first, then existing terminal slots.
    A negative source row denotes a fresh allocation. Query birth slots have
    already been reserved by native admission and occupy canonical source rows.
    Unused occurrences stay unbound; no physical page is allocated for them.
    """

    source_rows: torch.Tensor
    allocated_occurrences: torch.Tensor
    copy_occurrences: torch.Tensor
    copy_source_rows: torch.Tensor
    copy_position_pairs: torch.Tensor
    terminal_occurrences: torch.Tensor
    read_cached: torch.Tensor
    repositioned_cached: torch.Tensor

    @property
    def extra_page_count(self) -> int:
        return len(self.allocated_occurrences)

    def bind(self, canonical_slots, terminal_slots, allocated_slots):
        """Bind device indices without copying them back to the scheduler CPU."""
        n = len(self.read_cached)
        if (
            canonical_slots.ndim != 1
            or terminal_slots.ndim != 1
            or allocated_slots.ndim != 1
            or any(
                value.dtype not in (torch.int32, torch.int64)
                for value in (canonical_slots, terminal_slots, allocated_slots)
            )
            or canonical_slots.dtype != terminal_slots.dtype
            or canonical_slots.dtype != allocated_slots.dtype
            or len(canonical_slots) != n
            or len(terminal_slots) != n
            or len(allocated_slots) != self.extra_page_count
            or canonical_slots.device != terminal_slots.device
            or canonical_slots.device != allocated_slots.device
        ):
            raise ValueError("Occurrence slot buffers do not match the ownership plan")
        # CPU-only index preparation; one transfer covers reuse, fresh slots and
        # copy metadata. In particular, no CUDA nonzero/item is used here.
        rows = self.source_rows.numpy()
        reused = np.flatnonzero(rows >= 0)
        fields = (
            reused,
            rows[reused],
            self.allocated_occurrences.numpy(),
            self.copy_occurrences.numpy(),
            self.copy_source_rows.numpy(),
            self.copy_position_pairs.numpy().reshape(-1),
        )
        offsets = np.r_[0, np.cumsum([len(field) for field in fields])]
        packed = torch.from_numpy(np.concatenate(fields).astype(np.int64, copy=False))
        packed = packed.to(canonical_slots.device, non_blocking=True)
        reuse_ids, reuse_rows, fresh_ids, copy_ids, copy_rows, pairs = (
            packed[a:b] for a, b in pairwise(offsets)
        )
        sources = torch.cat((canonical_slots, terminal_slots))
        slots = canonical_slots.new_full((len(rows),), -1)
        slots[reuse_ids] = sources[reuse_rows]
        slots[fresh_ids] = allocated_slots
        return (
            slots,
            sources[copy_rows].to(torch.int32),
            slots[copy_ids].to(torch.int32),
            pairs.reshape(-1, 2).to(torch.int32),
        )


def plan_occurrence_materialization(
    window: OccurrenceWindow,
    canonical_positions: torch.Tensor,
    canonical_present: torch.Tensor,
    canonical_owned: torch.Tensor,
    terminal_present: torch.Tensor,
    terminal_keep: torch.Tensor,
    *,
    query_start: int,
    query_end: int,
    exact_prefix_len: int,
) -> OccurrenceMaterialization:
    """Select minimum occurrence allocations under explicit native ownership.

    Canonical metadata covers the prefix before this forward. A retained terminal
    slot already belongs to the target version. A borrowed Retry source after the
    *selected source's* exact prefix must be copied before target publication,
    even at the same position (a bit-copy, not another RoPE round trip).

    ``terminal_keep`` is a raw-token mask supplied by the cache policy. Holes may
    remain holes only when neither this forward's read set nor publication needs
    them. Missing required sources are returned as an error for recovery planning;
    they must never be bound to the padded slot 0.
    """
    n = len(window.birth_occurrences)
    if not 0 <= exact_prefix_len <= query_start < query_end <= n:
        raise ValueError("Invalid occurrence ownership/query boundaries")
    for value, size, dtype in (
        (canonical_positions, query_start, torch.int32),
        (canonical_present, query_start, torch.bool),
        (canonical_owned, query_start, torch.bool),
        (terminal_present, query_start, torch.bool),
        (terminal_keep, query_end, torch.bool),
    ):
        if value.device.type != "cpu" or value.dtype != dtype or value.shape != (size,):
            raise ValueError("Occurrence ownership must use aligned CPU metadata")
    if (
        int(window.segment_query_starts[0]) != query_start
        or int(window.segment_query_ends[-1]) != query_end
    ):
        raise ValueError("Occurrence ownership does not cover the planned queries")
    raw = window.occurrence_raw_tokens.numpy()
    position = window.occurrence_positions.numpy()
    birth = window.birth_occurrences.numpy()
    terminal = window.terminal_occurrences.numpy()
    count = len(raw)
    present = np.zeros(n, dtype=np.bool_)
    present[:query_start] = canonical_present.numpy()
    present[query_start:query_end] = True
    owned = np.zeros(n, dtype=np.bool_)
    owned[:query_start] = canonical_owned.numpy()
    owned[query_start:query_end] = True
    canonical_pos = position[birth].copy()
    canonical_pos[:query_start] = canonical_positions.numpy()
    terminal_live = np.zeros(n, dtype=np.bool_)
    terminal_live[:query_start] = terminal_present.numpy()
    terminal_pos = position[terminal]

    read = np.zeros(count, dtype=np.bool_)
    read[window.segment_key_occurrences.numpy()] = True
    required = read.copy()
    required[birth[query_start:query_end]] = True
    publish_raw = np.flatnonzero(terminal_keep.numpy())
    publish_ids = terminal[publish_raw]
    required[publish_ids] = True
    ids = np.flatnonzero(required)
    selected_raw = raw[ids]
    selected_pos = position[ids]
    terminal_match = terminal_live[selected_raw] & (
        terminal_pos[selected_raw] == selected_pos
    )
    canonical_match = present[selected_raw] & (
        canonical_pos[selected_raw] == selected_pos
    )
    # A page cannot become owned by two unrelated native Radix branches. Same
    # position does not imply same ownership; exact prefix pages remain borrowed.
    publishing = np.zeros(count, dtype=np.bool_)
    publishing[publish_ids] = True
    force_own = (
        publishing[ids]
        & (selected_raw >= exact_prefix_len)
        & ~owned[selected_raw]
        & ~terminal_match
    )
    reuse = terminal_match | (canonical_match & ~force_own)
    sources = np.where(terminal_match, n + selected_raw, selected_raw)
    rows = np.full(count, -1, dtype=np.int64)
    rows[ids[reuse]] = sources[reuse]
    allocated = ids[~reuse]
    copied_raw = raw[allocated]
    # Prefer the canonical source, keeping each copy independent of other copies
    # in this layer. Newly computed queries use their native birth write slots.
    copy_from_terminal = ~present[copied_raw] & terminal_live[copied_raw]
    missing = ~present[copied_raw] & ~terminal_live[copied_raw]
    if np.any(missing):
        raise ValueError(
            "Occurrence recovery required for raw tokens "
            + repr(np.unique(copied_raw[missing]).tolist())
        )
    copy_rows = np.where(copy_from_terminal, n + copied_raw, copied_raw)
    old = np.where(
        copy_from_terminal, terminal_pos[copied_raw], canonical_pos[copied_raw]
    )
    pairs = np.column_stack((old, position[allocated])).astype(np.int32)
    retained = np.full(query_end, -1, dtype=np.int32)
    retained[publish_raw] = publish_ids
    read_cached = np.zeros(n, dtype=np.bool_)
    read_raw = raw[np.flatnonzero(read)]
    read_cached[read_raw[read_raw < query_start]] = True
    rotated_cached = np.zeros(n, dtype=np.bool_)
    changed_reads = (
        read[ids]
        & (selected_pos != canonical_pos[selected_raw])
        & (selected_raw < query_start)
    )
    rotated_cached[selected_raw[changed_reads]] = True
    return OccurrenceMaterialization(
        torch.from_numpy(rows),
        torch.from_numpy(allocated),
        torch.from_numpy(allocated.copy()),
        torch.from_numpy(copy_rows),
        torch.from_numpy(pairs),
        torch.from_numpy(retained),
        torch.from_numpy(read_cached),
        torch.from_numpy(rotated_cached),
    )


@dataclass(frozen=True)
class OccurrenceState:
    """Request-local page ownership between prefill forwards.

    CPU rows identify entries in ``slots``, never physical page numbers. This
    permits alias tracking and retirement without a device-to-host page read.
    Borrowed rows remain protected by the caller's Radix source leases. Each
    successor replaces its predecessor as owner; forward snapshots only retain
    tensor references and do not independently own allocator pages.
    """

    slots: torch.Tensor
    owned: torch.Tensor
    canonical_rows: torch.Tensor
    terminal_rows: torch.Tensor
    canonical_positions: torch.Tensor
    exact_prefix_len: int
    # Independent SWA validity for each physical owner row. Full-only models
    # leave this absent; a Full hit never implies an SWA hit.
    swa_resident: torch.Tensor | None = None

    @classmethod
    def from_match(
        cls, slots, positions, *, exact_prefix_len, resident=None, swa_resident=None
    ):
        if (
            slots.ndim != 1
            or slots.dtype not in (torch.int32, torch.int64)
            or positions.device.type != "cpu"
            or positions.dtype != torch.int32
            or positions.shape != slots.shape
            or not 0 <= exact_prefix_len <= len(slots)
        ):
            raise ValueError("Invalid Context match ownership metadata")
        if resident is not None and (
            resident.device.type != "cpu"
            or resident.dtype != torch.bool
            or resident.shape != slots.shape
        ):
            raise ValueError("Context residency must be an aligned CPU bool vector")
        if swa_resident is not None and (
            swa_resident.device.type != "cpu"
            or swa_resident.dtype != torch.bool
            or swa_resident.shape != slots.shape
        ):
            raise ValueError("Context SWA residency must be an aligned CPU bool vector")
        rows = torch.arange(len(slots), dtype=torch.int64)
        if resident is not None:
            rows[~resident] = -1
        return cls(
            slots,
            torch.zeros(len(slots), dtype=torch.bool),
            rows,
            torch.full((len(slots),), -1, dtype=torch.int64),
            positions.clone(),
            exact_prefix_len,
            swa_resident.clone() if swa_resident is not None else None,
        )

    def reuse_match_gap(
        self,
        matched_slots,
        matched_positions,
        resident,
        end,
        *,
        exact_prefix_len,
        swa_resident=None,
    ):
        """Advance over a resident gap between recovery query intervals.

        The original matched path must remain leased. Gap slots are borrowed
        canonical sources, not target-version pages: the next forward makes
        any required COW/RoPE copies before cache publication. This operation
        allocates metadata only and never retires or allocates a physical page.
        """
        start = len(self.canonical_rows)
        if (
            not start <= end <= len(matched_slots)
            or matched_positions.device.type != "cpu"
            or matched_positions.dtype != torch.int32
            or resident.device.type != "cpu"
            or resident.dtype != torch.bool
            or matched_positions.shape != matched_slots.shape
            or resident.shape != matched_slots.shape
            or matched_slots.ndim != 1
            or matched_slots.dtype != self.slots.dtype
            or matched_slots.device != self.slots.device
            or not self.exact_prefix_len <= exact_prefix_len <= len(matched_slots)
        ):
            raise ValueError("Invalid matched gap ownership metadata")
        if (self.swa_resident is None) != (swa_resident is None) or (
            swa_resident is not None
            and (
                swa_resident.device.type != "cpu"
                or swa_resident.dtype != torch.bool
                or swa_resident.shape != matched_slots.shape
            )
        ):
            raise ValueError("Context matched gap must preserve SWA residency metadata")
        if end == start:
            return self
        count = end - start
        rows = torch.arange(len(self.slots), len(self.slots) + count, dtype=torch.int64)
        rows[~resident[start:end]] = -1
        return OccurrenceState(
            torch.cat((self.slots, matched_slots[start:end])),
            torch.cat((self.owned, torch.zeros(count, dtype=torch.bool))),
            torch.cat((self.canonical_rows, rows)),
            torch.cat(
                (self.terminal_rows, torch.full((count,), -1, dtype=torch.int64))
            ),
            torch.cat((self.canonical_positions, matched_positions[start:end])),
            min(end, exact_prefix_len),
            torch.cat((self.swa_resident, swa_resident[start:end]))
            if swa_resident is not None
            else None,
        )

    def prefill_terminal_keep(self, layout, query_end):
        start = len(self.canonical_rows)
        # Match mini's recovery policy: historical queries may read a dropped
        # token's birth KV, but no future decode needs its terminal cache copy.
        # advance() retains those birth sources until their final query and
        # retires private pages only after the completed forward.
        keep = layout.keep_mask[:query_end].clone()
        keep[:start] &= (self.canonical_rows >= 0) | (
            self.terminal_rows >= 0
        )
        if query_end < len(layout.positions):
            # Do not keep a third (final-position) version while a historical
            # stage still needs birth and intermediate-position KV. Publish
            # holes until the final forward; active birth sources stay private
            # and live, so the final copy still rotates directly from its source.
            available = (
                layout.birth_positions[:query_end]
                == layout.positions[:query_end]
            )
            available[:start] = (
                self.canonical_positions
                == layout.positions[:start]
            ) | (self.terminal_rows >= 0)
            keep &= available
        return keep

    def plan(self, window, terminal_keep):
        start = len(self.canonical_rows)
        canonical = self.canonical_rows.numpy()
        present = canonical >= 0
        owned = np.zeros(start, dtype=np.bool_)
        owned[present] = self.owned.numpy()[canonical[present]]
        return plan_occurrence_materialization(
            window,
            self.canonical_positions,
            torch.from_numpy(present),
            torch.from_numpy(owned),
            self.terminal_rows >= 0,
            terminal_keep,
            query_start=start,
            query_end=int(window.segment_query_ends[-1]),
            exact_prefix_len=self.exact_prefix_len,
        )

    def advance(self, window, plan, query_slots, extra_slots, visible_until):
        """Bind one admitted forward and return its successor plus retirement.

        ``query_slots`` were allocated by native extend admission; extra slots
        must also have been charged before dispatch. No allocator operation is
        performed here. Retired pages remain live until this forward completes,
        even when the successor is prepared on an overlapping scheduler stream.
        """
        n = len(window.birth_occurrences)
        start, end = len(self.canonical_rows), len(plan.terminal_occurrences)
        if (
            query_slots.ndim != 1
            or extra_slots.ndim != 1
            or len(query_slots) != end - start
            or len(extra_slots) != plan.extra_page_count
            or query_slots.dtype != self.slots.dtype
            or extra_slots.dtype != self.slots.dtype
            or (len(self.slots) and query_slots.device != self.slots.device)
            or extra_slots.device != query_slots.device
            or visible_until.device.type != "cpu"
            or visible_until.dtype != torch.int32
            or visible_until.shape != (n,)
        ):
            raise ValueError("Context admission does not cover the selected window")
        old_count = len(self.slots)
        old_slots = self.slots if old_count else query_slots[:0]
        all_slots = torch.cat((old_slots, query_slots, extra_slots))
        all_owned = np.r_[
            self.owned.numpy(), np.ones(len(query_slots) + len(extra_slots), dtype=bool)
        ]
        canonical = np.full(n, -1, dtype=np.int64)
        canonical[:start] = self.canonical_rows.numpy()
        canonical[start:end] = np.arange(old_count, old_count + len(query_slots))
        terminal = np.full(n, -1, dtype=np.int64)
        terminal[:start] = self.terminal_rows.numpy()
        source_rows = np.r_[canonical, terminal]
        rows = plan.source_rows.numpy()
        occurrences = np.full(len(rows), -1, dtype=np.int64)
        reuse = rows >= 0
        occurrences[reuse] = source_rows[rows[reuse]]
        if np.any(occurrences[reuse] < 0):
            raise ValueError("Context ownership plan references a missing source")
        fresh = np.arange(old_count + len(query_slots), len(all_slots))
        occurrences[plan.allocated_occurrences.numpy()] = fresh
        copy_rows = source_rows[plan.copy_source_rows.numpy()]
        if np.any(copy_rows < 0):
            raise ValueError("Context copy references a missing canonical source")
        all_swa = None
        unused_swa = np.empty(0, dtype=np.int64)
        if self.swa_resident is not None:
            all_swa = np.r_[
                self.swa_resident.numpy(),
                np.ones(len(query_slots) + len(extra_slots), dtype=np.bool_),
            ]
            copy_destinations = occurrences[plan.copy_occurrences.numpy()]
            all_swa[copy_destinations] = all_swa[copy_rows]
            # Only new private destinations can be released here. An absent
            # borrowed source's SWA mapping is already owned by the cache.
            unused_swa = copy_destinations[~all_swa[copy_destinations]]
        selected_terminal = plan.terminal_occurrences.numpy()
        terminal = np.full(end, -1, dtype=np.int64)
        kept = selected_terminal >= 0
        terminal[kept] = occurrences[selected_terminal[kept]]
        canonical = canonical[:end]
        positions = window.occurrence_positions[window.birth_occurrences[:end]].clone()
        positions[:start] = self.canonical_positions
        same_position = (
            positions.numpy()
            == window.occurrence_positions[window.terminal_occurrences[:end]].numpy()
        )
        # A same-position COW is bit-identical. Future reads can use the target
        # owner instead of pinning a second borrowed copy on the Retry branch.
        canonical = np.where(kept & same_position, terminal, canonical)
        # Drops are monotonic. A source that no later prefill query can read is
        # no longer needed. Decode reads terminal positions, not birth versions.
        future_read = visible_until.numpy()[:end] > end
        if end == n:
            future_read[:] = False
        canonical = np.where(future_read, canonical, -1)
        retained = np.unique(np.r_[canonical[canonical >= 0], terminal[terminal >= 0]])
        live = np.zeros(len(all_slots), dtype=bool)
        live[retained] = True
        retired = np.flatnonzero(all_owned & ~live)
        remap = np.full(len(all_slots), -1, dtype=np.int64)
        remap[retained] = np.arange(len(retained))
        next_canonical = np.full(end, -1, dtype=np.int64)
        next_terminal = np.full(end, -1, dtype=np.int64)
        next_canonical[canonical >= 0] = remap[canonical[canonical >= 0]]
        next_terminal[kept] = remap[terminal[kept]]
        # Pack all gathers once. Index preparation and alias decisions stay CPU.
        fields = (
            np.maximum(occurrences, 0),
            copy_rows,
            occurrences[plan.copy_occurrences.numpy()],
            retained,
            retired,
            plan.copy_position_pairs.numpy().reshape(-1),
            unused_swa,
        )
        offsets = np.r_[0, np.cumsum([len(field) for field in fields])]
        packed = torch.from_numpy(np.concatenate(fields).astype(np.int64, copy=False))
        packed = packed.to(query_slots.device, non_blocking=True)
        occ, src, dst, keep, release, pairs, unused = (
            packed[a:b] for a, b in pairwise(offsets)
        )
        successor = OccurrenceState(
            all_slots[keep],
            torch.from_numpy(all_owned[retained]),
            torch.from_numpy(next_canonical),
            torch.from_numpy(next_terminal),
            positions,
            self.exact_prefix_len,
            torch.from_numpy(all_swa[retained]) if all_swa is not None else None,
        )
        return OccurrenceAdvance(
            successor,
            all_slots[occ],
            all_slots[src].to(torch.int32),
            all_slots[dst].to(torch.int32),
            pairs.reshape(-1, 2).to(torch.int32),
            all_slots[release],
            all_slots[unused] if all_swa is not None else None,
            torch.from_numpy(all_swa[retired]) if all_swa is not None else None,
        )

    def terminal_slots(self):
        """Raw-order final-position table; holes keep the explicit -1 sentinel."""
        rows = self.terminal_rows.numpy()
        present = np.flatnonzero(rows >= 0)
        packed = torch.from_numpy(np.r_[present, rows[present]]).to(
            self.slots.device, non_blocking=True
        )
        ids, source = packed[: len(present)], packed[len(present) :]
        result = self.slots.new_full((len(rows),), -1)
        result[ids] = self.slots[source]
        return result

    def publish(self, matched_terminal_slots, *, cache_len=None, swa_resident=None):
        """Transfer terminal ownership after native Radix insert and rematch.

        Native insert may free duplicate input slots. Canonical aliases must be
        rebound to the resulting cache slots at the same time. Nonterminal birth
        sources keep their independent request ownership. The caller must hold
        the new Radix lease before releasing any previous source lease.
        """
        rows = self.terminal_rows.numpy()
        if (
            matched_terminal_slots.shape != rows.shape
            or matched_terminal_slots.dtype != self.slots.dtype
            or matched_terminal_slots.device != self.slots.device
        ):
            raise ValueError("Context publication must cover the raw terminal table")
        if cache_len is None:
            cache_len = len(rows)
        if not 0 <= cache_len <= len(rows):
            raise ValueError("Context cache publication length exceeds the raw table")
        present = np.flatnonzero(rows[:cache_len] >= 0)
        target = rows[present]
        if len(np.unique(target)) != len(target):
            raise ValueError("Distinct raw tokens cannot publish the same KV owner")
        packed = torch.from_numpy(np.r_[target, present]).to(
            self.slots.device, non_blocking=True
        )
        dst, src = packed[: len(target)], packed[len(target) :]
        slots = self.slots.clone()
        slots[dst] = matched_terminal_slots[src]
        owned = self.owned.clone()
        owned[target] = False
        swa = self.swa_resident
        if swa_resident is not None:
            if (
                swa is None
                or swa_resident.device.type != "cpu"
                or swa_resident.dtype != torch.bool
                or swa_resident.shape != rows.shape
            ):
                raise ValueError(
                    "Context publication SWA metadata must cover the raw table"
                )
            swa = swa.clone()
            swa[target] = swa_resident[present]
        return OccurrenceState(
            slots,
            owned,
            self.canonical_rows,
            self.terminal_rows,
            self.canonical_positions,
            self.exact_prefix_len,
            swa,
        )

    def drop_borrowed_raw(self, dropped):
        """Forget proven-unused cache owners before their Full pages can evict.

        Private versions remain owned until the normal completion receipt frees
        them. The surviving raw coordinates never shift when a page disappears.
        """
        if (
            dropped.device.type != "cpu"
            or dropped.dtype != torch.bool
            or dropped.shape != self.terminal_rows.shape
        ):
            raise ValueError("Context Drop ownership mask must cover the raw state")
        canonical = self.canonical_rows.numpy().copy()
        terminal = self.terminal_rows.numpy().copy()
        changed = False
        for rows in (canonical, terminal):
            candidates = np.flatnonzero(dropped.numpy() & (rows >= 0))
            candidates = candidates[~self.owned.numpy()[rows[candidates]]]
            if len(candidates):
                rows[candidates] = -1
                changed = True
        if not changed:
            return self
        live = self.owned.numpy().copy()
        live[canonical[canonical >= 0]] = True
        live[terminal[terminal >= 0]] = True
        retained = np.flatnonzero(live)
        remap = np.full(len(live), -1, dtype=np.int64)
        remap[retained] = np.arange(len(retained))
        for rows in (canonical, terminal):
            present = rows >= 0
            rows[present] = remap[rows[present]]
        indices = torch.from_numpy(retained).to(self.slots.device, non_blocking=True)
        return OccurrenceState(
            self.slots[indices],
            self.owned[retained],
            torch.from_numpy(canonical),
            torch.from_numpy(terminal),
            self.canonical_positions,
            self.exact_prefix_len,
            self.swa_resident[retained] if self.swa_resident is not None else None,
        )

    def terminal_swa_residency(self):
        if self.swa_resident is None:
            return None
        rows = self.terminal_rows
        present = rows >= 0
        result = torch.zeros(len(rows), dtype=torch.bool)
        result[present] = self.swa_resident[rows[present]]
        return result

    def live_swa_ranges(self, start: int, end: int):
        """Read only the newly retiring CPU interval, never the full history.

        Generated slots beyond this prefill state own native SWA peers. The
        caller's monotonic eviction floor excludes already retired peers.
        """
        from sglang.srt.context_system.recovery import mask_ranges

        if end <= start:
            return []
        live = np.ones(end - start, dtype=np.bool_)
        rows = self.terminal_rows[start:end].numpy()
        present = rows >= 0
        live[: len(rows)] = present
        if self.swa_resident is not None:
            indices = np.flatnonzero(present)
            live[indices] = self.swa_resident.numpy()[rows[indices]]
        return [(start + a, start + b) for a, b in mask_ranges(live)]

    def private_slots(self):
        """Request pages only; borrowed Radix pages are released through leases."""
        rows = torch.from_numpy(np.flatnonzero(self.owned.numpy())).to(
            self.slots.device, non_blocking=True
        )
        return self.slots[rows]

    def nonterminal_private_residency(self):
        """CPU SWA ownership for the private sources outside the raw table."""
        if self.swa_resident is None:
            return None
        selected = self.owned.clone()
        rows = self.terminal_rows
        selected[rows[rows >= 0]] = False
        return self.swa_resident[selected]

    def nonterminal_private_slots(self):
        """Private sources outside the raw table handled by native cache release."""
        selected = self.owned.numpy().copy()
        rows = self.terminal_rows.numpy()
        selected[rows[rows >= 0]] = False
        indices = torch.from_numpy(np.flatnonzero(selected)).to(
            self.slots.device, non_blocking=True
        )
        return self.slots[indices]


@dataclass(frozen=True)
class OccurrenceAdvance:
    state: OccurrenceState
    occurrence_slots: torch.Tensor
    copy_source_slots: torch.Tensor
    copy_destination_slots: torch.Tensor
    copy_position_pairs: torch.Tensor
    # Caller releases exactly once, after the forward's native completion event.
    retired_slots: torch.Tensor
    # These fresh SWA peers have no valid source and no reader. Release only
    # their SWA mapping before binding the forward, keeping Full ownership.
    unused_swa_slots: torch.Tensor | None = None
    retired_swa_resident: torch.Tensor | None = None


def free_context_slots(allocator, slots, swa_resident=None):
    """Release private page_size=1 owners without inspecting GPU mappings.

    SWA validity is carried with CPU ownership from allocation through copies.
    The ordinary/all-valid case keeps the allocator's existing fast path.
    """
    if swa_resident is None:
        allocator.free(slots)
        return
    if (
        swa_resident.device.type != "cpu"
        or swa_resident.dtype != torch.bool
        or swa_resident.shape != slots.shape
    ):
        raise ValueError("Context free SWA metadata must cover the owner slots")
    if len(slots) == 0:
        return
    valid = swa_resident.numpy()
    if valid.all():
        allocator.free(slots)
    elif not valid.any():
        allocator.free_full(slots)
    else:
        live, dead = np.flatnonzero(valid), np.flatnonzero(~valid)
        indices = torch.from_numpy(np.r_[live, dead]).to(
            slots.device, non_blocking=True
        )
        allocator.free(slots[indices[: len(live)]])
        allocator.free_full(slots[indices[len(live) :]])


@dataclass
class ContextPrefillCompletion:
    """Result-batch receipt; consume after its native GPU completion event."""

    retired_slots: torch.Tensor
    usage: object
    read_cached: torch.Tensor
    repositioned_cached: torch.Tensor
    query_count: int
    completed: bool = False
    retired_swa_resident: torch.Tensor | None = None

    def __post_init__(self):
        # Retraction may be prepared before this overlapped result is consumed.
        self.initial_match = not self.usage.recomputing

    def complete(self, allocator):
        # Copies of ScheduleBatch share the same receipt. Duplicate notification
        # must neither free pages twice nor count the same queries twice.
        if self.completed:
            return
        self.usage.record_prefill(
            self.read_cached,
            self.repositioned_cached,
            self.query_count,
            initial_match=self.initial_match,
        )
        free_context_slots(allocator, self.retired_slots, self.retired_swa_resident)
        self.completed = True


@dataclass
class ContextDecodeCompletion:
    usage: object
    completed: bool = False

    def complete(self, allocator):
        if not self.completed:
            self.usage.record_decode()
            self.completed = True
