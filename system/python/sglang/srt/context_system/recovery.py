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

import heapq
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import torch


class DropEvictionCandidates:
    """Persistent Drop-first heaps with bounded lazy invalidation.

    The tree updates a candidate only when its locks, children, residency or
    recency change. Reclaim never scans the tree or reads device page indices.
    ``kind=0`` denotes an ordinary leaf and ``kind=1`` a proven Drop edge,
    including a Drop edge that has become a leaf.
    """

    def __init__(self):
        self.heaps = ([], [])
        self.entries = {}
        self.version = 0

    def update(self, node, kind, priority=None):
        self.entries.pop(node.id, None)
        if kind is not None:
            self.version += 1
            entry = (priority, node.id, self.version, node)
            self.entries[node.id] = (kind, entry)
            heapq.heappush(self.heaps[kind], entry)
        if sum(map(len, self.heaps)) > 2 * len(self.entries) + 64:
            self.heaps = tuple(
                [entry for k, entry in self.entries.values() if k == kind]
                for kind in (0, 1)
            )
            for heap in self.heaps:
                heapq.heapify(heap)

    def pop(self):
        for kind in (1, 0):
            heap = self.heaps[kind]
            while heap:
                entry = heapq.heappop(heap)
                if self.entries.get(entry[1]) == (kind, entry):
                    del self.entries[entry[1]]
                    return kind, entry[-1]
        return None


def mask_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def proven_skip_ranges(
    matched_records: torch.Tensor, required_raw: torch.Tensor
) -> list[tuple[int, int]]:
    """Only a Delta on this matched path authorizes releasing an ancestor KV."""
    if len(matched_records) == 0:
        return []
    records = matched_records.numpy()
    if records.ndim != 2 or records.shape[1] != 3:
        return []
    real = records[:, 0] == 0
    raw_keys = np.flatnonzero(real)
    count = len(raw_keys)
    required = required_raw.numpy()
    if required.dtype != np.bool_ or len(required) != count:
        raise ValueError("Drop lock demand must cover the matched raw prefix.")
    dropped = np.zeros(count, dtype=np.bool_)
    raw_before = np.cumsum(real)
    for key in np.flatnonzero(records[:, 0] == 1):
        start, end = -int(records[key, 1]) - 1, -int(records[key, 2]) - 1
        if not 0 <= start < end <= int(raw_before[key]):
            raise ValueError("Matched Delta references a non-ancestor token range.")
        dropped[start:end] = True
    skip = np.zeros(len(records), dtype=np.bool_)
    skip[raw_keys] = dropped & ~required
    return mask_ranges(skip)


@dataclass(frozen=True)
class RecoveryPlan:
    # Continuous intervals use the existing prefill attention path. Resident
    # gaps between intervals are reused, including the matched suffix.
    intervals: tuple[tuple[int, int], ...]
    required_prefix: torch.Tensor
    matched_length: int
    # Initial resident tokens actually reused, excluding restored source versions.
    reusable_prefix: torch.Tensor

    @property
    def start(self) -> int:
        return self.intervals[0][0]

    def remaining_queries(self, cursor: int) -> int:
        return sum(
            end - max(start, cursor) for start, end in self.intervals if cursor < end
        )

    def next_interval(self, cursor: int) -> tuple[int, int]:
        for start, end in self.intervals:
            if cursor < end:
                return max(start, cursor), end
        raise ValueError("Recovery cursor is past the query stream.")


def sliding_window_starts(
    layout, visible_until: torch.Tensor, window: int
) -> torch.Tensor:
    """First possible raw key for each query's inclusive true-position window.

    Reposition stages use sorted surviving positions. Drop-only boundaries do
    not change positions; expiry is checked by the dependency walk itself.
    Work is vectorized per stage and proportional to the compiler's live-token
    transitions, rather than building a query-by-key mask.
    """
    if window <= 0:
        raise ValueError("Context SWA window must be positive")
    birth = layout.birth_positions.numpy()
    stages = layout.birth_stages.numpy()
    expiry = visible_until.numpy()
    n = len(birth)
    if len(expiry) != n:
        raise ValueError("Context SWA expiry must cover every query")
    positions = birth.copy()
    starts = np.empty(n, dtype=np.int32)
    offsets = layout.transition_offsets.numpy()
    transition_raw = layout.transition_raw_tokens.numpy()
    transition_positions = layout.transition_new_positions.numpy()
    boundaries = np.r_[0, np.flatnonzero(np.diff(stages)) + 1, n]
    active = np.empty(0, dtype=np.int64)
    applied = 0
    for a, b in pairwise(boundaries):
        stage = int(stages[a])
        if stage > applied:
            # A token can transition several times across stages without a
            # birth query. Apply in stage order so duplicate raw IDs are exact.
            for k in range(applied, stage):
                left, right = offsets[k : k + 2]
                positions[transition_raw[left:right]] = transition_positions[left:right]
            applied = stage
        active = np.r_[active[expiry[active] > a], np.arange(a, b)]
        ranks = np.searchsorted(positions[active], birth[a:b] - (window - 1))
        starts[a:b] = active[ranks]
    return torch.from_numpy(starts)


def sliding_window_read_mask(
    query_starts: torch.Tensor,
    visible_until: torch.Tensor,
    intervals: tuple[tuple[int, int], ...],
    prefix_length: int,
) -> torch.Tensor:
    """Union of cached SWA reads, with no query-by-key matrix.

    Between decreases in the window's first raw key, the earliest selected
    query after a token has the widest possible window for that token. Later
    queries cannot undo its Drop. Evaluate only the potential read range of
    each monotonic run, including disjoint historical repair intervals.
    """
    starts, expiry = query_starts.numpy(), visible_until.numpy()
    n = len(starts)
    if (
        query_starts.device.type != "cpu"
        or query_starts.dtype != torch.int32
        or visible_until.device.type != "cpu"
        or visible_until.dtype != torch.int32
        or len(expiry) != n
        or not 0 <= prefix_length <= n
    ):
        raise ValueError("Context SWA read metadata must cover CPU query positions")
    selected = np.zeros(n, dtype=np.bool_)
    for a, b in intervals:
        if not 0 <= a <= b <= n:
            raise ValueError("Context SWA query interval is out of range")
        selected[a:b] = True
    required = np.zeros(prefix_length, dtype=np.bool_)
    boundaries = np.r_[0, np.flatnonzero(np.diff(starts) < 0) + 1, n]
    for a, b in pairwise(boundaries):
        queries = np.flatnonzero(selected[a:b]) + a
        if len(queries) == 0:
            continue
        lo, hi = int(starts[queries[0]]), min(int(queries[-1]), prefix_length)
        if lo >= hi:
            continue
        raw = np.arange(lo, hi)
        query = queries[np.searchsorted(queries, raw, side="right")]
        required[lo:hi] |= (starts[query] <= raw) & (query < expiry[raw])
    return torch.from_numpy(required)


class _SWAQueryDemand:
    """Prefix-min Fenwick index of already planned later queries.

    Reverse recovery inserts only queries after the current token, so the
    prefix ending at that token's Drop event is exactly its live query range.
    Initialization is vectorized; each newly repaired query costs O(log N).
    """

    def __init__(self, starts: np.ndarray, matched: int):
        self.starts = starts
        n = len(starts)
        self.tree = np.full(n + 1, n, dtype=np.int32)
        self.tree[matched + 1 :] = starts[matched:]
        width = 1
        while width <= n:
            parents = np.arange(width * 2, n + 1, width * 2)
            self.tree[parents] = np.minimum(
                self.tree[parents], self.tree[parents - width]
            )
            width *= 2

    def reads(self, raw: int, expiry: int) -> bool:
        cursor = min(expiry, len(self.starts))
        while cursor:
            if self.tree[cursor] <= raw:
                return True
            cursor -= cursor & -cursor
        return False

    def add(self, query: int):
        cursor = query + 1
        start = self.starts[query]
        while cursor < len(self.tree):
            self.tree[cursor] = min(self.tree[cursor], start)
            cursor += cursor & -cursor


def plan_recovery(
    resident: torch.Tensor,
    visible_until: torch.Tensor,
    input_length: int,
    rewind_sources: torch.Tensor | None = None,
    incompatible_sources: torch.Tensor | None = None,
    *,
    swa_resident: torch.Tensor | None = None,
    swa_query_starts: torch.Tensor | None = None,
    swa_terminal_required: torch.Tensor | None = None,
) -> RecoveryPlan:
    """Close missing KV dependencies backwards without scanning the Radix tree.

    A missing token t is needed iff a planned later query q < expiry[t]
    reads it. A reverse scan over missing tokens closes this relation: once a
    token is needed it becomes the earliest query for all preceding tokens.
    """
    if (
        resident.device.type != "cpu"
        or resident.dtype != torch.bool
        or resident.ndim != 1
    ):
        raise ValueError("Recovery residency must be a CPU bool vector.")
    if (
        visible_until.device.type != "cpu"
        or visible_until.dtype != torch.int32
        or visible_until.ndim != 1
    ):
        raise ValueError("Recovery visibility must be a CPU int32 vector.")
    for source_mask in (
        rewind_sources,
        incompatible_sources,
        swa_resident,
        swa_terminal_required,
    ):
        if source_mask is not None and (
            source_mask.device.type != "cpu"
            or source_mask.dtype != torch.bool
            or source_mask.shape != resident.shape
        ):
            raise ValueError(
                "Recovery source masks must match the CPU residency vector."
            )
    present = resident.numpy()
    expiry = visible_until.numpy()
    matched = len(present)
    if not 0 <= matched < input_length or len(expiry) < input_length:
        raise ValueError("Recovery metadata must leave an uncached query suffix.")
    if np.any(expiry[:input_length] <= np.arange(input_length)):
        raise ValueError("A token must remain visible to its own birth query.")
    rewind = (
        np.zeros(matched, dtype=np.bool_)
        if rewind_sources is None
        else rewind_sources.numpy()
    )
    incompatible = (
        np.zeros(matched, dtype=np.bool_)
        if incompatible_sources is None
        else incompatible_sources.numpy()
    )
    if (swa_resident is None) != (swa_query_starts is None):
        raise ValueError("SWA recovery requires both residency and query windows")
    swa_demand = None
    swa_starts = None
    swa_suffix_missing = False
    swa_present = np.ones(matched, dtype=np.bool_)
    if swa_resident is not None:
        if (
            swa_query_starts.device.type != "cpu"
            or swa_query_starts.dtype != torch.int32
            or swa_query_starts.shape != (input_length,)
            or torch.any(swa_query_starts < 0)
            or torch.any(swa_query_starts > torch.arange(input_length))
        ):
            raise ValueError("Invalid Context SWA query window metadata")
        swa_present = swa_resident.numpy()
        if np.any(~swa_present):
            swa_starts = swa_query_starts.numpy()
            # Query suffix is continuous. Prefix minima answer all live suffix
            # window demands in one vectorized pass; cold old SWA tombstones
            # outside those windows must not trigger a per-token Python scan.
            minima = np.minimum.accumulate(swa_starts[matched:])
            ends = np.minimum(expiry[:matched], input_length) - matched - 1
            swa_suffix_missing = np.any(
                ~swa_present
                & (ends >= 0)
                & (minima[np.maximum(ends, 0)] <= np.arange(matched))
            )
    terminal_missing = np.zeros(matched, dtype=np.bool_)
    if swa_terminal_required is not None:
        if swa_resident is None:
            raise ValueError("Terminal SWA recovery requires residency metadata")
        terminal_missing = swa_terminal_required.numpy() & ~swa_present
        swa_suffix_missing |= bool(terminal_missing.any())
    suffix_demand = expiry[:matched] > matched
    if not swa_suffix_missing and not np.any((~present | incompatible) & suffix_demand):
        # No suffix query needs an absent/version-incompatible KV. Rewind-only
        # sources are irrelevant until a historical query actually needs repair.
        return RecoveryPlan(
            ((matched, input_length),),
            torch.from_numpy(suffix_demand),
            matched,
            torch.from_numpy(present.copy()),
        )
    if swa_starts is not None:
        swa_demand = _SWAQueryDemand(swa_starts, matched)
    missing = np.flatnonzero(~present | rewind | incompatible | ~swa_present)
    needed = np.zeros(input_length, dtype=np.bool_)
    needed[matched:] = True
    earliest = matched
    for raw in missing[::-1]:
        full_missing = (
            not present[raw]
            or incompatible[raw]
            or (earliest < matched and rewind[raw])
        )
        full_needed = full_missing and expiry[raw] > earliest
        swa_needed = terminal_missing[raw] or (
            not swa_present[raw]
            and swa_demand is not None
            and swa_demand.reads(int(raw), int(expiry[raw]))
        )
        if full_needed or swa_needed:
            needed[raw] = True
            earliest = int(raw)
            if swa_demand is not None:
                swa_demand.add(int(raw))
    # A resident token is held only when some planned query can read it.
    next_query = np.minimum.accumulate(
        np.where(needed, np.arange(input_length), input_length)[::-1]
    )[::-1]
    required = expiry[:matched] > next_query[1 : matched + 1]
    required |= needed[:matched]
    return RecoveryPlan(
        tuple(mask_ranges(needed)),
        torch.from_numpy(required),
        matched,
        torch.from_numpy(present & ~needed[:matched]),
    )
