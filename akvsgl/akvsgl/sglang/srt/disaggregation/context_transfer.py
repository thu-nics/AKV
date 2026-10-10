"""Page-size-one Context handoff over the native P-to-D KV transport.

Raw request coordinates stay intact, while transport and physical allocation
contain only final active versions. No decode output KV is sent back to P.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from sglang.srt.context_system.request_storage import request_row
from sglang.srt.context_system.occurrence import ContextDecodeLayout, OccurrenceState
from sglang.srt.context_system.usage import ContextUsage, ContextUsageSnapshot


@dataclass(frozen=True)
class ContextTransferPlan:
    decode: ContextDecodeLayout
    signature: tuple[int, int]

    @classmethod
    def build(cls, program, device):
        digest = hashlib.blake2b(digest_size=8)
        # Include the full event identity, not just final token text/positions.
        for value in (
            program.layout.records,
            program.layout.keep_mask,
            program.visible_until,
        ):
            digest.update(value.numpy().tobytes())
        signature = tuple(int(x) for x in np.frombuffer(digest.digest(), dtype="<i4"))
        return cls(ContextDecodeLayout.from_layout(program.layout, device), signature)

    @property
    def active_count(self):
        return len(self.decode.raw_indices)

    def swa_start(self, window):
        # The first D query is at next_position. Native SWA includes keys at
        # query_position - window, so preserve that boundary token too.
        return int(
            np.searchsorted(self.decode.positions, self.decode.next_position - window)
        )

    def slots(self, req, pool, *, window=None):
        start = self.swa_start(window) if window is not None else 0
        return request_row(pool, req.kv.req_pool_idx)[
            self.decode.device_indices[start : self.active_count]
        ]

    def full_chunk(self, raw_start, raw_end, *, last_chunk, terminal_rows=None, owned=None):
        """Map a completed raw prefix into consecutive final-version pages.

        Keep one page for the final send: native transports infer completion
        from the cumulative page count, when the metadata/SWA payload is ready.
        Returned raw cursor never advances past a withheld page.
        """
        begin, end = np.searchsorted(self.decode.raw_indices, (raw_start, raw_end))
        if terminal_rows is not None:
            raw = self.decode.raw_indices[begin:end]
            rows = terminal_rows.numpy()[raw]
            ready = rows >= 0
            if last_chunk:
                if not ready.all():
                    raise ValueError("Context final transfer has missing terminal KV")
            else:
                # Private copies may be deduplicated by a later publication.
                # Only stable cache owners can outlive an asynchronous send;
                # the final send is protected by native inflight retirement.
                if owned is not None:
                    ready[ready] = ~owned.numpy()[rows[ready]]
                missing = np.flatnonzero(~ready)
                if len(missing):
                    end = int(begin) + int(missing[0])
        if not last_chunk:
            end = min(int(end), max(0, self.active_count - 1))
        cursor = raw_end
        if end < self.active_count:
            cursor = min(cursor, int(self.decode.raw_indices[end]))
        return int(begin), int(end), cursor

    def header(self):
        return (2, self.active_count, self.decode.next_position, *self.signature)


def transfer_plan(req, device):
    program = req.context_recompute_program or req.context_program
    current = getattr(req, "context_transfer_plan", None)
    if current is None or current.decode.prompt_length != len(program.layout.positions):
        current = ContextTransferPlan.build(program, device)
        req.context_transfer_plan = current
    return current


def prepare_decode_transfer_plan(req, device, raw_length):
    base = len(req.origin_input_ids)
    if raw_length > base and (
        req.context_recompute_program is None
        or len(req.context_recompute_program.layout.positions) != raw_length
    ):
        req.context_recompute_program = req.context_program.with_generated(
            req.output_ids[: raw_length - base]
        )
    return transfer_plan(req, device)


def request_active_slots(req, pool, raw_end, *, window=None):
    """Rare offload/load paths must never send Full holes to a pool copy."""
    view = req.context_decode_layout
    if view is None:
        view = ContextDecodeLayout.from_layout(
            req.context_program.layout, pool.req_to_token.device
        )
    start = 0
    generated_start = view.prompt_length
    if window is not None:
        lower = view.next_position + raw_end - view.prompt_length - window
        start = int(np.searchsorted(view.positions, lower))
        generated_start += max(0, lower - view.next_position)
    prefix = request_row(pool, req.kv.req_pool_idx)[
        view.device_indices[start : len(view.raw_indices)]
    ]
    return torch.cat(
        (prefix, request_row(pool, req.kv.req_pool_idx)[generated_start:raw_end])
    )


def request_active_lengths(req, raw_end, window):
    """Admission reads immutable CPU metadata, without rebuilding generated IR."""
    view = req.context_decode_layout
    generated = max(0, raw_end - view.prompt_length)
    full = len(view.raw_indices) + generated
    lower = view.next_position + generated - window
    start = int(np.searchsorted(view.positions, lower))
    return full, len(view.raw_indices) - start + min(generated, window)


def select_missing_transfer(source, destination, reusable):
    """Filter one peer's chunk without mutating shared source chunk offsets."""
    if source.shape != destination.shape or reusable.shape != source.shape:
        raise ValueError("Context transfer bitmap and page coordinates disagree")
    if reusable.dtype != np.bool_:
        raise ValueError("Context transfer reuse bitmap must be boolean")
    missing = ~reusable
    return source[missing], destination[missing]


@dataclass(frozen=True)
class ContextDecodeReuse:
    """Same Radix Retry match as P, projected onto final active versions.

    Missing active pages are fetched from P independently of later resident
    pages. A changed key suffix always gets private pages, even at unchanged
    positions, so two Radix branches never acquire the same physical owner.
    """

    source_slots: torch.Tensor
    reusable: np.ndarray
    borrowed: np.ndarray
    copy_indices: np.ndarray
    position_pairs: torch.Tensor

    @classmethod
    def build(cls, req, plan, *, window=None):
        raw = plan.decode.raw_indices
        matched = len(req.prefix_indices)
        reusable = raw < matched
        resident = req.context_resident
        if resident is not None:
            reusable[reusable] &= resident.numpy()[raw[reusable]]
        if window is not None:
            # Match native PD: receive the SWA tail into fresh owners. The
            # transport sends that state as one payload, so borrowing any of
            # its destinations would overwrite a shared cache page.
            reusable[plan.swa_start(window):] = False
        source = req.context_source_positions
        if source is None:
            source = torch.arange(matched, dtype=torch.int32)
        borrowed = reusable.copy()
        indices = np.flatnonzero(reusable)
        old = source.numpy()[raw[indices]]
        new = plan.decode.positions[indices]
        copy = (old != new) | (raw[indices] >= req.context_exact_prefix_len)
        borrowed[indices[copy]] = False
        return cls(
            req.prefix_indices,
            reusable,
            borrowed,
            indices[copy],
            torch.from_numpy(np.stack((old[copy], new[copy]), axis=1).astype(np.int32)),
        )

    @property
    def allocation_count(self):
        return len(self.borrowed) - int(self.borrowed.sum())


def allocate_context_destination(req, allocator, pool, *, window=None):
    """Allocate compact Full/SWA owners, then scatter once into the raw table."""
    plan = transfer_plan(req, allocator.device)
    count = plan.active_count
    reuse = getattr(req, "context_decode_reuse", None)
    allocation_count = count if reuse is None else reuse.allocation_count
    if window is None:
        allocated = allocator.alloc(allocation_count)
        slots = allocated
        swa = None
    else:
        start = plan.swa_start(window)
        if reuse is not None and reuse.reusable[start:].any():
            raise ValueError("Context SWA transfer destinations must be private")
        allocated = allocator.alloc_context_swa_tail(allocation_count, count - start)
        slots = allocated
        swa = torch.arange(count) >= start
    if slots is None:
        raise RuntimeError(
            "Context PD destination allocation exceeds admitted capacity"
        )
    owned = torch.ones(count, dtype=torch.bool)
    if reuse is not None:
        owned = torch.from_numpy(~reuse.borrowed)
        private_raw = plan.decode.raw_indices[~reuse.borrowed]
        if len(private_raw):
            # Native SWA eviction treats cache_protected_len as already owned
            # by the tree. A freshly received tail is private even on an exact
            # cache hit; excluding it from that prefix also keeps its SWA peers
            # live until insertion adopts or frees the duplicate destinations.
            req.context_exact_prefix_len = min(
                req.context_exact_prefix_len, int(private_raw[0])
            )
        slots = torch.empty(count, dtype=torch.int64, device=allocator.device)
        slots[owned.to(device=allocator.device)] = allocated
        borrowed = torch.from_numpy(np.flatnonzero(reuse.borrowed)).to(allocator.device)
        raw = plan.decode.device_indices[borrowed].to(torch.int64)
        slots[borrowed] = reuse.source_slots[raw]
    rows = torch.full((plan.decode.prompt_length,), -1, dtype=torch.int64)
    rows[torch.from_numpy(plan.decode.raw_indices.astype(np.int64))] = torch.arange(
        count
    )
    program = req.context_recompute_program or req.context_program
    req.context_state = OccurrenceState(
        slots,
        owned,
        rows,
        rows.clone(),
        program.layout.positions,
        min(req.context_exact_prefix_len, plan.decode.prompt_length) if reuse is not None else 0,
        swa,
    )
    req.context_decode_layout = plan.decode
    req.context_prefill_started = True
    req.context_cache_published = reuse is None
    terminal = req.context_state.terminal_slots()
    pool.write((req.kv.req_pool_idx, slice(0, len(terminal))), terminal)
    req.kv.kv_allocated_len = len(terminal)
    return slots


def materialize_decode_reuse(req, model_runner):
    """COW uses the existing all-layer RoPE kernel, before metadata publication."""
    reuse = getattr(req, "context_decode_reuse", None)
    if reuse is None or not len(reuse.copy_indices):
        return
    from sglang.srt.layers.attention.context_backend import ContextModelBinding

    binding = getattr(model_runner, "context_model_binding", None)
    if binding is None:
        binding = ContextModelBinding(
            model_runner.model, model_runner.token_to_kv_pool,
            model_runner.kv_index_translator, page_size=1,
        )
        model_runner.context_model_binding = binding
    device = req.context_state.slots.device
    indices = torch.from_numpy(reuse.copy_indices).to(device)
    raw = req.context_decode_layout.device_indices[indices].to(torch.int64)
    binding.reposition_existing(
        reuse.source_slots[raw].to(torch.int32),
        req.context_state.slots[indices].to(torch.int32),
        reuse.position_pairs.to(device),
    )


def write_context_metadata(req, row):
    """Use the nine spare int32 cells in native cached_tokens metadata."""
    if req.context_program is None:
        row[7:16] = 0
        return
    plan = transfer_plan(req, req.context_state.slots.device)
    usage = req.context_usage.snapshot()
    values = (
        *plan.header(),
        usage.cached_tokens,
        usage.repos_tokens,
        usage.drop_skipped_tokens,
        usage.actual_prefill_tokens,
    )
    if any(not -(1 << 31) <= x < (1 << 31) for x in values):
        raise ValueError("Context PD metadata exceeds int32 transport range")
    row[7:16] = torch.tensor(values, dtype=row.dtype, device=row.device)


def commit_context_metadata(req, row, device):
    values = tuple(int(x) for x in row[7:16].tolist())
    if req.context_program is None:
        if any(values):
            raise ValueError("Unexpected Context PD metadata for an ordinary request")
        return
    plan = transfer_plan(req, device)
    if values[:5] != plan.header() or any(x < 0 for x in values[5:]):
        raise ValueError("Context PD event/position identity or usage mismatch")
    snapshot = ContextUsageSnapshot(*values[5:], actual_decode_tokens=0)
    if req.context_usage is not None and getattr(
        req, "pd_rebootstrap_in_progress", False
    ):
        before = req.context_usage.snapshot()
        snapshot = ContextUsageSnapshot(
            before.cached_tokens,
            before.repos_tokens,
            before.drop_skipped_tokens,
            before.actual_prefill_tokens + snapshot.actual_prefill_tokens,
            before.actual_decode_tokens,
        )
    req.context_usage = ContextUsage.from_snapshot(snapshot)
