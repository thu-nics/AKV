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

import time
from array import array
from bisect import bisect_left
from dataclasses import dataclass, replace

import torch

TOKEN_KIND = 0
DELTA_KIND = 1
REPOSITION_KIND = 2


@dataclass(frozen=True)
class ContextLayout:
    drop_insert_offsets: torch.Tensor
    drop_range_offsets: torch.Tensor
    drop_ranges: torch.Tensor
    records: torch.Tensor
    virtual_mask: torch.Tensor
    key_to_token: torch.Tensor
    token_to_key: torch.Tensor
    positions: torch.Tensor
    keep_mask: torch.Tensor
    materialized_stage: torch.Tensor
    birth_positions: torch.Tensor
    birth_stages: torch.Tensor
    transition_offsets: torch.Tensor
    transition_raw_tokens: torch.Tensor
    transition_old_positions: torch.Tensor
    transition_new_positions: torch.Tensor
    effective_reposition_stages: torch.Tensor
    drop_event_to_key: torch.Tensor
    effective_repositions: torch.Tensor
    ignored_repositions: torch.Tensor
    next_position: int
    compile_ns: int

    @property
    def keys(self) -> torch.Tensor:
        return self.records


def append_generated_layout(layout: ContextLayout, token_ids) -> ContextLayout:
    """Extend a retracted prompt at its final state, without replaying events.

    This runs once per recomputation attempt, never on the decode hot path.
    Existing transitions and trailing event records keep their identity; output
    tokens are born at the original prompt's final Reposition stage.
    """
    tokens = torch.as_tensor(token_ids, dtype=torch.int64, device="cpu")
    if tokens.ndim != 1 or bool(torch.any((tokens < 0) | (tokens >= 2**31))):
        raise ValueError("Generated Context IDs must be an int32 token vector")
    count = len(tokens)
    if not count:
        return layout
    if layout.next_position + count > 2**31 - 1:
        raise ValueError("Generated Context positions exceed int32 range")
    n, record_count = len(layout.positions), len(layout.records)
    positions = torch.arange(
        layout.next_position, layout.next_position + count, dtype=torch.int32
    )
    stages = torch.full((count,), len(layout.transition_offsets) - 1, dtype=torch.int32)
    records = torch.stack((torch.zeros_like(positions), tokens.int(), positions), 1)
    return replace(
        layout,
        records=torch.cat((layout.records, records)),
        virtual_mask=torch.cat(
            (layout.virtual_mask, torch.zeros(count, dtype=torch.bool))
        ),
        key_to_token=torch.cat((layout.key_to_token, torch.arange(n, n + count))),
        token_to_key=torch.cat(
            (layout.token_to_key, torch.arange(record_count, record_count + count))
        ),
        positions=torch.cat((layout.positions, positions)),
        keep_mask=torch.cat((layout.keep_mask, torch.ones(count, dtype=torch.bool))),
        materialized_stage=torch.cat((layout.materialized_stage, stages)),
        birth_positions=torch.cat((layout.birth_positions, positions)),
        birth_stages=torch.cat((layout.birth_stages, stages)),
        next_position=layout.next_position + count,
    )


def _load_module():
    from sglang.kernels.ops.attention.context_plan import load_context_plan

    return load_context_plan()


def compile_context_layout(
    token_ids: torch.Tensor,
    drop_insert_offsets: torch.Tensor,
    drop_range_offsets: torch.Tensor,
    drop_ranges: torch.Tensor,
    reposition_raw_boundaries: torch.Tensor,
    reposition_insert_offsets: torch.Tensor,
) -> ContextLayout:
    """Compile raw-token events, preserving mini-sglang final-position semantics.

    Drop offsets are insertion points (the next query); intervals are half-open
    raw-token ranges. Reposition boundaries are the preceding raw token. Drop
    runs before Reposition at a shared insertion point. Physical KV/page IDs
    never occur in this immutable request program.

    No-feature requests must bypass this compiler at the serving boundary.
    Tensor fields are read-only by convention, including across TP processes.
    """

    vectors = {
        "token_ids": token_ids,
        "drop_insert_offsets": drop_insert_offsets,
        "drop_range_offsets": drop_range_offsets,
        "drop_ranges": drop_ranges,
        "reposition_raw_boundaries": reposition_raw_boundaries,
        "reposition_insert_offsets": reposition_insert_offsets,
    }
    for name, value in vectors.items():
        if (
            value.device.type != "cpu"
            or value.ndim != 1
            or value.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(f"{name} must be a one-dimensional CPU integer tensor.")
        # Validate before narrowing: a wrapped offset can address unrelated KV.
        if (
            value.dtype == torch.int64
            and len(value)
            and (int(value.min()) < -(2**31) or int(value.max()) >= 2**31)
        ):
            raise ValueError(f"{name} exceeds the signed int32 range.")
    token_ids = token_ids.contiguous()
    drop_insert_offsets = drop_insert_offsets.to(torch.int32).contiguous()
    drop_range_offsets = drop_range_offsets.to(torch.int32).contiguous()
    drop_ranges = drop_ranges.to(torch.int32).contiguous()
    reposition_raw_boundaries = reposition_raw_boundaries.to(torch.int32).contiguous()
    reposition_insert_offsets = reposition_insert_offsets.to(torch.int32).contiguous()

    compile_started_ns = time.perf_counter_ns()
    token_count = len(token_ids)
    reposition_count = len(reposition_raw_boundaries)
    transition_counts = torch.zeros(reposition_count, dtype=torch.int32, device="cpu")
    count_status = torch.zeros(2, dtype=torch.int64, device="cpu")
    _load_module().count_radix_reposition_transitions(
        token_count,
        drop_insert_offsets,
        drop_range_offsets,
        drop_ranges,
        reposition_raw_boundaries,
        reposition_insert_offsets,
        transition_counts,
        count_status,
    )
    if int(count_status[0]) == 1:
        boundary = int(count_status[1])
        raise ValueError(f"Reposition at raw boundary {boundary} has no active tokens.")
    transition_count = int(transition_counts.sum().item())
    range_count = len(drop_ranges) // 2
    capacity = token_count + range_count + reposition_count
    records = torch.empty((capacity, 3), dtype=torch.int32, device="cpu")
    virtual_mask = torch.empty(capacity, dtype=torch.bool, device="cpu")
    key_to_token = torch.empty(capacity, dtype=torch.int64, device="cpu")
    token_to_key = torch.empty(token_count, dtype=torch.int64, device="cpu")
    positions = torch.empty(token_count, dtype=torch.int32, device="cpu")
    keep_mask = torch.empty(token_count, dtype=torch.bool, device="cpu")
    materialized_stage = torch.empty(token_count, dtype=torch.int32, device="cpu")
    birth_positions = torch.empty(token_count, dtype=torch.int32, device="cpu")
    birth_stages = torch.empty(token_count, dtype=torch.int32, device="cpu")
    transition_offsets = torch.empty(
        reposition_count + 1, dtype=torch.int32, device="cpu"
    )
    transition_raw_tokens = torch.empty(
        transition_count, dtype=torch.int32, device="cpu"
    )
    transition_old_positions = torch.empty(
        transition_count, dtype=torch.int32, device="cpu"
    )
    transition_new_positions = torch.empty(
        transition_count, dtype=torch.int32, device="cpu"
    )
    effective_reposition_stages = torch.full(
        (reposition_count,), -1, dtype=torch.int32, device="cpu"
    )
    drop_event_to_key = torch.full(
        (len(drop_insert_offsets),), -1, dtype=torch.int64, device="cpu"
    )
    effective = torch.zeros(reposition_count, dtype=torch.bool, device="cpu")
    ignored = torch.zeros(reposition_count, dtype=torch.bool, device="cpu")
    status = torch.zeros(5, dtype=torch.int64, device="cpu")

    _load_module().compile_radix_reposition_layout(
        token_ids,
        drop_insert_offsets,
        drop_range_offsets,
        drop_ranges,
        reposition_raw_boundaries,
        reposition_insert_offsets,
        records,
        virtual_mask,
        key_to_token,
        token_to_key,
        positions,
        keep_mask,
        materialized_stage,
        birth_positions,
        birth_stages,
        transition_offsets,
        transition_raw_tokens,
        transition_old_positions,
        transition_new_positions,
        effective_reposition_stages,
        drop_event_to_key,
        effective,
        ignored,
        status,
    )

    if int(status[0]) == 1:
        boundary = int(status[4])
        raise ValueError(f"Reposition at raw boundary {boundary} has no active tokens.")
    if int(status[0]) == 2:
        token_id = int(status[4])
        raise ValueError(
            f"Reposition Radix token ID {token_id} is outside the non-negative int32 range."
        )

    key_len = int(status[1])
    effective_stage_count = int(status[2])
    return ContextLayout(
        drop_insert_offsets=drop_insert_offsets,
        drop_range_offsets=drop_range_offsets,
        drop_ranges=drop_ranges,
        records=records[:key_len],
        virtual_mask=virtual_mask[:key_len],
        key_to_token=key_to_token[:key_len],
        token_to_key=token_to_key,
        positions=positions,
        keep_mask=keep_mask,
        materialized_stage=materialized_stage,
        birth_positions=birth_positions,
        birth_stages=birth_stages,
        transition_offsets=transition_offsets[: effective_stage_count + 1],
        transition_raw_tokens=transition_raw_tokens,
        transition_old_positions=transition_old_positions,
        transition_new_positions=transition_new_positions,
        effective_reposition_stages=effective_reposition_stages,
        drop_event_to_key=drop_event_to_key,
        effective_repositions=effective,
        ignored_repositions=ignored,
        next_position=int(status[3]),
        compile_ns=time.perf_counter_ns() - compile_started_ns,
    )


def _load_text_match_module():
    from sglang.kernels.ops.attention.context_plan import load_context_text_match

    return load_context_text_match()


def prewarm_context_layout() -> None:
    """Warm both CPU JIT modules; no model or CUDA state is initialized."""
    _load_text_match_module()
    empty = torch.empty(0, dtype=torch.int32, device="cpu")
    compile_context_layout(
        torch.tensor([0], dtype=torch.int32, device="cpu"),
        empty,
        torch.zeros(1, dtype=torch.int32, device="cpu"),
        empty,
        empty,
        empty,
    )


@dataclass(frozen=True)
class ContextKeyData:
    """Shared CPU record storage for native, real-token-sized Radix edges.

    A unit owns virtual records immediately before its TOKEN record. Tail events
    have no KV and remain in the request program; they become leading records of
    the next real token when that token is appended. Final TOKEN positions still
    distinguish tail Reposition versions. Slices share append-only storage; published records are never rewritten.
    """

    records: array
    token_to_record: array
    special_tokens: array
    retry_records: array
    event_tokens: array
    positions: array

    @classmethod
    def from_layout(cls, layout: ContextLayout) -> ContextKeyData:
        # Bulk copies avoid boxing three integers for every raw token. The native
        # compiler's outputs are CPU int32 / int64 and already range checked.
        records = array("i")
        records.frombytes(layout.records.numpy().tobytes())
        token_map = array("q")
        token_map.frombytes(layout.token_to_key.numpy().tobytes())
        ids = layout.token_to_key
        raw = torch.arange(len(ids), dtype=torch.int64)
        preceding = torch.cat((torch.tensor([-1]), ids[:-1])) if len(ids) else ids
        special = (ids != preceding + 1) | (layout.positions != raw)
        special_tokens = array("q")
        special_tokens.frombytes(torch.nonzero(special).flatten().numpy().tobytes())
        retry = layout.records.clone()
        retry[ids, 2] = 0
        retry_records = array("i")
        retry_records.frombytes(retry.numpy().tobytes())
        event_tokens = array("q")
        event_tokens.frombytes(
            torch.nonzero(ids != preceding + 1).flatten().numpy().tobytes()
        )
        positions = array("i")
        positions.frombytes(layout.positions.numpy().tobytes())
        return cls(
            records, token_map, special_tokens, retry_records, event_tokens, positions
        )

    def append_tokens(self, token_ids: array, *, next_position: int) -> None:
        """Append decode identity without copying or recompiling the prompt.

        A trailing Drop is already in records and becomes the leading event of
        the first appended token. Existing token spans and tree slices remain
        unchanged. Validate the complete suffix before mutating shared storage.
        """
        count = len(token_ids)
        if not count:
            return
        if (
            next_position < 0
            or next_position + count > 2**31
            or any(token < 0 or token >= 2**31 for token in token_ids)
        ):
            raise ValueError("Context decode tokens/positions exceed int32 range")
        for offset, token in enumerate(token_ids):
            raw = len(self.token_to_record)
            record = len(self.records) // 3
            preceding = self.token_to_record[-1] + 1 if raw else 0
            position = next_position + offset
            has_event = record != preceding
            if has_event:
                self.event_tokens.append(raw)
            if has_event or position != raw:
                self.special_tokens.append(raw)
            self.records.extend((TOKEN_KIND, token, position))
            self.retry_records.extend((TOKEN_KIND, token, 0))
            self.token_to_record.append(record)
            self.positions.append(position)

    def record_start(self, raw: int) -> int:
        return 0 if raw == 0 else self.token_to_record[raw - 1] + 1

    def record_span(self, start: int, count: int) -> tuple[int, int]:
        first = self.record_start(start)
        last = self.token_to_record[start + count - 1] + 1 if count else first
        return first, last

    def plain_prefix(self, start: int, count: int, *, retry: bool = False) -> int:
        boundaries = self.event_tokens if retry else self.special_tokens
        index = bisect_left(boundaries, start)
        if index == len(boundaries):
            return count
        return min(count, boundaries[index] - start)

    def match(
        self,
        other: ContextKeyData,
        start: int,
        offset: int,
        count: int,
        *,
        retry: bool = False,
    ) -> int:
        """Exact structured LCP, returned in real tokens (never virtual slots)."""
        if not count:
            return 0
        a, ae = self.record_span(start, count)
        b, be = other.record_span(offset, count)
        n = min(ae - a, be - b)
        left = self.retry_records if retry else self.records
        right = other.retry_records if retry else other.records
        lo, step = 0, 1
        while lo < n:
            hi = min(lo + step, n)
            if left[3 * (a + lo) : 3 * (a + hi)] != right[3 * (b + lo) : 3 * (b + hi)]:
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if (
                        left[3 * (a + lo) : 3 * (a + mid)]
                        == right[3 * (b + lo) : 3 * (b + mid)]
                    ):
                        lo = mid
                    else:
                        hi = mid
                break
            lo = hi
            step *= 2
        # Only completely equal TOKEN records contribute KV. A mismatched event
        # cannot accidentally count the following token as a cache hit.
        return min(count, bisect_left(self.token_to_record, a + lo) - start)

    def child_records(self, start: int, count: int) -> tuple[int, ...]:
        begin, end = self.record_span(start, count)
        return tuple(self.records[3 * begin : 3 * end])
