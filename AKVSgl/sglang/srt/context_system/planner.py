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

from bisect import bisect_right
from dataclasses import dataclass, fields
from typing import Any

import torch

from .ir import ContextLayout, compile_context_layout
from .provenance import TemplateTokenProvenance
from .rules import (
    DropCompileContext,
    DropRule,
    TokenDropEvents,
    _matchable_content,
    find_all,
)


@dataclass(frozen=True)
class TokenRepositionEvents:
    raw_boundaries: torch.Tensor
    insert_offsets: torch.Tensor


def resolve_reposition_token_boundaries(
    reposition_ids: list[int] | None,
    owner_ranges: dict[int, list[tuple[int, int]]],
    public_to_normalized_owner: dict[int, int],
) -> TokenRepositionEvents:
    """Translate public message IDs into exact raw-token boundaries."""

    raw_ids = reposition_ids or []
    if any(
        isinstance(raw_id, bool) or not isinstance(raw_id, int) for raw_id in raw_ids
    ):
        raise ValueError("reposition must contain integer message IDs.")
    if raw_ids != sorted(set(raw_ids)):
        raise ValueError(
            "reposition must contain strictly increasing unique message IDs."
        )
    boundaries: list[int] = []
    for raw_id in raw_ids:
        owner = public_to_normalized_owner.get(raw_id)
        if owner is None:
            raise ValueError(
                f"reposition message ID {raw_id} is outside the conversation."
            )
        ranges = owner_ranges.get(owner)
        if not ranges:
            raise ValueError(
                f"Cannot map reposition message ID {raw_id} into the token stream."
            )
        boundaries.append(max(end for _, end in ranges) - 1)
    if boundaries != sorted(set(boundaries)):
        raise ValueError(
            "Chat template ownership does not preserve Reposition boundary order."
        )
    raw_boundaries = torch.tensor(boundaries, dtype=torch.int32, device="cpu")
    return TokenRepositionEvents(raw_boundaries, raw_boundaries + 1)


class TokenEventCompiler:
    @staticmethod
    def _build_owner_position_ranges(
        owners: list[int] | torch.Tensor,
    ) -> dict[int, list[tuple[int, int]]]:
        """Return exact full-token ranges for every provenance owner."""

        ranges: dict[int, list[tuple[int, int]]] = {}
        owner_tensor = torch.as_tensor(owners, dtype=torch.int32, device="cpu")
        if owner_tensor.ndim != 1:
            raise ValueError(
                "Token provenance owners must be a one-dimensional vector."
            )
        if len(owner_tensor) == 0:
            return ranges
        changes = (
            torch.nonzero(owner_tensor[1:] != owner_tensor[:-1], as_tuple=False).view(
                -1
            )
            + 1
        )
        starts = torch.cat((torch.zeros(1, dtype=torch.int64), changes.to(torch.int64)))
        ends = torch.cat(
            (
                changes.to(torch.int64),
                torch.tensor([len(owner_tensor)], dtype=torch.int64),
            )
        )
        range_owners = owner_tensor[starts].tolist()
        for owner, start, end in zip(
            range_owners,
            starts.tolist(),
            ends.tolist(),
            strict=True,
        ):
            ranges.setdefault(owner, []).append((start, end))
        return ranges

    @staticmethod
    def _canonicalize_position_ranges(
        ranges: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if not ranges:
            return []
        normalized = sorted((int(start), int(end)) for start, end in ranges)
        merged: list[tuple[int, int]] = []
        for start, end in normalized:
            if start < 0 or end <= start:
                raise ValueError(f"Invalid token-position Drop range: [{start}, {end})")
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    @classmethod
    def _build_position_range_drop_plan(
        cls,
        event_ranges: dict[int, list[tuple[int, int]]],
        query_epoch: list[int],
    ) -> TokenDropEvents:
        """Compile rule-produced absolute ranges into the generic delta wire."""

        delta_by_pos: dict[int, list[tuple[int, int]]] = {}
        covered_ranges: list[tuple[int, int]] = []
        for event_n in sorted(event_ranges):
            canonical = cls._canonicalize_position_ranges(event_ranges[event_n])
            newly_effective = cls._subtract_position_ranges(canonical, covered_ranges)
            covered_ranges = cls._canonicalize_position_ranges(
                covered_ranges + canonical
            )
            if not newly_effective:
                continue
            insertion_pos = bisect_right(query_epoch, event_n)
            if any(end > insertion_pos for _, end in newly_effective):
                raise ValueError(
                    "A Drop event cannot hide a token before that token has been computed: "
                    f"event={event_n}, insertion_pos={insertion_pos}, ranges={newly_effective}"
                )
            delta_by_pos.setdefault(insertion_pos, []).extend(newly_effective)

        event_positions: list[int] = []
        range_offsets: list[int] = [0]
        flat_ranges: list[tuple[int, int]] = []
        visible_until = torch.full(
            (len(query_epoch),),
            torch.iinfo(torch.int32).max,
            dtype=torch.int32,
            device="cpu",
        )
        for insertion_pos in sorted(delta_by_pos):
            canonical = cls._canonicalize_position_ranges(delta_by_pos[insertion_pos])
            if not canonical:
                continue
            event_positions.append(insertion_pos)
            flat_ranges.extend(canonical)
            range_offsets.append(len(flat_ranges))
            for start, end in canonical:
                visible_until[start:end] = torch.minimum(
                    visible_until[start:end],
                    torch.tensor(insertion_pos, dtype=torch.int32, device="cpu"),
                )
        return TokenDropEvents(
            event_insert_offsets=torch.tensor(
                event_positions, dtype=torch.int32, device="cpu"
            ),
            range_offsets=torch.tensor(range_offsets, dtype=torch.int32, device="cpu"),
            raw_ranges=torch.tensor(
                flat_ranges, dtype=torch.int32, device="cpu"
            ).reshape(-1),
            full_token_visible_until=visible_until,
        )

    @staticmethod
    def _subtract_position_ranges(
        ranges: list[tuple[int, int]],
        covered: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        for start, end in ranges:
            fragments = [(start, end)]
            for cover_start, cover_end in covered:
                next_fragments: list[tuple[int, int]] = []
                for frag_start, frag_end in fragments:
                    if cover_end <= frag_start or cover_start >= frag_end:
                        next_fragments.append((frag_start, frag_end))
                        continue
                    if frag_start < cover_start:
                        next_fragments.append((frag_start, cover_start))
                    if cover_end < frag_end:
                        next_fragments.append((cover_end, frag_end))
                fragments = next_fragments
            result.extend(fragments)
        return result

    @staticmethod
    def _rendered_source_start(
        provenance: TemplateTokenProvenance,
        *,
        owner: int,
        source: str,
        field: str,
        prefer_latest: bool = False,
    ) -> int:
        candidates = []
        for start, end in find_all(provenance.rendered_text, [source])[0]:
            if all(
                candidate == owner for candidate in provenance.char_owners[start:end]
            ):
                candidates.append(start)
        if prefer_latest and candidates:
            return candidates[-1]
        if len(candidates) != 1:
            raise ValueError(
                f"Cannot map {field} for messages[{owner}] uniquely into the "
                "canonical chat template"
            )
        return candidates[0]

    @classmethod
    def _token_ranges_for_char_spans(
        cls,
        provenance: TemplateTokenProvenance,
        *,
        owner: int,
        spans: list[tuple[int, int]],
        field: str,
        boundary_mode: str = "contained",
        allow_empty: bool = False,
    ) -> list[tuple[int, int]]:
        if boundary_mode not in {"contained", "overlap"}:
            raise ValueError(f"Unsupported token boundary mode: {boundary_mode}")
        selected: list[int] = []
        for token_id, (start, end) in enumerate(provenance.offsets):
            if provenance.owners[token_id] != owner or start == end:
                continue
            if boundary_mode == "contained":
                matched = any(
                    start >= span_start and end <= span_end
                    for span_start, span_end in spans
                )
            else:
                matched = any(
                    start < span_end and end > span_start
                    for span_start, span_end in spans
                )
            if matched:
                selected.append(token_id)
        if not selected:
            if allow_empty:
                return []
            raise ValueError(
                f"{field} for messages[{owner}] contains no complete token; "
                "boundary-crossing tokens are kept"
            )
        return cls._position_ranges_from_ids(selected)

    @classmethod
    def _position_ranges_from_ids(cls, token_ids: list[int]) -> list[tuple[int, int]]:
        if not token_ids:
            return []
        token_ids = sorted(set(token_ids))
        ranges: list[tuple[int, int]] = []
        start = previous = token_ids[0]
        for token_id in token_ids[1:]:
            if token_id == previous + 1:
                previous = token_id
                continue
            ranges.append((start, previous + 1))
            start = previous = token_id
        ranges.append((start, previous + 1))
        return cls._canonicalize_position_ranges(ranges)

    @staticmethod
    def _query_epochs_from_owners(
        owners: list[int] | torch.Tensor, message_count: int
    ) -> list[int]:
        owner_tensor = torch.as_tensor(owners, dtype=torch.int32, device="cpu")
        if owner_tensor.ndim != 1:
            raise ValueError(
                "Token provenance owners must be a one-dimensional vector."
            )
        epochs = torch.clamp(owner_tensor, min=0, max=message_count)
        if len(epochs) > 1 and bool(torch.any(epochs[1:] < epochs[:-1]).item()):
            raise RuntimeError(
                "Chat template reordered messages; cannot construct monotonic Drop events."
            )
        return epochs.tolist()


@dataclass(frozen=True)
class ContextProgram:
    layout: ContextLayout
    visible_until: torch.Tensor

    def with_generated(self, token_ids) -> ContextProgram:
        from .ir import append_generated_layout

        layout = append_generated_layout(self.layout, token_ids)
        if layout is self.layout:
            return self
        # Survivors remain visible throughout generation. A trailing Drop at
        # the prompt boundary stays effective for the very first output query.
        expiry = torch.where(
            self.layout.keep_mask, torch.iinfo(torch.int32).max, self.visible_until
        )
        return ContextProgram(
            layout,
            torch.cat(
                (
                    expiry,
                    torch.full(
                        (len(token_ids),),
                        torch.iinfo(torch.int32).max,
                        dtype=torch.int32,
                    ),
                )
            ),
        )

    def to_wire(self) -> dict[str, Any]:
        """Use SGLang's native tensor-buffer IPC, without per-token Python lists.

        Compilation happens once at the tokenizer boundary. Scheduler/TP workers
        reconstruct views of the transported CPU tensors, not the event program.
        The tensors are immutable by convention throughout the request lifetime.
        """
        return {
            "version": 3,
            "layout": vars(self.layout),
            "visible_until": self.visible_until,
        }

    def to_json_wire(self) -> dict[str, Any]:
        """Bounded binary JSON for rare native PD rebootstrap HTTP requests."""
        import base64

        def encode(value):
            if not isinstance(value, torch.Tensor):
                return value
            return {
                "shape": list(value.shape),
                "data": base64.b64encode(value.numpy().tobytes()).decode("ascii"),
            }

        return {
            "version": 4,
            "layout": {
                name: encode(value) for name, value in vars(self.layout).items()
            },
            "visible_until": encode(self.visible_until),
        }

    @classmethod
    def from_wire(cls, wire: dict[str, Any], input_ids) -> ContextProgram:
        """Check the internal payload before binding it to a scheduler request."""
        if (
            isinstance(wire, dict)
            and type(wire.get("version")) is int
            and wire["version"] == 4
        ):
            import base64
            import math

            scalar = {"next_position", "compile_ns"}
            boolean = {
                "virtual_mask",
                "keep_mask",
                "effective_repositions",
                "ignored_repositions",
            }
            wide = {"key_to_token", "token_to_key", "drop_event_to_key"}

            def decode(name, value):
                if name in scalar:
                    return value
                if not isinstance(value, dict) or set(value) != {"shape", "data"}:
                    raise ValueError("Invalid Context JSON tensor")
                shape = value["shape"]
                if (
                    not isinstance(shape, list)
                    or len(shape) != (2 if name == "records" else 1)
                    or any(type(n) is not int or n < 0 for n in shape)
                    or not isinstance(value["data"], str)
                ):
                    raise ValueError("Invalid Context JSON tensor shape")
                dtype = (
                    torch.bool
                    if name in boolean
                    else torch.int64
                    if name in wide
                    else torch.int32
                )
                raw = base64.b64decode(value["data"], validate=True)
                width = torch.empty((), dtype=dtype).element_size()
                if len(raw) != math.prod(shape) * width:
                    raise ValueError("Context JSON tensor byte length mismatch")
                if dtype == torch.bool and any(b > 1 for b in raw):
                    raise ValueError("Invalid Context JSON boolean")
                return (
                    torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)
                    if raw
                    else torch.empty(shape, dtype=dtype)
                )

            if set(wire) != {"version", "layout", "visible_until"} or not isinstance(
                wire["layout"], dict
            ):
                raise ValueError("Invalid Context JSON wire")
            wire = {
                "version": 3,
                "layout": {
                    name: decode(name, value) for name, value in wire["layout"].items()
                },
                "visible_until": decode("visible_until", wire["visible_until"]),
            }
        if (
            not isinstance(wire, dict)
            or set(wire) != {"version", "layout", "visible_until"}
            or type(wire["version"]) is not int
            or wire["version"] != 3
        ):
            raise ValueError("Unsupported Context program wire version")
        data = wire["layout"]
        if not isinstance(data, dict) or set(data) != {
            field.name for field in fields(ContextLayout)
        }:
            raise ValueError("Context layout fields do not match the wire schema")
        layout = ContextLayout(**data)
        n = len(input_ids)
        scalar_names = {"next_position", "compile_ns"}
        bool_names = {
            "virtual_mask",
            "keep_mask",
            "effective_repositions",
            "ignored_repositions",
        }
        i64_names = {"key_to_token", "token_to_key", "drop_event_to_key"}
        for name, value in data.items():
            if name in scalar_names:
                if type(value) is not int or value < 0:
                    raise ValueError(f"Invalid Context scalar {name}")
                continue
            dtype = (
                torch.bool
                if name in bool_names
                else (torch.int64 if name in i64_names else torch.int32)
            )
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or (
                    value.dtype != dtype
                    or not value.is_contiguous()
                    or value.ndim != (2 if name == "records" else 1)
                )
            ):
                raise ValueError(f"Invalid Context tensor {name}")
        records = layout.records
        if records.shape[1] != 3:
            raise ValueError("Context key records must have three columns")
        for name in (
            "token_to_key",
            "positions",
            "keep_mask",
            "materialized_stage",
            "birth_positions",
            "birth_stages",
        ):
            if len(data[name]) != n:
                raise ValueError(f"Context {name} does not cover the input tokens")
        real_keys = torch.nonzero(records[:, 0] == 0, as_tuple=True)[0]
        if not torch.equal(real_keys, layout.token_to_key) or not torch.equal(
            records[real_keys, 1].to(torch.int64),
            torch.as_tensor(input_ids, dtype=torch.int64),
        ):
            raise ValueError(
                "Context program does not describe the canonical input IDs"
            )
        expiry = wire["visible_until"]
        if (
            not isinstance(expiry, torch.Tensor)
            or expiry.device.type != "cpu"
            or (
                expiry.dtype != torch.int32
                or expiry.ndim != 1
                or len(expiry) != n
                or not expiry.is_contiguous()
            )
            or bool(torch.any(expiry <= torch.arange(n)))
        ):
            raise ValueError("Invalid Context visibility metadata")
        if not torch.equal(layout.positions, records[real_keys, 2]) or bool(
            torch.any(layout.positions < 0) | torch.any(layout.birth_positions < 0)
        ):
            raise ValueError("Context positions disagree with the Radix key")
        return cls(layout, expiry)


def compile_chat_program(
    raw_messages: list[dict[str, Any]],
    provenance: TemplateTokenProvenance,
    drop_rule: DropRule | None,
    reposition: list[int] | None,
) -> ContextProgram:
    """Compile one canonical native-template result; never render a prefix again.

    The caller supplies messages in public ID order and provenance with those
    same IDs. Keep-text callers supply the full history selected by the rule.
    This function is called only after the no-feature fast-path decision.
    """
    compiler = TokenEventCompiler
    message_count = len(raw_messages)
    epochs = compiler._query_epochs_from_owners(provenance.owners, message_count)
    owner_ranges = compiler._build_owner_position_ranges(provenance.owners)
    event_ranges = {}
    if drop_rule is not None:

        context = DropCompileContext(
            raw_messages=raw_messages,
            owner_ranges=owner_ranges,
            provenance=provenance,
            normalized_message_count=message_count,
            normalize_content=lambda content: _matchable_content(
                {"content": content}, field="messages"
            ),
            rendered_source_start=compiler._rendered_source_start,
            token_ranges_for_char_spans=compiler._token_ranges_for_char_spans,
            canonicalize_ranges=compiler._canonicalize_position_ranges,
            position_ranges_from_ids=compiler._position_ranges_from_ids,
        )
        event_ranges = drop_rule.position_events(context)
    events = compiler._build_position_range_drop_plan(event_ranges, epochs)
    repos = resolve_reposition_token_boundaries(
        reposition, owner_ranges, {i: i for i in range(message_count)}
    )
    layout = compile_context_layout(
        torch.tensor(provenance.input_ids, dtype=torch.int32),
        events.event_insert_offsets,
        events.range_offsets,
        events.raw_ranges,
        repos.raw_boundaries,
        repos.insert_offsets,
    )
    return ContextProgram(layout, events.full_token_visible_until)
