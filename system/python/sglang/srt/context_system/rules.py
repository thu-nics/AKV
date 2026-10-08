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

# Public parse errors intentionally remain ValueError for the serving HTTP 400 contract.
# ruff: noqa: TRY004
from __future__ import annotations

import functools
import json
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

MAX_DROP_MESSAGES = 4096
MAX_SELECTORS_PER_MESSAGE = 1024


def _role(message: Mapping[str, Any]) -> str:
    role = str(message.get("role", "")).lower()
    return "tool" if role == "function" else role


def _content(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        raise ValueError(
            "text_drop requires every selected message content to be a string or null"
        )
    return content


def _matchable_content(message: Mapping[str, Any], *, field: str) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part_id, part in enumerate(content):
            if not isinstance(part, Mapping):
                raise ValueError(f"{field}.content[{part_id}] must be an object")
            part_type = part.get("type")
            value = part.get("text") if part_type in {"text", "input_text"} else None
            if part_type == "thinking":
                value = part.get("thinking")
            if not isinstance(value, str):
                raise ValueError(
                    f"{field}.content supports only text, input_text, and thinking parts"
                )
            parts.append(value)
        return "".join(parts)
    raise ValueError(f"{field}.content must be a string, text-part list, or null")


def _protocol_fingerprint(message: Mapping[str, Any]) -> str:
    role = _role(message)
    protocol: dict[str, Any] = {"role": role}
    if role == "assistant":
        protocol["reasoning_content"] = message.get("reasoning_content")
        protocol["tool_calls"] = message.get("tool_calls")
    elif role == "tool":
        protocol["name"] = message.get("name")
        protocol["tool_call_id"] = message.get("tool_call_id")
    elif message.get("name") is not None:
        protocol["name"] = message.get("name")
    return json.dumps(
        protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{field} must be a positive integer")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be a positive integer")
    if isinstance(value, str) and value.strip() != str(result):
        raise ValueError(f"{field} must be a positive integer")
    return result


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer")
    if value == 0 or value == "0":
        return 0
    return _positive_int(value, field=field)


@dataclass(frozen=True)
class _TextSelection:
    segments: tuple[str, ...]
    occurrences: tuple[int, ...]
    spans: tuple[tuple[int, int], ...]
    whole_message: bool


@dataclass(frozen=True)
class TokenDropEvents:
    """Token-level Drop events shared by tokenizer and Radix compilation."""

    event_insert_offsets: torch.Tensor
    range_offsets: torch.Tensor
    raw_ranges: torch.Tensor
    full_token_visible_until: torch.Tensor


@dataclass(frozen=True)
class DropCompileContext:
    """Tokenizer services needed by every DropRule implementation."""

    raw_messages: Sequence[Mapping[str, Any]]
    owner_ranges: Mapping[int, Sequence[tuple[int, int]]]
    provenance: Any
    normalized_message_count: int
    normalize_content: Callable[[Any], str]
    rendered_source_start: Callable[..., int]
    token_ranges_for_char_spans: Callable[..., list[tuple[int, int]]]
    canonicalize_ranges: Callable[[Sequence[tuple[int, int]]], list[tuple[int, int]]]
    position_ranges_from_ids: Callable[[list[int]], list[tuple[int, int]]]


@dataclass(frozen=True)
class MessageDropRule:
    """Drop complete chat-template message ownership ranges at named triggers."""

    drop_messages: dict[int, tuple[int, ...]]
    type: str = "message_drop"

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
    ) -> MessageDropRule:
        raw = payload.get("drop_messages")
        if not isinstance(raw, Mapping):
            raise ValueError(
                "message_drop.drop_messages must be an object of trigger-to-ID lists"
            )
        normalized: dict[int, tuple[int, ...]] = {}
        unknown = set(payload) - {"type", "drop_messages"}
        if unknown:
            raise ValueError(f"message_drop has unsupported fields: {sorted(unknown)}")
        for raw_trigger, raw_ids in raw.items():
            trigger = _nonnegative_int(raw_trigger, field="message_drop trigger")
            if trigger >= (1 << 63):
                raise ValueError(
                    "message_drop trigger is outside the signed int64 range"
                )
            if not isinstance(raw_ids, list):
                raise ValueError(
                    f"message_drop.drop_messages[{trigger}] must be a list"
                )
            ids: list[int] = []
            for raw_id in raw_ids:
                message_id = _nonnegative_int(raw_id, field="message_drop message ID")
                if message_id >= (1 << 63):
                    raise ValueError(
                        "message_drop message ID is outside the signed int64 range"
                    )
                if message_id > trigger:
                    raise ValueError(
                        f"message_drop event {trigger} cannot drop future message {message_id}"
                    )
                if trigger < len(messages) and message_id >= len(messages):
                    raise ValueError(
                        f"message_drop message ID {message_id} is outside the current message range"
                    )
                ids.append(message_id)
            normalized[trigger] = tuple(ids)
        return cls(normalized)

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "drop_messages": {
                str(trigger): list(message_ids)
                for trigger, message_ids in self.drop_messages.items()
            },
        }

    def position_events(
        self, context: DropCompileContext
    ) -> dict[int, list[tuple[int, int]]]:
        events: dict[int, list[tuple[int, int]]] = {}
        effective: set[int] = set()
        for trigger in sorted(self.drop_messages):
            if trigger >= context.normalized_message_count:
                continue
            newly_effective = set(self.drop_messages[trigger]) - effective
            effective.update(newly_effective)
            ranges = [
                item
                for message_id in sorted(newly_effective)
                for item in context.owner_ranges.get(message_id, ())
            ]
            if ranges:
                events[trigger] = ranges
        return events


@dataclass(frozen=True)
class TextDropRule:
    """Drop user-selected raw-content substrings after the latest user Prefill."""

    drop_messages: tuple[dict[str, Any], ...]
    selections: tuple[_TextSelection | None, ...]
    trigger_message_id: int
    type: str = "text_drop"

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
    ) -> TextDropRule:
        unknown_payload = set(payload) - {
            "type",
            "drop_messages",
            "_trigger_message_id",
        }
        if unknown_payload:
            raise ValueError(
                f"text_drop has unsupported fields: {sorted(unknown_payload)}"
            )
        raw_entries = payload.get("drop_messages")
        if not isinstance(raw_entries, list):
            raise ValueError(
                "text_drop.drop_messages must be a list aligned with messages"
            )
        if len(raw_entries) != len(messages):
            raise ValueError(
                "text_drop.drop_messages must have exactly the same length as messages"
            )
        if len(raw_entries) > MAX_DROP_MESSAGES:
            raise ValueError(f"text_drop supports at most {MAX_DROP_MESSAGES} messages")

        latest_user = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if _role(messages[index]) == "user"
            ),
            None,
        )
        internal_trigger = payload.get("_trigger_message_id")
        if internal_trigger is not None:
            latest_user = int(internal_trigger)
        if latest_user is None:
            raise ValueError("text_drop requires at least one user message")

        normalized_entries: list[dict[str, Any]] = []
        selections: list[_TextSelection | None] = []
        for index, (entry, message) in enumerate(
            zip(raw_entries, messages, strict=True)
        ):
            if not isinstance(entry, Mapping):
                raise ValueError(f"text_drop.drop_messages[{index}] must be an object")
            unknown = set(entry) - {"role", "content", "occurrence"}
            if unknown:
                raise ValueError(
                    f"text_drop.drop_messages[{index}] has unsupported fields: {sorted(unknown)}"
                )
            role = str(entry.get("role", "")).lower()
            expected_role = str(message.get("role", "")).lower()
            if role != expected_role:
                raise ValueError(
                    f"text_drop.drop_messages[{index}].role must match messages[{index}].role"
                )
            raw_selector = entry.get("content")
            occurrence_supplied = "occurrence" in entry
            raw_occurrence = entry.get("occurrence")

            if raw_selector is None:
                segments: list[str] = []
            elif isinstance(raw_selector, str):
                segments = [] if raw_selector == "" else [raw_selector]
            elif isinstance(raw_selector, list) and all(
                isinstance(item, str) for item in raw_selector
            ):
                if len(raw_selector) > MAX_SELECTORS_PER_MESSAGE:
                    raise ValueError(
                        f"text_drop.drop_messages[{index}] has too many content selectors"
                    )
                empty = [item == "" for item in raw_selector]
                if any(empty) and not all(empty):
                    raise ValueError(
                        f"text_drop.drop_messages[{index}].content cannot mix empty "
                        "and non-empty strings"
                    )
                segments = [] if all(empty) else list(raw_selector)
            else:
                raise ValueError(
                    f"text_drop.drop_messages[{index}].content must be null, a string, or list[str]"
                )

            if occurrence_supplied and not segments:
                raise ValueError(
                    f"text_drop.drop_messages[{index}].occurrence is invalid without content"
                )
            if not segments:
                normalized_entries.append(
                    {"role": expected_role, "content": raw_selector}
                )
                selections.append(None)
                continue

            if occurrence_supplied:
                if isinstance(raw_selector, str):
                    if isinstance(raw_occurrence, list):
                        raise ValueError(
                            f"text_drop.drop_messages[{index}].occurrence must be one "
                            "positive integer"
                        )
                    occurrences = [
                        _positive_int(
                            raw_occurrence, field=f"drop_messages[{index}].occurrence"
                        )
                    ]
                else:
                    if not isinstance(raw_occurrence, list) or len(
                        raw_occurrence
                    ) != len(segments):
                        raise ValueError(
                            f"text_drop.drop_messages[{index}].occurrence must provide "
                            "one value per content segment"
                        )
                    occurrences = [
                        _positive_int(
                            value, field=f"drop_messages[{index}].occurrence[{part}]"
                        )
                        for part, value in enumerate(raw_occurrence)
                    ]
            else:
                occurrences = [1] * len(segments)

            source = _content(message)
            matches = find_all(source, segments)
            spans: list[tuple[int, int]] = []
            for part, (segment, occurrence, candidates) in enumerate(
                zip(segments, occurrences, matches, strict=True)
            ):
                if occurrence > len(candidates):
                    raise ValueError(
                        "text_drop selector is not the requested occurrence of a substring of "
                        f"messages[{index}].content (selector {part}, occurrence {occurrence})"
                    )
                spans.append(candidates[occurrence - 1])
            covered = _merge_ranges(spans)
            whole_message = bool(source) and covered == [(0, len(source))]
            normalized = {"role": expected_role, "content": raw_selector}
            if occurrence_supplied:
                normalized["occurrence"] = raw_occurrence
            normalized_entries.append(normalized)
            selections.append(
                _TextSelection(
                    tuple(segments), tuple(occurrences), tuple(spans), whole_message
                )
            )
        return cls(tuple(normalized_entries), tuple(selections), latest_user)

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "drop_messages": [dict(entry) for entry in self.drop_messages],
            "_trigger_message_id": self.trigger_message_id,
        }

    def position_events(
        self, context: DropCompileContext
    ) -> dict[int, list[tuple[int, int]]]:
        trigger = self.trigger_message_id
        if trigger >= context.normalized_message_count:
            return {}
        ranges: list[tuple[int, int]] = []
        for raw_message_id, selection in enumerate(self.selections):
            if selection is None:
                continue
            owner = raw_message_id
            if selection.whole_message:
                ranges.extend(context.owner_ranges.get(owner, ()))
                continue
            source = context.normalize_content(
                context.raw_messages[raw_message_id].get("content")
            )
            rendered_start = context.rendered_source_start(
                context.provenance,
                owner=owner,
                source=source,
                field="content",
            )
            rendered_spans = [
                (rendered_start + start, rendered_start + end)
                for start, end in selection.spans
            ]
            ranges.extend(
                context.token_ranges_for_char_spans(
                    context.provenance,
                    owner=owner,
                    spans=rendered_spans,
                    field="text_drop content",
                    allow_empty=True,
                )
            )
        canonical = context.canonicalize_ranges(ranges)
        return {trigger: canonical} if canonical else {}


@dataclass(frozen=True)
class KeepTextDropRule:
    """Keep an ordered visible-text projection over a complete Radix history."""

    full_messages: tuple[dict[str, Any], ...]
    keep_spans: tuple[tuple[int, int] | None, ...]
    force: bool = False
    use_visible_as_full: bool = False
    fallback_reason: str | None = None
    type: str = "keep_text_drop"

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
        *,
        allow_internal: bool = False,
    ) -> KeepTextDropRule:
        internal = "_keep_spans" in payload
        allowed = (
            {"type", "force", "_keep_spans"}
            if internal and allow_internal
            else {"type", "full_messages", "force"}
        )
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(
                f"keep_text_drop has unsupported fields: {sorted(unknown)}"
            )

        force = payload.get("force", False)
        if not isinstance(force, bool):
            raise ValueError("keep_text_drop.force must be a boolean")
        if internal:
            if not allow_internal:
                raise ValueError("drop_rule contains a reserved internal field")
            raw_full: Sequence[Mapping[str, Any]] = messages
        else:
            public_full = payload.get("full_messages")
            if not isinstance(public_full, list):
                raise ValueError("keep_text_drop.full_messages must be a list")
            if not all(isinstance(message, Mapping) for message in public_full):
                raise ValueError(
                    "every keep_text_drop.full_messages entry must be an object"
                )
            raw_full = public_full
        if not raw_full:
            raise ValueError("keep_text_drop.full_messages must not be empty")
        if len(raw_full) > MAX_DROP_MESSAGES:
            raise ValueError(
                f"keep_text_drop supports at most {MAX_DROP_MESSAGES} full_messages"
            )
        full_messages = tuple(dict(message) for message in raw_full)
        for message_id, message in enumerate(full_messages):
            role = _role(message)
            if role not in {"system", "user", "assistant", "tool"}:
                raise ValueError(
                    f"keep_text_drop.full_messages[{message_id}].role is invalid"
                )
            _matchable_content(
                message, field=f"keep_text_drop.full_messages[{message_id}]"
            )

        if internal:
            raw_spans = payload.get("_keep_spans")
            if not isinstance(raw_spans, list) or len(raw_spans) != len(full_messages):
                raise ValueError(
                    "internal keep_text_drop._keep_spans must align with full_messages"
                )
            keep_spans: list[tuple[int, int] | None] = []
            for message_id, (raw_span, message) in enumerate(
                zip(raw_spans, full_messages, strict=True)
            ):
                if raw_span is None:
                    keep_spans.append(None)
                    continue
                if (
                    not isinstance(raw_span, list)
                    or len(raw_span) != 2
                    or any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in raw_span
                    )
                ):
                    raise ValueError(
                        f"internal keep_text_drop._keep_spans[{message_id}] is invalid"
                    )
                start, end = raw_span
                source = _matchable_content(
                    message, field=f"keep_text_drop.full_messages[{message_id}]"
                )
                if start < 0 or end < start or end > len(source):
                    raise ValueError(
                        f"internal keep_text_drop._keep_spans[{message_id}] is out of range"
                    )
                keep_spans.append((start, end))
            return cls(full_messages, tuple(keep_spans), force=force)

        if not messages:
            raise ValueError("keep_text_drop requires at least one visible message")
        if len(messages) > len(full_messages):
            reason = "visible messages cannot outnumber keep_text_drop.full_messages"
            if force:
                return cls(
                    tuple(dict(message) for message in messages),
                    (),
                    force=True,
                    use_visible_as_full=True,
                    fallback_reason=reason,
                )
            raise ValueError(reason)

        full_contents = [
            _matchable_content(
                message, field=f"keep_text_drop.full_messages[{message_id}]"
            )
            for message_id, message in enumerate(full_messages)
        ]
        visible_contents = [
            _matchable_content(message, field=f"messages[{message_id}]")
            for message_id, message in enumerate(messages)
        ]
        fingerprints = [
            *(_protocol_fingerprint(message) for message in full_messages),
            *(_protocol_fingerprint(message) for message in messages),
        ]
        key_ids = {
            value: key_id for key_id, value in enumerate(dict.fromkeys(fingerprints))
        }
        full_keys = [key_ids[value] for value in fingerprints[: len(full_messages)]]
        visible_keys = [key_ids[value] for value in fingerprints[len(full_messages) :]]
        try:
            matches = find_ordered_latest(
                full_contents,
                visible_contents,
                source_keys=full_keys,
                pattern_keys=visible_keys,
            )
        except ValueError as exc:
            reason = str(exc)
            if force:
                return cls(
                    tuple(dict(message) for message in messages),
                    (),
                    force=True,
                    use_visible_as_full=True,
                    fallback_reason=reason,
                )
            raise ValueError(f"keep_text_drop projection failed: {reason}") from exc

        keep_spans: list[tuple[int, int] | None] = [None] * len(full_messages)
        for source_id, start, end in matches:
            keep_spans[source_id] = (start, end)
        return cls(full_messages, tuple(keep_spans), force=force)

    def to_wire(self) -> dict[str, Any]:
        if self.use_visible_as_full:
            raise RuntimeError(
                "forced keep_text_drop fallback has no Drop wire payload"
            )
        return {
            "type": self.type,
            "force": self.force,
            "_keep_spans": [
                list(span) if span is not None else None for span in self.keep_spans
            ],
        }

    def position_events(
        self, context: DropCompileContext
    ) -> dict[int, list[tuple[int, int]]]:
        ranges: list[tuple[int, int]] = []
        for raw_message_id, keep_span in enumerate(self.keep_spans):
            owner = raw_message_id
            if keep_span is None:
                ranges.extend(context.owner_ranges.get(owner, ()))
                continue

            source = context.normalize_content(
                context.raw_messages[raw_message_id].get("content")
            )
            if not source:
                continue
            rendered_start = context.rendered_source_start(
                context.provenance,
                owner=owner,
                source=source,
                field="keep_text_drop content",
                prefer_latest=True,
            )
            full_span = (rendered_start, rendered_start + len(source))
            selected_span = (
                rendered_start + keep_span[0],
                rendered_start + keep_span[1],
            )
            content_ranges = context.token_ranges_for_char_spans(
                context.provenance,
                owner=owner,
                spans=[full_span],
                field="keep_text_drop full content",
                boundary_mode="contained",
                allow_empty=True,
            )
            kept_ranges = context.token_ranges_for_char_spans(
                context.provenance,
                owner=owner,
                spans=[selected_span],
                field="keep_text_drop selected content",
                boundary_mode="overlap",
                allow_empty=keep_span[0] == keep_span[1],
            )
            content_ids = {
                token_id
                for start, end in content_ranges
                for token_id in range(start, end)
            }
            kept_ids = {
                token_id for start, end in kept_ranges for token_id in range(start, end)
            }
            ranges.extend(
                context.position_ranges_from_ids(list(content_ids - kept_ids))
            )

        trigger = max(context.normalized_message_count - 1, 0)
        canonical = context.canonicalize_ranges(ranges)
        return {trigger: canonical} if canonical else {}


@dataclass(frozen=True)
class ThinkingDropRule:
    """Retain structured assistant thinking in the full stream, then drop its KV."""

    thinking_by_message: dict[int, str]
    type: str = "thinking_drop"

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
    ) -> ThinkingDropRule:
        allowed = {"type"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"thinking_drop has unsupported fields: {sorted(unknown)}")
        thinking: dict[int, str] = {}
        for index, message in enumerate(messages):
            if _role(message) != "assistant":
                continue
            structured = message.get("reasoning_content")
            if structured is not None and not isinstance(structured, str):
                raise ValueError(
                    f"messages[{index}].reasoning_content must be a string or null"
                )
            structured = structured or None
            inline = _extract_leading_think(_content(message), message_id=index)
            if structured is not None and inline is not None:
                raise ValueError(
                    f"messages[{index}] cannot provide both reasoning_content and "
                    "a leading <think> block"
                )
            source = structured if structured is not None else inline
            if source:
                thinking[index] = source
        if not thinking:
            raise ValueError(
                "thinking_drop requires at least one assistant reasoning_content or "
                "leading <think> block"
            )
        return cls(thinking)

    def to_wire(self) -> dict[str, Any]:
        return {"type": self.type}

    def position_events(
        self, context: DropCompileContext
    ) -> dict[int, list[tuple[int, int]]]:
        events: dict[int, list[tuple[int, int]]] = {}
        for raw_message_id, source in self.thinking_by_message.items():
            owner = raw_message_id
            if owner >= context.normalized_message_count:
                continue
            rendered_start = context.rendered_source_start(
                context.provenance,
                owner=owner,
                source=source,
                field="thinking",
            )
            ranges = context.token_ranges_for_char_spans(
                context.provenance,
                owner=owner,
                spans=[(rendered_start, rendered_start + len(source))],
                field="thinking",
            )
            events[owner] = context.canonicalize_ranges(ranges)
        return events


DropRule = MessageDropRule | TextDropRule | KeepTextDropRule | ThinkingDropRule


def parse_drop_rule(
    payload: Mapping[str, Any] | None,
    messages: Sequence[Mapping[str, Any]],
    *,
    legacy_drop_message: Mapping[Any, Any] | None = None,
    allow_internal: bool = False,
) -> DropRule | None:
    if payload is not None and legacy_drop_message is not None:
        raise ValueError(
            "drop_rule and legacy drop_message cannot be supplied together"
        )
    if payload is None:
        if legacy_drop_message is None:
            return None
        payload = {"type": "message_drop", "drop_messages": legacy_drop_message}
    if not isinstance(payload, Mapping):
        raise ValueError("drop_rule must be an object")
    if any(str(field).startswith("_") for field in payload) and not allow_internal:
        raise ValueError("drop_rule contains a reserved internal field")
    rule_type = payload.get("type")
    if rule_type == "message_drop":
        return MessageDropRule.from_payload(payload, messages)
    if rule_type == "text_drop":
        return TextDropRule.from_payload(payload, messages)
    if rule_type == "keep_text_drop":
        return KeepTextDropRule.from_payload(
            payload, messages, allow_internal=allow_internal
        )
    if rule_type == "thinking_drop":
        return ThinkingDropRule.from_payload(payload, messages)
    raise ValueError(
        "drop_rule.type must be one of: message_drop, text_drop, "
        "keep_text_drop, thinking_drop"
    )


def _extract_leading_think(content: str, *, message_id: int) -> str | None:
    has_tag = "<think>" in content or "</think>" in content
    if not has_tag:
        return None
    if not content.startswith("<think>"):
        raise ValueError(
            f"messages[{message_id}] has a non-leading or malformed <think> block"
        )
    close = content.find("</think>", len("<think>"))
    if close < 0:
        raise ValueError(f"messages[{message_id}] has an unclosed <think> block")
    reasoning = content[len("<think>") : close]
    remainder = content[close + len("</think>") :]
    if (
        "<think>" in reasoning
        or "</think>" in reasoning
        or "<think>" in remainder
        or "</think>" in remainder
    ):
        raise ValueError(
            f"messages[{message_id}] has nested or multiple <think> blocks"
        )
    return reasoning


def _merge_ranges(ranges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


# Native UTF-8 selection. Fail explicitly on compiler failure in production;
# reference fallback is available only when the caller opts in.


MAX_TEXT_BYTES = 16 * 1024 * 1024
MAX_PATTERNS = 4096
MAX_PATTERN_BYTES = 1024 * 1024
MAX_MATCHES = 1_000_000


@functools.cache
def _load_text_match_module():
    from sglang.kernels.ops.attention.context_plan import load_context_text_match

    return load_context_text_match()


def _validate_inputs(text: str, patterns: Sequence[str], max_matches: int) -> None:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if len(patterns) > MAX_PATTERNS:
        raise ValueError(f"too many text-drop patterns; maximum is {MAX_PATTERNS}")
    text_bytes = text.encode("utf-8")
    if len(text_bytes) > MAX_TEXT_BYTES:
        raise ValueError(
            f"text is too large for text matching; maximum is {MAX_TEXT_BYTES} bytes"
        )
    total_pattern_bytes = 0
    for pattern in patterns:
        if not isinstance(pattern, str):
            raise TypeError("every pattern must be a string")
        if not pattern:
            raise ValueError("empty patterns are not matchable")
        total_pattern_bytes += len(pattern.encode("utf-8"))
    if total_pattern_bytes > MAX_PATTERN_BYTES:
        raise ValueError(
            "text-drop patterns are too large; "
            f"maximum combined UTF-8 size is {MAX_PATTERN_BYTES} bytes"
        )
    if max_matches <= 0 or max_matches > MAX_MATCHES:
        raise ValueError(f"max_matches must be in [1, {MAX_MATCHES}]")


def _byte_to_char_boundaries(text: str) -> dict[int, int]:
    boundaries = {0: 0}
    byte_pos = 0
    for char_pos, char in enumerate(text, start=1):
        byte_pos += len(char.encode("utf-8"))
        boundaries[byte_pos] = char_pos
    return boundaries


def _decode_byte_matches(
    text: str,
    pattern_count: int,
    matches: Sequence[tuple[int, int, int]],
) -> list[list[tuple[int, int]]]:
    boundaries = _byte_to_char_boundaries(text)
    result: list[list[tuple[int, int]]] = [[] for _ in range(pattern_count)]
    for pattern_id, byte_start, byte_end in matches:
        if byte_start not in boundaries or byte_end not in boundaries:
            raise RuntimeError("text matcher returned a span inside a UTF-8 code point")
        result[pattern_id].append((boundaries[byte_start], boundaries[byte_end]))
    for spans in result:
        spans.sort()
    return result


def find_all_reference(
    text: str,
    patterns: Sequence[str],
    *,
    max_matches: int = MAX_MATCHES,
) -> list[list[tuple[int, int]]]:
    """Aho-Corasick reference implementation with overlapping matches."""

    patterns = list(patterns)
    _validate_inputs(text, patterns, max_matches)
    if not patterns:
        return []

    encoded_patterns = [pattern.encode("utf-8") for pattern in patterns]
    transitions: list[dict[int, int]] = [{}]
    failure = [0]
    output: list[list[int]] = [[]]
    for pattern_id, pattern in enumerate(encoded_patterns):
        node = 0
        for byte in pattern:
            next_node = transitions[node].get(byte)
            if next_node is None:
                next_node = len(transitions)
                transitions[node][byte] = next_node
                transitions.append({})
                failure.append(0)
                output.append([])
            node = next_node
        output[node].append(pattern_id)

    queue: deque[int] = deque(transitions[0].values())
    while queue:
        node = queue.popleft()
        for byte, child in transitions[node].items():
            queue.append(child)
            fallback = failure[node]
            while fallback and byte not in transitions[fallback]:
                fallback = failure[fallback]
            failure[child] = transitions[fallback].get(byte, 0)
            output[child].extend(output[failure[child]])

    raw_matches: list[tuple[int, int, int]] = []
    node = 0
    for byte_end, byte in enumerate(text.encode("utf-8"), start=1):
        while node and byte not in transitions[node]:
            node = failure[node]
        node = transitions[node].get(byte, 0)
        for pattern_id in output[node]:
            byte_start = byte_end - len(encoded_patterns[pattern_id])
            raw_matches.append((pattern_id, byte_start, byte_end))
            if len(raw_matches) > max_matches:
                raise ValueError(
                    f"text matching exceeds the maximum of {max_matches} matches"
                )
    return _decode_byte_matches(text, len(patterns), raw_matches)


def find_all(
    text: str,
    patterns: Sequence[str],
    *,
    max_matches: int = MAX_MATCHES,
    allow_fallback: bool = False,
) -> list[list[tuple[int, int]]]:
    """Find all overlapping UTF-8 pattern occurrences in O(input + patterns + output)."""

    patterns = list(patterns)
    _validate_inputs(text, patterns, max_matches)
    if not patterns:
        return []

    source_bytes = text.encode("utf-8")
    pattern_bytes = [pattern.encode("utf-8") for pattern in patterns]
    flat_patterns = b"".join(pattern_bytes)
    offsets = [0]
    for pattern in pattern_bytes:
        offsets.append(offsets[-1] + len(pattern))

    source = _bytes_to_int_tensor(source_bytes)
    flat = _bytes_to_int_tensor(flat_patterns)
    pattern_offsets = torch.tensor(offsets, dtype=torch.int64, device="cpu")
    capacity = min(max_matches, max(1, len(source_bytes) * len(patterns)))
    output = torch.empty((capacity, 3), dtype=torch.int64, device="cpu")
    try:
        count = int(
            _load_text_match_module().aho_find_all(
                source, flat, pattern_offsets, output, int(capacity)
            )
        )
    except Exception:
        if not allow_fallback:
            raise
        return find_all_reference(text, patterns, max_matches=max_matches)
    if count < 0:
        raise ValueError(f"text matching exceeds the maximum of {max_matches} matches")
    raw = [tuple(map(int, row)) for row in output[:count].tolist()]
    return _decode_byte_matches(text, len(patterns), raw)


def _validate_ordered_inputs(
    sources: Sequence[str],
    patterns: Sequence[str],
    source_keys: Sequence[int],
    pattern_keys: Sequence[int],
) -> tuple[list[bytes], list[bytes]]:
    if len(sources) != len(source_keys):
        raise ValueError("source_keys must provide one key per source")
    if len(patterns) != len(pattern_keys):
        raise ValueError("pattern_keys must provide one key per pattern")
    if len(sources) > MAX_PATTERNS or len(patterns) > MAX_PATTERNS:
        raise ValueError(
            f"ordered text matching supports at most {MAX_PATTERNS} entries"
        )
    if not all(isinstance(source, str) for source in sources):
        raise TypeError("every source must be a string")
    if not all(isinstance(pattern, str) for pattern in patterns):
        raise TypeError("every pattern must be a string")
    encoded_sources = [source.encode("utf-8") for source in sources]
    encoded_patterns = [pattern.encode("utf-8") for pattern in patterns]
    if sum(map(len, encoded_sources)) > MAX_TEXT_BYTES:
        raise ValueError(
            f"ordered text sources are too large; maximum is {MAX_TEXT_BYTES} bytes"
        )
    if sum(map(len, encoded_patterns)) > MAX_PATTERN_BYTES:
        raise ValueError(
            "ordered text patterns are too large; "
            f"maximum combined UTF-8 size is {MAX_PATTERN_BYTES} bytes"
        )
    if not all(
        isinstance(key, int) and not isinstance(key, bool) for key in source_keys
    ):
        raise TypeError("every source key must be an integer")
    if not all(
        isinstance(key, int) and not isinstance(key, bool) for key in pattern_keys
    ):
        raise TypeError("every pattern key must be an integer")
    return encoded_sources, encoded_patterns


def _bytes_to_int_tensor(value: bytes) -> torch.Tensor:
    if not value:
        return torch.empty(0, dtype=torch.int32, device="cpu")
    return torch.frombuffer(bytearray(value), dtype=torch.uint8).to(dtype=torch.int32)


def find_ordered_latest_reference(
    sources: Sequence[str],
    patterns: Sequence[str],
    *,
    source_keys: Sequence[int],
    pattern_keys: Sequence[int],
) -> list[tuple[int, int, int]]:
    """Match ordered patterns to distinct sources, preferring the latest valid spans."""

    sources = list(sources)
    patterns = list(patterns)
    source_keys = list(source_keys)
    pattern_keys = list(pattern_keys)
    _validate_ordered_inputs(sources, patterns, source_keys, pattern_keys)

    result: list[tuple[int, int, int] | None] = [None] * len(patterns)
    source_id = len(sources) - 1
    for pattern_id in range(len(patterns) - 1, -1, -1):
        pattern = patterns[pattern_id]
        while source_id >= 0:
            if source_keys[source_id] == pattern_keys[pattern_id]:
                start = sources[source_id].rfind(pattern)
                if start >= 0:
                    result[pattern_id] = (source_id, start, start + len(pattern))
                    source_id -= 1
                    break
            source_id -= 1
        if result[pattern_id] is None:
            raise ValueError(
                f"ordered text pattern {pattern_id} has no compatible match in full_messages"
            )
    return [match for match in result if match is not None]


def find_ordered_latest(
    sources: Sequence[str],
    patterns: Sequence[str],
    *,
    source_keys: Sequence[int],
    pattern_keys: Sequence[int],
    allow_fallback: bool = False,
) -> list[tuple[int, int, int]]:
    """Linear right-to-left ordered matching with a CPU AOT KMP implementation."""

    sources = list(sources)
    patterns = list(patterns)
    source_keys = list(source_keys)
    pattern_keys = list(pattern_keys)
    source_bytes, pattern_bytes = _validate_ordered_inputs(
        sources, patterns, source_keys, pattern_keys
    )
    if not patterns:
        return []

    flat_sources = b"".join(source_bytes)
    flat_patterns = b"".join(pattern_bytes)
    source_offsets = [0]
    pattern_offsets = [0]
    for source in source_bytes:
        source_offsets.append(source_offsets[-1] + len(source))
    for pattern in pattern_bytes:
        pattern_offsets.append(pattern_offsets[-1] + len(pattern))

    source = _bytes_to_int_tensor(flat_sources)
    pattern = _bytes_to_int_tensor(flat_patterns)
    source_offset_tensor = torch.tensor(source_offsets, dtype=torch.int64, device="cpu")
    pattern_offset_tensor = torch.tensor(
        pattern_offsets, dtype=torch.int64, device="cpu"
    )
    source_key_tensor = torch.tensor(source_keys, dtype=torch.int64, device="cpu")
    pattern_key_tensor = torch.tensor(pattern_keys, dtype=torch.int64, device="cpu")
    output = torch.empty((len(patterns), 3), dtype=torch.int64, device="cpu")
    try:
        missing = int(
            _load_text_match_module().ordered_latest_find(
                source,
                source_offset_tensor,
                source_key_tensor,
                pattern,
                pattern_offset_tensor,
                pattern_key_tensor,
                output,
            )
        )
    except Exception:
        if not allow_fallback:
            raise
        return find_ordered_latest_reference(
            sources,
            patterns,
            source_keys=source_keys,
            pattern_keys=pattern_keys,
        )
    if missing >= 0:
        raise ValueError(
            f"ordered text pattern {missing} has no compatible match in full_messages"
        )

    byte_boundaries: dict[int, dict[int, int]] = {}
    result: list[tuple[int, int, int]] = []
    for source_id, byte_start, byte_end in output.tolist():
        source_id = int(source_id)
        byte_start = int(byte_start)
        byte_end = int(byte_end)
        boundaries = byte_boundaries.setdefault(
            source_id, _byte_to_char_boundaries(sources[source_id])
        )
        if byte_start not in boundaries or byte_end not in boundaries:
            raise RuntimeError(
                "ordered text matcher returned a span inside a UTF-8 code point"
            )
        result.append((source_id, boundaries[byte_start], boundaries[byte_end]))
    return result
