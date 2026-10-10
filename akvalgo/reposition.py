"""Cumulative position compaction at completed logical-call boundaries."""

import itertools
from dataclasses import dataclass

from .types import ChatTokenCounter, PolicyRequest


@dataclass(frozen=True)
class RepositionConfig:
    reposition_after_call: int | None = None
    reposition_interval_calls: int | None = None
    reposition_trigger_tokens: int | None = None
    token_counter: ChatTokenCounter | None = None

    def __post_init__(self):
        for name in ("reposition_after_call", "reposition_interval_calls", "reposition_trigger_tokens"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or null")
        if self.reposition_interval_calls is not None and self.reposition_after_call is None:
            raise ValueError("reposition_interval_calls requires reposition_after_call")
        if self.reposition_trigger_tokens is not None:
            if self.reposition_after_call is not None:
                raise ValueError("Use either token or call-based reposition")
            if self.token_counter is None:
                raise ValueError("Token reposition requires a token counter")


def message_tokens(layout, message_id):
    if layout.message_token_counts is not None:
        return layout.message_token_counts[message_id]
    return layout.message_ends[message_id] - (layout.message_ends[message_id - 1] if message_id else 0)


def reposition_events(
    request: PolicyRequest,
    schedule,
    *,
    after_call=None,
    interval_calls=None,
    trigger_tokens=None,
    token_counter=None,
    prompt_end_message_ids_by_boundary=None,
):
    if after_call is None and trigger_tokens is None:
        return None, {}
    boundaries = tuple(request.completed_call_message_ids)
    if (
        len(boundaries) != request.call_index
        or any(
            type(b) is not int
            or not 0 <= b < len(request.messages)
            or request.messages[b]["role"] not in ("tool", "user")
            or b + 1 < len(request.messages)
            and request.messages[b + 1]["role"] == "tool"
            for b in boundaries
        )
        or any(a >= b for a, b in itertools.pairwise(boundaries))
    ):
        raise ValueError("Reposition requires complete ordered logical-call boundaries")
    if trigger_tokens is None:
        calls = (
            (after_call,)
            if interval_calls is None
            else range(after_call, request.call_index + 1, interval_calls)
        )
        return tuple(boundaries[i - 1] for i in calls if i <= request.call_index) or None, {}

    assert token_counter is not None
    controller_ids = prompt_end_message_ids_by_boundary or {}
    assert all(
        b in boundaries and end == b + 1 and request.messages[end]["role"] == "user"
        for b, end in controller_ids.items()
    )
    if boundaries:
        end = controller_ids.get(boundaries[-1], boundaries[-1])
        if any(
            message["role"] != "assistant"
            or message.get("content")
            or not str(message.get("reasoning_content") or "").strip()
            or message.get("tool_calls")
            for message in request.messages[end + 1 :]
        ):
            raise ValueError("Token reposition requires the current generation boundary")
    layout = token_counter.message_layout(
        request.messages,
        request.tools,
        enable_thinking=request.enable_thinking,
        reasoning_effort=request.reasoning_effort,
        api_style=request.api_style,
    )
    compacted_ids, events, audits = set(), [], []
    for call, boundary in enumerate(boundaries, 1):
        compacted = sum(message_tokens(layout, i) for i in compacted_ids)
        end = controller_ids.get(boundary, boundary)
        position = layout.message_ends[end] + layout.generation_tokens - compacted
        dropped = {i for trigger, ids in schedule.items() if trigger <= boundary for i in ids}
        dropped_tokens = sum(message_tokens(layout, i) for i in dropped)
        if position > trigger_tokens and dropped_tokens > compacted:
            events.append(boundary)
            audits.append(
                {
                    "call": call,
                    "message_id": boundary,
                    "position_tokens_before": position,
                    "position_tokens_after": position - dropped_tokens + compacted,
                    "compacted_drop_tokens": dropped_tokens,
                }
            )
            compacted_ids = dropped
    return tuple(events) or None, {
        "reposition_trigger_tokens": trigger_tokens,
        "reposition_token_metric": "post_reposition_position_length_including_generation_header",
        "reposition_raw_prompt_tokens": layout.prompt_tokens,
        "reposition_position_tokens": layout.prompt_tokens
        - sum(message_tokens(layout, i) for i in compacted_ids),
        "reposition_token_events": audits,
    }
