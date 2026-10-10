"""Stateless policy adaptation of final Chat Completions JSON payloads."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

from .messages import atomic_message_units
from .types import ContextDecision, JsonDict, PolicyRequest


def check_context_request(payload: Mapping) -> None:
    if any(payload.get(name) is not None for name in ("drop_message", "drop_rule", "reposition")):
        raise ValueError("Pass a canonical request without existing context instructions")
    if "extra_body" in payload:
        raise ValueError("Pass the final HTTP payload, with SDK extra_body already merged")


def adapt_request(
    payload: Mapping,
    *,
    api_style: str | None = None,
    completed_call_message_ids: Sequence[int] | None = None,
    prompt_end_message_ids: Mapping[int, int] | None = None,
) -> PolicyRequest:
    """Extract policy facts from the exact message order sent to the server.

    Native tool-result batches supply their own boundaries. Other observation
    formats and controller suffixes require explicit harness metadata, including
    all historical boundaries. Nothing is remembered between calls. Rendering
    parameters come from the final JSON payload; exact token layouts remain the
    model-specific token counter's responsibility.
    """
    check_context_request(payload)
    messages = copy.deepcopy(tuple(payload["messages"]))
    tools = copy.deepcopy(tuple(payload.get("tools") or ()))
    template = payload.get("chat_template_kwargs") or {}
    thinking = template.get("enable_thinking")
    effort = payload.get("reasoning_effort")
    if not messages:
        raise ValueError("Canonical history must be nonempty")
    native = {unit[-1] for unit in atomic_message_units(messages, 0) if len(unit) > 1}
    boundaries = (
        tuple(sorted(native)) if completed_call_message_ids is None else tuple(completed_call_message_ids)
    )
    if any(
        type(b) is not int
        or not 0 <= b < len(messages)
        or messages[b]["role"] not in ("tool", "user")
        or messages[b]["role"] == "tool"
        and b not in native
        for b in boundaries
    ) or any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("Logical boundaries must be complete, ordered tool batches or user observations")
    if not native <= set(boundaries):
        raise ValueError("Logical boundaries must include every completed native tool batch")
    suffixes = dict(prompt_end_message_ids or {})
    if any(
        type(boundary) is not int
        or type(end) is not int
        or boundary not in boundaries
        or end != boundary + 1
        or end >= len(messages)
        or messages[end]["role"] != "user"
        for boundary, end in suffixes.items()
    ):
        raise ValueError("Prompt suffix must be the user message immediately after a boundary")
    return PolicyRequest(
        call_index=len(boundaries),
        messages=messages,
        tools=tools,
        enable_thinking=thinking,
        reasoning_effort=effort,
        api_style=api_style,
        completed_call_message_ids=boundaries,
        prompt_end_message_ids=suffixes,
    )


def apply_context(payload: Mapping, decision: ContextDecision) -> JsonDict:
    """Return a private final HTTP payload, retaining all unrelated fields.

    IDs refer to this payload's message order. Send the result without further
    message insertion, merging or reordering. The caller owns canonical history.
    """
    check_context_request(payload)
    prepared = copy.deepcopy(dict(payload))
    if decision.messages is not None:
        prepared["messages"] = copy.deepcopy(list(decision.messages))

    def message_id(value):
        if type(value) is not int or not 0 <= value < len(prepared["messages"]):
            raise ValueError("Context IDs must identify current payload messages")
        return value

    if decision.drop_message is not None:
        schedule = {}
        for trigger, ids in decision.drop_message.items():
            trigger = message_id(trigger)
            schedule[str(trigger)] = [message_id(value) for value in ids]
            if any(value > trigger for value in ids):
                raise ValueError("Drop events cannot reference future messages")
        prepared["drop_message"] = schedule
    if decision.reposition is not None:
        values = [message_id(value) for value in decision.reposition]
        if not values or values != sorted(set(values)):
            raise ValueError("reposition must contain increasing unique message IDs")
        prepared["reposition"] = values
    return prepared
