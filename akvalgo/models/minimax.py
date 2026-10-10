"""MiniMax tool schemas and shared malformed-argument replay copies."""

from __future__ import annotations

import json
from collections.abc import Iterable

from akvalgo.types import JsonDict


def reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def adapt_tool_call_history(messages: Iterable[JsonDict]) -> tuple[list[JsonDict], int]:
    """Wrap non-object/invalid JSON strings only in the replay representation.

    Preserve message order, IDs, reasoning and original argument bytes. Valid
    object strings remain verbatim; adapting an already adapted view is a no-op.
    The repair count describes the wire view, not executable tool attempts.
    """
    adapted = []
    repairs = 0
    for message in messages:
        calls = message.get("tool_calls")
        if message.get("role") != "assistant" or not isinstance(calls, list):
            adapted.append(message)
            continue
        replay_calls = []
        changed = False
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else None
            if not isinstance(arguments, str):
                replay_calls.append(call)
                continue
            try:
                value = json.loads(arguments, parse_constant=reject_json_constant)
            except ValueError:
                value = None
            if isinstance(value, dict):
                replay_calls.append(call)
                continue
            replay_calls.append(
                {
                    **call,
                    "function": {
                        **function,
                        "arguments": json.dumps(
                            {"_malformed_arguments": arguments},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                }
            )
            repairs += 1
            changed = True
        adapted.append({**message, "tool_calls": replay_calls} if changed else message)
    return adapted, repairs


def prepare_minimax_tools(tools):
    """Match SGLang's Function.model_dump before MiniMax renders whole schemas."""
    prepared = []
    for tool in tools:
        raw = tool["function"]
        function = {
            "description": raw.get("description"),
            "name": raw["name"],
            "parameters": raw.get("parameters"),
            "strict": raw.get("strict", False),
        }
        defer = raw.get("defer_loading", tool.get("defer_loading"))
        if defer is not None:
            function["defer_loading"] = defer
        prepared.append({"type": tool.get("type", "function"), "function": function})
    return prepared or None
