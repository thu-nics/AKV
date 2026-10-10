"""Canonical roles and complete atomic units shared by context policies."""

from collections.abc import Mapping, Sequence
from typing import Any

from akvalgo.types import RECOVERED_TOOL_CALLS_KEY, JsonDict

CANONICAL_ROLES = ("system", "developer", "user", "assistant", "tool")


def history_tool_calls(message: Mapping[str, Any]) -> list[JsonDict]:
    """Read executable calls without replacing the provider's replay fields."""
    return list(message.get("tool_calls") or message.get(RECOVERED_TOOL_CALLS_KEY, {}).get("tool_calls", ()))


def message_for_wire(message: Mapping[str, Any]) -> JsonDict:
    """Exclude client execution metadata from model inputs."""
    return {key: value for key, value in message.items() if key != RECOVERED_TOOL_CALLS_KEY}


def canonical_role(message: JsonDict) -> str:
    role = message.get("role")
    if role not in CANONICAL_ROLES:
        raise ValueError("history contains an unknown message role")
    return role


def atomic_message_units(messages: Sequence[JsonDict], start: int) -> tuple[tuple[int, ...], ...]:
    units: list[tuple[int, ...]] = []
    index = start
    while index < len(messages):
        message = messages[index]
        role = canonical_role(message)
        if role == "tool":
            raise ValueError(f"history contains orphan tool message at {index}")
        tool_calls = history_tool_calls(message)
        if role != "assistant" or not tool_calls:
            units.append((index,))
            index += 1
            continue
        expected = {
            str(call.get("id"))
            for call in tool_calls
            if isinstance(call, dict) and call.get("id") is not None
        }
        if len(expected) != len(tool_calls):
            raise ValueError("assistant tool_calls must have unique IDs")
        grouped = [index]
        seen: list[str] = []
        index += 1
        while index < len(messages) and canonical_role(messages[index]) == "tool":
            tool_id = str(messages[index].get("tool_call_id"))
            if tool_id not in expected:
                raise ValueError(f"tool result {index} does not match the preceding tool calls")
            grouped.append(index)
            seen.append(tool_id)
            index += 1
        if len(seen) != len(expected) or set(seen) != expected:
            raise ValueError("assistant tool-call batch has incomplete tool results")
        units.append(tuple(grouped))
    return tuple(units)
