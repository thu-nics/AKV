"""Shared policy, message, token-layout and HTTP result types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

JsonDict = dict[str, Any]
RECOVERED_TOOL_CALLS_KEY = "contextualize_recovered_tool_calls"


@dataclass(frozen=True)
class BackendTraceContext:
    """Filesystem destination and identity for one physical backend call.

    ``artifact_dir`` must be unique for the call. The backend creates it and
    refuses to reuse an existing directory, which prevents concurrent calls or
    retries from overwriting one another. ``metadata`` is audit-only and is
    never included in the provider request.
    """

    artifact_dir: Path
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str
    server_name: str | None = None

    def as_openai(self) -> JsonDict:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }

    def as_recovered(self) -> JsonDict:
        """Serialize client-side recovery metadata without changing wire calls."""

        value = self.as_openai()
        if self.server_name is not None:
            value["server_name"] = self.server_name
        return value


@dataclass(frozen=True)
class GeneratedMessage:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    role: str = "assistant"
    reasoning_content: str | None = None
    recovered_tool_calls: tuple[ToolCall, ...] = ()
    recovered_tool_call_source: str | None = None

    @property
    def execution_tool_calls(self) -> tuple[ToolCall, ...]:
        return self.tool_calls or self.recovered_tool_calls

    def as_openai(self, *, include_reasoning_content: bool = False) -> JsonDict:
        message: JsonDict = {"role": self.role, "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [call.as_openai() for call in self.tool_calls]
        if include_reasoning_content and isinstance(self.reasoning_content, str):
            message["reasoning_content"] = self.reasoning_content
        return message

    def as_history(self, *, include_reasoning_content: bool = False) -> JsonDict:
        """Keep recovered calls as client metadata; preserve their source text."""
        message = self.as_openai(
            include_reasoning_content=(
                include_reasoning_content or self.recovered_tool_call_source == "reasoning_content"
            )
        )
        if self.recovered_tool_calls:
            message[RECOVERED_TOOL_CALLS_KEY] = {
                "source": self.recovered_tool_call_source,
                "tool_calls": [call.as_recovered() for call in self.recovered_tool_calls],
            }
        return message


@dataclass(frozen=True)
class GenerationResult:
    message: GeneratedMessage
    finish_reason: str | None
    latency_seconds: float
    usage: JsonDict | None = None
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyRequest:
    """Canonical input facts for one drop/reposition decision."""

    messages: Sequence[JsonDict]
    call_index: int = 0
    tools: Sequence[JsonDict] = ()
    enable_thinking: bool | None = None
    reasoning_effort: str | None = None
    api_style: str | None = None
    completed_call_message_ids: Sequence[int] = ()
    prompt_end_message_ids: Mapping[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextDecision:
    """A wire-history replacement or cumulative model-side context instructions."""

    drop_message: dict[int, tuple[int, ...]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    messages: tuple[JsonDict, ...] | None = None
    reposition: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ChatMessageTokenLayout:
    """Canonical message end offsets, including the injected protocol prefix."""

    message_ends: tuple[int, ...]
    generation_tokens: int
    message_token_counts: tuple[int, ...] | None = None

    @property
    def prompt_tokens(self) -> int:
        return self.message_ends[-1] + self.generation_tokens


class ChatTokenCounter(Protocol):
    def count(
        self,
        messages: Sequence[JsonDict],
        tools: Sequence[JsonDict],
        *,
        enable_thinking: bool | None,
        reasoning_effort: str | None,
        api_style: str | None,
    ) -> int: ...

    def message_layout(
        self,
        messages: Sequence[JsonDict],
        tools: Sequence[JsonDict],
        *,
        enable_thinking: bool | None,
        reasoning_effort: str | None,
        api_style: str | None,
    ) -> ChatMessageTokenLayout: ...


class BackendError(RuntimeError):
    """The inference service returned invalid data or an HTTP error."""

    def __init__(
        self,
        message: str,
        *,
        response_body: bytes | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.response_body = response_body
        self.http_status = http_status


class ContextLengthError(BackendError):
    """The service rejected input length, output budget, or execution positions
    outside the model context window."""
