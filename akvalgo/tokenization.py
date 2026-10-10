"""Optional token counting for MiniMax, MiroThinker and GPT-OSS."""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from pathlib import Path

from .messages import message_for_wire
from .models.minimax import adapt_tool_call_history
from .models.mirothinker import (
    MIROTHINKER_MCP_PROTOCOL,
    current_mirothinker_mcp_date,
    prepare_mirothinker_mcp_request,
)
from .types import ChatMessageTokenLayout, ChatTokenCounter, JsonDict


class LocalChatTemplateTokenCounter:
    """Select the matching local encoder and return counts or message layouts.

    The caller must use the same rendering settings and protocol preparation
    for counting and sending. Explicit tool_protocol preparation operates on a
    rendering copy; it never updates the caller's HTTP request.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        tool_protocol: str | None = None,
        tool_protocol_date: str | None = None,
    ) -> None:
        if tool_protocol not in {None, MIROTHINKER_MCP_PROTOCOL}:
            raise ValueError("token counter tool_protocol must be null or mirothinker_mcp")
        self.tool_protocol = tool_protocol
        self.tool_protocol_date = tool_protocol_date or current_mirothinker_mcp_date()
        self.model_path = Path(model_path)
        for name in ("tokenizer.json", "chat_template.jinja"):
            path = self.model_path / name
            if not path.is_file():
                raise FileNotFoundError(path)
        config_path = self.model_path / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        self._model_type = config.get("model_type")
        self._count_lock = threading.Lock()
        self._encoder = None
        if self._model_type == "gpt_oss":
            self.tokenizer_protocol = (
                "openai_harmony_gpt_oss_mirothinker_mcp_v1"
                if tool_protocol == MIROTHINKER_MCP_PROTOCOL
                else "openai_harmony_gpt_oss"
            )
        else:
            from .templates import JinjaEncoder

            self._encoder = JinjaEncoder(self.model_path, self._model_type)
            self.tokenizer_protocol = (
                "local_jinja_chat_template_provenance_mirothinker_mcp_v1"
                if tool_protocol == MIROTHINKER_MCP_PROTOCOL
                else "local_jinja_chat_template_provenance_v1"
            )

    def prepare_messages(self, messages, tools):
        messages = [message_for_wire(message) for message in messages]
        if self._model_type == "minimax_m2" or self.tool_protocol == MIROTHINKER_MCP_PROTOCOL:
            messages, _ = adapt_tool_call_history(messages)
        if self.tool_protocol != MIROTHINKER_MCP_PROTOCOL:
            return messages, list(tools), 0
        prepared = prepare_mirothinker_mcp_request(messages, tools, today=self.tool_protocol_date)
        return list(prepared.messages), list(prepared.tools), prepared.canonical_message_offset

    def message_layout(
        self,
        messages: Sequence[JsonDict],
        tools: Sequence[JsonDict],
        *,
        enable_thinking: bool | None,
        reasoning_effort: str | None,
        api_style: str | None,
    ) -> ChatMessageTokenLayout:
        if api_style not in {"minisglang", "sglang"}:
            raise ValueError("token reposition requires minisglang or sglang")
        logical_messages = list(messages)
        if not logical_messages:
            raise ValueError("token reposition requires messages")
        messages, tools, offset = self.prepare_messages(logical_messages, tools)
        if self._model_type == "gpt_oss":
            if self.tool_protocol == MIROTHINKER_MCP_PROTOCOL:
                raise ValueError(
                    "mirothinker_mcp token reposition requires a MiroThinker (Qwen3-based) model"
                )
            from .models.gpt_oss import HarmonyEncoder

            with self._count_lock:
                return HarmonyEncoder(reasoning_effort).message_layout(
                    messages, tools, enable_thinking=enable_thinking, logical_messages=logical_messages
                )
        with self._count_lock:
            layout = self._encoder.message_layout(
                messages, tools, enable_thinking=enable_thinking, api_style=api_style
            )
        end = offset + len(logical_messages)
        return ChatMessageTokenLayout(
            layout.message_ends[offset:end],
            layout.generation_tokens,
            layout.message_token_counts[offset:end],
        )

    def count(
        self,
        messages: Sequence[JsonDict],
        tools: Sequence[JsonDict],
        *,
        enable_thinking: bool | None,
        reasoning_effort: str | None,
        api_style: str | None,
    ) -> int:
        if not messages:
            raise ValueError("chat-template token counting requires messages")
        if api_style not in {"minisglang", "sglang"}:
            raise ValueError("tokenizer api_style must be minisglang or sglang")
        messages, tools, _ = self.prepare_messages(messages, tools)
        if self._model_type == "gpt_oss":
            if self.tool_protocol == MIROTHINKER_MCP_PROTOCOL:
                raise ValueError("mirothinker_mcp token counting requires a MiroThinker (Qwen3-based) model")
            from .models.gpt_oss import HarmonyEncoder

            with self._count_lock:
                ids, _, _ = HarmonyEncoder(reasoning_effort).render_tokens(
                    messages, tools, enable_thinking=enable_thinking
                )
                return len(ids)
        with self._count_lock:
            return self._encoder.count(messages, tools, enable_thinking=enable_thinking, api_style=api_style)


__all__ = ["ChatMessageTokenLayout", "ChatTokenCounter", "LocalChatTemplateTokenCounter"]
