"""Token-native GPT-OSS chat parsing, adapted from mini-sglang ce3ce58.

Parse generated token IDs once for both reasoning and function calls. In
particular, a recipient takes precedence over the analysis channel.
"""
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

import json
import re
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List

from sglang.srt.environ import envs


def _allowed(name, names):
    return name in names or (bool(names) and envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get())


def _tool_name(tool: Dict[str, Any]) -> str | None:
    function = tool.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    return name if isinstance(name, str) and name else None


def _tool_call(
    name: str, arguments: Any, index: int, call_id: str | None = None
) -> dict:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    return {
        "id": call_id or f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "index": index,
        "function": {"name": name, "arguments": arguments},
    }


@dataclass(frozen=True)
class ParsedResponse:
    content: str = ""
    reasoning_content: str = ""
    tool_calls: List[Dict[str, Any]] | None = None


@dataclass(frozen=True)
class StreamPiece:
    content: str = ""
    reasoning_content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


@lru_cache(maxsize=1)
def _harmony_encoding():
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding

    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


class _HarmonyStringFallback:
    """Tolerant full-text fallback for legacy non-canonical Harmony headers."""

    _TERMINATORS = ("<|end|>", "<|call|>", "<|return|>")

    def __init__(
        self, tools: List[Dict[str, Any]], stream_reasoning: bool = True
    ) -> None:
        self.names = {_tool_name(tool) for tool in tools}
        self.stream_reasoning = stream_reasoning
        self.buffer = ""
        self.reasoning_buffer = ""
        self.state = "seek"
        self.channel: str | None = None
        self.recipient: str | None = None
        self.tool_index = 0

    @staticmethod
    def _hold_partial(text: str, markers: tuple[str, ...]) -> tuple[str, str]:
        hold = 0
        for marker in markers:
            for size in range(1, min(len(text), len(marker) - 1) + 1):
                if marker.startswith(text[-size:]):
                    hold = max(hold, size)
        return (text[:-hold], text[-hold:]) if hold else (text, "")

    def _piece_for_content(self, content: str, *, terminal: bool) -> StreamPiece:
        if self.recipient:
            if not terminal:
                return StreamPiece()
            name = self.recipient.removeprefix("functions.")
            if not _allowed(name, self.names):
                return StreamPiece()
            arguments = content if content.strip() else "{}"
            call = _tool_call(name, arguments, self.tool_index)
            self.tool_index += 1
            return StreamPiece(tool_calls=[call])
        if self.channel == "analysis":
            if not self.stream_reasoning:
                self.reasoning_buffer += content
                if not terminal:
                    return StreamPiece()
                content = self.reasoning_buffer
                self.reasoning_buffer = ""
            return StreamPiece(reasoning_content=content)
        if self.channel in {"final", "commentary"}:
            return StreamPiece(content=content)
        return StreamPiece()

    def feed(self, text: str) -> StreamPiece:
        self.buffer += text
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: list[dict] = []

        def append(piece: StreamPiece) -> None:
            content_parts.append(piece.content)
            reasoning_parts.append(piece.reasoning_content)
            calls.extend(piece.tool_calls)

        while self.buffer:
            if self.state == "seek":
                marker = self.buffer.find("<|channel|>")
                if marker < 0:
                    # Harmony output normally starts with structural markers.
                    # Preserve a possible split marker and pass through genuine
                    # plain text produced by compatibility templates.
                    emitted, self.buffer = self._hold_partial(
                        self.buffer, ("<|start|>", "<|channel|>")
                    )
                    if "<|" not in emitted and not emitted.endswith("assistant"):
                        content_parts.append(emitted)
                    break
                prefix = self.buffer[:marker]
                recipient = re.search(r"\bto=([^\s<]+)", prefix)
                self.recipient = recipient.group(1) if recipient else None
                self.buffer = self.buffer[marker + len("<|channel|>") :]
                self.state = "header"
                continue

            if self.state == "header":
                marker = self.buffer.find("<|message|>")
                if marker < 0:
                    break
                header = self.buffer[:marker].strip()
                self.buffer = self.buffer[marker + len("<|message|>") :]
                self.channel = header.split(None, 1)[0].lower() if header else None
                recipient = re.search(r"\bto=([^\s<]+)", header)
                if recipient:
                    self.recipient = recipient.group(1)
                self.state = "content"
                continue

            positions = [
                (self.buffer.find(marker), marker)
                for marker in self._TERMINATORS
                if self.buffer.find(marker) >= 0
            ]
            if not positions:
                if self.recipient:
                    break
                emitted, self.buffer = self._hold_partial(
                    self.buffer, self._TERMINATORS
                )
                append(self._piece_for_content(emitted, terminal=False))
                break
            position, marker = min(positions, key=lambda item: item[0])
            append(self._piece_for_content(self.buffer[:position], terminal=True))
            self.buffer = self.buffer[position + len(marker) :]
            self.state = "seek"
            self.channel = self.recipient = None

        return StreamPiece("".join(content_parts), "".join(reasoning_parts), calls)

    def finish(self) -> StreamPiece:
        if self.state == "content":
            piece = self._piece_for_content(self.buffer, terminal=True)
        else:
            piece = StreamPiece()
        self.buffer = ""
        return piece


def _harmony_message_text(message: Any) -> str:
    return "".join(
        text
        for part in getattr(message, "content", ())
        if isinstance((text := getattr(part, "text", None)), str)
    )


def _parsed_harmony_messages(
    messages: List[Any],
    tools: List[Dict[str, Any]],
) -> ParsedResponse:
    names = {name for tool in tools if (name := _tool_name(tool)) is not None}
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    calls: list[dict] = []
    for message in messages:
        channel = getattr(message, "channel", None)
        recipient = getattr(message, "recipient", None)
        content = _harmony_message_text(message)
        if recipient:
            name = str(recipient).removeprefix("functions.")
            if _allowed(name, names):
                calls.append(
                    _tool_call(name, content if content.strip() else "{}", len(calls))
                )
        elif channel == "analysis":
            reasoning_parts.append(content)
        elif channel in {"final", "commentary", None}:
            content_parts.append(content)
    return ParsedResponse(
        content="".join(content_parts),
        reasoning_content="".join(reasoning_parts),
        tool_calls=calls or None,
    )


def _parse_harmony_text_compat(
    text: str, tools: List[Dict[str, Any]]
) -> ParsedResponse:
    """Parse a complete decoded response, with one bounded legacy fallback."""

    from openai_harmony import HarmonyError, Role, StreamableParser

    encoding = _harmony_encoding()
    completion = text.removeprefix("<|start|>assistant")
    parser = StreamableParser(encoding, Role.ASSISTANT, strict=False)
    try:
        for token in encoding.encode(completion, allowed_special="all"):
            parser.process(int(token))
        parser.process_eos()
        return _parsed_harmony_messages(parser.messages, tools)
    except HarmonyError:
        # Older clients and fixtures may place the recipient after the content
        # type. The reference parser rejects that ordering, but it is
        # unambiguous, so recover it without making it the production path.
        fallback = _HarmonyStringFallback(tools)
        pieces = (fallback.feed(text), fallback.finish())
        calls = [call for piece in pieces for call in piece.tool_calls]
        return ParsedResponse(
            content="".join(piece.content for piece in pieces),
            reasoning_content="".join(piece.reasoning_content for piece in pieces),
            tool_calls=calls or None,
        )


class HarmonyChatParser:
    """Token-native GPT-OSS response parser backed by openai-harmony."""

    def __init__(
        self, tools: List[Dict[str, Any]], stream_reasoning: bool = True
    ) -> None:
        from openai_harmony import Role, StreamableParser

        self.tools = tools
        self.names = {name for tool in tools if (name := _tool_name(tool)) is not None}
        self.stream_reasoning = stream_reasoning
        self.encoding = _harmony_encoding()
        self.parser = StreamableParser(self.encoding, Role.ASSISTANT, strict=False)
        self.mode: str | None = None
        self.text_buffer = ""
        self.all_tokens: list[int] = []
        self.message_count = 0
        self.tool_index = 0
        self.emitted_content = ""
        self.emitted_reasoning = ""
        self.failed = False
        self.finished = False
        self.received_tokens = 0

    def feed_output(self, token_ids, *, incremental=False, completion_tokens=None):
        if token_ids is None:
            raise ValueError("GPT-OSS Harmony parsing requires output_ids")
        if incremental:
            delta = token_ids
            if completion_tokens is not None:
                delta = delta[: max(0, completion_tokens - self.received_tokens)]
        else:
            if len(token_ids) < self.received_tokens:
                raise ValueError("GPT-OSS cumulative output_ids shrank")
            delta = token_ids[self.received_tokens :]
        self.received_tokens += len(delta)
        return self.feed_tokens(delta)

    @staticmethod
    def _join(pieces: List[StreamPiece]) -> StreamPiece:
        return StreamPiece(
            content="".join(piece.content for piece in pieces),
            reasoning_content="".join(piece.reasoning_content for piece in pieces),
            tool_calls=[call for piece in pieces for call in piece.tool_calls],
        )

    def _record(self, piece: StreamPiece) -> StreamPiece:
        self.emitted_content += piece.content
        self.emitted_reasoning += piece.reasoning_content
        return piece

    def _completed_message_piece(self, message: Any) -> StreamPiece:
        channel = getattr(message, "channel", None)
        recipient = getattr(message, "recipient", None)
        content = _harmony_message_text(message)
        if recipient:
            name = str(recipient).removeprefix("functions.")
            if not _allowed(name, self.names):
                return StreamPiece()
            call = _tool_call(
                name,
                content if content.strip() else "{}",
                self.tool_index,
            )
            self.tool_index += 1
            return StreamPiece(tool_calls=[call])
        if channel == "analysis" and not self.stream_reasoning:
            return StreamPiece(reasoning_content=content)
        return StreamPiece()

    def _collect_completed(self) -> StreamPiece:
        messages = self.parser.messages
        pieces = [
            self._completed_message_piece(message)
            for message in messages[self.message_count :]
        ]
        self.message_count = len(messages)
        return self._join(pieces)

    def _process_token(self, token: int) -> StreamPiece:
        from openai_harmony import HarmonyError

        if self.failed:
            return StreamPiece()
        try:
            self.parser.process(int(token))
        except HarmonyError:
            self.failed = True
            return StreamPiece()

        pieces: list[StreamPiece] = []
        delta = self.parser.last_content_delta
        if delta:
            channel = self.parser.current_channel
            recipient = self.parser.current_recipient
            if recipient is None and channel == "analysis" and self.stream_reasoning:
                pieces.append(StreamPiece(reasoning_content=delta))
            elif recipient is None and channel in {"final", "commentary", None}:
                pieces.append(StreamPiece(content=delta))
        pieces.append(self._collect_completed())
        return self._record(self._join(pieces))

    def feed_tokens(self, token_ids: List[int]) -> StreamPiece:
        if self.finished:
            raise RuntimeError("Cannot feed Harmony tokens after finish().")
        if self.mode == "text":
            buffered = self.encoding.encode(self.text_buffer, allowed_special="all")
            self.text_buffer = ""
            self.mode = "tokens"
            prefix = self.feed_tokens([int(token) for token in buffered])
        else:
            self.mode = "tokens"
            prefix = StreamPiece()

        pieces = [prefix]
        for token in token_ids:
            value = int(token)
            self.all_tokens.append(value)
            pieces.append(self._process_token(value))
        return self._join(pieces)

    def feed_text(self, text: str) -> StreamPiece:
        if self.finished:
            raise RuntimeError("Cannot feed Harmony text after finish().")
        if self.mode == "tokens":
            # Production replies carry both the token id and its decoded delta.
            # Once token mode is active, the decoded copy must not be parsed twice.
            return StreamPiece()
        self.mode = "text"
        self.text_buffer += text
        return StreamPiece()

    @staticmethod
    def _remaining(value: str, emitted: str) -> str:
        if not emitted:
            return value
        return value[len(emitted) :] if value.startswith(emitted) else ""

    def _fallback_remaining(self) -> StreamPiece:
        decoded = self.encoding.decode_utf8(self.all_tokens)
        parsed = _parse_harmony_text_compat(decoded, self.tools)
        calls = list(parsed.tool_calls or [])[self.tool_index :]
        return StreamPiece(
            content=self._remaining(parsed.content, self.emitted_content),
            reasoning_content=self._remaining(
                parsed.reasoning_content,
                self.emitted_reasoning,
            ),
            tool_calls=calls,
        )

    def finish(self, finish_reason="stop") -> StreamPiece:
        piece = self._finish()
        if finish_reason != "stop":
            return StreamPiece(piece.content, piece.reasoning_content)
        return piece

    def _finish(self) -> StreamPiece:
        from openai_harmony import HarmonyError

        if self.finished:
            return StreamPiece()
        self.finished = True
        if self.mode != "tokens":
            parsed = _parse_harmony_text_compat(self.text_buffer, self.tools)
            return StreamPiece(
                content=parsed.content,
                reasoning_content=parsed.reasoning_content,
                tool_calls=list(parsed.tool_calls or []),
            )
        if self.failed:
            return self._fallback_remaining()
        try:
            self.parser.process_eos()
        except HarmonyError:
            self.failed = True
            return self._fallback_remaining()
        return self._record(self._collect_completed())
