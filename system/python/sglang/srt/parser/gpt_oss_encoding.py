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
#
"""GPT-OSS Harmony encoding vendored from mini-SGLang.

Source: python/minisgl/tokenizer/tokenize.py, revision 89d8a9fd22a2784a8989bdd7b80e7a232d5e877e.
Keep token IDs and canonical message ownership aligned with the evaluation
encoder when updating this copy. No inference dependency here.
"""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from sglang.srt.context_system.provenance import TemplateTokenProvenance


@dataclass(frozen=True)
class _HarmonyComponentOwnership:
    owner: int
    sources: tuple[tuple[int, str], ...] = ()
    is_analysis: bool = False


@dataclass(frozen=True)
class _HarmonyPrompt:
    conversation: Any
    components: list[Any]
    ownership: list[_HarmonyComponentOwnership]


class HarmonyEncoder:
    def __init__(self, reasoning_effort=None, *, preserve_thinking=False):
        self._preserve_harmony_thinking = preserve_thinking
        self._reasoning_effort = reasoning_effort

    def render_tokens(self, messages, tools=None, *, enable_thinking=None):
        return self._render_harmony_message_drop(
            messages, tools=tools, enable_thinking=enable_thinking
        )

    def render(self, messages, tools=None, *, enable_thinking=None):
        ids, owners, _ = self.render_tokens(
            messages, tools=tools, enable_thinking=enable_thinking
        )
        return self._build_harmony_provenance(ids, owners)

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_harmony_encoding():
        from openai_harmony import HarmonyEncodingName, load_harmony_encoding

        return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def _build_harmony_prompt(
        self,
        messages: List[dict[str, Any]],
        *,
        enable_thinking: bool | None,
        tools: List[Dict[str, Any]] | None,
    ) -> _HarmonyPrompt:
        from openai_harmony import (
            Author,
            Conversation,
            DeveloperContent,
            ReasoningEffort,
            Role,
            SystemContent,
            ToolDescription,
        )
        from openai_harmony import (
            Message as HarmonyMessage,
        )

        effort_text = self._reasoning_effort
        if effort_text is None and enable_thinking is False:
            effort_text = "low"
        effort = {
            "low": ReasoningEffort.LOW,
            "medium": ReasoningEffort.MEDIUM,
            "high": ReasoningEffort.HIGH,
        }.get(str(effort_text or "medium").lower())
        if effort is None:
            raise ValueError("reasoning_effort must be one of: low, medium, high")

        harmony_messages = [
            HarmonyMessage.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new()
                .with_reasoning_effort(effort)
                .with_conversation_start_date(date.today().isoformat())
                .with_required_channels(["analysis", "commentary", "final"]),
            )
        ]
        ownership = [_HarmonyComponentOwnership(owner=-1)]

        instruction_sources = tuple(
            (raw_message_id, content)
            for raw_message_id, raw in enumerate(messages)
            if str(raw.get("role", "")).lower() in {"system", "developer"}
            and (content := self._normalize_message_content(raw.get("content")))
        )
        developer = DeveloperContent.new()
        if instruction_sources:
            developer.with_instructions(
                "\n\n".join(content for _, content in instruction_sources)
            )
        descriptions = []
        for tool in tools or []:
            fn = tool.get("function", tool) if isinstance(tool, dict) else {}
            if not isinstance(fn, dict) or not fn.get("name"):
                continue
            descriptions.append(
                ToolDescription.new(
                    str(fn["name"]),
                    str(fn.get("description") or ""),
                    parameters=fn.get("parameters"),
                )
            )
        if descriptions:
            developer.with_function_tools(descriptions)
        if instruction_sources or descriptions:
            harmony_messages.append(
                HarmonyMessage.from_role_and_content(Role.DEVELOPER, developer)
            )
            ownership.append(
                _HarmonyComponentOwnership(
                    owner=instruction_sources[0][0] if instruction_sources else -1,
                    sources=instruction_sources,
                )
            )

        tool_names: dict[str, str] = {}
        for raw_message_id, raw in enumerate(messages):
            role = str(raw.get("role", "user")).lower()
            if role in {"system", "developer"}:
                continue
            content = self._normalize_message_content(raw.get("content"))
            if role == "user":
                harmony_messages.append(
                    HarmonyMessage.from_role_and_content(Role.USER, content)
                )
                ownership.append(_HarmonyComponentOwnership(owner=raw_message_id))
                continue
            if role == "assistant":
                calls = raw.get("tool_calls") or []
                if calls and content:
                    harmony_messages.append(
                        HarmonyMessage.from_role_and_content(
                            Role.ASSISTANT, content
                        ).with_channel("commentary")
                    )
                    ownership.append(_HarmonyComponentOwnership(owner=raw_message_id))
                reasoning = raw.get("reasoning")
                legacy_reasoning = raw.get("reasoning_content")
                if (
                    isinstance(reasoning, str)
                    and isinstance(legacy_reasoning, str)
                    and reasoning != legacy_reasoning
                ):
                    raise ValueError(
                        "GPT-OSS assistant reasoning and reasoning_content must match "
                        "when both are provided."
                    )
                if not isinstance(reasoning, str):
                    reasoning = legacy_reasoning
                if isinstance(reasoning, str) and reasoning:
                    harmony_messages.append(
                        HarmonyMessage.from_role_and_content(
                            Role.ASSISTANT, reasoning
                        ).with_channel("analysis")
                    )
                    ownership.append(
                        _HarmonyComponentOwnership(
                            owner=raw_message_id,
                            is_analysis=True,
                        )
                    )
                if content and not calls:
                    harmony_messages.append(
                        HarmonyMessage.from_role_and_content(
                            Role.ASSISTANT, content
                        ).with_channel("final")
                    )
                    ownership.append(_HarmonyComponentOwnership(owner=raw_message_id))
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    fn = call.get("function")
                    if not isinstance(fn, dict) or not fn.get("name"):
                        continue
                    name = str(fn["name"])
                    if call.get("id") is not None:
                        tool_names[str(call["id"])] = name
                    arguments = fn.get("arguments", "{}")
                    if not isinstance(arguments, str):
                        arguments = self._json_dumps(arguments)
                    harmony_messages.append(
                        HarmonyMessage.from_role_and_content(Role.ASSISTANT, arguments)
                        .with_channel("commentary")
                        .with_recipient(f"functions.{name}")
                        .with_content_type("json")
                    )
                    ownership.append(_HarmonyComponentOwnership(owner=raw_message_id))
                continue
            if role in {"tool", "function"}:
                name = raw.get("name") or tool_names.get(
                    str(raw.get("tool_call_id", ""))
                )
                if not name:
                    raise ValueError(
                        "GPT-OSS tool results require name or a matching tool_call_id."
                    )
                harmony_messages.append(
                    HarmonyMessage.from_author_and_content(
                        Author.new(Role.TOOL, f"functions.{name}"), content
                    )
                    .with_channel("commentary")
                    .with_recipient("assistant")
                )
                ownership.append(_HarmonyComponentOwnership(owner=raw_message_id))
                continue
            raise ValueError(f"Unsupported GPT-OSS Harmony role: {role}")

        prompt = _HarmonyPrompt(
            conversation=Conversation.from_messages(harmony_messages),
            components=harmony_messages,
            ownership=ownership,
        )
        # Completed turns no longer contribute analysis to the next prompt.
        # Explicit thinking_drop retains it so Context can compile its KV drop.
        return (
            prompt
            if self._preserve_harmony_thinking
            else self._drop_harmony_analysis_before_last_final(prompt)
        )

    @staticmethod
    def _drop_harmony_analysis_before_last_final(
        prompt: _HarmonyPrompt,
    ) -> _HarmonyPrompt:
        """Match vLLM's long-history cleanup while retaining component ownership."""

        from openai_harmony import Conversation

        last_final = -1
        for component_id in range(len(prompt.components) - 1, -1, -1):
            component = prompt.components[component_id]
            role = getattr(getattr(component, "author", None), "role", None)
            role = getattr(role, "value", role)
            if str(role).lower() == "assistant" and component.channel == "final":
                last_final = component_id
                break
        if last_final < 0:
            return prompt

        keep_ids = [
            component_id
            for component_id, component in enumerate(prompt.components)
            if not (component_id < last_final and component.channel == "analysis")
        ]
        if len(keep_ids) == len(prompt.components):
            return prompt

        components = [prompt.components[component_id] for component_id in keep_ids]
        return _HarmonyPrompt(
            conversation=Conversation.from_messages(components),
            components=components,
            ownership=[prompt.ownership[component_id] for component_id in keep_ids],
        )

    def _render_harmony_message_drop(
        self,
        messages: List[dict[str, Any]],
        *,
        enable_thinking: bool | None,
        tools: List[Dict[str, Any]] | None,
    ) -> tuple[List[int], List[int], int]:
        """Render once and recover message owners from Harmony protocol boundaries."""

        from openai_harmony import RenderConversationConfig, Role

        prompt = self._build_harmony_prompt(
            messages,
            enable_thinking=enable_thinking,
            tools=tools,
        )
        encoding = self._get_harmony_encoding()
        input_ids = [
            int(token_id)
            for token_id in encoding.render_conversation_for_completion(
                prompt.conversation,
                Role.ASSISTANT,
                RenderConversationConfig(auto_drop_analysis=False),
            )
        ]

        special_names = {
            token_id: encoding.decode([token_id])
            for token_id in set(input_ids)
            if encoding.is_special_token(token_id)
        }
        starts = [
            position
            for position, token_id in enumerate(input_ids)
            if special_names.get(token_id) == "<|start|>"
        ]
        if not starts:
            raise RuntimeError("Harmony render contains no message boundary tokens.")

        complete_ranges = list(zip(starts[:-1], starts[1:]))
        generation_start = starts[-1]
        if any(
            special_names.get(input_ids[position])
            in {"<|end|>", "<|call|>", "<|return|>"}
            for position in range(generation_start, len(input_ids))
        ):
            raise RuntimeError("Harmony completion render has no generation prompt.")
        expected: list[_HarmonyComponentOwnership] = []
        component_id = 0
        for start, end in complete_ranges:
            header = encoding.decode(input_ids[start:end]).split("<|message|>", 1)[0]
            is_analysis = "<|channel|>analysis" in header
            while (
                component_id < len(prompt.ownership)
                and prompt.ownership[component_id].is_analysis
                and not is_analysis
            ):
                component_id += 1
            if (
                component_id >= len(prompt.ownership)
                or prompt.ownership[component_id].is_analysis != is_analysis
            ):
                raise RuntimeError(
                    "Harmony analysis filtering changed the native message stream; "
                    "cannot align message ownership."
                )
            expected.append(prompt.ownership[component_id])
            component_id += 1
        if any(not item.is_analysis for item in prompt.ownership[component_id:]):
            raise RuntimeError(
                "Harmony analysis filtering changed the native message stream; "
                "cannot align message ownership."
            )

        owners = [-1] * len(input_ids)
        decode_bytes = getattr(getattr(encoding, "_inner", None), "decode_bytes", None)
        for (start, end), component in zip(complete_ranges, expected):
            owners[start:end] = [component.owner] * (end - start)
            if len(component.sources) < 2:
                continue
            if decode_bytes is None:
                raise RuntimeError(
                    "Harmony byte decoding is required to split merged system messages."
                )

            token_bytes = [
                bytes(decode_bytes([token_id])) for token_id in input_ids[start:end]
            ]
            offsets = [0]
            for value in token_bytes:
                offsets.append(offsets[-1] + len(value))
            rendered = b"".join(token_bytes)
            byte_owners = [component.owner] * len(rendered)
            cursor = 0
            previous_owner = component.owner
            for source_owner, source in component.sources:
                needle = source.encode("utf-8")
                source_start = rendered.find(needle, cursor)
                if source_start < 0:
                    raise RuntimeError(
                        "Harmony changed system/developer message text; "
                        "cannot construct exact ownership."
                    )
                source_end = source_start + len(needle)
                byte_owners[cursor:source_start] = [previous_owner] * (
                    source_start - cursor
                )
                byte_owners[source_start:source_end] = [source_owner] * len(needle)
                cursor = source_end
                previous_owner = source_owner
            byte_owners[cursor:] = [previous_owner] * (len(rendered) - cursor)

            previous_token_owner = component.owner
            for local_id, (byte_start, byte_end) in enumerate(
                zip(offsets[:-1], offsets[1:])
            ):
                if byte_start == byte_end:
                    token_owner = previous_token_owner
                else:
                    token_owner = byte_owners[byte_start]
                owners[start + local_id] = token_owner
                previous_token_owner = token_owner

        owners[generation_start:] = [len(messages)] * (
            len(input_ids) - generation_start
        )
        return input_ids, owners, generation_start

    def _build_harmony_provenance(
        self,
        input_ids: List[int],
        owners: List[int],
    ) -> TemplateTokenProvenance:
        """Recover character offsets from Harmony bytes without retokenizing."""

        from sglang.srt.context_system.provenance import TemplateTokenProvenance

        encoding = self._get_harmony_encoding()
        decode_bytes = getattr(getattr(encoding, "_inner", None), "decode_bytes", None)
        if decode_bytes is None:
            raise RuntimeError("Harmony byte decoding is required for keep_text_drop.")
        token_bytes = [bytes(decode_bytes([token_id])) for token_id in input_ids]
        byte_offsets = [0]
        byte_owners: List[int] = []
        for owner, value in zip(owners, token_bytes, strict=True):
            byte_offsets.append(byte_offsets[-1] + len(value))
            byte_owners.extend([owner] * len(value))
        rendered_bytes = b"".join(token_bytes)
        rendered_text = rendered_bytes.decode("utf-8")
        char_byte_offsets = [0]
        char_owners: List[int] = []
        cross_owner_tokens = 0
        byte_pos = 0
        for char in rendered_text:
            encoded = char.encode("utf-8")
            char_owners.append(byte_owners[byte_pos] if encoded else 0)
            byte_pos += len(encoded)
            char_byte_offsets.append(byte_pos)

        offsets: List[tuple[int, int]] = []
        for token_id, (byte_start, byte_end) in enumerate(
            zip(byte_offsets[:-1], byte_offsets[1:], strict=True)
        ):
            char_start = max(bisect_right(char_byte_offsets, byte_start) - 1, 0)
            char_end = bisect_left(char_byte_offsets, byte_end)
            offsets.append((char_start, char_end))
            if byte_end > byte_start and any(
                owner != owners[token_id] for owner in byte_owners[byte_start:byte_end]
            ):
                cross_owner_tokens += 1
        return TemplateTokenProvenance(
            input_ids=list(input_ids),
            owners=list(owners),
            offsets=offsets,
            rendered_text=rendered_text,
            char_owners=char_owners,
            cross_owner_tokens=cross_owner_tokens,
        )

    def _normalize_message_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if not isinstance(part, dict):
                    raise ValueError("Message content parts must be objects.")
                part_type = part.get("type")
                if part_type in {"text", "input_text"} and isinstance(
                    part.get("text"), str
                ):
                    parts.append(part["text"])
                elif part_type == "thinking" and isinstance(part.get("thinking"), str):
                    parts.append(part["thinking"])
                else:
                    raise ValueError(
                        "MiniSGL currently supports only text content parts in chat messages."
                    )
            return "".join(parts)
        return self._json_dumps(content)

    @staticmethod
    def _json_dumps(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
