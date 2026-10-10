"""Shared Jinja rendering and token ownership for MiniMax and MiroThinker."""

from __future__ import annotations

import copy
import json
import re
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from jinja2 import Environment, nodes
from jinja2.visitor import NodeTransformer
from tokenizers import Tokenizer

from akvalgo.types import ChatMessageTokenLayout, JsonDict

from .models.minimax import prepare_minimax_tools

# Match SGLang's pure MiniMax formatting macro. Instrumenting its temporary
# string corrupts markers when the template later splits it at </think>.
_MINIMAX_VISIBLE_TEXT = """
{% macro visible_text(content) %}
{% if content is string %}{{ content }}
{% elif content is iterable and content is not mapping %}
{% for item in content %}
{% if item is mapping and item.type == 'text' %}{{ item.text }}
{% elif item is string %}{{ item }}{% endif %}
{% endfor %}
{% else %}{{ content }}{% endif %}
{% endmacro %}
"""


def macro_structure(node: nodes.Node) -> str:
    class StripWhitespace(NodeTransformer):
        def visit_TemplateData(self, item, *args, **kwargs):
            return None if not item.data.strip() else item

        def visit_Output(self, item, *args, **kwargs):
            item = self.generic_visit(item, *args, **kwargs)
            return item if item.nodes else None

    return StripWhitespace().visit(copy.deepcopy(node)).dump()


@dataclass(frozen=True)
class ChatTemplateTokenProvenance:
    """Rendered token IDs and their canonical message owners."""

    input_ids: tuple[int, ...]
    owners: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    rendered_text: str
    cross_owner_tokens: int


class TraceTemplateOutputs(NodeTransformer):
    """Wrap template output nodes with active message-loop markers."""

    def __init__(self, *, minimax: bool = False) -> None:
        self._loop_vars: list[str] = []
        self._minimax = minimax

    def visit_Macro(self, node: nodes.Macro, *args, **kwargs):
        if self._minimax and node.name == "visible_text":
            expected = next(Environment().parse(_MINIMAX_VISIBLE_TEXT).find_all(nodes.Macro))
            if macro_structure(node) != macro_structure(expected):
                raise ValueError("Unrecognized MiniMax visible_text macro")
            return node
        return self.generic_visit(node, *args, **kwargs)

    @staticmethod
    def target_names(target: nodes.Node) -> list[str]:
        if isinstance(target, nodes.Name):
            return [target.name]
        if isinstance(target, (nodes.List, nodes.Tuple)):
            names: list[str] = []
            for item in target.items:
                names.extend(TraceTemplateOutputs.target_names(item))
            return names
        return []

    def visit_For(self, node: nodes.For, *args, **kwargs):
        names = self.target_names(node.target)
        self._loop_vars.extend(names)
        try:
            return self.generic_visit(node, *args, **kwargs)
        finally:
            if names:
                del self._loop_vars[-len(names) :]

    @staticmethod
    def references_generation_prompt(node: nodes.Node) -> bool:
        if isinstance(node, nodes.Name) and node.name == "add_generation_prompt":
            return True
        return any(
            isinstance(candidate, nodes.Name) and candidate.name == "add_generation_prompt"
            for candidate in node.find_all(nodes.Name)
        )

    def marker_call(self, phase: str, lineno: int) -> nodes.Call:
        loop_values = [nodes.Name(name, "load") for name in reversed(self._loop_vars)]
        return nodes.Call(
            nodes.Name("_contextualize_owner_marker", "load"),
            [nodes.Const(phase), *loop_values],
            [],
            None,
            None,
        ).set_lineno(lineno)

    def marker_output(self, phase: str, lineno: int) -> nodes.Output:
        return nodes.Output([self.marker_call(phase, lineno)]).set_lineno(lineno)

    def visit_If(self, node: nodes.If, *args, **kwargs):
        traces_generation = self.references_generation_prompt(node.test)
        node = self.generic_visit(node, *args, **kwargs)
        if traces_generation:
            node.body = [
                self.marker_output("G", node.lineno),
                *node.body,
                self.marker_output("H", node.lineno),
            ]
        return node

    def visit_Output(self, node: nodes.Output, *args, **kwargs):
        traces_generation = self.references_generation_prompt(node)
        node = self.generic_visit(node, *args, **kwargs)
        traced_nodes = [
            self.marker_call("B", node.lineno),
            *node.nodes,
            self.marker_call("E", node.lineno),
        ]
        if traces_generation:
            traced_nodes = [
                self.marker_call("G", node.lineno),
                *traced_nodes,
                self.marker_call("H", node.lineno),
            ]
        node.nodes = traced_nodes
        return node


class ChatTemplateProvenanceRenderer:
    """Render a Jinja chat template while retaining exact message ownership."""

    def __init__(self, environment: Environment, chat_template: str) -> None:
        minimax = "ns.last_user_index" in chat_template and "<minimax:tool_call>" in chat_template
        traced_ast = TraceTemplateOutputs(minimax=minimax).visit(environment.parse(chat_template))
        code = environment.compile(traced_ast)
        self._template = environment.template_class.from_code(
            environment,
            code,
            environment.globals,
            None,
        )

    @staticmethod
    def parse_character_owners(
        traced_text: str,
        marker_pattern: re.Pattern[str],
        *,
        message_count: int,
        add_generation_prompt: bool,
    ) -> tuple[str, list[int]]:
        clean_parts: list[str] = []
        owners: list[int] = []
        active: list[int] = []
        generation_depth = 0
        cursor = 0

        def active_owner() -> int:
            if generation_depth > 0:
                return message_count
            return next((owner for owner in reversed(active) if owner >= 0), -1)

        for match in marker_pattern.finditer(traced_text):
            chunk = traced_text[cursor : match.start()]
            clean_parts.append(chunk)
            owners.extend([active_owner()] * len(chunk))

            phase = match.group(1)
            marker_owner = int(match.group(2))
            if phase == "B":
                active.append(marker_owner)
            elif phase == "E":
                if not active:
                    raise RuntimeError("Unbalanced chat-template provenance marker.")
                active.pop()
            elif phase == "G":
                generation_depth += 1
            else:
                if generation_depth == 0:
                    raise RuntimeError("Unbalanced generation-prompt provenance marker.")
                generation_depth -= 1
            cursor = match.end()

        tail = traced_text[cursor:]
        clean_parts.append(tail)
        owners.extend([active_owner()] * len(tail))
        if active or generation_depth:
            raise RuntimeError("Unbalanced chat-template provenance marker.")

        clean_text = "".join(clean_parts)
        if not owners:
            return clean_text, owners

        known = [index for index, owner in enumerate(owners) if owner >= 0]
        if not known:
            if message_count > 1:
                raise RuntimeError("Chat template emitted multiple messages outside traceable message loops.")
            fallback = message_count if add_generation_prompt and message_count == 0 else 0
            return clean_text, [fallback] * len(owners)

        first_known = known[0]
        leading_owner = 0 if message_count > 0 else owners[first_known]
        owners[:first_known] = [leading_owner] * first_known

        previous_owner = owners[first_known]
        for index in range(first_known + 1, len(owners)):
            if owners[index] < 0:
                owners[index] = previous_owner
            else:
                previous_owner = owners[index]

        if add_generation_prompt and message_count not in owners:
            last_known = known[-1]
            if last_known + 1 < len(owners):
                owners[last_known + 1 :] = [message_count] * (len(owners) - last_known - 1)
        return clean_text, owners

    def render(
        self,
        tokenizer: Any,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        add_generation_prompt: bool,
        enable_thinking: bool | None,
        template_globals: dict[str, Any] | None = None,
    ) -> ChatTemplateTokenProvenance:
        nonce = uuid.uuid4().hex
        marker_prefix = f"\x00contextualize-owner-{nonce}:"
        owner_by_object = {id(message): message_id for message_id, message in enumerate(messages)}

        def owner_marker(phase: str, *loop_values: Any) -> str:
            owner = -1
            for value in loop_values:
                candidate = owner_by_object.get(id(value))
                if candidate is not None:
                    owner = candidate
                    break
            return f"{marker_prefix}{phase}:{owner}\x00"

        render_values = dict(template_globals or {})
        if enable_thinking is not None:
            render_values["enable_thinking"] = enable_thinking
        traced = self._template.render(
            messages=messages,
            tools=tools,
            documents=None,
            add_generation_prompt=add_generation_prompt,
            _contextualize_owner_marker=owner_marker,
            **render_values,
        )
        marker_pattern = re.compile(re.escape(marker_prefix) + r"([BEGH]):(-?\d+)\x00")
        rendered_text, char_owners = self.parse_character_owners(
            traced,
            marker_pattern,
            message_count=len(messages),
            add_generation_prompt=add_generation_prompt,
        )
        encoded = tokenizer.encode(rendered_text, add_special_tokens=False)
        input_ids = tuple(int(token_id) for token_id in encoded.ids)
        offsets = tuple((int(start), int(end)) for start, end in encoded.offsets)

        owners: list[int] = []
        cross_owner_tokens = 0
        previous_owner = 0
        for token_index, (start, end) in enumerate(offsets):
            if start < 0 or end < start or end > len(char_owners):
                raise RuntimeError(f"Token {token_index} has an invalid character offset.")
            if start == end:
                owner = previous_owner
            else:
                token_owners = char_owners[start:end]
                owner = token_owners[0]
                if any(candidate != owner for candidate in token_owners[1:]):
                    cross_owner_tokens += 1
            owners.append(owner)
            previous_owner = owner

        return ChatTemplateTokenProvenance(
            input_ids=input_ids,
            owners=tuple(owners),
            offsets=offsets,
            rendered_text=rendered_text,
            cross_owner_tokens=cross_owner_tokens,
        )


def json_dumps(value: Any, **kwargs: Any) -> str:
    return json.dumps(value, **{"ensure_ascii": False, **kwargs})


def strftime_now(format_string: str) -> str:
    return datetime.now().strftime(format_string)


def compact_json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def prepare_chat_template_messages(
    messages: Sequence[JsonDict],
    *,
    api_style: str = "minisglang",
) -> list[JsonDict]:
    """Decode tool arguments while preserving each server's message view."""
    prepared: list[JsonDict] = []
    for raw in messages:
        role = str(raw.get("role", "user")).lower()
        if role == "function":
            role = "tool"
        raw_content = raw.get("content")
        content = (
            ""
            if raw_content is None
            else raw_content
            if isinstance(raw_content, str)
            else compact_json_dumps(raw_content)
        )
        message: JsonDict = {"role": role, "content": content}
        if api_style == "sglang":
            message = {**raw, "content": "" if raw_content is None else raw_content}
            # Match serving_chat.normalize_tool_content for text tool results.
            if (
                role == "tool"
                and isinstance(raw_content, list)
                and all(
                    isinstance(part, str) or (isinstance(part, dict) and part.get("type") == "text")
                    for part in raw_content
                )
            ):
                message["content"] = " ".join(
                    part if isinstance(part, str) else part.get("text", "") for part in raw_content
                )
        if role == "assistant" and isinstance(raw.get("reasoning_content"), str):
            message["reasoning_content"] = raw["reasoning_content"]
        if role == "assistant" and isinstance(raw.get("tool_calls"), list):
            calls: list[JsonDict] = []
            for raw_call in raw["tool_calls"]:
                if not isinstance(raw_call, dict):
                    continue
                call = dict(raw_call)
                function = call.get("function")
                if isinstance(function, dict):
                    function = dict(function)
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            function["arguments"] = json.loads(arguments)
                        except json.JSONDecodeError:
                            pass
                    call["function"] = function
                calls.append(call)
            message["tool_calls"] = calls
        if role == "tool":
            if raw.get("tool_call_id") is not None:
                message["tool_call_id"] = str(raw["tool_call_id"])
            if raw.get("name") is not None:
                message["name"] = str(raw["name"])
        prepared.append(message)

    # mini-SGLang passes tool definitions to the model chat template. Do not
    # also synthesize them into the system message or they are counted twice.
    return prepared


class JinjaEncoder:
    """Keep a local tokenizer, chat template and ownership renderer together."""

    def __init__(self, path: Path, model_type: str | None):
        self.model_type = model_type
        tokenizer_path = path / "tokenizer.json"
        template_path = path / "chat_template.jinja"
        environment = Environment(trim_blocks=True, lstrip_blocks=True)
        environment.filters["tojson"] = json_dumps
        environment.globals["strftime_now"] = strftime_now
        template_source = template_path.read_text(encoding="utf-8")
        self._template = environment.from_string(template_source)
        self._provenance_renderer = ChatTemplateProvenanceRenderer(
            environment,
            template_source,
        )
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer_config_path = path / "tokenizer_config.json"
        tokenizer_config = (
            json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
            if tokenizer_config_path.is_file()
            else {}
        )
        self._template_globals = {
            key: value
            for key, value in tokenizer_config.items()
            if key.endswith("_token") and isinstance(value, str)
        }

    def prepare_tools(self, tools, *, api_style):
        if self.model_type == "minimax_m2" and api_style == "sglang":
            return prepare_minimax_tools(tools)
        return list(tools) or None

    def message_layout(self, messages, tools, *, enable_thinking, api_style):
        # MiroThinker uses Qwen3 architecture IDs in its local model config.
        if self.model_type not in {"qwen3", "qwen3_moe", "minimax_m2"}:
            raise ValueError(
                "token reposition requires the MiroThinker (Qwen3/Qwen3-MoE) or MiniMax M2 architecture"
            )
        prepared = prepare_chat_template_messages(messages, api_style=api_style)
        assert self._provenance_renderer is not None
        assert self._template is not None
        assert self._tokenizer is not None
        template_tools = self.prepare_tools(tools, api_style=api_style)
        provenance = self._provenance_renderer.render(
            self._tokenizer,
            prepared,
            tools=template_tools,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            template_globals=self._template_globals,
        )
        rendered = self._template.render(
            messages=prepared,
            tools=template_tools,
            documents=None,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            **self._template_globals,
        )
        if provenance.rendered_text != rendered:
            raise RuntimeError("chat-template provenance changed the canonical prompt")
        wire_message_count = len(prepared)
        ends = {owner: index + 1 for index, owner in enumerate(provenance.owners)}
        counts = Counter(provenance.owners)
        wire_message_ends = []
        wire_message_token_counts = []
        for message_id in range(wire_message_count):
            if not counts[message_id]:
                raise ValueError(f"chat template did not render canonical message {message_id}")
            wire_message_ends.append(ends[message_id])
            wire_message_token_counts.append(counts[message_id])
        if any(left >= right for left, right in pairwise(wire_message_ends)):
            raise ValueError("chat template does not preserve canonical message order")
        generation_positions = [
            index for index, owner in enumerate(provenance.owners) if owner == wire_message_count
        ]
        if not generation_positions:
            raise ValueError("chat template emitted no generation prompt")
        if generation_positions != list(range(generation_positions[0], len(provenance.input_ids))):
            raise ValueError("generation prompt is not a final token suffix")
        return ChatMessageTokenLayout(
            tuple(wire_message_ends),
            len(generation_positions),
            tuple(wire_message_token_counts),
        )

    def count(self, messages, tools, *, enable_thinking, api_style):
        template_messages = prepare_chat_template_messages(messages, api_style=api_style)
        template_tools = self.prepare_tools(tools, api_style=api_style)
        rendered = self._template.render(
            messages=template_messages,
            tools=template_tools,
            documents=None,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            **self._template_globals,
        )
        return len(self._tokenizer.encode(rendered, add_special_tokens=False).ids)
