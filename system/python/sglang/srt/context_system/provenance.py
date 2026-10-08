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

import re
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from jinja2 import nodes
from jinja2.visitor import NodeTransformer
from transformers.utils.chat_template_utils import _compile_jinja_template


# This pure formatting macro returns text that MiniMax subsequently splits at
# </think>. Markers inside that temporary string would be split too. Validate
# its AST before leaving it uninstrumented; the enclosing message Output still
# assigns ownership to the final rendered text. Other macros keep tracing.
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


def _macro_structure(node):
    """Ignore formatting-only whitespace, never expressions or literal content."""
    class StripWhitespace(NodeTransformer):
        def visit_TemplateData(self, item, *args, **kwargs):
            return None if not item.data.strip() else item

        def visit_Output(self, item, *args, **kwargs):
            item = self.generic_visit(item, *args, **kwargs)
            return item if item.nodes else None

    import copy

    return StripWhitespace().visit(copy.deepcopy(node)).dump()


@dataclass(frozen=True)
class TemplateTokenProvenance:
    input_ids: list[int]
    owners: list[int]
    offsets: list[tuple[int, int]]
    rendered_text: str
    char_owners: list[int]
    cross_owner_tokens: int


class _TraceTemplateOutputs(NodeTransformer):
    """Wrap template output nodes with markers carrying active loop variables."""

    def __init__(self, *, minimax: bool = False) -> None:
        self._loop_vars: list[str] = []
        self._minimax = minimax

    def visit_Macro(self, node: nodes.Macro, *args, **kwargs):
        if self._minimax and node.name == "visible_text":
            from jinja2 import Environment

            expected = next(Environment().parse(_MINIMAX_VISIBLE_TEXT).find_all(nodes.Macro))
            if _macro_structure(node) != _macro_structure(expected):
                raise ValueError("Unrecognized MiniMax visible_text macro")
            return node
        return self.generic_visit(node, *args, **kwargs)

    @staticmethod
    def _target_names(target: nodes.Node) -> list[str]:
        if isinstance(target, nodes.Name):
            return [target.name]
        if isinstance(target, (nodes.List, nodes.Tuple)):
            names: list[str] = []
            for item in target.items:
                names.extend(_TraceTemplateOutputs._target_names(item))
            return names
        return []

    def visit_For(self, node: nodes.For, *args, **kwargs):
        names = self._target_names(node.target)
        self._loop_vars.extend(names)
        try:
            return self.generic_visit(node, *args, **kwargs)
        finally:
            if names:
                del self._loop_vars[-len(names) :]

    @staticmethod
    def _references_generation_prompt(node: nodes.Node) -> bool:
        if isinstance(node, nodes.Name) and node.name == "add_generation_prompt":
            return True
        return any(
            isinstance(candidate, nodes.Name)
            and candidate.name == "add_generation_prompt"
            for candidate in node.find_all(nodes.Name)
        )

    def _marker_call(self, phase: str, lineno: int) -> nodes.Call:
        loop_values = [nodes.Name(name, "load") for name in reversed(self._loop_vars)]
        return nodes.Call(
            nodes.Name("_sglang_context_owner_marker", "load"),
            [nodes.Const(phase), *loop_values],
            [],
            None,
            None,
        ).set_lineno(lineno)

    def _marker_output(self, phase: str, lineno: int) -> nodes.Output:
        return nodes.Output([self._marker_call(phase, lineno)]).set_lineno(lineno)

    def visit_If(self, node: nodes.If, *args, **kwargs):
        traces_generation = self._references_generation_prompt(node.test)
        node = self.generic_visit(node, *args, **kwargs)
        if traces_generation:
            node.body = [
                self._marker_output("G", node.lineno),
                *node.body,
                self._marker_output("H", node.lineno),
            ]
        return node

    def visit_Output(self, node: nodes.Output, *args, **kwargs):
        traces_generation = self._references_generation_prompt(node)
        node = self.generic_visit(node, *args, **kwargs)
        traced_nodes = [
            self._marker_call("B", node.lineno),
            *node.nodes,
            self._marker_call("E", node.lineno),
        ]
        if traces_generation:
            traced_nodes = [
                self._marker_call("G", node.lineno),
                *traced_nodes,
                self._marker_call("H", node.lineno),
            ]
        node.nodes = traced_nodes
        return node


@lru_cache(maxsize=32)
def _compile_traced_template(chat_template: str):
    compiled = _compile_jinja_template(chat_template)
    environment = compiled.environment
    minimax = "ns.last_user_index" in chat_template and "<minimax:tool_call>" in chat_template
    traced_ast = _TraceTemplateOutputs(minimax=minimax).visit(environment.parse(chat_template))
    code = environment.compile(traced_ast)
    return environment.template_class.from_code(
        environment,
        code,
        environment.globals,
        None,
    )


def _render_traced_template(
    tokenizer,
    chat_template: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    add_generation_prompt: bool,
    enable_thinking: bool | None,
    extra_template_kwargs: dict[str, Any] | None,
) -> tuple[str, str, re.Pattern[str]]:
    nonce = uuid.uuid4().hex
    marker_prefix = f"\x00sglang-context-owner-{nonce}:"
    owner_by_object = {id(message): msg_id for msg_id, message in enumerate(messages)}

    def owner_marker(phase: str, *loop_values: Any) -> str:
        owner = -1
        for value in loop_values:
            candidate = owner_by_object.get(id(value))
            if candidate is not None:
                owner = candidate
                break
        return f"{marker_prefix}{phase}:{owner}\x00"

    template_kwargs = dict(getattr(tokenizer, "special_tokens_map", {}))
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    template_kwargs.update(extra_template_kwargs or {})
    traced = _compile_traced_template(chat_template).render(
        messages=messages,
        tools=tools,
        documents=None,
        add_generation_prompt=add_generation_prompt,
        _sglang_context_owner_marker=owner_marker,
        **template_kwargs,
    )
    pattern = re.compile(re.escape(marker_prefix) + r"([BEGH]):(-?\d+)\x00")
    return traced, marker_prefix, pattern


def _parse_character_owners(
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
    if len(owners) != len(clean_text):
        raise RuntimeError(
            "Character ownership length does not match canonical chat text."
        )
    if len(owners) == 0:
        return clean_text, owners

    known = [idx for idx, owner in enumerate(owners) if owner >= 0]
    if not known:
        if message_count > 1:
            raise RuntimeError(
                "Chat template emitted multiple messages outside traceable message loops."
            )
        fallback = message_count if add_generation_prompt and message_count == 0 else 0
        owners = [fallback] * len(owners)
        return clean_text, owners

    first_known = known[0]
    first_owner = owners[first_known]
    leading_owner = 0 if message_count > 0 else first_owner
    owners[:first_known] = [leading_owner] * first_known

    previous_owner = owners[first_known]
    for idx in range(first_known + 1, len(owners)):
        if owners[idx] < 0:
            owners[idx] = previous_owner
        else:
            previous_owner = owners[idx]

    if add_generation_prompt and message_count not in owners:
        last_known = known[-1]
        if last_known + 1 < len(owners):
            owners[last_known + 1 :] = [message_count] * (len(owners) - last_known - 1)
    return clean_text, owners


def _encode_with_offsets(tokenizer, text, *, add_special_tokens):
    if bool(getattr(tokenizer, "is_fast", True)):
        # Preserve the tokenizer's native default for an assistant continuation.
        kwargs = {} if add_special_tokens else {"add_special_tokens": False}
        encoded = tokenizer(text, return_offsets_mapping=True, **kwargs)
        input_ids = [int(token_id) for token_id in encoded["input_ids"]]
        offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    else:
        # Original Qwen uses tiktoken. Decode exact token bytes from the single
        # canonical encoding, never independently tokenize message fragments.
        decoder = getattr(getattr(tokenizer, "tokenizer", None), "decode_single_token_bytes", None)
        if not callable(decoder):
            raise RuntimeError("Drop Message ownership requires offset_mapping or exact token bytes")
        input_ids = [int(i) for i in tokenizer.encode(text, add_special_tokens=add_special_tokens)]
        pieces = [decoder(i) for i in input_ids]
        if b"".join(pieces) != text.encode("utf-8"):
            raise RuntimeError("Tokenizer bytes do not reproduce the canonical template")
        from bisect import bisect_left, bisect_right

        char_bytes = [0]
        for char in text:
            char_bytes.append(char_bytes[-1] + len(char.encode("utf-8")))
        offsets = []
        cursor = 0
        for piece in pieces:
            start = bisect_right(char_bytes, cursor) - 1
            cursor += len(piece)
            offsets.append((start, bisect_left(char_bytes, cursor)))
    return input_ids, offsets


def build_template_token_provenance(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    add_generation_prompt: bool,
    enable_thinking: bool | None,
    chat_template: str | None = None,
    template_kwargs: dict[str, Any] | None = None,
) -> TemplateTokenProvenance:
    if chat_template is None:
        chat_template = tokenizer.get_chat_template(tools=tools)
    if template_kwargs and template_kwargs.get("preserve_thinking_history", False):
        from sglang.srt.context_system.thinking_template import prepare_thinking_history

        messages, template_kwargs = prepare_thinking_history(
            tokenizer, messages, tools, {**template_kwargs, "chat_template": chat_template}
        )
        template_kwargs = dict(template_kwargs)
        chat_template = template_kwargs.pop("chat_template")
    traced_text, _, marker_pattern = _render_traced_template(
        tokenizer,
        chat_template,
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        extra_template_kwargs=template_kwargs,
    )
    canonical_text, char_owners = _parse_character_owners(
        traced_text,
        marker_pattern,
        message_count=len(messages),
        add_generation_prompt=add_generation_prompt,
    )

    return _token_provenance(tokenizer, canonical_text, char_owners, add_special_tokens=False)


def _token_provenance(tokenizer, canonical_text, char_owners, *, add_special_tokens):
    input_ids, offsets = _encode_with_offsets(tokenizer, canonical_text, add_special_tokens=add_special_tokens)
    owners: list[int] = []
    cross_owner_tokens = 0
    previous_owner = 0
    for token_idx, (start, end) in enumerate(offsets):
        if start < 0 or end < start or end > len(char_owners):
            raise RuntimeError(f"Token {token_idx} has an invalid character offset.")
        if start == end:
            owner = previous_owner
        else:
            token_owners = char_owners[start:end]
            if any(owner < 0 for owner in token_owners):
                raise RuntimeError(
                    f"Token {token_idx} covers an unowned template character."
                )
            owner = token_owners[0]
            if any(candidate != owner for candidate in token_owners[1:]):
                cross_owner_tokens += 1
        owners.append(owner)
        previous_owner = owner

    return TemplateTokenProvenance(
        input_ids=input_ids,
        owners=owners,
        offsets=offsets,
        rendered_text=canonical_text,
        char_owners=char_owners,
        cross_owner_tokens=cross_owner_tokens,
    )


def build_conversation_token_provenance(tokenizer, conv, messages, canonical_text):
    """Trace native ChatML text boundaries, then encode the full prompt once.

    Legacy Qwen can use SGLang's existing --chat-template chatml. Prefix
    rendering verifies append-only boundaries including role/separator text;
    it never tokenizes messages independently or guesses from content matches.
    """
    from sglang.srt.parser.conversation import SeparatorStyle

    if conv.sep_style != SeparatorStyle.CHATML:
        raise ValueError("Context conversation provenance requires native ChatML")
    systems = [i for i, message in enumerate(messages) if message["role"] == "system"]
    if systems not in ([], [0]) or conv.offset:
        raise ValueError("Context ChatML requires at most one leading system message")
    message_owners = [i for i in range(len(messages)) if i not in systems]
    if len(conv.messages) == len(message_owners) + 1 and conv.messages[-1][1] is None:
        message_owners.append(len(messages))
    if len(conv.messages) != len(message_owners):
        raise ValueError("Native conversation did not preserve the message sequence")
    prefix = conv.copy()
    prefix.messages = []
    system_text = prefix.get_prompt()
    if not canonical_text.startswith(system_text):
        raise ValueError("Native ChatML system prefix changed during rendering")
    char_owners = [0] * len(system_text)
    for message, owner in zip(conv.messages, message_owners):
        prefix.messages.append(message)
        rendered = prefix.get_prompt()
        # continue_final_message may strip the final separator.
        end = min(len(rendered), len(canonical_text))
        if not canonical_text.startswith(rendered[:end]) or end < len(char_owners):
            raise ValueError("Native ChatML rendering is not append-only")
        char_owners.extend([owner] * (end - len(char_owners)))
    char_owners.extend([len(messages)] * (len(canonical_text) - len(char_owners)))
    return _token_provenance(tokenizer, canonical_text, char_owners, add_special_tokens=True)


def append_assistant_prefix(
    trace: TemplateTokenProvenance, tokenizer, text: str, *, owner: int
) -> TemplateTokenProvenance:
    """Mirror native continue_final_message's separate encode and BOS removal.

    Do not re-tokenize the joined text: SGLang deliberately appends independently
    encoded assistant-prefix IDs. Their character offsets still point into the
    joined text, so text Drop sees the same content and exact token boundaries.
    """
    ids, offsets = _encode_with_offsets(tokenizer, text, add_special_tokens=True)
    if ids and ids[0] == tokenizer.bos_token_id:
        ids = ids[1:]
        offsets = offsets[1:]
    if any(start < 0 or end < start or end > len(text) for start, end in offsets):
        raise ValueError("Assistant prefix has an invalid tokenizer offset")
    offset = len(trace.rendered_text)
    return TemplateTokenProvenance(
        input_ids=trace.input_ids + ids,
        owners=trace.owners + [owner] * len(ids),
        offsets=trace.offsets
        + [(start + offset, end + offset) for start, end in offsets],
        rendered_text=trace.rendered_text + text,
        char_owners=trace.char_owners + [owner] * len(text),
        cross_owner_tokens=trace.cross_owner_tokens,
    )
