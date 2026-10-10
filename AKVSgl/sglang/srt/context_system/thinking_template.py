"""Request-local history retention using the selected native Jinja template.

Model execution, tool parsing and sampling remain native. Only recognized
history-elision guards change, and no tokenizer-global template is mutated.
"""

import re
from functools import lru_cache

from jinja2 import Environment


@lru_cache(maxsize=16)
def retained_template(template):
    marker = "{# SGLANG_CONTEXT_THINKING_HISTORY_V1:"
    if template.startswith(marker):
        family = template[len(marker) :].split(" #}", 1)[0]
        if family in ("gpt-oss", "qwen", "minimax", "native"):
            return template, family
    Environment().parse(template)
    if "future_final_message.found" in template and "<|start|>assistant" in template:
        guard = "and not future_final_message.found"
        if template.count(guard) != 2:
            raise ValueError("Unrecognized GPT-OSS thinking history guards")
        patched = template.replace(
            guard, "and (preserve_thinking_history or not future_final_message.found)"
        )
        final = '{{- "<|start|>assistant<|channel|>final<|message|>" + message.content + "<|end|>" }}'
        if patched.count(final) != 1:
            raise ValueError("Unrecognized GPT-OSS historical final-message template")
        patched = patched.replace(
            final,
            (
                "{%- if preserve_thinking_history and message.thinking %}"
                '{{- "<|start|>assistant<|channel|>analysis<|message|>" + message.thinking + "<|end|>" }}'
                "{%- endif %}" + final
            ),
        )
        # Also support this template through upstream's --chat-template option,
        # which receives OpenAI reasoning_content without our Python adapter.
        patched = patched.replace(
            "message.thinking", "(message.reasoning_content or message.thinking)"
        )
        family = "gpt-oss"
    elif "ns.last_user_index" in template and "<minimax:tool_call>" in template:
        patched, count = re.subn(
            r"({%-?\s*if\s+)reasoning_content\s+and\s+loop\.index0\s*>\s*ns\.last_user_index(\s*-?%})",
            r"\1reasoning_content and (preserve_thinking_history or loop.index0 > ns.last_user_index)\2",
            template,
        )
        if count != 1:
            raise ValueError("Unrecognized MiniMax thinking history guard")
        family = "minimax"
    elif "ns.last_query_index" in template and "reasoning_content" in template:
        patched, count = re.subn(
            r"({%-?\s*if\s+)loop\.index0\s*>\s*ns\.last_query_index(\s*-?%})",
            r"\1preserve_thinking_history or loop.index0 > ns.last_query_index\2",
            template,
        )
        if count != 1:
            raise ValueError("Unrecognized Qwen thinking history guard")
        family = "qwen"
    else:
        # Templates with native retention still receive the public preference.
        # ThinkingDrop's exact provenance check rejects silently omitted text.
        patched, family = template, "native"
    Environment().parse(patched)
    return marker + family + " #}" + patched, family


def prepare_thinking_history(tokenizer, messages, tools, template_kwargs):
    """Return a private message view and kwargs, retaining native formatting."""
    if not template_kwargs.get("preserve_thinking_history", False):
        return messages, template_kwargs
    template = template_kwargs.get("chat_template") or tokenizer.get_chat_template(
        tools=tools
    )
    patched, family = retained_template(template)
    if family == "gpt-oss":
        messages = [dict(message) for message in messages]
        for message in messages:
            reasoning = message.get("reasoning_content")
            if message.get("role") == "assistant" and reasoning:
                if message.get("thinking") not in (None, "", reasoning):
                    raise ValueError(
                        "Conflicting thinking and reasoning_content fields"
                    )
                message["thinking"] = reasoning
    return messages, {**template_kwargs, "chat_template": patched}
