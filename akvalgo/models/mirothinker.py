"""MiroThinker MCP request preparation and text tool-call recovery."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from akvalgo.types import JsonDict, ToolCall

MIROTHINKER_MCP_PROTOCOL = "mirothinker_mcp"
MIROTHINKER_MCP_PROTOCOL_VERSION = "mirothinker-mcp-v6"


def current_mirothinker_mcp_date() -> str:
    """Return the UTC date embedded in one backend's MCP prompt."""

    return datetime.now(UTC).date().isoformat()


def render_mirothinker_mcp_system_prompt(
    tools: Iterable[JsonDict],
    *,
    today: str,
) -> str:
    """Render MiroThinker's official MCP tool prompt with request tools."""

    server_sections: dict[str, list[str]] = {}
    for tool in tools:
        function = tool.get("function")
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not name or not isinstance(parameters, Mapping):
            continue
        description = function.get("description")
        description_text = (
            description.strip()
            if isinstance(description, str) and description.strip()
            else "No description provided."
        )
        # Presentation metadata only; execution does not validate server names.
        server_name = tool.get("mcp_server_name", "default")
        server_sections.setdefault(server_name, []).append(
            "\n".join(
                (
                    f"### Tool name: {name}",
                    f"Description: {description_text}",
                    "",
                    "Input JSON schema: " + json.dumps(parameters, ensure_ascii=False),
                )
            )
        )

    rendered_tools = "\n\n".join(
        f"## Server name: {server_name}\n" + "\n\n".join(sections)
        for server_name, sections in server_sections.items()
    )
    return "\n".join(
        (
            "In this environment you have access to a set of tools you can use to answer the user's question.",
            "You only have access to the tools provided below. You can only use one tool per message, and will receive the result of that tool in the user's next response. You use tools step-by-step to accomplish a given task, with each tool-use informed by the result of the previous tool-use. Today is: "
            + today,
            "# Tool-Use Formatting Instructions",
            "Tool-use is formatted using XML-style tags. The tool-use is enclosed in <use_mcp_tool></use_mcp_tool> and each parameter is similarly enclosed within its own set of tags.",
            "The Model Context Protocol (MCP) connects to servers that provide additional tools and resources to extend your capabilities. You can use the server's tools via the `use_mcp_tool`.",
            "Description:",
            "Request to use a tool provided by a MCP server. Each MCP server can provide multiple tools with different capabilities. Tools have defined input schemas that specify required and optional parameters.",
            "Parameters:",
            "- server_name: (required) The name of the MCP server providing the tool",
            "- tool_name: (required) The name of the tool to execute",
            "- arguments: (required) A JSON object containing the tool's input parameters, following the tool's input schema, quotes within string must be properly escaped, ensure it's valid JSON",
            "Usage:",
            "<use_mcp_tool>",
            "<server_name>server name here</server_name>",
            "<tool_name>tool name here</tool_name>",
            "<arguments>",
            "{",
            '  "param1": "value1",',
            '  "param2": "value2 \\"escaped string\\""',
            "}",
            "</arguments>",
            "</use_mcp_tool>",
            "Important Notes:",
            "- Tool-use must be placed **at the end** of your response, **top-level**, and not nested within other tags.",
            "- Always adhere to this format for the tool use to ensure proper parsing and execution.",
            "String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.",
            "Here are the functions available in JSONSchema format:",
            rendered_tools,
            "# General Objective",
            "",
            "You accomplish a given task iteratively, breaking it down into clear steps and working through them methodically.",
        )
    ).strip()


@dataclass(frozen=True)
class ToolProtocolPreparation:
    """Wire messages and canonical-to-wire message-ID mapping."""

    messages: tuple[JsonDict, ...]
    tools: tuple[JsonDict, ...]
    canonical_message_count: int
    canonical_message_offset: int

    def wire_message_id(self, canonical_message_id: int) -> int:
        if (
            type(canonical_message_id) is not int
            or canonical_message_id < 0
            or canonical_message_id >= self.canonical_message_count
        ):
            raise ValueError("tool protocol message ID must identify a canonical message")
        return canonical_message_id + self.canonical_message_offset


def prepare_mirothinker_mcp_request(
    messages: Sequence[Mapping[str, object]],
    tools: Sequence[JsonDict],
    *,
    today: str,
) -> ToolProtocolPreparation:
    """Prepare the exact MiroThinker MCP wire history.

    Tool-free internal requests, including full-summary calls, retain their
    original system prompt. Agent requests merge the MCP instructions into an
    existing leading system message or insert one when the benchmark starts
    directly with a user message.
    """

    prepared = [dict(message) for message in messages]
    if not tools:
        return ToolProtocolPreparation(
            messages=tuple(prepared),
            tools=(),
            canonical_message_count=len(prepared),
            canonical_message_offset=0,
        )

    prompt = render_mirothinker_mcp_system_prompt(tools, today=today)
    if prepared and prepared[0].get("role") == "system":
        existing = prepared[0].get("content")
        existing_text = existing if isinstance(existing, str) else ""
        prepared[0]["content"] = f"{existing_text.rstrip()}\n\n{prompt}" if existing_text.strip() else prompt
        offset = 0
    else:
        prepared.insert(0, {"role": "system", "content": prompt})
        offset = 1
    return ToolProtocolPreparation(
        messages=tuple(prepared),
        tools=(),
        canonical_message_count=len(messages),
        canonical_message_offset=offset,
    )


MIRO_TOOL_BLOCK = re.compile(
    r"(?:"
    r"<tool_call>\s*(?P<qwen>.*?)\s*</tool_call>"
    r"|<<tool_call>>\s*(?P<double_qwen>.*?)\s*<</tool_call>>"
    r"|<tool>\s*(?P<legacy>.*?)\s*(?:</tool>|</use_mcp_tool>)"
    r"|<use_mcp_tool>\s*(?P<mcp>.*?)\s*</use_mcp_tool>"
    r")",
    re.IGNORECASE | re.DOTALL,
)


def request_tool_names(tools: Iterable[JsonDict]) -> frozenset[str]:
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function")
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return frozenset(names)


def json_object(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def tag_values(body: str, tag: str) -> list[str]:
    return [
        value.strip()
        for value in re.findall(
            rf"<{tag}>\s*(.*?)\s*</{tag}>",
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )
    ]


def normalize_miro_tool_arguments(
    name: str,
    arguments: dict[str, object],
) -> dict[str, object]:
    """Apply narrowly scoped aliases used by MiroThinker's official tools."""

    if name == "search" and set(arguments) == {"q"} and isinstance(arguments["q"], str):
        return {"query": arguments["q"]}
    return arguments


def parse_miro_tool_body(
    body: str,
    offered_names: frozenset[str],
) -> tuple[str, str] | None:
    """Return one complete, offered tool call from a Miro/Qwen tag body."""

    try:
        candidate = json.loads(body.strip())
    except json.JSONDecodeError:
        candidate = None
    if isinstance(candidate, dict):
        name = candidate.get("name")
        arguments = json_object(candidate.get("arguments"))
        if isinstance(name, str) and name in offered_names and arguments is not None:
            arguments = normalize_miro_tool_arguments(name, arguments)
            return name, json.dumps(
                arguments,
                ensure_ascii=False,
                separators=(",", ":"),
            )

    argument_values = tag_values(body, "arguments")
    if len(argument_values) != 1:
        return None
    arguments = json_object(argument_values[0])
    if arguments is None:
        return None
    matching_tool_names = {name for name in tag_values(body, "tool_name") if name in offered_names}
    if len(matching_tool_names) != 1:
        return None
    name = next(iter(matching_tool_names))
    arguments = normalize_miro_tool_arguments(name, arguments)
    return name, json.dumps(
        arguments,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def single_miro_tag_value(body: str, tag: str) -> str:
    """Return one unambiguous XML field, or an empty failure sentinel."""

    values = tag_values(body, tag)
    return values[0] if len(values) == 1 else ""


def parse_miro_mcp_body(body: str) -> tuple[str, str, str]:
    """Extract one fully bounded MCP XML attempt without validating it."""

    server_name = single_miro_tag_value(body, "server_name")
    name = single_miro_tag_value(body, "tool_name")
    raw_arguments = single_miro_tag_value(body, "arguments")
    arguments = json_object(raw_arguments)
    if arguments is None:
        return server_name, name, raw_arguments
    arguments = normalize_miro_tool_arguments(name, arguments)
    return (
        server_name,
        name,
        json.dumps(
            arguments,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


def recover_miro_tool_calls(
    content: str | None,
    tools: Iterable[JsonDict],
) -> tuple[str | None, tuple[ToolCall, ...]]:
    """Recover complete MCP attempts and valid legacy/Qwen tool tags."""

    if not isinstance(content, str) or not content:
        return content, ()
    offered_names = request_tool_names(tools)
    if not offered_names:
        return content, ()

    calls: list[ToolCall] = []
    visible_parts: list[str] = []
    cursor = 0
    for match in MIRO_TOOL_BLOCK.finditer(content):
        body = next(
            (
                value
                for value in (
                    match.group("qwen"),
                    match.group("double_qwen"),
                    match.group("legacy"),
                    match.group("mcp"),
                )
                if value is not None
            ),
            "",
        )
        server_name: str | None = None
        if match.group("mcp") is not None:
            server_name, name, arguments = parse_miro_mcp_body(body)
        else:
            parsed = parse_miro_tool_body(body, offered_names)
            if parsed is None:
                continue
            name, arguments = parsed
        visible_parts.append(content[cursor : match.start()])
        calls.append(
            ToolCall(
                f"call_miro_{len(calls)}",
                name,
                arguments,
                server_name=server_name,
            )
        )
        cursor = match.end()
    if not calls:
        return content, ()
    visible_parts.append(content[cursor:])
    visible = "".join(visible_parts)
    return (visible if visible.strip() else None), tuple(calls)
