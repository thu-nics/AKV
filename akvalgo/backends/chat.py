"""Send one final Chat Completions payload over native HTTP/SSE."""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Mapping
from contextlib import nullcontext
from math import isfinite

from akvalgo.types import (
    BackendError,
    BackendTraceContext,
    GeneratedMessage,
    GenerationResult,
    JsonDict,
    ToolCall,
)

from .capture import CallCapture
from .transport import backend_error, chat_url, encode_body, iter_sse_data, stream_http


def positive_timeout(value):
    if type(value) not in (int, float) or not isfinite(value) or value <= 0:
        raise ValueError("timeout must be a positive finite number")
    return value


def stream_string(value):
    if not isinstance(value, str):
        raise BackendError("SSE text and tool-call fragments must be strings")
    return value


def send_chat_request(
    base_url: str,
    payload: JsonDict,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = 600,
    trace_context: BackendTraceContext | None = None,
) -> GenerationResult:
    """Send a final JSON body and collect one complete SSE response.

    The caller owns message conversion, policy application, tools and retries.
    Payload fields are preserved, with no model-specific request or response
    adaptation. Capture is optional; a successful stream must end with [DONE].
    """
    url = chat_url(base_url)
    timeout = positive_timeout(timeout)
    if payload.get("stream") is not True:
        raise ValueError("The SSE client requires a payload with stream=true")
    payload = copy.deepcopy(payload)
    headers = dict(headers or {})
    capture = (
        CallCapture(trace_context, url=url, payload=encode_body(payload), headers=headers, timeout=timeout)
        if trace_context is not None
        else nullcontext(None)
    )
    with capture as record:
        started = time.monotonic()
        chunks = iter(stream_http(url, payload, headers=headers, timeout=timeout))
        try:
            data = collect(record.capture_stream(chunks) if record else chunks, record=record)
            result = make_result(data, started)
            if record:
                record.result = result
            return result
        finally:
            close = getattr(chunks, "close", None)
            if close is not None:
                close()


def collect(chunks, *, record=None):
    data = {
        "content": [],
        "reasoning_content": [],
        "calls": {},
        "usage": None,
        "finish_reason": None,
        "response_id": None,
        "server_metrics": None,
        "sglext": {},
        "matched_stop": None,
        "tool_call_missing": False,
        "sse_chunks": 0,
    }
    for text in iter_sse_data(chunks):
        if text == "[DONE]":
            if record:
                record.record_event(text, None)
            return data
        try:
            chunk = json.loads(text)
        except json.JSONDecodeError as error:
            raise BackendError("chat backend returned invalid SSE JSON") from error
        if not isinstance(chunk, dict):
            raise BackendError("chat backend SSE data must be an object")
        if chunk.get("error") is not None or chunk.get("object") == "error":
            raise backend_error("chat backend stream error: " + json.dumps(chunk, ensure_ascii=False))
        choices = chunk.get("choices") or []
        if not isinstance(choices, list) or len(choices) > 1:
            raise BackendError("Expected a single completion choice")
        choice = choices[0] if choices else {}
        if not isinstance(choice, dict) or choice.get("index", 0) not in (None, 0):
            raise BackendError("Invalid completion choice")
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            raise BackendError("SSE delta must be an object")
        if record:
            record.record_event(text, chunk)
        data["sse_chunks"] += 1
        if chunk.get("id") is not None:
            data["response_id"] = str(chunk["id"])
        for name in ("usage", "server_metrics"):
            if chunk.get(name) is not None:
                if not isinstance(chunk[name], dict):
                    raise BackendError(f"SSE {name} must be an object or null")
                data[name] = chunk[name]
        if isinstance(chunk.get("sglext"), dict):
            data["sglext"].update(chunk["sglext"])
        for name in ("finish_reason", "matched_stop"):
            if choice.get(name) is not None:
                data[name] = str(choice[name])
        data["tool_call_missing"] |= bool(choice.get("tool_call_missing") or delta.get("tool_call_missing"))
        if delta.get("content") is not None:
            data["content"].append(stream_string(delta["content"]))
        reasoning = delta.get("reasoning_content")
        if reasoning is None:
            reasoning = delta.get("reasoning")
        if reasoning is not None:
            data["reasoning_content"].append(stream_string(reasoning))
        calls = delta.get("tool_calls") or []
        if not isinstance(calls, list):
            raise BackendError("tool_calls must be a list")
        for fallback, item in enumerate(calls):
            if not isinstance(item, dict):
                raise BackendError("tool_call fragment must be an object")
            index = item.get("index") if item.get("index") is not None else fallback
            function = item.get("function") or {}
            if type(index) is not int or index < 0 or not isinstance(function, dict):
                raise BackendError("Invalid tool_call index or function")
            call = data["calls"].setdefault(index, {"id": "", "name": "", "arguments": ""})
            for key, value in (
                ("id", item.get("id")),
                ("name", function.get("name")),
                ("arguments", function.get("arguments")),
            ):
                if value is not None:
                    call[key] += stream_string(value)
    raise BackendError("chat backend SSE stream ended before [DONE]")


def make_result(data, started):
    calls = tuple(
        ToolCall(call["id"] or f"call_{index}", call["name"], call["arguments"])
        for index, call in sorted(data["calls"].items())
    )
    metadata = {
        name: data[name]
        for name in (
            "response_id",
            "finish_reason",
            "server_metrics",
            "sglext",
            "matched_stop",
            "tool_call_missing",
            "sse_chunks",
        )
    }
    return GenerationResult(
        GeneratedMessage(
            content="".join(data["content"]) if data["content"] else None,
            reasoning_content="".join(data["reasoning_content"]) if data["reasoning_content"] else None,
            tool_calls=calls,
        ),
        data["finish_reason"],
        time.monotonic() - started,
        usage=data["usage"],
        metadata=metadata,
    )
