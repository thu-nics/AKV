"""Optional durable per-request HTTP/SSE evidence for an external harness."""

from __future__ import annotations

import gzip
import json
import os
import time
import uuid
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from akvalgo.types import BackendTraceContext


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path: Path, data: bytes):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def redacted_headers(headers):
    sensitive = {"authorization", "proxy-authorization", "api-key", "x-api-key", "openai-api-key"}
    return {name: "<redacted>" if name.lower() in sensitive else value for name, value in headers.items()}


class CallCapture:
    """Files belong to one physical call; an existing destination is an error.

    Filesystem errors propagate as OSError so the harness treats failed evidence
    writes as a run failure. A partial response remains available after failure.
    """

    def __init__(self, context: BackendTraceContext, *, url, payload, headers, timeout):
        self.path, self.trace = Path(context.artifact_dir), context.metadata
        self.url, self.payload, self.headers, self.timeout = url, payload, headers, timeout
        self.result = None
        self.stack = ExitStack()
        self.response_bytes = self.chunks = self.events = self.outputs = 0
        self.first_chunk = self.first_event = self.first_output = self.last_output = self.done = None

    def __enter__(self):
        json.dumps(self.trace, allow_nan=False)
        self.path.mkdir(parents=True, mode=0o700, exist_ok=False)
        atomic_write(self.path / "request.json.gz", gzip.compress(self.payload, mtime=0))
        try:
            streams = []
            for name in ("response.sse.partial.gz", "events.partial.jsonl.gz"):
                raw = self.stack.enter_context(
                    os.fdopen(os.open(self.path / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")
                )
                streams.append(self.stack.enter_context(gzip.GzipFile(fileobj=raw, mode="wb", mtime=0)))
            self.response, self.event_stream = streams
        except BaseException:
            self.stack.close()
            raise
        self.started = time.monotonic()
        self.started_at = utc_now()
        return self

    def write_bytes(self, data):
        if data and self.first_chunk is None:
            self.first_chunk = (time.monotonic(), utc_now())
        self.response.write(data)
        self.response_bytes += len(data)
        self.chunks += 1

    def capture_stream(self, chunks):
        for chunk in chunks:
            self.write_bytes(chunk)
            yield chunk

    def record_event(self, text, parsed):
        stamp = (time.monotonic(), utc_now())
        self.first_event = self.first_event or stamp
        choices = (parsed or {}).get("choices") or []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        delta = choice.get("delta") or {}
        fields = [
            name for name in ("content", "reasoning", "reasoning_content", "tool_calls") if delta.get(name)
        ]
        if fields:
            self.first_output = self.first_output or stamp
            self.last_output = stamp
        if text == "[DONE]":
            self.done = stamp
        value = {
            "event_index": self.events,
            "output_index": self.outputs if fields else None,
            "received_at": stamp[1],
            "elapsed_seconds": stamp[0] - self.started,
            "event": "done" if text == "[DONE]" else "data",
            "output_fields": fields,
            "finish_reason": choice.get("finish_reason"),
            "has_usage": isinstance((parsed or {}).get("usage"), dict),
            "data": parsed if parsed is not None else text,
        }
        self.event_stream.write(
            (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8", "backslashreplace")
        )
        self.events += 1
        self.outputs += bool(fields)

    def __exit__(self, kind, error, traceback):
        try:
            body = getattr(error, "response_body", None)
            if body and not self.response_bytes:
                self.write_bytes(body)
        finally:
            self.stack.close()
        complete = error is None and self.result is not None
        response_name, events_name = "response.sse.partial.gz", "events.partial.jsonl.gz"
        if complete:
            os.replace(self.path / response_name, self.path / "response.sse.gz")
            os.replace(self.path / events_name, self.path / "events.jsonl.gz")
            response_name, events_name = "response.sse.gz", "events.jsonl.gz"
        tokens = (self.result.usage or {}).get("completion_tokens") if self.result is not None else None
        tpot = None
        if type(tokens) is int and tokens > 1 and self.first_output and self.done:
            tpot = (self.done[0] - self.first_output[0]) / (tokens - 1)
        timing = {
            "started_at": self.started_at,
            "finished_at": utc_now(),
            "duration_seconds": time.monotonic() - self.started,
            "tpot_seconds": tpot,
            "completion_tokens": tokens,
        }
        for name, elapsed_name, stamp in (
            ("first_response_chunk", "time_to_first_response_chunk_seconds", self.first_chunk),
            ("first_sse_event", "time_to_first_sse_event_seconds", self.first_event),
            ("first_output", "ttft_seconds", self.first_output),
            ("last_output", "time_to_last_output_seconds", self.last_output),
            ("stream_completed", "user_latency_seconds", self.done),
        ):
            timing[name + "_at"] = stamp[1] if stamp else None
            timing[elapsed_name] = stamp[0] - self.started if stamp else None
        result = None
        if self.result is not None:
            result = asdict(self.result)
            result["message"]["tool_calls"] = [call.as_openai() for call in self.result.message.tool_calls]
            result["message"]["recovered_tool_calls"] = [
                call.as_recovered() for call in self.result.message.recovered_tool_calls
            ]
        call = {
            "schema_version": 2,
            "trace": self.trace,
            "status": "completed" if complete else "failed",
            "request": {
                "url": self.url,
                "headers": redacted_headers(self.headers),
                "timeout_seconds": self.timeout,
                "body_path": "request.json.gz",
                "body_bytes": len(self.payload),
            },
            "response": {
                "body_path": response_name,
                "body_bytes": self.response_bytes,
                "transport_chunks": self.chunks,
                "complete": complete,
                "events_path": events_name,
                "sse_events": self.events,
                "output_events": self.outputs,
            },
            "timing": timing,
            "result": result,
            "error": {
                "type": kind.__name__,
                "message": str(error),
                "http_status": getattr(error, "http_status", None),
            }
            if error
            else None,
        }
        atomic_write(
            self.path / "call.json",
            (json.dumps(call, ensure_ascii=False, indent=2) + "\n").encode("utf-8", "backslashreplace"),
        )
