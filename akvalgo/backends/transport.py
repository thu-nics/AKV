"""Native HTTP and incremental SSE framing; no inference or SDK dependency."""

from __future__ import annotations

import codecs
import http.client
import ipaddress
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator

from akvalgo.types import BackendError, ContextLengthError


def encode_body(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8", errors="backslashreplace")


def backend_error(message: str, *, response_body=None, http_status=None) -> BackendError:
    limits = (
        "longer than the model's context length",
        "exceeds the model's maximum context length",
        "exceeds the model context length",
        "exceeds the usable context length",
        "An occurrence execution position exceeds the model/RoPE limit",
        '"context_length_exceeded"',
    )
    kind = ContextLengthError if any(text in message for text in limits) else BackendError
    return kind(message, response_body=response_body, http_status=http_status)


def chat_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("base_url must be an HTTP(S) endpoint without credentials, query or fragment")
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + ("/chat/completions" if normalized.endswith("/v1") else "/v1/chat/completions")


def loopback_url(url: str) -> bool:
    host = urllib.parse.urlsplit(url).hostname
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def stream_http(url: str, payload: dict, *, headers: dict, timeout: float) -> Iterator[bytes]:
    """Stream HTTP bytes, bypassing environment proxies only for loopback."""
    request = urllib.request.Request(
        url,
        data=encode_body(payload),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream", **headers},
        method="POST",
    )
    handlers = [urllib.request.ProxyHandler({})] if loopback_url(url) else []
    try:
        with urllib.request.build_opener(*handlers).open(request, timeout=timeout) as response:
            while chunk := response.read1(65536):
                yield chunk
    except urllib.error.HTTPError as error:
        body = error.read()
        raise backend_error(
            f"chat backend HTTP {error.code}: {body.decode('utf-8', errors='replace')}",
            response_body=body,
            http_status=error.code,
        ) from error
    except (urllib.error.URLError, OSError, http.client.HTTPException) as error:
        raise BackendError(f"chat backend transport failed: {error}") from error


def iter_sse_data(chunks: Iterable[bytes]) -> Iterator[str]:
    """Frame events across arbitrary byte, UTF-8, CRLF and HTTP chunk boundaries."""
    decoder = codecs.getincrementaldecoder("utf-8-sig")("strict")
    buffer = ""
    data = []

    def line_value(line):
        if not line:
            if data:
                value = "\n".join(data)
                data.clear()
                return value
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
        elif line == "data":
            data.append("")
        return None

    try:
        for chunk in chunks:
            buffer += decoder.decode(chunk)
            while match := re.search(r"\r\n|[\r\n]", buffer):
                if match.group() == "\r" and match.end() == len(buffer):
                    break
                line, buffer = buffer[: match.start()], buffer[match.end() :]
                value = line_value(line)
                if value is not None:
                    yield value
        buffer += decoder.decode(b"", final=True)
        for line in re.split(r"\r\n|[\r\n]", buffer):
            value = line_value(line)
            if value is not None:
                yield value
        if data:
            yield "\n".join(data)
    except UnicodeDecodeError as error:
        raise BackendError("chat backend returned invalid UTF-8 SSE data") from error
