"""SSE formatting and parsing helpers for M3 streaming.

Pure functions (no I/O) so unit tests exercise the wire format without a
server. Providers yield plain text chunks; this module owns the
``text/event-stream`` framing. Parsing tolerates fragmented network reads:
feed arbitrary byte-chunks into :class:`SseIncrementalDecoder` and it emits
complete events split on blank lines.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from src.models import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
)

DONE_PAYLOAD = "[DONE]"

ParsedSseEvent = dict[str, Any] | Literal["DONE"]


def format_chunk(
    *,
    chunk_id: str,
    created: int,
    model: str,
    provider: str,
    request_id: str,
    content: str | None = None,
    role: Literal["assistant"] | None = None,
    finish_reason: Literal["stop", "length"] | None = None,
) -> str:
    """Format one OpenAI-style ``chat.completion.chunk`` SSE frame.

    Typed models validate the shape; ``delta`` omits ``None`` fields while
    ``finish_reason`` is preserved as explicit ``null`` for content chunks
    (locked OpenAI-style wire format).
    """
    delta = ChatCompletionChunkDelta(role=role, content=content)
    choice = ChatCompletionChunkChoice(index=0, delta=delta, finish_reason=finish_reason)
    chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=model,
        choices=[choice],
        provider=provider,
        request_id=request_id,
    )
    dumped = chunk.model_dump()
    # Omit null delta fields (``{"role":..,"content":..}`` -> ``{}`` when empty)
    # but keep ``finish_reason: null`` for non-terminal chunks.
    raw_delta = dumped["choices"][0]["delta"]
    cleaned_delta = {k: v for k, v in raw_delta.items() if v is not None}
    dumped["choices"][0]["delta"] = cleaned_delta
    payload = json.dumps(dumped, separators=(",", ":"))
    return f"data: {payload}\n\n"


def format_done() -> str:
    """Terminal frame for clean completion. Never emitted after an error."""
    return f"data: {DONE_PAYLOAD}\n\n"


def format_error(
    *,
    code: str,
    message: str,
    provider: str | None,
    retryable: bool,
    request_id: str,
) -> str:
    """Mid-stream failure frame. The stream closes immediately after."""
    payload = json.dumps(
        {
            "error": {
                "code": code,
                "message": message,
                "provider": provider,
                "retryable": retryable,
                "request_id": request_id,
            }
        },
        separators=(",", ":"),
    )
    return f"data: {payload}\n\n"


def parse_sse_line(line: str) -> str:
    """Extract the payload from a single ``data: ...`` line.

    Raises ``ValueError`` for non-data lines (blank lines, comments, fields).
    """
    stripped = line.strip()
    if not stripped.startswith("data:"):
        raise ValueError(f"not an SSE data line: {line!r}")
    payload = stripped[len("data:") :]
    if payload.startswith(" "):
        payload = payload[1:]
    return payload


def parse_sse_payload(payload: str) -> ParsedSseEvent:
    """Parse a bare SSE payload: ``[DONE]`` sentinel or JSON object."""
    if payload.strip() == DONE_PAYLOAD:
        return "DONE"
    try:
        decoded: Any = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed SSE JSON payload: {payload!r}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"SSE payload must be a JSON object: {payload!r}")
    return decoded


class SseIncrementalDecoder:
    """Accumulates fragmented ``text/event-stream`` bytes into parsed events.

    Network reads do not align with SSE events, so callers feed arbitrary
    string fragments; complete events (terminated by a blank line) are
    returned parsed. Incomplete trailing data is buffered. Blank-only blocks
    are skipped; ``:`` comment lines are ignored per SSE convention.
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, data: str) -> list[ParsedSseEvent]:
        self._buffer += data
        # Normalize CRLF so both ``\\n\\n`` and ``\\r\\n\\r\\n`` terminate events.
        normalized = self._buffer.replace("\r\n", "\n")
        parts = normalized.split("\n\n")
        # Last part may be an incomplete event; keep it buffered.
        self._buffer = parts.pop()
        events: list[ParsedSseEvent] = []
        for raw in parts:
            parsed = self._parse_block(raw)
            if parsed is not None:
                events.append(parsed)
        return events

    def flush(self) -> list[ParsedSseEvent]:
        """Parse any remaining buffered data as a final event, if non-blank."""
        remaining = self._buffer
        self._buffer = ""
        if not remaining.strip():
            return []
        parsed = self._parse_block(remaining.replace("\r\n", "\n"))
        return [parsed] if parsed is not None else []

    def _parse_block(self, raw: str) -> ParsedSseEvent | None:
        data_lines: list[str] = []
        for line in raw.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(":"):
                continue  # SSE comment / heartbeat
            if stripped.startswith("data:"):
                data_lines.append(parse_sse_line(stripped))
            # Ignore other SSE fields (event:, id:, retry:) for M3 subset.
        if not data_lines:
            return None
        # Multiple data lines in one event join with newline per SSE spec.
        combined = "\n".join(data_lines)
        return parse_sse_payload(combined)


def parse_sse_stream(text: str) -> list[ParsedSseEvent]:
    """Parse a complete SSE body into events (convenience for tests)."""
    decoder = SseIncrementalDecoder()
    events = decoder.feed(text)
    events.extend(decoder.flush())
    return events
