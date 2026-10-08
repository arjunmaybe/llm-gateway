"""SSE formatting/parsing tests: wire format, fragmentation, malformed input."""

from __future__ import annotations

import json

import pytest

from src.proxy.sse_parser import (
    SseIncrementalDecoder,
    format_chunk,
    format_done,
    format_error,
    parse_sse_line,
    parse_sse_payload,
    parse_sse_stream,
)


def test_format_chunk_first_includes_role() -> None:
    raw = format_chunk(
        chunk_id="chatcmpl-abc",
        created=123,
        model="mock-a",
        provider="mock-a",
        request_id="req-1",
        content="hello",
        role="assistant",
    )
    assert raw.startswith("data: ")
    assert raw.endswith("\n\n")
    payload = json.loads(parse_sse_line(raw.strip().splitlines()[0]))
    assert payload["id"] == "chatcmpl-abc"
    assert payload["object"] == "chat.completion.chunk"
    assert payload["choices"][0]["delta"] == {"role": "assistant", "content": "hello"}
    assert payload["choices"][0]["finish_reason"] is None
    assert payload["provider"] == "mock-a"
    assert payload["request_id"] == "req-1"


def test_format_chunk_subsequent_omits_role() -> None:
    raw = format_chunk(
        chunk_id="chatcmpl-abc",
        created=123,
        model="mock-a",
        provider="mock-a",
        request_id="req-1",
        content="world",
    )
    payload = json.loads(parse_sse_line(raw.strip().splitlines()[0]))
    assert payload["choices"][0]["delta"] == {"content": "world"}
    assert "role" not in payload["choices"][0]["delta"]


def test_format_chunk_terminal_finish_reason() -> None:
    for reason in ("stop", "length"):
        raw = format_chunk(
            chunk_id="chatcmpl-abc",
            created=123,
            model="mock-a",
            provider="mock-a",
            request_id="req-1",
            finish_reason=reason,  # type: ignore[arg-type]
        )
        payload = json.loads(parse_sse_line(raw.strip().splitlines()[0]))
        assert payload["choices"][0]["delta"] == {}
        assert payload["choices"][0]["finish_reason"] == reason


def test_format_done() -> None:
    assert format_done() == "data: [DONE]\n\n"
    assert parse_sse_payload("[DONE]") == "DONE"


def test_format_error() -> None:
    raw = format_error(
        code="PROVIDER_UNAVAILABLE",
        message="boom",
        provider="mock-a",
        retryable=True,
        request_id="req-9",
    )
    payload = json.loads(parse_sse_line(raw.strip().splitlines()[0]))
    assert payload["error"]["code"] == "PROVIDER_UNAVAILABLE"
    assert payload["error"]["provider"] == "mock-a"
    assert payload["error"]["retryable"] is True
    assert payload["error"]["request_id"] == "req-9"


def test_parse_sse_line_rejects_non_data() -> None:
    with pytest.raises(ValueError):
        parse_sse_line(": comment")
    with pytest.raises(ValueError):
        parse_sse_line("")
    with pytest.raises(ValueError):
        parse_sse_line("event: message")
    # Leading space and missing space after colon both accepted.
    assert parse_sse_line("data:hello") == "hello"
    assert parse_sse_line("data: hello") == "hello"


def test_parse_sse_payload_malformed() -> None:
    with pytest.raises(ValueError):
        parse_sse_payload("{not json")
    with pytest.raises(ValueError):
        parse_sse_payload("[1,2,3]")  # must be object or [DONE]
    with pytest.raises(ValueError):
        parse_sse_payload("")


def test_decoder_multiple_events() -> None:
    body = format_chunk(
        chunk_id="c1",
        created=1,
        model="m",
        provider="p",
        request_id="r",
        content="a",
        role="assistant",
    )
    body += format_chunk(
        chunk_id="c1", created=1, model="m", provider="p", request_id="r", content="b"
    )
    body += format_done()
    events = parse_sse_stream(body)
    assert len(events) == 3
    assert events[-1] == "DONE"


def test_decoder_fragmented_input() -> None:
    body = format_chunk(
        chunk_id="c1",
        created=1,
        model="m",
        provider="p",
        request_id="r",
        content="hello world",
        role="assistant",
    ) + format_done()
    decoder = SseIncrementalDecoder()
    events: list[object] = []
    # Feed one character at a time: no assumption of read/event alignment.
    for ch in body:
        events.extend(decoder.feed(ch))
    events.extend(decoder.flush())
    assert len(events) == 2
    assert events[-1] == "DONE"


def test_decoder_ignores_blank_and_comments() -> None:
    decoder = SseIncrementalDecoder()
    events = decoder.feed(": heartbeat\n\n\n")
    assert events == []
    events = decoder.feed(format_done())
    assert events == ["DONE"]


def test_decoder_malformed_event_raises() -> None:
    decoder = SseIncrementalDecoder()
    with pytest.raises(ValueError):
        decoder.feed("data: {bad json\n\n")
