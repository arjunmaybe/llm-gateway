"""API streaming tests: SSE contract, fallback, errors, cancellation, telemetry."""

from __future__ import annotations

import json
import time

import httpx
from fastapi import FastAPI

from src.config import AppSettings, ProviderEntry, RoutingSettings
from src.main import create_app
from src.providers.mock import MockProviderSettings
from src.proxy.sse_parser import parse_sse_stream
from tests.conftest import chat_body, make_client


def make_stream_settings(
    *,
    fail_rate_a: float = 0.0,
    failure_script_a: tuple[str | None, ...] = (),
    stream_failure_after_a: int | None = None,
    stream_failure_mode_a: str | None = None,
    response_tokens_a: int = 16,
    stream_chunk_words_a: int = 2,
    latency_ms_a: float = 0.0,
    fail_rate_b: float = 0.0,
    response_tokens_b: int = 16,
) -> AppSettings:
    return AppSettings(
        routing=RoutingSettings(
            default_provider="mock-a",
            model_aliases={"mock": "mock-a", "mock-a": "mock-a", "mock-b": "mock-b"},
        ),
        providers=[
            ProviderEntry(
                name="mock-a",
                type="mock",
                enabled=True,
                priority=10,
                timeout_s=5.0,
                mock=MockProviderSettings(
                    latency_ms=latency_ms_a,
                    fail_rate=fail_rate_a,  # type: ignore[arg-type]
                    response_tokens=response_tokens_a,
                    failure_script=failure_script_a,  # type: ignore[arg-type]
                    stream_chunk_words=stream_chunk_words_a,
                    stream_failure_after=stream_failure_after_a,
                    stream_failure_mode=stream_failure_mode_a,  # type: ignore[arg-type]
                ),
            ),
            ProviderEntry(
                name="mock-b",
                type="mock",
                enabled=True,
                priority=20,
                timeout_s=5.0,
                mock=MockProviderSettings(
                    fail_rate=fail_rate_b,  # type: ignore[arg-type]
                    response_tokens=response_tokens_b,
                ),
            ),
        ],
    )


def stream_payload(**overrides: object) -> dict[str, object]:
    body = chat_body()
    body["stream"] = True
    body.update(overrides)
    return body


def _delta_contents(chunks: list[object]) -> str:
    """Join delta contents from parsed SSE chunk dicts."""
    parts: list[str] = []
    for item in chunks:
        assert isinstance(item, dict)
        choices = item["choices"]
        assert isinstance(choices, list)
        first = choices[0]
        assert isinstance(first, dict)
        delta = first["delta"]
        assert isinstance(delta, dict)
        content = delta.get("content")
        if isinstance(content, str) and content:
            parts.append(content)
    return " ".join(parts)


async def test_stream_200_and_sse_headers(app: FastAPI) -> None:
    async with make_client(app) as client:
        resp = await client.post("/v1/chat/completions", json=stream_payload())
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["x-request-id"]
    assert resp.headers["x-provider"] == "mock-a"
    # Total latency must not be exposed before the stream finishes.
    assert "x-gateway-latency-ms" not in resp.headers


async def test_stream_request_id_propagated(app: FastAPI) -> None:
    async with make_client(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=stream_payload(),
            headers={"x-request-id": "stream-fixed-1"},
        )
    assert resp.headers["x-request-id"] == "stream-fixed-1"
    events = parse_sse_stream(resp.text)
    first = events[0]
    assert isinstance(first, dict)
    assert first["request_id"] == "stream-fixed-1"


async def test_stream_openai_chunks_role_once_and_done(app: FastAPI) -> None:
    async with make_client(app) as client:
        resp = await client.post("/v1/chat/completions", json=stream_payload())
    events = parse_sse_stream(resp.text)
    assert events[-1] == "DONE"
    chunks = [e for e in events if isinstance(e, dict)]
    assert len(chunks) >= 3  # content chunks + terminal finish chunk
    # First content chunk carries role.
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "content" in chunks[0]["choices"][0]["delta"]
    # Subsequent content chunks omit role.
    for chunk in chunks[1:-1]:
        assert "role" not in chunk["choices"][0]["delta"]
        assert "content" in chunk["choices"][0]["delta"]
    # Terminal chunk carries finish_reason stop (no truncation by default).
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"][0]["delta"] == {}
    # Non-terminal chunks have explicit null finish_reason.
    for chunk in chunks[:-1]:
        assert chunk["choices"][0]["finish_reason"] is None
    # Object/model/provider shape.
    for chunk in chunks:
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["id"].startswith("chatcmpl-")
        assert chunk["provider"] == "mock-a"


async def test_stream_content_matches_non_streaming(app: FastAPI) -> None:
    async with make_client(app) as client:
        non_stream = await client.post("/v1/chat/completions", json=chat_body())
        streamed = await client.post("/v1/chat/completions", json=stream_payload())
    expected = non_stream.json()["choices"][0]["message"]["content"]
    events = parse_sse_stream(streamed.text)
    chunks = [e for e in events if isinstance(e, dict)]
    assert _delta_contents(chunks) == expected


async def test_stream_max_tokens_truncates_with_length() -> None:
    settings = make_stream_settings(response_tokens_a=20)
    stream_app = create_app(settings)
    body = stream_payload(max_tokens=5)
    async with make_client(stream_app) as client:
        resp = await client.post("/v1/chat/completions", json=body)
    events = parse_sse_stream(resp.text)
    assert events[-1] == "DONE"
    chunks = [e for e in events if isinstance(e, dict)]
    last = chunks[-1]
    assert isinstance(last, dict)
    choices = last["choices"]
    assert isinstance(choices, list)
    first_choice = choices[0]
    assert isinstance(first_choice, dict)
    assert first_choice["finish_reason"] == "length"
    assert len(_delta_contents(chunks).split()) == 5


async def test_stream_pre_first_byte_fallback() -> None:
    # mock-a fails before first byte; healthy mock-b serves the stream.
    settings = make_stream_settings(failure_script_a=("unavailable",))
    fallback_app = create_app(settings)
    async with make_client(fallback_app) as client:
        resp = await client.post("/v1/chat/completions", json=stream_payload())
    assert resp.status_code == 200
    assert resp.headers["x-provider"] == "mock-b"
    events = parse_sse_stream(resp.text)
    assert events[-1] == "DONE"
    first = events[0]
    assert isinstance(first, dict)
    assert first["provider"] == "mock-b"


async def test_stream_pre_first_byte_all_fail_returns_json_error() -> None:
    settings = make_stream_settings(failure_script_a=("unavailable",))
    # Make mock-b fail too via fail_rate.
    settings_both = make_stream_settings(
        failure_script_a=("unavailable", "unavailable", "unavailable"),
        fail_rate_b=1.0,
    )
    both_app = create_app(settings_both)
    async with make_client(both_app) as client:
        resp = await client.post("/v1/chat/completions", json=stream_payload())
    # Exhausted pre-first-byte: JSON error (headers never committed to SSE).
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "PROVIDER_UNAVAILABLE"
    _ = settings


async def test_stream_mid_failure_error_event_no_done_no_fallback() -> None:
    settings = make_stream_settings(
        response_tokens_a=20, stream_chunk_words_a=1, stream_failure_after_a=2
    )
    mid_app = create_app(settings)
    async with make_client(mid_app) as client:
        resp = await client.post("/v1/chat/completions", json=stream_payload())
    # Headers already committed as SSE; failure is an error event, not HTTP error.
    assert resp.status_code == 200
    assert resp.headers["x-provider"] == "mock-a"  # pinned, no fallback to mock-b
    assert "text/event-stream" in resp.headers["content-type"]
    events = parse_sse_stream(resp.text)
    # No [DONE] after a mid-stream error.
    assert "DONE" not in events
    assert len(events) == 3  # 2 content chunks + 1 error event
    assert isinstance(events[0], dict) and "choices" in events[0]
    assert isinstance(events[1], dict) and "choices" in events[1]
    err = events[2]
    assert isinstance(err, dict) and "error" in err
    assert err["error"]["provider"] == "mock-a"
    assert err["error"]["request_id"] == resp.headers["x-request-id"]
    assert err["error"]["code"] == "PROVIDER_UNAVAILABLE"


async def test_stream_mid_failure_after_all_content_no_done() -> None:
    settings = make_stream_settings(
        response_tokens_a=6, stream_chunk_words_a=2, stream_failure_after_a=3
    )
    mid_app = create_app(settings)
    async with make_client(mid_app) as client:
        resp = await client.post("/v1/chat/completions", json=stream_payload())
    events = parse_sse_stream(resp.text)
    assert "DONE" not in events
    assert isinstance(events[-1], dict) and "error" in events[-1]


async def test_stream_chunks_arrive_incrementally() -> None:
    """Client-observed TTFT: first byte fast; body parses to many SSE events."""
    settings = make_stream_settings(response_tokens_a=12, stream_chunk_words_a=1)
    inc_app = create_app(settings)
    async with make_client(inc_app) as client:
        start = time.perf_counter()
        async with client.stream("POST", "/v1/chat/completions", json=stream_payload()) as resp:
            assert resp.status_code == 200
            first_at: float | None = None
            async for _ in resp.aiter_text():
                if first_at is None:
                    first_at = time.perf_counter()
            assert first_at is not None
            ttft_ms = (first_at - start) * 1000.0
            total_ms = (time.perf_counter() - start) * 1000.0
            assert ttft_ms <= total_ms + 1.0
            assert ttft_ms < 5000.0
        # Full body (buffered by test transport) must contain many SSE events + DONE.
        # Incremental delivery itself is proven by proxy/mock unit tests.
        async with make_client(inc_app) as client2:
            full = await client2.post("/v1/chat/completions", json=stream_payload())
        events = parse_sse_stream(full.text)
        assert len([e for e in events if isinstance(e, dict)]) >= 4
        assert events[-1] == "DONE"


async def test_stream_client_cancellation_closes_cleanly() -> None:
    """Partial consumption then close must not hang or error."""
    settings = make_stream_settings(response_tokens_a=40, stream_chunk_words_a=1)
    cancel_app = create_app(settings)
    async with make_client(cancel_app) as client:
        async with client.stream(
            "POST", "/v1/chat/completions", json=stream_payload()
        ) as resp:
            assert resp.status_code == 200
            it = resp.aiter_text()
            first = await it.__anext__()
            assert "data:" in first
            await resp.aclose()
    # If cancellation hung or raised, this test would fail/time out.


async def test_stream_unknown_model_falls_back(app: FastAPI) -> None:
    async with make_client(app) as client:
        resp = await client.post("/v1/chat/completions", json=stream_payload(model="nope"))
    assert resp.status_code == 200
    assert resp.headers["x-provider"] == "mock-a"


async def test_non_streaming_regression_unchanged(app: FastAPI) -> None:
    """stream=false must keep exact M1/M2 JSON behavior."""
    async with make_client(app) as client:
        resp = await client.post("/v1/chat/completions", json=chat_body())
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["ttft_ms"] is None
    assert body["itl_ms"] is None
    assert body["cached"] is False
    assert resp.headers["x-gateway-latency-ms"]


async def test_stream_invalid_body_rejected(app: FastAPI) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        resp = await client.post("/v1/chat/completions", json={"model": "mock-a"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


def test_stream_chunk_model_shape() -> None:
    """Typed chunk models validate the locked wire format."""
    from src.models import ChatCompletionChunk

    raw = json.dumps(
        {
            "id": "chatcmpl-abc",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "mock-a",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": None}
            ],
            "provider": "mock-a",
            "request_id": "r",
        }
    )
    chunk = ChatCompletionChunk.model_validate_json(raw)
    assert chunk.choices[0].delta.role == "assistant"
