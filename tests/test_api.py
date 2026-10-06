"""Offline API tests via ASGITransport."""

from __future__ import annotations

from fastapi import FastAPI

from tests.conftest import chat_body, make_client


async def test_health(app: FastAPI) -> None:
    async with make_client(app) as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "mock-a" in body["providers"]
    assert resp.headers["x-request-id"]


async def test_ready(app: FastAPI) -> None:
    async with make_client(app) as client:
        resp = await client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert body["providers"]["mock-a"] is True


async def test_chat_happy_path(app: FastAPI) -> None:
    async with make_client(app) as client:
        resp = await client.post("/v1/chat/completions", json=chat_body())
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["provider"] == "mock-a"
    assert body["choices"][0]["message"]["content"].startswith("[mock:mock-a]")
    assert body["ttft_ms"] is None
    assert body["itl_ms"] is None
    assert body["cached"] is False
    assert body["request_id"] == resp.headers["x-request-id"]
    assert resp.headers["x-provider"] == "mock-a"
    assert float(resp.headers["x-gateway-latency-ms"]) >= 0.0


async def test_request_id_propagated(app: FastAPI) -> None:
    async with make_client(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=chat_body(),
            headers={"x-request-id": "fixed-id-123"},
        )
    assert resp.status_code == 200
    assert resp.headers["x-request-id"] == "fixed-id-123"
    assert resp.json()["request_id"] == "fixed-id-123"


async def test_stream_supported_returns_sse(app: FastAPI) -> None:
    """M3: stream=true returns SSE (replaces M1/M2 422 rejection)."""
    from src.proxy.sse_parser import parse_sse_stream

    payload = chat_body()
    payload["stream"] = True
    async with make_client(app) as client:
        resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    events = parse_sse_stream(resp.text)
    assert events
    assert events[-1] == "DONE"
    first = events[0]
    assert isinstance(first, dict)
    assert first["object"] == "chat.completion.chunk"


async def test_invalid_body_rejected(app: FastAPI) -> None:
    async with make_client(app) as client:
        resp = await client.post("/v1/chat/completions", json={"model": "mock-a"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


async def test_unknown_model_falls_back_to_default(app: FastAPI) -> None:
    async with make_client(app) as client:
        resp = await client.post("/v1/chat/completions", json=chat_body(model="nope"))
    assert resp.status_code == 200
    assert resp.json()["provider"] == "mock-a"


async def test_provider_failure_normalized(failing_app: FastAPI) -> None:
    failing_body = chat_body(model="mock-a")
    # M2: mock-a fails (fail_rate=1.0) so the request falls back to healthy mock-b.
    async with make_client(failing_app) as client:
        resp = await client.post("/v1/chat/completions", json=failing_body)
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "mock-b"
    assert resp.headers["x-provider"] == "mock-b"
    assert body["request_id"] == resp.headers["x-request-id"]
    assert "traceback" not in resp.text.lower()


async def test_all_providers_fail_normalized(all_failing_app: FastAPI) -> None:
    async with make_client(all_failing_app) as client:
        resp = await client.post("/v1/chat/completions", json=chat_body(model="mock-a"))
    assert resp.status_code == 502
    err = resp.json()["error"]
    assert err["provider"] == "mock-b"  # last attempted provider is reported
    assert err["retryable"] is True
    assert err["code"] == "PROVIDER_UNAVAILABLE"
    assert err["request_id"] == resp.headers["x-request-id"]
    assert "traceback" not in resp.text.lower()
