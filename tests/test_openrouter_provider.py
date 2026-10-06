"""OpenRouter provider tests. All HTTP is mocked — no real API calls."""

from __future__ import annotations

import json

import httpx
import pytest

from src.models import ChatMessage, NormalizedChatRequest
from src.providers.base import ProviderError
from src.providers.openrouter import OpenRouterProvider, OpenRouterSettings


def make_request(
    model: str = "mock-a", content: str = "hello"
) -> NormalizedChatRequest:
    return NormalizedChatRequest(
        request_id="req-or-1",
        model=model,
        provider="openrouter",
        messages=[ChatMessage(role="user", content=content)],
        temperature=0.0,
        max_tokens=None,
    )


def make_provider(
    handler: object, *, api_key: str = "test-key"
) -> OpenRouterProvider:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    client = httpx.AsyncClient(transport=transport)
    settings = OpenRouterSettings(api_key=api_key, model="test-model")
    return OpenRouterProvider("or-test", settings, client=client)


def chat_success_handler(request: httpx.Request) -> httpx.Response:
    assert request.headers["authorization"] == "Bearer test-key"
    body = json.loads(request.content.decode("utf-8"))
    assert body["model"] == "test-model"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["stream"] is False
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "Hi there"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        },
    )


async def test_chat_success() -> None:
    provider = make_provider(chat_success_handler)
    result = await provider.chat(make_request())
    assert result.provider == "or-test"
    assert result.model == "mock-a"
    assert result.content == "Hi there"
    assert result.usage.prompt_tokens == 5
    assert result.usage.completion_tokens == 3
    assert result.usage.total_tokens == 8
    assert result.latency_ms >= 0.0


async def test_chat_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    provider = OpenRouterProvider("or-nokey", OpenRouterSettings(api_key=""))
    with pytest.raises(ProviderError) as excinfo:
        await provider.chat(make_request())
    assert excinfo.value.code == "PROVIDER_NOT_CONFIGURED"
    assert excinfo.value.retryable is False


@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [
        (429, "PROVIDER_RATE_LIMITED", True),
        (500, "PROVIDER_UNAVAILABLE", True),
        (503, "PROVIDER_UNAVAILABLE", True),
        (401, "PROVIDER_REJECTED", False),
        (400, "PROVIDER_REJECTED", False),
        (404, "PROVIDER_REJECTED", False),
    ],
)
async def test_chat_error_mapping(status: int, code: str, retryable: bool) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(status, json={"error": {"message": "boom"}})

    provider = make_provider(handler)
    with pytest.raises(ProviderError) as excinfo:
        await provider.chat(make_request())
    err = excinfo.value
    assert err.code == code
    assert err.retryable is retryable
    assert err.provider == "or-test"


async def test_chat_timeout_maps_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow", request=request)

    provider = make_provider(handler)
    with pytest.raises(ProviderError) as excinfo:
        await provider.chat(make_request())
    assert excinfo.value.code == "PROVIDER_TIMEOUT"
    assert excinfo.value.retryable is True


async def test_chat_connection_error_maps_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    provider = make_provider(handler)
    with pytest.raises(ProviderError) as excinfo:
        await provider.chat(make_request())
    assert excinfo.value.code == "PROVIDER_CONNECTION_FAILED"
    assert excinfo.value.retryable is True


SSE_BODY = (
    'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
    'data: {"choices": [{"delta": {"content": " world"}}]}\n\n'
    "data: [DONE]\n\n"
)


async def test_chat_stream_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(
            200,
            content=SSE_BODY.encode("utf-8"),
            headers={"content-type": "text/event-stream"},
        )

    provider = make_provider(handler)
    chunks = [c async for c in provider.chat_stream(make_request())]
    assert chunks == ["Hello", " world"]


async def test_chat_stream_error_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    provider = make_provider(handler)
    with pytest.raises(ProviderError) as excinfo:
        [c async for c in provider.chat_stream(make_request())]
    assert excinfo.value.code == "PROVIDER_RATE_LIMITED"
    assert excinfo.value.retryable is True


async def test_health_check_ok() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models")
        return httpx.Response(200, json={"data": []})

    provider = make_provider(handler)
    status = await provider.health_check()
    assert status.healthy is True
    assert status.provider == "or-test"


async def test_health_check_unauthorized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    provider = make_provider(handler)
    status = await provider.health_check()
    assert status.healthy is False
    assert "unauthorized" in (status.error or "")


async def test_health_check_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    provider = OpenRouterProvider("or-nokey", OpenRouterSettings(api_key=""))
    status = await provider.health_check()
    assert status.healthy is False
