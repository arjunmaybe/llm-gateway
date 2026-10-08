"""Streaming proxy: incremental delivery, first-byte timeout, cleanup."""

from __future__ import annotations

import asyncio

import pytest

from src.errors import GatewayError
from src.models import ChatMessage, NormalizedChatRequest
from src.providers.base import ProviderError
from src.providers.mock import MockProvider, MockProviderSettings
from src.proxy.client import ProxyClient
from src.router.engine import RouteDecision


def make_request(request_id: str = "req-proxy-stream") -> NormalizedChatRequest:
    return NormalizedChatRequest(
        request_id=request_id,
        model="mock-a",
        provider="mock-a",
        messages=[ChatMessage(role="user", content="hi")],
        temperature=0.0,
        max_tokens=None,
    )


def make_route(provider: str = "mock-a") -> RouteDecision:
    return RouteDecision(provider_name=provider, reason="test", tried=[provider])


async def test_stream_forward_incremental_delivery() -> None:
    providers = {"mock-a": MockProvider("mock-a", MockProviderSettings(response_tokens=10))}
    proxy = ProxyClient(providers, {"mock-a": 5.0})
    seen: list[str] = []
    async for chunk in proxy.stream_forward(make_route(), make_request()):
        seen.append(chunk)
        # Proves incremental delivery: more than one chunk, none is the full text.
        assert chunk
    assert len(seen) >= 2
    assert len(" ".join(seen).split()) == 10


async def test_stream_forward_unknown_provider() -> None:
    proxy = ProxyClient({}, {})
    with pytest.raises(GatewayError) as excinfo:
        async for _ in proxy.stream_forward(make_route("ghost"), make_request()):
            pass
    assert excinfo.value.code == "UNKNOWN_PROVIDER"


async def test_stream_first_byte_timeout() -> None:
    providers = {"slow": MockProvider("slow", MockProviderSettings(latency_ms=300.0))}
    proxy = ProxyClient(providers, {"slow": 0.01})
    with pytest.raises(GatewayError) as excinfo:
        async for _ in proxy.stream_forward(make_route("slow"), make_request()):
            pass
    assert excinfo.value.code == "UPSTREAM_TIMEOUT"
    assert excinfo.value.status_code == 504
    assert excinfo.value.retryable is True


async def test_stream_does_not_timeout_healthy_long_stream() -> None:
    """First-byte timeout must not kill a healthy stream that streams past it.

    Provider yields slowly per chunk but first byte is fast; a whole-stream
    wait_for would incorrectly time out.
    """

    class SlowChunks(MockProvider):
        async def chat_stream(self, request: NormalizedChatRequest):  # type: ignore[override]
            for i in range(5):
                await asyncio.sleep(0.03)
                yield f"w{i}"

    providers = {"slow": SlowChunks("slow", MockProviderSettings())}
    # Timeout larger than first-chunk latency (30ms) but smaller than total (~150ms).
    proxy = ProxyClient(providers, {"slow": 0.08})
    seen = [chunk async for chunk in proxy.stream_forward(make_route("slow"), make_request())]
    assert seen == ["w0", "w1", "w2", "w3", "w4"]


async def test_stream_post_first_byte_failure_normalized() -> None:
    providers = {
        "mock-a": MockProvider(
            "mock-a",
            MockProviderSettings(
                response_tokens=12, stream_chunk_words=1, stream_failure_after=2
            ),
        )
    }
    proxy = ProxyClient(providers, {"mock-a": 5.0})
    seen: list[str] = []
    with pytest.raises(GatewayError) as excinfo:
        async for chunk in proxy.stream_forward(make_route(), make_request()):
            seen.append(chunk)
    assert len(seen) == 2
    assert excinfo.value.code == "PROVIDER_UNAVAILABLE"
    assert excinfo.value.provider == "mock-a"


async def test_stream_provider_error_preserved() -> None:
    class Boom(MockProvider):
        async def chat_stream(self, request: NormalizedChatRequest):  # type: ignore[override]
            yield "first"
            raise ProviderError(
                code="PROVIDER_RATE_LIMITED",
                message="slow down",
                provider="mock-a",
                retryable=True,
                status_code=429,
            )
            yield "unreachable"  # pragma: no cover

    proxy = ProxyClient({"mock-a": Boom("mock-a", MockProviderSettings())}, {"mock-a": 5.0})
    seen: list[str] = []
    with pytest.raises(GatewayError) as excinfo:
        async for chunk in proxy.stream_forward(make_route(), make_request()):
            seen.append(chunk)
    assert seen == ["first"]
    assert excinfo.value.code == "PROVIDER_RATE_LIMITED"
    assert excinfo.value.status_code == 429


async def test_stream_cleanup_on_early_break() -> None:
    closed: list[bool] = []

    class Tracked(MockProvider):
        async def chat_stream(self, request: NormalizedChatRequest):  # type: ignore[override]
            try:
                for i in range(10):
                    yield f"w{i}"
            finally:
                closed.append(True)

    proxy = ProxyClient({"mock-a": Tracked("mock-a", MockProviderSettings())}, {"mock-a": 5.0})
    stream = proxy.stream_forward(make_route(), make_request())
    first = await stream.__anext__()
    assert first == "w0"
    await stream.aclose()
    assert closed == [True]
