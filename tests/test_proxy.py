"""Proxy tests: timeout + error normalization."""

from __future__ import annotations

import pytest

from src.errors import GatewayError
from src.models import ChatMessage, NormalizedChatRequest
from src.providers.base import ProviderError
from src.providers.mock import MockProvider, MockProviderSettings
from src.proxy.client import ProxyClient
from src.router.engine import RouteDecision


def make_request() -> NormalizedChatRequest:
    return NormalizedChatRequest(
        request_id="req-proxy",
        model="mock-a",
        provider="mock-a",
        messages=[ChatMessage(role="user", content="hi")],
        temperature=0.0,
        max_tokens=None,
    )


def make_route(provider: str = "mock-a") -> RouteDecision:
    return RouteDecision(provider_name=provider, reason="test", tried=[provider])


async def test_forward_success() -> None:
    providers = {"mock-a": MockProvider("mock-a", MockProviderSettings())}
    proxy = ProxyClient(providers, {"mock-a": 5.0})
    result = await proxy.forward(make_route(), make_request())
    assert result.provider == "mock-a"
    assert result.content


async def test_unknown_provider() -> None:
    proxy = ProxyClient({}, {})
    with pytest.raises(GatewayError) as excinfo:
        await proxy.forward(make_route("ghost"), make_request())
    assert excinfo.value.code == "UNKNOWN_PROVIDER"
    assert excinfo.value.status_code == 500


async def test_timeout_maps_to_504() -> None:
    providers = {"slow": MockProvider("slow", MockProviderSettings(latency_ms=300.0))}
    proxy = ProxyClient(providers, {"slow": 0.01})
    with pytest.raises(GatewayError) as excinfo:
        await proxy.forward(make_route("slow"), make_request())
    assert excinfo.value.code == "UPSTREAM_TIMEOUT"
    assert excinfo.value.status_code == 504
    assert excinfo.value.retryable is True


async def test_provider_error_preserved() -> None:
    class Boom(MockProvider):
        async def chat(self, request: NormalizedChatRequest):  # type: ignore[override]
            raise ProviderError(
                code="PROVIDER_RATE_LIMITED",
                message="slow down",
                provider="mock-a",
                retryable=True,
                status_code=429,
            )

    proxy = ProxyClient({"mock-a": Boom("mock-a", MockProviderSettings())}, {"mock-a": 5.0})
    with pytest.raises(GatewayError) as excinfo:
        await proxy.forward(make_route(), make_request())
    assert excinfo.value.code == "PROVIDER_RATE_LIMITED"
    assert excinfo.value.status_code == 429
    assert excinfo.value.provider == "mock-a"
