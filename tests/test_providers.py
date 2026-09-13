"""Mock provider contract: every future provider must satisfy the same shape."""

from __future__ import annotations

import pytest

from src.models import ChatMessage, NormalizedChatRequest
from src.providers.base import ProviderError
from src.providers.mock import MockProvider, MockProviderSettings


def make_request(request_id: str = "req-1") -> NormalizedChatRequest:
    return NormalizedChatRequest(
        request_id=request_id,
        model="mock-a",
        provider="mock-a",
        messages=[ChatMessage(role="user", content="hello")],
        temperature=0.0,
        max_tokens=None,
    )


async def test_chat_shape_and_token_size() -> None:
    provider = MockProvider("mock-a", MockProviderSettings(response_tokens=10))
    result = await provider.chat(make_request())
    assert result.provider == "mock-a"
    assert len(result.content.split()) == 10
    assert result.usage.total_tokens == result.usage.prompt_tokens + result.usage.completion_tokens


async def test_deterministic_same_request_id() -> None:
    provider = MockProvider("mock-a", MockProviderSettings())
    first = await provider.chat(make_request("fixed"))
    second = await provider.chat(make_request("fixed"))
    assert first.content == second.content


async def test_failure_injection_preserves_provider() -> None:
    provider = MockProvider(
        "mock-a",
        MockProviderSettings(fail_rate=1.0, failure_mode="rate_limited"),
    )
    with pytest.raises(ProviderError) as excinfo:
        await provider.chat(make_request())
    assert excinfo.value.provider == "mock-a"
    assert excinfo.value.code == "PROVIDER_RATE_LIMITED"
    assert excinfo.value.retryable is True


async def test_health_check() -> None:
    healthy = MockProvider("mock-a", MockProviderSettings(fail_rate=0.0))
    assert (await healthy.health_check()).healthy is True
    sick = MockProvider("mock-a", MockProviderSettings(fail_rate=1.0))
    assert (await sick.health_check()).healthy is False
