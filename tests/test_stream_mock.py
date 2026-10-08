"""MockProvider streaming: determinism, chunking, failure injection."""

from __future__ import annotations

import pytest

from src.models import ChatMessage, NormalizedChatRequest
from src.providers.base import ProviderError
from src.providers.mock import MockProvider, MockProviderSettings


def make_request(request_id: str = "req-stream") -> NormalizedChatRequest:
    return NormalizedChatRequest(
        request_id=request_id,
        model="mock-a",
        provider="mock-a",
        messages=[ChatMessage(role="user", content="hello gateway")],
        temperature=0.0,
        max_tokens=None,
    )


async def collect(provider: MockProvider, request_id: str = "req-stream") -> list[str]:
    return [chunk async for chunk in provider.chat_stream(make_request(request_id))]


async def test_stream_chunks_reconstruct_full_content() -> None:
    provider = MockProvider("mock-a", MockProviderSettings(response_tokens=12))
    chunks = await collect(provider)
    assert len(chunks) >= 2
    full = await provider.chat(make_request("other-id"))
    # Same request content shape: word count matches response_tokens.
    assert len(" ".join(chunks).split()) == len(full.content.split())


async def test_stream_deterministic_same_request_id() -> None:
    provider = MockProvider("mock-a", MockProviderSettings())
    first = await collect(provider, "fixed-stream")
    second = await collect(provider, "fixed-stream")
    assert first == second


async def test_stream_chunk_size_configurable() -> None:
    small = MockProvider("mock-a", MockProviderSettings(response_tokens=12, stream_chunk_words=1))
    big = MockProvider("mock-a", MockProviderSettings(response_tokens=12, stream_chunk_words=6))
    small_chunks = await collect(small, "req-1")
    big_chunks = await collect(big, "req-1")
    assert len(small_chunks) > len(big_chunks)
    assert " ".join(small_chunks) == " ".join(big_chunks)


async def test_stream_first_byte_delay() -> None:
    import time

    provider = MockProvider("mock-a", MockProviderSettings(latency_ms=120.0))
    start = time.perf_counter()
    chunks = await collect(provider)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert chunks
    assert elapsed_ms >= 100.0


async def test_stream_failure_before_first_chunk_via_script() -> None:
    provider = MockProvider(
        "mock-a", MockProviderSettings(failure_script=("unavailable",))
    )
    with pytest.raises(ProviderError) as excinfo:
        await collect(provider)
    assert excinfo.value.code == "PROVIDER_UNAVAILABLE"


async def test_stream_failure_before_first_chunk_via_zero() -> None:
    provider = MockProvider(
        "mock-a", MockProviderSettings(stream_failure_after=0, stream_failure_mode="timeout")
    )
    with pytest.raises(ProviderError) as excinfo:
        await collect(provider)
    assert excinfo.value.code == "PROVIDER_TIMEOUT"


async def test_stream_failure_after_n_chunks() -> None:
    provider = MockProvider(
        "mock-a",
        MockProviderSettings(
            response_tokens=20, stream_chunk_words=1, stream_failure_after=3
        ),
    )
    seen: list[str] = []
    with pytest.raises(ProviderError):
        async for chunk in provider.chat_stream(make_request()):
            seen.append(chunk)
    assert len(seen) == 3


async def test_stream_failure_after_end() -> None:
    provider = MockProvider(
        "mock-a",
        MockProviderSettings(response_tokens=6, stream_chunk_words=2, stream_failure_after=3),
    )
    # 6 words / 2 per chunk = 3 chunks; failure_after=3 fails after all content.
    seen: list[str] = []
    with pytest.raises(ProviderError):
        async for chunk in provider.chat_stream(make_request()):
            seen.append(chunk)
    assert len(seen) == 3


async def test_non_streaming_unchanged_after_streaming_fields() -> None:
    provider = MockProvider("mock-a", MockProviderSettings(response_tokens=10))
    result = await provider.chat(make_request())
    assert len(result.content.split()) == 10
