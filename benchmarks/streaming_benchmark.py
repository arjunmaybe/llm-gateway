"""M3 streaming benchmark over the mock provider (offline).

Reports request count, TTFT (avg/p50/p95), ITL (avg/p50/p95), average total
stream duration, chunks/request, and interrupted streams. Numbers describe the
in-process mock only, not production performance.

Usage:
    py -3.11 benchmarks/streaming_benchmark.py --requests 50 --tokens 24
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

from src.config import AppSettings, ProviderEntry, RoutingSettings
from src.main import create_app
from src.models import ChatMessage, NormalizedChatRequest
from src.providers.mock import MockProvider, MockProviderSettings
from src.proxy.sse_parser import parse_sse_stream
from tests.conftest import chat_body, make_client


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1) + 0.5)))
    return ordered[rank]


async def _one_provider_sample(
    provider: MockProvider, request_id: str
) -> tuple[float, list[float], int]:
    """Direct provider timing (no HTTP buffering): TTFT, ITLs, chunk count."""
    req = NormalizedChatRequest(
        request_id=request_id,
        model="mock-a",
        provider=provider.name,
        messages=[ChatMessage(role="user", content=f"bench {request_id}")],
        temperature=0.0,
        max_tokens=None,
    )
    start_ns = time.perf_counter_ns()
    first_ns: int | None = None
    prev_ns: int | None = None
    itls: list[float] = []
    count = 0
    async for _ in provider.chat_stream(req):
        now_ns = time.perf_counter_ns()
        if first_ns is None:
            first_ns = now_ns
        elif prev_ns is not None:
            itls.append((now_ns - prev_ns) / 1e6)
        prev_ns = now_ns
        count += 1
    ttft_ms = ((first_ns - start_ns) / 1e6) if first_ns is not None else 0.0
    return ttft_ms, itls, count


async def _one_request(
    client: object, body: dict[str, object]
) -> tuple[float, float, int, bool]:
    """Run one HTTP streamed request. Returns (total, chunks, interrupted)."""
    import httpx  # local import keeps module import cheap

    assert isinstance(client, httpx.AsyncClient)
    start_ns = time.perf_counter_ns()
    chunks = 0
    interrupted = False
    async with client.stream("POST", "/v1/chat/completions", json=body) as resp:
        assert resp.status_code == 200
        text_parts: list[str] = []
        async for part in resp.aiter_text():
            text_parts.append(part)
        body_text = "".join(text_parts)
    total_ms = (time.perf_counter_ns() - start_ns) / 1e6
    try:
        events = parse_sse_stream(body_text)
        chunks = sum(1 for e in events if isinstance(e, dict) and "choices" in e)
        interrupted = "DONE" not in events and not any(
            isinstance(e, dict) and "error" in e for e in events
        )
    except ValueError:
        interrupted = True
    return total_ms, chunks, interrupted


async def run(*, requests: int, tokens: int, chunk_words: int) -> dict[str, float]:
    settings = AppSettings(
        routing=RoutingSettings(
            default_provider="mock-a",
            model_aliases={"mock-a": "mock-a", "mock-b": "mock-b"},
        ),
        providers=[
            ProviderEntry(
                name="mock-a",
                type="mock",
                enabled=True,
                priority=10,
                timeout_s=5.0,
                mock=MockProviderSettings(
                    response_tokens=tokens, stream_chunk_words=chunk_words
                ),
            ),
            ProviderEntry(
                name="mock-b",
                type="mock",
                enabled=True,
                priority=20,
                timeout_s=5.0,
                mock=MockProviderSettings(),
            ),
        ],
    )
    app = create_app(settings)
    provider = MockProvider(
        "mock-a",
        MockProviderSettings(response_tokens=tokens, stream_chunk_words=chunk_words),
    )
    ttfts: list[float] = []
    all_itls: list[float] = []
    totals: list[float] = []
    chunk_counts: list[int] = []
    interrupted = 0
    async with make_client(app) as client:
        for i in range(requests):
            # Provider-level TTFT/ITL (accurate, no HTTP test-transport buffering).
            ttft, itls, _ = await _one_provider_sample(provider, f"bench-prov-{i}")
            ttfts.append(ttft)
            all_itls.extend(itls)
            # HTTP end-to-end total duration + chunk framing.
            body = chat_body(model="mock-a")
            body["stream"] = True
            body["messages"] = [{"role": "user", "content": f"bench {i}"}]
            total, chunks, intr = await _one_request(client, body)
            totals.append(total)
            chunk_counts.append(chunks)
            interrupted += 1 if intr else 0
    return {
        "requests": float(requests),
        "avg_ttft_ms": float(statistics.fmean(ttfts)) if ttfts else 0.0,
        "p50_ttft_ms": _percentile(ttfts, 0.5),
        "p95_ttft_ms": _percentile(ttfts, 0.95),
        "avg_itl_ms": float(statistics.fmean(all_itls)) if all_itls else 0.0,
        "p50_itl_ms": _percentile(all_itls, 0.5),
        "p95_itl_ms": _percentile(all_itls, 0.95),
        "avg_total_ms": float(statistics.fmean(totals)) if totals else 0.0,
        "avg_chunks_per_request": float(statistics.fmean(chunk_counts)) if chunk_counts else 0.0,
        "interrupted": float(interrupted),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M3 mock streaming benchmark")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--chunk-words", type=int, default=2)
    args = parser.parse_args()
    summary = asyncio.run(
        run(requests=args.requests, tokens=args.tokens, chunk_words=args.chunk_words)
    )
    print("M3 streaming benchmark (mock provider only, not production performance):")
    for key, value in summary.items():
        print(f"  {key}: {value:.3f}")


if __name__ == "__main__":
    main()
