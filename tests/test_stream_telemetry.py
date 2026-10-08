"""Streaming TTFT/ITL telemetry: real timing, never fabricated.

Covers the M3 streaming path in ``src/main.py:_handle_streaming`` using the
deterministic mock provider:

- multi-chunk stream emits ``ttft_ms`` and ``itl_ms`` as positive floats and
  TTFT reflects at least the mock's configured first-byte delay;
- a single-chunk stream reports ``itl_ms=None`` (no inter-chunk gap exists)
  rather than a fabricated ``0.0``.

Values are captured from the ``gateway.stream`` structured log because
``create_app`` wires a ``NoOpMetricsRecorder`` into a closure (there is no
post-creation metrics injection point: ``app.state.metrics`` is never read
by the streaming path).
"""

from __future__ import annotations

from structlog.testing import capture_logs

from src.config import AppSettings, ProviderEntry, RoutingSettings
from src.main import create_app
from src.providers.mock import MockProviderSettings
from src.proxy.sse_parser import parse_sse_stream
from tests.conftest import chat_body, make_client

FIRST_BYTE_DELAY_MS = 60.0


def make_telemetry_settings(
    *,
    latency_ms: float = 0.0,
    response_tokens: int = 12,
    stream_chunk_words: int = 1,
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
                    latency_ms=latency_ms,
                    response_tokens=response_tokens,
                    stream_chunk_words=stream_chunk_words,
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


def stream_payload(**overrides: object) -> dict[str, object]:
    body = chat_body()
    body["stream"] = True
    body.update(overrides)
    return body


def _gateway_stream_events(logs: list[dict]) -> list[dict]:
    return [e for e in logs if e.get("event") == "gateway.stream"]


async def test_stream_emits_positive_ttft_and_itl() -> None:
    """Multi-chunk stream: TTFT/ITL are positive floats; TTFT covers the delay."""
    settings = make_telemetry_settings(
        latency_ms=FIRST_BYTE_DELAY_MS,
        response_tokens=12,
        stream_chunk_words=1,
    )
    app = create_app(settings)
    with capture_logs() as logs:
        async with make_client(app) as client:
            resp = await client.post("/v1/chat/completions", json=stream_payload())
    assert resp.status_code == 200
    events = parse_sse_stream(resp.text)
    assert events[-1] == "DONE"
    assert len([e for e in events if isinstance(e, dict)]) >= 4

    stream_logs = _gateway_stream_events(logs)
    assert len(stream_logs) == 1
    entry = stream_logs[0]
    assert entry.get("status") == "ok"
    ttft_ms = entry.get("ttft_ms")
    itl_ms = entry.get("itl_ms")
    assert isinstance(ttft_ms, float) and ttft_ms > 0.0
    assert isinstance(itl_ms, float) and itl_ms > 0.0
    assert ttft_ms >= FIRST_BYTE_DELAY_MS
    timing = entry.get("timing_ms")
    assert isinstance(timing, dict)
    assert timing.get("ttft") == ttft_ms or abs(timing["ttft"] - ttft_ms) < 0.0015


async def test_single_chunk_stream_reports_itl_none() -> None:
    """One content chunk: no inter-chunk gap, so itl_ms stays None (not 0.0)."""
    settings = make_telemetry_settings(
        response_tokens=1,
        stream_chunk_words=2,
    )
    app = create_app(settings)
    with capture_logs() as logs:
        async with make_client(app) as client:
            resp = await client.post("/v1/chat/completions", json=stream_payload())
    assert resp.status_code == 200
    events = parse_sse_stream(resp.text)
    assert events[-1] == "DONE"
    chunks = [e for e in events if isinstance(e, dict)]
    # Single content chunk + terminal finish chunk.
    assert len(chunks) == 2

    stream_logs = _gateway_stream_events(logs)
    assert len(stream_logs) == 1
    entry = stream_logs[0]
    assert entry.get("status") == "ok"
    ttft_ms = entry.get("ttft_ms")
    assert isinstance(ttft_ms, float) and ttft_ms > 0.0
    assert entry.get("itl_ms") is None
    timing = entry.get("timing_ms")
    assert isinstance(timing, dict)
    assert timing.get("itl") is None
