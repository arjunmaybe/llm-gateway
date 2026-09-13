"""Timing tests: monotonic measurement, streaming fields never faked."""

from __future__ import annotations

import pytest

from src.telemetry.latency import StreamingTiming, Timer


def test_timer_measures_milliseconds() -> None:
    timer = Timer()
    timer.start()
    elapsed = timer.stop()
    assert elapsed >= 0.0


def test_timer_requires_start() -> None:
    with pytest.raises(RuntimeError):
        Timer().stop()


def test_streaming_timing_defaults_to_none() -> None:
    timing = StreamingTiming()
    assert timing.ttft_ms is None
    assert timing.itl_ms is None
