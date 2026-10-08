"""Accurate monotonic timing. TTFT/ITL are streaming-only (M3) and never faked."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from types import TracebackType


class Timer:
    """``perf_counter_ns``-based timer reporting milliseconds."""

    def __init__(self) -> None:
        self._start_ns: int | None = None
        self._end_ns: int | None = None

    def start(self) -> None:
        self._start_ns = time.perf_counter_ns()
        self._end_ns = None

    def stop(self) -> float:
        if self._start_ns is None:
            raise RuntimeError("Timer.stop() called before start()")
        self._end_ns = time.perf_counter_ns()
        return self.elapsed_ms

    @property
    def elapsed_ms(self) -> float:
        if self._start_ns is None:
            raise RuntimeError("Timer not started")
        end_ns = self._end_ns if self._end_ns is not None else time.perf_counter_ns()
        return (end_ns - self._start_ns) / 1e6

    def __enter__(self) -> Timer:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()


@dataclass(frozen=True)
class StreamingTiming:
    """Measured streaming timing: time-to-first-byte and inter-chunk latency.

    Populated with real ``perf_counter_ns`` measurements on the M3 streaming
    path (see ``src/main.py:_handle_streaming``): ``ttft_ms`` covers request
    entry to first-chunk acquisition, ``itl_ms`` is the mean gap between
    subsequent chunks. A single-chunk stream has no inter-chunk gap, so
    ``itl_ms`` stays ``None`` rather than a fabricated ``0.0``.

    Non-streaming responses MUST still report ``ttft_ms=None`` and
    ``itl_ms=None``. Fabricating streaming metrics is a bug.
    """

    ttft_ms: float | None = None
    itl_ms: float | None = None


class LatencyTracker:
    """Minimal per-provider rolling latency average (last-N mean).

    Keyed by provider name. ``ResilientExecutor`` records each *successful*
    attempt's latency; only successes are recorded so a fast-failing provider
    never looks attractive. ``RouterEngine`` reads ``average()`` for scoring;
    providers with no samples return ``None`` and score neutrally on latency.
    Synchronous with no awaits: safe on a single asyncio event loop.
    """

    def __init__(self, window: int = 20) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._window = window
        self._samples: dict[str, deque[float]] = {}

    @property
    def window(self) -> int:
        return self._window

    def record(self, provider: str, latency_ms: float) -> None:
        samples = self._samples.get(provider)
        if samples is None:
            samples = deque(maxlen=self._window)
            self._samples[provider] = samples
        samples.append(max(latency_ms, 0.0))

    def average(self, provider: str) -> float | None:
        """Mean of the last ``window`` successful samples, or ``None``."""
        samples = self._samples.get(provider)
        if not samples:
            return None
        return sum(samples) / len(samples)

    def averages(self) -> dict[str, float]:
        return {
            name: sum(samples) / len(samples)
            for name, samples in self._samples.items()
            if samples
        }
