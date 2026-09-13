"""Accurate monotonic timing. TTFT/ITL are streaming-only (M3) and never faked."""

from __future__ import annotations

import time
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
    """Reserved for M3 streaming.

    M1 rule: non-streaming responses MUST report ``ttft_ms=None`` and
    ``itl_ms=None``. Fabricating streaming metrics is a bug, so M1 constructs
    this with both fields ``None`` and never populates them.
    """

    ttft_ms: float | None = None
    itl_ms: float | None = None
