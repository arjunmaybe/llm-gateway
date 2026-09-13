"""Metrics seam. Prometheus exposition lands in M5."""

from __future__ import annotations

import abc


class MetricsRecorder(abc.ABC):
    @abc.abstractmethod
    def increment(self, name: str, *, provider: str = "", code: str = "") -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def observe_latency(self, name: str, value_ms: float, *, provider: str = "") -> None:
        raise NotImplementedError


class NoOpMetricsRecorder(MetricsRecorder):
    def increment(self, name: str, *, provider: str = "", code: str = "") -> None:
        _ = (name, provider, code)

    def observe_latency(self, name: str, value_ms: float, *, provider: str = "") -> None:
        _ = (name, value_ms, provider)
