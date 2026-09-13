"""Circuit breaker: NoOp default plus the M2 resilient state machine."""

from __future__ import annotations

import abc
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker(abc.ABC):
    """Stability seam so routing code never changes when M2 lands."""

    @abc.abstractmethod
    def can_execute(self, provider: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def record_success(self, provider: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def record_failure(self, provider: str) -> None:
        raise NotImplementedError

    def state_of(self, provider: str) -> CircuitState:
        """Current state for telemetry. Defaults to closed (e.g. NoOp)."""
        _ = provider
        return CircuitState.CLOSED


class NoOpCircuitBreaker(CircuitBreaker):
    """M1 default: always closed, records nothing."""

    def can_execute(self, provider: str) -> bool:
        _ = provider
        return True

    def record_success(self, provider: str) -> None:
        _ = provider

    def record_failure(self, provider: str) -> None:
        _ = provider


@dataclass
class _BreakerRecord:
    """Per-provider breaker state. Mutated only by synchronous methods."""

    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    opened_at: float | None = None
    half_open_inflight: int = 0


class ResilientCircuitBreaker(CircuitBreaker):
    """M2 provider-specific breaker: CLOSED -> OPEN -> HALF_OPEN -> CLOSED.

    Deterministic consecutive-failure counting (no EWMA): ``failure_threshold``
    consecutive failures open the circuit; after ``recovery_timeout_s`` a
    bounded number of half-open probes (``half_open_max_inflight``) is
    admitted; a probe success closes the circuit, a probe failure reopens it.
    Any success resets the consecutive counter.

    Concurrency note: all methods are synchronous and contain no awaits, so
    on a single asyncio event loop each call runs atomically — no lock is
    needed for check-and-mutate transitions. ``clock`` (monotonic seconds)
    is injectable for deterministic tests.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 30.0,
        half_open_max_inflight: int = 1,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_timeout_s <= 0:
            raise ValueError("recovery_timeout_s must be > 0")
        if half_open_max_inflight < 1:
            raise ValueError("half_open_max_inflight must be >= 1")
        self._failure_threshold = failure_threshold
        self._recovery_timeout_s = recovery_timeout_s
        self._half_open_max_inflight = half_open_max_inflight
        self._clock = clock if clock is not None else time.monotonic
        self._records: dict[str, _BreakerRecord] = {}

    def _record(self, provider: str) -> _BreakerRecord:
        record = self._records.get(provider)
        if record is None:
            record = _BreakerRecord()
            self._records[provider] = record
        return record

    def state_of(self, provider: str) -> CircuitState:
        return self._record(provider).state

    def can_execute(self, provider: str) -> bool:
        record = self._record(provider)
        if record.state is CircuitState.CLOSED:
            return True
        if record.state is CircuitState.OPEN:
            opened_at = record.opened_at
            if opened_at is None or self._clock() - opened_at < self._recovery_timeout_s:
                return False
            record.state = CircuitState.HALF_OPEN
            record.half_open_inflight = 0
        if record.half_open_inflight >= self._half_open_max_inflight:
            return False
        record.half_open_inflight += 1
        return True

    def record_success(self, provider: str) -> None:
        record = self._record(provider)
        record.consecutive_failures = 0
        if record.state is CircuitState.HALF_OPEN:
            record.state = CircuitState.CLOSED
            record.half_open_inflight = 0

    def record_failure(self, provider: str) -> None:
        record = self._record(provider)
        record.consecutive_failures += 1
        if record.state is CircuitState.HALF_OPEN:
            record.state = CircuitState.OPEN
            record.opened_at = self._clock()
            record.half_open_inflight = 0
        elif record.state is CircuitState.CLOSED:
            if record.consecutive_failures >= self._failure_threshold:
                record.state = CircuitState.OPEN
                record.opened_at = self._clock()
        # Already OPEN: keep the original opened_at so stray reports
        # cannot extend the outage window.

