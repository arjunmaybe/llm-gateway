"""Circuit breaker unit tests: deterministic state machine via injected clock."""

from __future__ import annotations

import pytest

from src.router.circuit_breaker import (
    CircuitState,
    NoOpCircuitBreaker,
    ResilientCircuitBreaker,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_breaker(clock: FakeClock) -> ResilientCircuitBreaker:
    return ResilientCircuitBreaker(
        failure_threshold=3, recovery_timeout_s=10.0, half_open_max_inflight=1, clock=clock
    )


def test_closed_allows_requests() -> None:
    breaker = make_breaker(FakeClock())
    assert breaker.state_of("mock-a") is CircuitState.CLOSED
    assert breaker.can_execute("mock-a") is True


def test_unknown_provider_defaults_closed() -> None:
    breaker = make_breaker(FakeClock())
    assert breaker.can_execute("never-seen") is True
    assert breaker.state_of("never-seen") is CircuitState.CLOSED


def test_below_threshold_stays_closed() -> None:
    breaker = make_breaker(FakeClock())
    breaker.record_failure("mock-a")
    breaker.record_failure("mock-a")
    assert breaker.state_of("mock-a") is CircuitState.CLOSED
    assert breaker.can_execute("mock-a") is True


def test_threshold_reached_opens() -> None:
    breaker = make_breaker(FakeClock())
    breaker.record_failure("mock-a")
    breaker.record_failure("mock-a")
    breaker.record_failure("mock-a")
    assert breaker.state_of("mock-a") is CircuitState.OPEN


def test_open_rejects_requests() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock)
    for _ in range(3):
        breaker.record_failure("mock-a")
    clock.advance(9.9)
    assert breaker.can_execute("mock-a") is False
    assert breaker.state_of("mock-a") is CircuitState.OPEN


def test_recovery_timeout_admits_half_open_probe() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock)
    for _ in range(3):
        breaker.record_failure("mock-a")
    clock.advance(10.0)
    assert breaker.can_execute("mock-a") is True
    assert breaker.state_of("mock-a") is CircuitState.HALF_OPEN


def test_successful_probe_closes() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock)
    for _ in range(3):
        breaker.record_failure("mock-a")
    clock.advance(10.0)
    assert breaker.can_execute("mock-a") is True
    breaker.record_success("mock-a")
    assert breaker.state_of("mock-a") is CircuitState.CLOSED
    assert breaker.can_execute("mock-a") is True


def test_failed_probe_reopens() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock)
    for _ in range(3):
        breaker.record_failure("mock-a")
    clock.advance(10.0)
    assert breaker.can_execute("mock-a") is True
    breaker.record_failure("mock-a")
    assert breaker.state_of("mock-a") is CircuitState.OPEN
    assert breaker.can_execute("mock-a") is False
    # A full recovery window is required again before the next probe.
    clock.advance(10.0)
    assert breaker.can_execute("mock-a") is True


def test_half_open_concurrency_limit() -> None:
    clock = FakeClock()
    breaker = ResilientCircuitBreaker(
        failure_threshold=1, recovery_timeout_s=10.0, half_open_max_inflight=1, clock=clock
    )
    breaker.record_failure("mock-a")
    clock.advance(10.0)
    assert breaker.can_execute("mock-a") is True  # probe permit granted
    assert breaker.can_execute("mock-a") is False  # second probe blocked
    breaker.record_success("mock-a")
    assert breaker.can_execute("mock-a") is True  # closed again


def test_success_resets_consecutive_failures() -> None:
    breaker = make_breaker(FakeClock())
    breaker.record_failure("mock-a")
    breaker.record_failure("mock-a")
    breaker.record_success("mock-a")
    breaker.record_failure("mock-a")
    breaker.record_failure("mock-a")
    assert breaker.state_of("mock-a") is CircuitState.CLOSED
    breaker.record_failure("mock-a")
    assert breaker.state_of("mock-a") is CircuitState.OPEN


def test_open_keeps_original_opened_at() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock)
    for _ in range(3):
        breaker.record_failure("mock-a")
    clock.advance(5.0)
    breaker.record_failure("mock-a")  # stray report while open
    clock.advance(5.0)  # 10s since the original opening
    assert breaker.can_execute("mock-a") is True


def test_providers_are_independent() -> None:
    breaker = make_breaker(FakeClock())
    for _ in range(3):
        breaker.record_failure("mock-a")
    assert breaker.state_of("mock-a") is CircuitState.OPEN
    assert breaker.state_of("mock-b") is CircuitState.CLOSED
    assert breaker.can_execute("mock-b") is True


def test_invalid_settings_rejected() -> None:
    with pytest.raises(ValueError):
        ResilientCircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError):
        ResilientCircuitBreaker(recovery_timeout_s=0.0)
    with pytest.raises(ValueError):
        ResilientCircuitBreaker(half_open_max_inflight=0)


def test_noop_reports_closed() -> None:
    assert NoOpCircuitBreaker().state_of("anything") is CircuitState.CLOSED
