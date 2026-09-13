"""Passive health tracking tests: executor outcome reporting to the registry."""

from __future__ import annotations

from src.router.health import HealthRegistry
from tests.test_fallback import build_chain
from tests.test_retry import make_request


async def test_success_keeps_provider_healthy() -> None:
    executor, health, _, _ = build_chain()
    assert health.is_healthy("mock-a") is True
    await executor.execute(make_request())
    assert health.is_healthy("mock-a") is True
    assert health.readiness() == {"mock-a": True, "mock-b": True}


async def test_retryable_failure_does_not_mark_unhealthy() -> None:
    # Transient faults are the breaker's job; the registry stays green so a
    # recovered provider needs no external signal to re-enter rotation.
    executor, health, _, _ = build_chain(script_a=("unavailable", None))
    await executor.execute(make_request())
    assert health.is_healthy("mock-a") is True
    assert health.snapshot()["mock-a"].healthy is True


async def test_permanent_failure_marks_provider_unhealthy() -> None:
    executor, health, _, _ = build_chain(script_a=("permanent",))
    result = await executor.execute(make_request())
    assert result.response.provider == "mock-b"
    assert health.is_healthy("mock-a") is False
    assert health.snapshot()["mock-a"].error == "PROVIDER_REJECTED"
    assert health.readiness()["mock-a"] is False


async def test_recovery_via_explicit_healthy_mark() -> None:
    # M2 has no active probing: an unhealthy mark sticks until an external
    # signal (future prober, operator, config reload) re-marks healthy.
    # This test pins that seam contract.
    executor, health, _, _ = build_chain(script_a=("permanent", "permanent"))
    await executor.execute(make_request())
    assert health.is_healthy("mock-a") is False
    health.mark_healthy("mock-a")
    assert health.is_healthy("mock-a") is True
    result = await executor.execute(make_request())
    # mock-a still rejects; health is routing input, not fate
    assert result.response.provider == "mock-b"
    assert health.is_healthy("mock-a") is False  # permanent failure re-marks


def test_registry_defaults_to_unknown_unhealthy() -> None:
    registry = HealthRegistry(["mock-a"])
    assert registry.is_healthy("ghost") is False
