"""Router unit tests: static priority + health + breaker seams."""

from __future__ import annotations

import pytest

from src.errors import GatewayError
from src.router.circuit_breaker import NoOpCircuitBreaker
from src.router.engine import RouterEngine
from src.router.health import HealthRegistry


def make_engine() -> tuple[RouterEngine, HealthRegistry]:
    health = HealthRegistry(["mock-a", "mock-b"])
    engine = RouterEngine(
        priority=["mock-a", "mock-b"],
        default_provider="mock-a",
        model_aliases={"mock-b": "mock-b"},
        health=health,
        breaker=NoOpCircuitBreaker(),
    )
    return engine, health


def test_alias_routes_to_mock_b() -> None:
    engine, _ = make_engine()
    decision = engine.select_provider(model="mock-b", request_id="r1")
    assert decision.provider_name == "mock-b"


def test_default_model_routes_to_default() -> None:
    engine, _ = make_engine()
    decision = engine.select_provider(model="unknown-model", request_id="r1")
    assert decision.provider_name == "mock-a"


def test_unhealthy_provider_skipped() -> None:
    engine, health = make_engine()
    health.mark_unhealthy("mock-a", "boom")
    decision = engine.select_provider(model="mock-a", request_id="r1")
    assert decision.provider_name == "mock-b"
    assert "mock-a" in decision.tried


def test_no_healthy_provider_raises_503() -> None:
    engine, health = make_engine()
    health.mark_unhealthy("mock-a", "x")
    health.mark_unhealthy("mock-b", "y")
    with pytest.raises(GatewayError) as excinfo:
        engine.select_provider(model="mock-a", request_id="r1")
    assert excinfo.value.status_code == 503
    assert excinfo.value.code == "NO_HEALTHY_PROVIDER"
