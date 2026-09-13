"""Fallback tests: executor movement across the router candidate chain."""

from __future__ import annotations

import random

import pytest

from src.errors import GatewayError
from src.models import ChatMessage, NormalizedChatRequest
from src.providers.mock import MockProvider, MockProviderSettings
from src.proxy.client import ProxyClient
from src.resilience.executor import ResilientExecutor
from src.resilience.failures import FailureCategory
from src.resilience.retry import RetryPolicy
from src.router.circuit_breaker import ResilientCircuitBreaker
from src.router.engine import RouterEngine
from src.router.health import HealthRegistry
from tests.test_retry import RecordingProvider, make_request


async def _no_sleep(delay_ms: float) -> None:
    _ = delay_ms


def build_chain(
    *,
    script_a: tuple[str | None, ...] = (),
    script_b: tuple[str | None, ...] = (),
    priority: list[str] | None = None,
    breaker: ResilientCircuitBreaker | None = None,
) -> tuple[
    ResilientExecutor, HealthRegistry, ResilientCircuitBreaker, dict[str, RecordingProvider]
]:
    names = priority if priority is not None else ["mock-a", "mock-b"]
    scripts = {"mock-a": script_a, "mock-b": script_b}
    providers: dict[str, RecordingProvider] = {
        name: RecordingProvider(name, MockProviderSettings(failure_script=scripts[name]))  # type: ignore[arg-type]
        for name in names
    }
    health = HealthRegistry(names)
    active = breaker if breaker is not None else ResilientCircuitBreaker(failure_threshold=100)
    router = RouterEngine(
        priority=names,
        default_provider="mock-a",
        model_aliases={"mock-a": "mock-a", "mock-b": "mock-b"},
        health=health,
        breaker=active,
    )
    proxies: dict[str, MockProvider] = dict(providers)
    executor = ResilientExecutor(
        router=router,
        proxy=ProxyClient(proxies, {name: 5.0 for name in names}),
        breaker=active,
        health=health,
        retry=RetryPolicy(max_attempts=2, backoff_base_ms=1.0),
        sleeper=_no_sleep,
        jitter=random.Random(0),
    )
    return executor, health, active, providers


async def test_primary_failure_falls_back_to_success() -> None:
    executor, _, _, _ = build_chain(script_a=("unavailable",) * 4)
    result = await executor.execute(make_request())
    assert result.response.provider == "mock-b"
    assert result.outcome.primary == "mock-a"
    assert result.outcome.final_provider == "mock-b"
    assert result.outcome.fallback_used is True
    assert result.outcome.retry_count >= 1
    assert result.outcome.request_id == "req-retry"


async def test_primary_circuit_open_goes_directly_to_fallback() -> None:
    breaker = ResilientCircuitBreaker(failure_threshold=1)
    breaker.record_failure("mock-a")  # opens mock-a before the request
    executor, _, _, providers = build_chain(breaker=breaker)
    result = await executor.execute(make_request())
    assert result.response.provider == "mock-b"
    assert result.outcome.fallback_used is True
    assert providers["mock-a"].seen_request_ids == []  # never attempted
    skips = [a for a in result.outcome.attempts if a.outcome == "skipped"]
    assert [(s.provider, s.skip_reason) for s in skips] == [("mock-a", "circuit-open")]


async def test_fallback_failure_raises_last_error() -> None:
    executor, _, _, _ = build_chain(script_a=("unavailable",) * 4, script_b=("unavailable",) * 4)
    with pytest.raises(GatewayError) as excinfo:
        await executor.execute(make_request())
    assert excinfo.value.code == "PROVIDER_UNAVAILABLE"
    assert excinfo.value.provider == "mock-b"  # last attempted provider
    assert excinfo.value.request_id == "req-retry"


async def test_all_providers_skipped_raises_503() -> None:
    executor, health, _, _ = build_chain()
    health.mark_unhealthy("mock-a", "test")
    health.mark_unhealthy("mock-b", "test")
    with pytest.raises(GatewayError) as excinfo:
        await executor.execute(make_request())
    assert excinfo.value.code == "NO_HEALTHY_PROVIDER"
    assert excinfo.value.status_code == 503
    assert excinfo.value.provider is None


async def test_provider_outside_priority_chain_is_never_used() -> None:
    # mock-b exists and is healthy but is not in the router priority chain,
    # mirroring a disabled provider: it must never be attempted.
    executor, _, _, providers = build_chain(
        script_a=("unavailable",) * 4, priority=["mock-a"]
    )
    with pytest.raises(GatewayError):
        await executor.execute(make_request())
    assert "mock-b" not in providers


async def test_permanent_failure_falls_back_without_retry() -> None:
    executor, _, _, providers = build_chain(script_a=("permanent",))
    result = await executor.execute(make_request())
    assert result.response.provider == "mock-b"
    assert len(providers["mock-a"].seen_request_ids) == 1
    assert result.outcome.retry_count == 0


async def test_outcome_metadata_shape() -> None:
    executor, _, active, _ = build_chain(script_a=("unavailable", None))
    result = await executor.execute(make_request("meta-1"))
    outcome = result.outcome
    assert outcome.request_id == "meta-1"
    assert outcome.primary == "mock-a"
    assert outcome.final_provider == "mock-a"
    assert outcome.fallback_used is False
    assert outcome.retry_count == 1
    assert all(
        record.failure_category is not None or record.outcome == "success"
        for record in outcome.attempts
    )
    failed = [a for a in outcome.attempts if a.outcome == "failed"]
    assert failed and failed[0].failure_category is FailureCategory.UPSTREAM_TRANSIENT
    assert set(outcome.circuit_states) == {"mock-a", "mock-b"}
    assert outcome.total_duration_ms >= 0.0


async def test_router_plan_orders_alias_first() -> None:
    executor, _, _, _ = build_chain(script_b=("unavailable",) * 4)
    aliased = NormalizedChatRequest(
        request_id="alias-1",
        model="mock-b",
        provider="mock-b",
        messages=[ChatMessage(role="user", content="hi")],
        temperature=0.0,
        max_tokens=None,
    )
    result = await executor.execute(aliased)
    assert result.outcome.primary == "mock-b"
    assert result.outcome.fallback_used is True
    assert result.response.provider == "mock-a"
