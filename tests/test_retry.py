"""Retry tests: policy math plus executor retry behavior with scripted mocks."""

from __future__ import annotations

import random

import pytest

from src.errors import GatewayError
from src.models import ChatMessage, NormalizedChatRequest
from src.providers.mock import MockProvider, MockProviderSettings
from src.proxy.client import ProxyClient
from src.resilience.executor import ResilientExecutor
from src.resilience.failures import FailureCategory
from src.resilience.retry import BackoffSleeper, RetryPolicy
from src.router.circuit_breaker import ResilientCircuitBreaker
from src.router.engine import RouterEngine
from src.router.health import HealthRegistry


def make_request(request_id: str = "req-retry") -> NormalizedChatRequest:
    return NormalizedChatRequest(
        request_id=request_id,
        model="mock-a",
        provider="mock-a",
        messages=[ChatMessage(role="user", content="hi")],
        temperature=0.0,
        max_tokens=None,
    )


class RecordingProvider(MockProvider):
    """Captures inbound request IDs to prove stability across attempts."""

    def __init__(self, name: str, settings: MockProviderSettings) -> None:
        super().__init__(name, settings)
        self.seen_request_ids: list[str] = []

    async def chat(self, request: NormalizedChatRequest):  # type: ignore[override]
        self.seen_request_ids.append(request.request_id)
        return await super().chat(request)


def build(
    *,
    script_a: tuple[str | None, ...] = (),
    priority: list[str] | None = None,
    retry: RetryPolicy | None = None,
    provider_cls: type[MockProvider] = MockProvider,
) -> tuple[ResilientExecutor, list[float], dict[str, MockProvider]]:
    names = priority if priority is not None else ["mock-a", "mock-b"]
    providers: dict[str, MockProvider] = {}
    timeouts: dict[str, float] = {}
    for name in names:
        script = script_a if name == "mock-a" else ()
        providers[name] = provider_cls(name, MockProviderSettings(failure_script=script))  # type: ignore[arg-type]
        timeouts[name] = 5.0
    health = HealthRegistry(names)
    breaker = ResilientCircuitBreaker(failure_threshold=100)
    router = RouterEngine(
        priority=names,
        default_provider="mock-a",
        model_aliases={"mock-a": "mock-a", "mock-b": "mock-b"},
        health=health,
        breaker=breaker,
    )
    proxy = ProxyClient(providers, timeouts)
    sleeps: list[float] = []

    async def recorder(delay_ms: float) -> None:
        sleeps.append(delay_ms)

    executor = ResilientExecutor(
        router=router,
        proxy=proxy,
        breaker=breaker,
        health=health,
        retry=retry if retry is not None else RetryPolicy(max_attempts=3, backoff_base_ms=10.0),
        sleeper=recorder,
        jitter=random.Random(0),
    )
    return executor, sleeps, providers


async def test_immediate_success_no_retry() -> None:
    executor, sleeps, _ = build()
    result = await executor.execute(make_request())
    assert result.response.provider == "mock-a"
    assert result.outcome.retry_count == 0
    assert result.outcome.fallback_used is False
    assert result.outcome.final_provider == "mock-a"
    assert sleeps == []


async def test_retryable_failure_retried_then_succeeds() -> None:
    executor, sleeps, _ = build(script_a=("unavailable", None))
    result = await executor.execute(make_request())
    assert result.response.provider == "mock-a"
    assert result.outcome.retry_count == 1
    assert result.outcome.fallback_used is False
    assert len([a for a in result.outcome.attempts if a.provider == "mock-a"]) == 2
    assert len(sleeps) == 1
    assert 0.0 <= sleeps[0] <= 10.0


async def test_retry_exhaustion_raises_last_error() -> None:
    executor, sleeps, _ = build(
        script_a=("unavailable",) * 5,
        priority=["mock-a"],
        retry=RetryPolicy(max_attempts=3, backoff_base_ms=10.0),
    )
    with pytest.raises(GatewayError) as excinfo:
        await executor.execute(make_request())
    assert excinfo.value.code == "PROVIDER_UNAVAILABLE"
    assert excinfo.value.provider == "mock-a"
    assert excinfo.value.request_id == "req-retry"
    assert len(sleeps) == 2  # bounded: max_attempts - 1 backoffs


async def test_non_retryable_failure_does_not_retry() -> None:
    executor, sleeps, providers = build(script_a=("permanent",), provider_cls=RecordingProvider)
    result = await executor.execute(make_request())
    assert result.response.provider == "mock-b"  # fell back without retrying mock-a
    assert result.outcome.fallback_used is True
    assert sleeps == []
    mock_a_records = [a for a in result.outcome.attempts if a.provider == "mock-a"]
    assert len(mock_a_records) == 1
    assert mock_a_records[0].failure_category is FailureCategory.PERMANENT
    seen = providers["mock-a"].seen_request_ids  # type: ignore[attr-defined]
    assert len(seen) == 1  # exactly one call, no retry


async def test_request_id_preserved_across_attempts() -> None:
    executor, _, providers = build(script_a=("unavailable", None), provider_cls=RecordingProvider)
    await executor.execute(make_request("stable-id"))
    seen = providers["mock-a"].seen_request_ids  # type: ignore[attr-defined]
    assert seen == ["stable-id", "stable-id"]


async def test_elapsed_budget_stops_retries() -> None:
    state = {"now": 0.0}
    sleeps: list[float] = []

    async def advancing_sleeper(delay_ms: float) -> None:
        sleeps.append(delay_ms)
        state["now"] += 1.0  # each backoff burns a full second of budget

    providers = {"mock-a": MockProvider("mock-a", MockProviderSettings(fail_rate=1.0))}
    names = ["mock-a"]
    health = HealthRegistry(names)
    breaker = ResilientCircuitBreaker(failure_threshold=100)
    router = RouterEngine(
        priority=names,
        default_provider="mock-a",
        model_aliases={},
        health=health,
        breaker=breaker,
    )
    executor = ResilientExecutor(
        router=router,
        proxy=ProxyClient(providers, {"mock-a": 5.0}),
        breaker=breaker,
        health=health,
        retry=RetryPolicy(max_attempts=10, backoff_base_ms=50.0, max_elapsed_ms=100.0),
        sleeper=advancing_sleeper,
        jitter=random.Random(0),
        clock=lambda: state["now"],
    )
    with pytest.raises(GatewayError):
        await executor.execute(make_request())
    assert len(sleeps) == 1  # budget (100ms) consumed after the first backoff


def test_policy_backoff_sequence_and_cap() -> None:
    policy = RetryPolicy(max_attempts=6, backoff_base_ms=50.0, backoff_max_ms=200.0)
    assert policy.backoff_ms(attempt_index=1) == 50.0
    assert policy.backoff_ms(attempt_index=2) == 100.0
    assert policy.backoff_ms(attempt_index=3) == 200.0
    assert policy.backoff_ms(attempt_index=4) == 200.0  # capped


def test_policy_should_retry_matrix() -> None:
    policy = RetryPolicy(max_attempts=3, max_elapsed_ms=1000.0)
    assert policy.should_retry(category=FailureCategory.TIMEOUT, attempt_index=1, elapsed_ms=0.0)
    assert policy.should_retry(
        category=FailureCategory.RATE_LIMITED, attempt_index=2, elapsed_ms=10.0
    )
    assert not policy.should_retry(
        category=FailureCategory.TIMEOUT, attempt_index=3, elapsed_ms=0.0
    )  # attempts exhausted
    assert not policy.should_retry(
        category=FailureCategory.PERMANENT, attempt_index=1, elapsed_ms=0.0
    )
    assert not policy.should_retry(
        category=FailureCategory.TIMEOUT, attempt_index=1, elapsed_ms=1000.0
    )  # budget spent


def test_policy_validation() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError):
        RetryPolicy(backoff_base_ms=0.0)
    with pytest.raises(ValueError):
        RetryPolicy(backoff_base_ms=100.0, backoff_max_ms=10.0)


async def test_backoff_sleeper_full_jitter_bounds() -> None:
    seen: list[float] = []

    async def recorder(delay_ms: float) -> None:
        seen.append(delay_ms)

    sleeper = BackoffSleeper(sleeper=recorder, jitter=random.Random(42))
    slept = await sleeper.sleep_before_retry(backoff_ceiling_ms=100.0)
    assert 0.0 <= slept <= 100.0
    assert seen == [slept]
