"""Fault-tolerant execution: health/circuit gating, retry, and fallback.

M2 invariant: execution always completes to a full :class:`ProviderResponse`
(or raises) **before** response serialization. Retry and fallback therefore
happen strictly pre-first-byte. Once response bytes reach the client (M3
streaming), transparent provider replacement is impossible — see
``docs/routing.md``. This module must never grow streaming logic.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Literal, Protocol

from src.errors import GatewayError
from src.models import NormalizedChatRequest
from src.providers.base import ProviderResponse
from src.proxy.client import ProxyClient
from src.resilience.failures import (
    FailureCategory,
    affects_health,
    allows_fallback,
    classify,
    counts_toward_breaker,
)
from src.resilience.retry import BackoffSleeper, RetryPolicy, Sleeper
from src.router.circuit_breaker import CircuitBreaker, CircuitState
from src.router.engine import RouteDecision, RouterEngine
from src.router.health import HealthRegistry
from src.telemetry.latency import LatencyTracker

AttemptOutcome = Literal["success", "failed", "skipped"]
SkipReason = Literal["unhealthy", "circuit-open"]


@dataclass(frozen=True)
class AttemptRecord:
    """One gated-or-executed step against a single provider candidate."""

    provider: str
    attempt_index: int  # 1-based within that provider; 0 for skips
    outcome: AttemptOutcome
    failure_category: FailureCategory | None = None
    skip_reason: SkipReason | None = None
    latency_ms: float = 0.0


@dataclass(frozen=True)
class ExecutionOutcome:
    """Structured M2 execution metadata for logs / future telemetry."""

    request_id: str
    primary: str
    final_provider: str
    attempts: tuple[AttemptRecord, ...] = ()
    retry_count: int = 0
    fallback_used: bool = False
    circuit_states: Mapping[str, CircuitState] = field(default_factory=dict)
    total_duration_ms: float = 0.0


@dataclass(frozen=True)
class ExecuteResult:
    response: ProviderResponse
    outcome: ExecutionOutcome


class TracerLike(Protocol):
    """Structural subset of the gateway tracer used by the executor."""

    def span(self, name: str, **attrs: str) -> AbstractContextManager[None]: ...


class MetricsLike(Protocol):
    """Structural subset of the gateway metrics recorder."""

    def increment(self, name: str, *, provider: str = "", code: str = "") -> None: ...

    def observe_latency(self, name: str, value_ms: float, *, provider: str = "") -> None: ...


def _no_healthy_provider(request_id: str) -> GatewayError:
    return GatewayError(
        code="NO_HEALTHY_PROVIDER",
        message="no healthy providers available",
        status_code=503,
        provider=None,
        retryable=True,
        request_id=request_id,
    )


class ResilientExecutor:
    """Runs a router candidate chain with retry + fallback.

    Division of labor: the router decides *which* providers may be tried
    (ordered names); this executor handles *how* (gating, attempts, backoff,
    outcome recording). Provider HTTP stays in :class:`ProxyClient`;
    single-attempt normalization stays at the proxy boundary.
    """

    def __init__(
        self,
        *,
        router: RouterEngine,
        proxy: ProxyClient,
        breaker: CircuitBreaker,
        health: HealthRegistry,
        retry: RetryPolicy | None = None,
        metrics: MetricsLike | None = None,
        tracer: TracerLike | None = None,
        sleeper: Sleeper | None = None,
        jitter: random.Random | None = None,
        clock: Callable[[], float] | None = None,
        latency_tracker: LatencyTracker | None = None,
    ) -> None:
        self._router = router
        self._proxy = proxy
        self._breaker = breaker
        self._health = health
        self._retry = retry if retry is not None else RetryPolicy()
        self._metrics = metrics
        self._tracer = tracer
        self._backoff = BackoffSleeper(sleeper=sleeper, jitter=jitter)
        self._clock = clock if clock is not None else time.monotonic
        self._latency_tracker = latency_tracker

    async def execute(self, request: NormalizedChatRequest) -> ExecuteResult:
        start = self._clock()
        candidates = self._router.plan(model=request.model)
        if not candidates:
            raise _no_healthy_provider(request.request_id)
        primary = candidates[0]

        attempts: list[AttemptRecord] = []
        retry_count = 0
        fallback_counted = False
        last_error: GatewayError | None = None

        for name in candidates:
            if not self._health.is_healthy(name):
                attempts.append(
                    AttemptRecord(
                        provider=name, attempt_index=0, outcome="skipped", skip_reason="unhealthy"
                    )
                )
                continue
            if not self._breaker.can_execute(name):
                attempts.append(
                    AttemptRecord(
                        provider=name,
                        attempt_index=0,
                        outcome="skipped",
                        skip_reason="circuit-open",
                    )
                )
                continue
            if name != primary and not fallback_counted:
                self._metric("gateway_fallbacks_total", provider=name)
                fallback_counted = True
            outcome_or_error, retries = await self._attempt_provider(
                name, request, candidates, attempts, start
            )
            retry_count += retries
            if not isinstance(outcome_or_error, GatewayError):
                return ExecuteResult(
                    response=outcome_or_error,
                    outcome=self._outcome(
                        request=request,
                        primary=primary,
                        final_provider=name,
                        attempts=attempts,
                        retry_count=retry_count,
                        candidates=candidates,
                        start=start,
                    ),
                )
            last_error = outcome_or_error
            if not allows_fallback(classify(last_error)):
                raise last_error

        if last_error is not None:
            raise last_error
        raise _no_healthy_provider(request.request_id)

    async def _attempt_provider(
        self,
        name: str,
        request: NormalizedChatRequest,
        candidates: list[str],
        attempts: list[AttemptRecord],
        start: float,
    ) -> tuple[ProviderResponse | GatewayError, int]:
        """Attempt one provider up to the retry budget.

        Returns the response (or terminal error) plus retries performed.
        Appends one :class:`AttemptRecord` per attempt to ``attempts``.
        """
        retries = 0
        attempt = 0
        if request.provider == name:
            scoped = request
        else:
            scoped = request.model_copy(update={"provider": name})
        route = RouteDecision(
            provider_name=name,
            reason="primary" if name == candidates[0] else "fallback",
            tried=list(candidates),
        )
        while True:
            attempt += 1
            begun = self._clock()
            try:
                result = await self._forward(route, scoped, provider=name, attempt=attempt)
            except GatewayError as exc:
                latency_ms = (self._clock() - begun) * 1000.0
                category = classify(exc)
                self._metric("gateway_provider_attempts_total", provider=name, code=exc.code)
                if affects_health(category):
                    self._health.mark_unhealthy(name, exc.code)
                    attempts.append(
                        AttemptRecord(
                            provider=name,
                            attempt_index=attempt,
                            outcome="failed",
                            failure_category=category,
                            latency_ms=latency_ms,
                        )
                    )
                    return exc, retries
                if not counts_toward_breaker(category):
                    attempts.append(
                        AttemptRecord(
                            provider=name,
                            attempt_index=attempt,
                            outcome="failed",
                            failure_category=category,
                            latency_ms=latency_ms,
                        )
                    )
                    return exc, retries
                was_open = self._breaker.state_of(name)
                self._breaker.record_failure(name)
                if self._breaker.state_of(name) is CircuitState.OPEN and (
                    was_open is not CircuitState.OPEN
                ):
                    self._metric("gateway_breaker_opens_total", provider=name, code=exc.code)
                attempts.append(
                    AttemptRecord(
                        provider=name,
                        attempt_index=attempt,
                        outcome="failed",
                        failure_category=category,
                        latency_ms=latency_ms,
                    )
                )
                elapsed_ms = (self._clock() - start) * 1000.0
                if not self._retry.should_retry(
                    category=category, attempt_index=attempt, elapsed_ms=elapsed_ms
                ):
                    return exc, retries
                await self._backoff.sleep_before_retry(
                    backoff_ceiling_ms=self._retry.backoff_ms(attempt_index=attempt)
                )
                retries += 1
                self._metric("gateway_retries_total", provider=name, code=exc.code)
                continue
            latency_ms = (self._clock() - begun) * 1000.0
            self._breaker.record_success(name)
            self._health.mark_healthy(name)
            if self._latency_tracker is not None:
                self._latency_tracker.record(name, latency_ms)
            self._metric("gateway_provider_attempts_total", provider=name, code="ok")
            self._metrics_latency(name, latency_ms)
            attempts.append(
                AttemptRecord(
                    provider=name,
                    attempt_index=attempt,
                    outcome="success",
                    latency_ms=latency_ms,
                )
            )
            return result, retries

    async def _forward(
        self, route: RouteDecision, request: NormalizedChatRequest, *, provider: str, attempt: int
    ) -> ProviderResponse:
        if self._tracer is None:
            return await self._proxy.forward(route, request)
        with self._tracer.span("provider.forward", provider=provider, attempt=str(attempt)):
            return await self._proxy.forward(route, request)

    def _outcome(
        self,
        *,
        request: NormalizedChatRequest,
        primary: str,
        final_provider: str,
        attempts: list[AttemptRecord],
        retry_count: int,
        candidates: list[str],
        start: float,
    ) -> ExecutionOutcome:
        fallback_used = any(
            record.provider != primary and record.outcome != "skipped" for record in attempts
        )
        return ExecutionOutcome(
            request_id=request.request_id,
            primary=primary,
            final_provider=final_provider,
            attempts=tuple(attempts),
            retry_count=retry_count,
            fallback_used=fallback_used,
            circuit_states={name: self._breaker.state_of(name) for name in candidates},
            total_duration_ms=(self._clock() - start) * 1000.0,
        )

    def _metric(self, name: str, *, provider: str = "", code: str = "") -> None:
        if self._metrics is not None:
            self._metrics.increment(name, provider=provider, code=code)

    def _metrics_latency(self, provider: str, latency_ms: float) -> None:
        if self._metrics is not None:
            self._metrics.observe_latency(
                "gateway_provider_attempt_ms", latency_ms, provider=provider
            )
