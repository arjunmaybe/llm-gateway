"""Static priority router (M1). Scoring / adaptive policies arrive in M2."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.errors import GatewayError
from src.router.circuit_breaker import CircuitBreaker
from src.router.health import HealthRegistry


class RouteDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_name: str
    reason: str
    tried: list[str] = Field(default_factory=list)


class RouterEngine:
    """Selects a provider by static priority, skipping unhealthy/open ones."""

    def __init__(
        self,
        *,
        priority: list[str],
        default_provider: str,
        model_aliases: dict[str, str],
        health: HealthRegistry,
        breaker: CircuitBreaker,
    ) -> None:
        self._priority = list(priority)
        self._default_provider = default_provider
        self._model_aliases = dict(model_aliases)
        self._health = health
        self._breaker = breaker

    def select_provider(self, *, model: str, request_id: str) -> RouteDecision:
        _ = request_id
        candidates = self._ordered_candidates(model)
        tried: list[str] = []
        for name in candidates:
            tried.append(name)
            if not self._health.is_healthy(name):
                continue
            if not self._breaker.can_execute(name):
                continue
            reason = "default" if name == self._default_provider else "alias-or-priority"
            if name == candidates[0] and name != self._default_provider:
                reason = "model-alias"
            return RouteDecision(provider_name=name, reason=reason, tried=tried)
        raise GatewayError(
            code="NO_HEALTHY_PROVIDER",
            message="no healthy providers available",
            status_code=503,
            provider=None,
            retryable=True,
            request_id=request_id,
        )

    def _ordered_candidates(self, model: str) -> list[str]:
        preferred = self._model_aliases.get(model)
        ordered = list(self._priority)
        if preferred is not None and preferred in ordered:
            ordered.remove(preferred)
            ordered.insert(0, preferred)
        elif model in ordered:
            ordered.remove(model)
            ordered.insert(0, model)
        return ordered
