"""Priority router with multi-factor scoring for the candidate chain.

M2 adds the fallback candidate chain; scoring orders it, alias pins first.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.errors import GatewayError
from src.router.circuit_breaker import CircuitBreaker
from src.router.health import HealthRegistry
from src.router.scorer import ScoredCandidate, ScoringWeights, rank_candidates
from src.telemetry.latency import LatencyTracker


class RouteDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_name: str
    reason: str
    tried: list[str] = Field(default_factory=list)


class RouterEngine:
    """Selects a provider by score among healthy/closed candidates.

    Score order is the fallback chain (see :meth:`plan`); static priority
    order remains the tie-breaker; an explicit model alias still pins its
    target.
    """

    def __init__(
        self,
        *,
        priority: list[str],
        default_provider: str,
        model_aliases: dict[str, str],
        health: HealthRegistry,
        breaker: CircuitBreaker,
        costs: dict[str, float] | None = None,
        qualities: dict[str, float] | None = None,
        latency_tracker: LatencyTracker | None = None,
        scoring_weights: ScoringWeights | None = None,
    ) -> None:
        self._priority = list(priority)
        self._default_provider = default_provider
        self._model_aliases = dict(model_aliases)
        self._health = health
        self._breaker = breaker
        # Scoring inputs. All optional so existing constructions keep working;
        # missing entries fall back to free cost, neutral quality, no samples.
        self._costs = dict(costs) if costs is not None else {}
        self._qualities = dict(qualities) if qualities is not None else {}
        self._latency_tracker = latency_tracker
        self._scoring_weights = (
            scoring_weights if scoring_weights is not None else ScoringWeights()
        )

    def plan(self, *, model: str) -> list[str]:
        """Ordered candidate chain for a model: alias-first, then by score.

        M2: returns every candidate without health/breaker filtering. The
        executor applies health and breaker gates per attempt, so a breaker
        that recovers mid-request is still honored. An explicit model alias
        (or provider name used as model) still pins its target first —
        same as :meth:`select_provider` — and the remaining candidates are
        ordered by :func:`rank_candidates` score; ties keep static priority
        order. With no alias hit the whole chain is score-ordered.
        """
        ordered = self._ordered_candidates(model)
        if not ordered:
            return []
        if self._is_alias_hit(model):
            head, rest = ordered[0], ordered[1:]
            if not rest:
                return ordered
            ranked_rest = rank_candidates(
                [self._scored_candidate(name) for name in rest],
                self._scoring_weights,
            )
            return [head] + ranked_rest
        return rank_candidates(
            [self._scored_candidate(name) for name in ordered],
            self._scoring_weights,
        )

    def select_provider(self, *, model: str, request_id: str) -> RouteDecision:
        candidates = self.plan(model=model)
        tried: list[str] = []
        eligible: list[str] = []
        for name in candidates:
            tried.append(name)
            if not self._health.is_healthy(name):
                continue
            if not self._breaker.can_execute(name):
                continue
            eligible.append(name)
        if not eligible:
            raise GatewayError(
                code="NO_HEALTHY_PROVIDER",
                message="no healthy providers available",
                status_code=503,
                provider=None,
                retryable=True,
                request_id=request_id,
            )
        # Explicit model alias (or provider name used as model) still wins
        # over scoring when its target is eligible — same as static routing.
        if self._is_alias_hit(model) and candidates[0] in eligible:
            picked = candidates[0]
        else:
            picked = self._best_by_score(eligible)
        reason = "default" if picked == self._default_provider else "alias-or-priority"
        if picked == candidates[0] and picked != self._default_provider:
            reason = "model-alias"
        return RouteDecision(provider_name=picked, reason=reason, tried=tried)

    def _is_alias_hit(self, model: str) -> bool:
        """Whether ``model`` explicitly pins the front candidate (alias or name)."""
        preferred = self._model_aliases.get(model)
        if preferred is not None and preferred in self._priority:
            return True
        return model in self._priority

    def _scored_candidate(self, name: str) -> ScoredCandidate:
        """Build the scorer input for one provider name."""
        tracker = self._latency_tracker
        return ScoredCandidate(
            name=name,
            latency_ms=tracker.average(name) if tracker is not None else None,
            cost_per_1k_tokens=self._costs.get(name, 0.0),
            quality=self._qualities.get(name, 0.5),
        )

    def _best_by_score(self, eligible: list[str]) -> str:
        """Highest composite score among eligible names; ties keep given order."""
        ranked = rank_candidates(
            [self._scored_candidate(name) for name in eligible],
            self._scoring_weights,
        )
        return ranked[0]

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
