"""Multi-factor provider scoring: latency vs cost vs quality.

One-sentence formula: score = (w_lat * latency_score + w_cost * cost_score
+ w_qual * quality) / (w_lat + w_cost + w_qual), where latency_score and
cost_score are 1 minus the min-max normalization of that factor across the
eligible candidates (lowest raw value scores 1, highest scores 0, ties score 1)
and quality is the provider's 0-1 ``quality_weight`` used as-is; the highest
score wins, ties keep static priority order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringWeights:
    """Weights for the three 0-1 score components.

    Equal defaults mean equally weighted. All weights must be >= 0; an
    all-zero triple falls back to equal weighting instead of dividing by zero.
    """

    latency: float = 1.0
    cost: float = 1.0
    quality: float = 1.0

    def __post_init__(self) -> None:
        if self.latency < 0.0 or self.cost < 0.0 or self.quality < 0.0:
            raise ValueError("scoring weights must be >= 0")

    def normalized(self) -> tuple[float, float, float]:
        total = self.latency + self.cost + self.quality
        if total <= 0.0:
            return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        return (self.latency / total, self.cost / total, self.quality / total)


@dataclass(frozen=True)
class ScoredCandidate:
    """One eligible provider's raw scoring inputs."""

    name: str
    latency_ms: float | None  # rolling avg; None = no samples yet
    cost_per_1k_tokens: float
    quality: float  # 0-1 quality_weight; higher is better


def _invert_min_max(value: float, minimum: float, maximum: float) -> float:
    """Map ``value`` to 0-1 with lower raw values scoring higher.

    The minimum scores 1, the maximum scores 0; a degenerate (all-equal)
    range scores 1 for everyone so the factor becomes a tie.
    """
    if maximum <= minimum:
        return 1.0
    clamped = min(max(value, minimum), maximum)
    return (maximum - clamped) / (maximum - minimum)


def score_provider(
    *,
    latency_ms: float,
    cost_per_1k_tokens: float,
    quality: float,
    min_latency_ms: float,
    max_latency_ms: float,
    min_cost: float,
    max_cost: float,
    weights: ScoringWeights | None = None,
) -> float:
    """Score one provider in 0-1 (higher is better).

    ``latency_ms``/``cost_per_1k_tokens`` are min-max normalized against the
    eligible candidate set's ``min_*``/``max_*`` bounds and inverted (lower is
    better); ``quality`` is clamped to 0-1 and used directly (higher is
    better). The three components combine as a weighted average using
    ``weights`` (default equally weighted).
    """
    w = weights if weights is not None else ScoringWeights()
    w_lat, w_cost, w_qual = w.normalized()
    latency_score = _invert_min_max(latency_ms, min_latency_ms, max_latency_ms)
    cost_score = _invert_min_max(
        max(cost_per_1k_tokens, 0.0), min_cost, max(max_cost, min_cost)
    )
    quality_score = min(max(quality, 0.0), 1.0)
    return w_lat * latency_score + w_cost * cost_score + w_qual * quality_score


def rank_candidates(
    candidates: Sequence[ScoredCandidate],
    weights: ScoringWeights | None = None,
) -> list[str]:
    """Order eligible candidate names by descending composite score.

    Providers with no latency samples (``None``) are assigned the mean of the
    known averages, or 0.0 when nobody has samples (which ties the latency
    factor for everyone). Ties keep the input order, so callers pass
    candidates in static priority order and get deterministic fallback
    behavior. Health/circuit-breaker filtering happens before this call —
    every candidate passed in is eligible, and the top name is the pick.
    """
    names = [c.name for c in candidates]
    if not names:
        return []
    w = weights if weights is not None else ScoringWeights()
    known = [c.latency_ms for c in candidates if c.latency_ms is not None]
    lat_fill = sum(known) / len(known) if known else 0.0
    latencies = [
        c.latency_ms if c.latency_ms is not None else lat_fill for c in candidates
    ]
    costs = [max(c.cost_per_1k_tokens, 0.0) for c in candidates]
    min_lat, max_lat = min(latencies), max(latencies)
    min_cost, max_cost = min(costs), max(costs)
    scored = [
        (
            score_provider(
                latency_ms=lat,
                cost_per_1k_tokens=c.cost_per_1k_tokens,
                quality=c.quality,
                min_latency_ms=min_lat,
                max_latency_ms=max_lat,
                min_cost=min_cost,
                max_cost=max_cost,
                weights=w,
            ),
            c.name,
        )
        for c, lat in zip(candidates, latencies)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)  # stable: ties keep input order
    return [name for _, name in scored]
