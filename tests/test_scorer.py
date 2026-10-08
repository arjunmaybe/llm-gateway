"""Scorer tests: each factor decides in isolation; gates still exclude."""

from __future__ import annotations

from src.router.circuit_breaker import ResilientCircuitBreaker
from src.router.engine import RouterEngine
from src.router.health import HealthRegistry
from src.router.scorer import ScoredCandidate, ScoringWeights, rank_candidates, score_provider
from src.telemetry.latency import LatencyTracker

GENERIC_MODEL = "generic-model"  # hits no alias and matches no provider name


def make_engine(
    *,
    priority: list[str] | None = None,
    latencies: dict[str, float] | None = None,
    costs: dict[str, float] | None = None,
    qualities: dict[str, float] | None = None,
    unhealthy: tuple[str, ...] = (),
    open_breaker: tuple[str, ...] = (),
    model_aliases: dict[str, str] | None = None,
    weights: ScoringWeights | None = None,
) -> RouterEngine:
    names = list(priority) if priority is not None else ["prov-a", "prov-b"]
    health = HealthRegistry(names)
    for name in unhealthy:
        health.mark_unhealthy(name, "test")
    breaker = ResilientCircuitBreaker(failure_threshold=1)
    for name in open_breaker:
        breaker.record_failure(name)  # threshold=1 -> OPEN immediately
    tracker = LatencyTracker()
    for name, ms in (latencies or {}).items():
        for _ in range(5):
            tracker.record(name, ms)
    return RouterEngine(
        priority=names,
        default_provider=names[0],
        model_aliases=dict(model_aliases) if model_aliases is not None else {},
        health=health,
        breaker=breaker,
        costs=dict(costs) if costs is not None else {},
        qualities=dict(qualities) if qualities is not None else {},
        latency_tracker=tracker,
        scoring_weights=weights,
    )


def select(engine: RouterEngine, model: str = GENERIC_MODEL) -> str:
    return engine.select_provider(model=model, request_id="r-scorer").provider_name


def test_lower_latency_wins_when_cost_quality_equal() -> None:
    engine = make_engine(
        priority=["prov-slow", "prov-fast"],  # static order favors the slow one
        latencies={"prov-slow": 500.0, "prov-fast": 50.0},
        costs={"prov-slow": 0.0, "prov-fast": 0.0},
        qualities={"prov-slow": 0.5, "prov-fast": 0.5},
    )
    assert select(engine) == "prov-fast"


def test_lower_cost_wins_when_latency_quality_equal() -> None:
    engine = make_engine(
        priority=["prov-expensive", "prov-cheap"],
        costs={"prov-expensive": 0.09, "prov-cheap": 0.01},
        qualities={"prov-expensive": 0.5, "prov-cheap": 0.5},
    )
    assert select(engine) == "prov-cheap"


def test_higher_quality_wins_when_latency_cost_equal() -> None:
    engine = make_engine(
        priority=["prov-lowq", "prov-highq"],
        costs={"prov-lowq": 0.0, "prov-highq": 0.0},
        qualities={"prov-lowq": 0.2, "prov-highq": 0.9},
    )
    assert select(engine) == "prov-highq"


def test_unhealthy_provider_excluded_regardless_of_score() -> None:
    engine = make_engine(
        latencies={"prov-a": 10.0, "prov-b": 1000.0},
        costs={"prov-a": 0.0, "prov-b": 0.99},
        qualities={"prov-a": 1.0, "prov-b": 0.0},
        unhealthy=("prov-a",),
    )
    assert select(engine) == "prov-b"


def test_open_breaker_provider_excluded_regardless_of_score() -> None:
    engine = make_engine(
        latencies={"prov-a": 10.0, "prov-b": 1000.0},
        costs={"prov-a": 0.0, "prov-b": 0.99},
        qualities={"prov-a": 1.0, "prov-b": 0.0},
        open_breaker=("prov-a",),
    )
    assert select(engine) == "prov-b"


def test_explicit_alias_still_beats_a_better_score() -> None:
    engine = make_engine(
        latencies={"prov-a": 1000.0, "prov-b": 10.0},
        costs={"prov-a": 0.99, "prov-b": 0.0},
        qualities={"prov-a": 0.0, "prov-b": 1.0},
        model_aliases={"pinned": "prov-a"},
    )
    assert select(engine, model="pinned") == "prov-a"


def test_score_provider_orders_single_factor_correctly() -> None:
    weights = ScoringWeights()
    fast = score_provider(
        latency_ms=50.0,
        cost_per_1k_tokens=0.05,
        quality=0.5,
        min_latency_ms=50.0,
        max_latency_ms=500.0,
        min_cost=0.05,
        max_cost=0.05,
        weights=weights,
    )
    slow = score_provider(
        latency_ms=500.0,
        cost_per_1k_tokens=0.05,
        quality=0.5,
        min_latency_ms=50.0,
        max_latency_ms=500.0,
        min_cost=0.05,
        max_cost=0.05,
        weights=weights,
    )
    assert fast > slow


def test_rank_candidates_tie_keeps_static_order() -> None:
    assert rank_candidates([]) == []
    assert rank_candidates(
        [
            ScoredCandidate(
                name="prov-a", latency_ms=None, cost_per_1k_tokens=0.0, quality=0.5
            ),
            ScoredCandidate(
                name="prov-b", latency_ms=None, cost_per_1k_tokens=0.0, quality=0.5
            ),
        ]
    ) == ["prov-a", "prov-b"]


def test_plan_orders_by_score_not_static_order() -> None:
    engine = make_engine(
        priority=["prov-a", "prov-b"],
        costs={"prov-a": 0.09, "prov-b": 0.01},
        qualities={"prov-a": 0.2, "prov-b": 0.9},
    )
    assert engine.plan(model=GENERIC_MODEL) == ["prov-b", "prov-a"]


def test_plan_alias_pins_provider_first() -> None:
    engine = make_engine(
        priority=["prov-a", "prov-b"],
        costs={"prov-a": 0.09, "prov-b": 0.01},
        qualities={"prov-a": 0.2, "prov-b": 0.9},
        model_aliases={"pinned": "prov-a"},
    )
    assert engine.plan(model="pinned") == ["prov-a", "prov-b"]
