"""Live OpenRouter primary -> fallback demo through the gateway's REAL router.

Uses RouterEngine + ResilientExecutor + ProxyClient (the same stack wired in
``src/main.py:create_app``), NOT ``OpenRouterProvider.chat()`` directly.

Phase 1: one request through the primary provider (openrouter-primary).
Phase 2: simulate a primary failure by rebuilding the PRIMARY provider with an
         invalid model slug, then send the same request again and watch the
         executor automatically fall back to openrouter-fallback.

How to simulate a primary failure manually (same idea, without code):
    1. In configs/gateway.yaml, temporarily set openrouter-primary's model to
       something bogus, e.g. ``model: "invalid/no-such-model-xyz:free"``.
    2. Restart the gateway / re-run this script and send a chat request.
    3. OpenRouter answers 400/404 -> PROVIDER_REJECTED (permanent) ->
       executor falls back to openrouter-fallback within the same request.
    4. Restore the real model slug afterwards.

Run:
    set OPENROUTER_API_KEY=...   (Windows cmd)
    python live_openrouter_test.py
Requires network access to https://openrouter.ai.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from src.config import load_settings
from src.errors import GatewayError
from src.main import build_providers
from src.models import ChatMessage, NormalizedChatRequest
from src.providers.openrouter import OpenRouterProvider, OpenRouterSettings
from src.proxy.client import ProxyClient
from src.resilience.executor import ResilientExecutor
from src.resilience.retry import RetryPolicy
from src.router.circuit_breaker import ResilientCircuitBreaker
from src.router.engine import RouterEngine
from src.router.health import HealthRegistry
from src.router.scorer import ScoringWeights
from src.telemetry.latency import LatencyTracker

PRIMARY_NAME = "openrouter-primary"
FALLBACK_NAME = "openrouter-fallback"
PRIORITY = [PRIMARY_NAME, FALLBACK_NAME]

# Bogus slug -> OpenRouter returns 4xx -> PROVIDER_REJECTED (permanent, but
# still fallback-eligible per src/resilience/failures.py:allows_fallback).
INVALID_MODEL_SLUG = "invalid/no-such-model-xyz:free"


def build_stack(primary_model_override: str | None = None):
    """Build the real gateway stack restricted to the two OpenRouter providers.

    Returns (executor, providers, router, settings). A fresh HealthRegistry +
    breaker is created each call so Phase 2 retries the (broken) primary
    first instead of skipping it due to Phase 1 marks.
    """
    settings = load_settings(Path("configs/gateway.yaml"))
    providers = build_providers(settings)
    # Restrict the candidate chain to just the OpenRouter pair so the demo is
    # deterministic: [openrouter-primary, openrouter-fallback].
    providers = {k: v for k, v in providers.items() if k in PRIORITY}
    missing = [n for n in PRIORITY if n not in providers]
    if missing:
        raise RuntimeError(
            f"missing providers {missing} — check configs/gateway.yaml "
            "(both must exist and be enabled: true)"
        )

    if primary_model_override is not None:
        current: OpenRouterProvider = providers[PRIMARY_NAME]  # type: ignore[assignment]
        base_settings: OpenRouterSettings = current.settings
        providers[PRIMARY_NAME] = OpenRouterProvider(
            PRIMARY_NAME,
            OpenRouterSettings(
                api_key=base_settings.api_key,
                model=primary_model_override,
                base_url=base_settings.base_url,
                timeout_s=base_settings.timeout_s,
                site_url=base_settings.site_url,
                app_name=base_settings.app_name,
            ),
            api_key=os.getenv("OPENROUTER_API_KEY") or None,
        )

    timeouts = {
        e.name: e.timeout_s for e in settings.providers if e.name in PRIORITY
    }
    health = HealthRegistry(PRIORITY)
    breaker = ResilientCircuitBreaker(
        failure_threshold=settings.resilience.circuit_breaker.failure_threshold,
        recovery_timeout_s=settings.resilience.circuit_breaker.recovery_timeout_s,
        half_open_max_inflight=settings.resilience.circuit_breaker.half_open_max_inflight,
    )
    enabled_ordered = settings.enabled_providers_in_priority_order()
    latency_tracker = LatencyTracker()
    router = RouterEngine(
        priority=PRIORITY,
        default_provider=PRIMARY_NAME,
        model_aliases={},
        health=health,
        breaker=breaker,
        costs={p.name: p.cost_per_1k_tokens for p in enabled_ordered},
        qualities={p.name: p.quality_weight for p in enabled_ordered},
        latency_tracker=latency_tracker,
        scoring_weights=ScoringWeights(
            latency=settings.scoring.weight_latency,
            cost=settings.scoring.weight_cost,
            quality=settings.scoring.weight_quality,
        ),
    )
    proxy = ProxyClient(providers, timeouts, default_timeout_s=30.0)
    executor = ResilientExecutor(
        router=router,
        proxy=proxy,
        breaker=breaker,
        health=health,
        retry=RetryPolicy(
            max_attempts=settings.resilience.retry.max_attempts,
            backoff_base_ms=settings.resilience.retry.backoff_base_ms,
            backoff_max_ms=settings.resilience.retry.backoff_max_ms,
            max_elapsed_ms=settings.resilience.retry.max_elapsed_ms,
        ),
    )
    return executor, providers, router, settings


def show_outcome(tag: str, executed) -> None:
    o = executed.outcome
    r = executed.response
    print(f"[{tag}] final_provider : {r.provider}")
    print(f"[{tag}] primary        : {o.primary}")
    print(f"[{tag}] fallback_used  : {o.fallback_used}")
    print(f"[{tag}] retry_count    : {o.retry_count}")
    print(f"[{tag}] content        : {r.content!r}")
    print(f"[{tag}] usage          : {r.usage}")
    print(f"[{tag}] latency_ms     : {round(r.latency_ms, 1)}")
    print(f"[{tag}] attempts:")
    for a in o.attempts:
        extra = ""
        if a.failure_category is not None:
            extra = f" category={a.failure_category.value}"
        if a.skip_reason is not None:
            extra = f" skip={a.skip_reason}"
        print(
            f"[{tag}]   - provider={a.provider} "
            f"attempt={a.attempt_index} outcome={a.outcome}{extra} "
            f"latency_ms={round(a.latency_ms, 1)}"
        )


async def main() -> None:
    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit(
            "OPENROUTER_API_KEY is not set — export it first, e.g.\n"
            '  set OPENROUTER_API_KEY=sk-or-...   (Windows cmd)\n'
            "  $env:OPENROUTER_API_KEY='sk-or-...'  (PowerShell)"
        )
    if os.getenv("GATEWAY_OPENROUTER_MODEL"):
        print(
            "WARNING: GATEWAY_OPENROUTER_MODEL is set — config.py overlays it "
            "onto EVERY openrouter provider, which would hide the distinct "
            "primary/fallback models for this demo. Consider unsetting it."
        )

    settings = load_settings(Path("configs/gateway.yaml"))
    print("== providers from configs/gateway.yaml ==")
    for e in settings.providers:
        if e.type == "openrouter":
            print(
                f"   name={e.name} enabled={e.enabled} "
                f"priority={e.priority} model={e.openrouter.model}"
            )
    print(f"   DEFAULT_MODEL check: see src/providers/openrouter.py")
    print()

    # ---- Phase 1: healthy primary, request goes through the real router ----
    executor, providers, router, _ = build_stack()
    print("== health_check (direct adapter probe, informational only) ==")
    for name in PRIORITY:
        status = await providers[name].health_check()  # type: ignore[union-attr]
        print(f"   {name}: healthy={status.healthy} error={status.error}")
    print()
    print(f"== router plan for model 'test-model' -> {router.plan(model='test-model')}")
    print("== Phase 1: request through PRIMARY via gateway router (executor) ==")
    req1 = NormalizedChatRequest(
        request_id="live-primary-1",
        model="test-model",  # any string: priority order gives [primary, fallback]
        provider=PRIMARY_NAME,
        messages=[ChatMessage(role="user", content="Reply with exactly one word: pong")],
        temperature=0.0,
        max_tokens=20,
    )
    try:
        executed = await executor.execute(req1)
    except GatewayError as exc:
        print(f"Phase 1 FAILED: code={exc.code} provider={exc.provider}: {exc.message}")
        raise SystemExit(1)
    show_outcome("phase1", executed)
    if executed.response.provider == PRIMARY_NAME and not executed.outcome.fallback_used:
        print("   -> served by PRIMARY as intended (no fallback needed).")
    else:
        print(
            f"   -> note: primary did not serve this request "
            f"(final={executed.response.provider}, "
            f"fallback_used={executed.outcome.fallback_used}). "
            "This is still the real router doing its job — e.g. a 429 "
            "rate-limit on the free-tier primary correctly triggered "
            "automatic fallback. Re-run later for a clean primary hit."
        )
    print()

    # ---- Phase 2: break the primary, same router must fall back ----
    print("== Phase 2: simulate PRIMARY failure ==")
    print(f"   rebuilding '{PRIMARY_NAME}' with invalid model: {INVALID_MODEL_SLUG}")
    print("   (this mimics temporarily pointing primary at a dead slug in YAML)")
    executor2, providers2, router2, _ = build_stack(
        primary_model_override=INVALID_MODEL_SLUG
    )
    print(f"   router plan -> {router2.plan(model='test-model')}")
    req2 = NormalizedChatRequest(
        request_id="live-fallback-1",
        model="test-model",
        provider=PRIMARY_NAME,
        messages=[ChatMessage(role="user", content="Reply with exactly one word: pong")],
        temperature=0.0,
        max_tokens=20,
    )
    try:
        executed2 = await executor2.execute(req2)
    except GatewayError as exc:
        print(f"Phase 2 FAILED: code={exc.code} provider={exc.provider}: {exc.message}")
        raise SystemExit(1)
    show_outcome("phase2", executed2)
    assert executed2.outcome.fallback_used is True, "expected fallback_used=True"
    assert executed2.response.provider == FALLBACK_NAME, (
        f"expected fallback {FALLBACK_NAME}, got {executed2.response.provider}"
    )
    print()
    print("DONE: Phase 2 deterministically proved automatic primary -> fallback.")


if __name__ == "__main__":
    asyncio.run(main())
