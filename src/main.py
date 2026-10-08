"""FastAPI ingress: request IDs, timing, routing, proxy, normalized errors."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import httpx
import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src import __version__
from src.cache.index import VectorIndex
from src.cache.manager import (
    CacheManager,
    NoOpCacheManager,
    SemanticCacheManager,
    build_prompt_key,
    build_prompt_text,
)
from src.config import AppSettings, load_settings
from src.errors import GatewayError
from src.models import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionResponse,
    ChatRequest,
    GatewayErrorEnvelope,
    GatewayErrorPayload,
    HealthResponse,
    NormalizedChatRequest,
    ReadyResponse,
    Usage,
)
from src.providers.base import ProviderAdapter
from src.providers.mock import MockProvider
from src.providers.openrouter import OpenRouterProvider
from src.proxy.client import ProxyClient
from src.proxy.sse_parser import format_chunk, format_done, format_error
from src.resilience.executor import ResilientExecutor
from src.resilience.failures import (
    affects_health,
    allows_fallback,
    classify,
    counts_toward_breaker,
)
from src.resilience.retry import RetryPolicy
from src.router.circuit_breaker import CircuitState, ResilientCircuitBreaker
from src.router.engine import RouteDecision, RouterEngine
from src.router.health import HealthRegistry
from src.router.prober import HealthProber
from src.router.scorer import ScoringWeights
from src.telemetry.latency import LatencyTracker, StreamingTiming, Timer
from src.telemetry.metrics import NoOpMetricsRecorder
from src.telemetry.tracer import NoOpTracer

REQUEST_ID_HEADER = "x-request-id"


def configure_logging(level: str) -> None:
    """Lightweight JSON logging to stdout. No external sinks in M1."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(),
    )


def _new_request_id() -> str:
    return uuid.uuid4().hex


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Propagates or mints ``x-request-id`` and exposes ``request.state`` timing."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or _new_request_id()
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        timer = Timer()
        timer.start()
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


def _request_id_of(request: Request) -> str:
    raw: Any = getattr(request.state, "request_id", None)
    if isinstance(raw, str) and raw:
        return raw
    return _new_request_id()


def _percentile(sorted_vals: list[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..1). Returns None for empty input."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = int(q * (len(sorted_vals) - 1) + 0.5)
    rank = max(0, min(rank, len(sorted_vals) - 1))
    return sorted_vals[rank]


def _itl_summary(samples: list[float]) -> dict[str, float | None]:
    """Summarize inter-chunk latencies (chunk-level, not true token boundaries)."""
    if not samples:
        return {"avg": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(samples)
    avg = sum(samples) / len(samples)
    return {
        "avg": avg,
        "p50": _percentile(ordered, 0.5),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def build_providers(settings: AppSettings) -> dict[str, ProviderAdapter]:
    providers: dict[str, ProviderAdapter] = {}
    for entry in settings.providers:
        if entry.type == "mock":
            providers[entry.name] = MockProvider(entry.name, entry.mock)
        elif entry.type == "openrouter":
            providers[entry.name] = OpenRouterProvider(
                entry.name,
                entry.openrouter,
                api_key=os.getenv("OPENROUTER_API_KEY") or None,
            )
    return providers


def build_cache(settings: AppSettings) -> CacheManager:
    """Wire the real cache. ``noop`` stays offline-safe; memory/semantic loads embedder."""
    backend = settings.cache.backend
    if backend == "noop":
        return NoOpCacheManager()
    try:
        from src.cache.embedder import TextEmbedder

        embedder = TextEmbedder(model_name=settings.cache.embedding_model)
        embed_fn = embedder.embed
    except Exception as exc:
        structlog.get_logger("gateway").warning(
            "gateway.cache_fallback_noop", backend=backend, error=str(exc)
        )
        return NoOpCacheManager()
    return SemanticCacheManager(
        embed_fn=embed_fn,
        threshold=settings.cache.semantic_threshold,
        index=VectorIndex(),
    )


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """App factory. Pass explicit settings in tests; defaults load YAML+env."""
    resolved = settings if settings is not None else load_settings()
    configure_logging(resolved.logging.level)
    log = structlog.get_logger("gateway")

    providers = build_providers(resolved)
    enabled_ordered = resolved.enabled_providers_in_priority_order()
    priority = [p.name for p in enabled_ordered if p.name in providers]
    health_registry = HealthRegistry([p.name for p in enabled_ordered])
    breaker = ResilientCircuitBreaker(
        failure_threshold=resolved.resilience.circuit_breaker.failure_threshold,
        recovery_timeout_s=resolved.resilience.circuit_breaker.recovery_timeout_s,
        half_open_max_inflight=resolved.resilience.circuit_breaker.half_open_max_inflight,
    )
    latency_tracker = LatencyTracker()
    router = RouterEngine(
        priority=priority,
        default_provider=resolved.routing.default_provider,
        model_aliases=dict(resolved.routing.model_aliases),
        health=health_registry,
        breaker=breaker,
        costs={p.name: p.cost_per_1k_tokens for p in enabled_ordered},
        qualities={p.name: p.quality_weight for p in enabled_ordered},
        latency_tracker=latency_tracker,
        scoring_weights=ScoringWeights(
            latency=resolved.scoring.weight_latency,
            cost=resolved.scoring.weight_cost,
            quality=resolved.scoring.weight_quality,
        ),
    )
    proxy = ProxyClient(
        providers,
        {p.name: p.timeout_s for p in enabled_ordered},
        default_timeout_s=5.0,
    )
    cache = build_cache(resolved)
    metrics = NoOpMetricsRecorder()
    tracer = NoOpTracer()
    prober: HealthProber | None = None
    if resolved.health.probe_enabled:
        prober = HealthProber(
            providers=providers,
            enabled=[p.name for p in enabled_ordered],
            health=health_registry,
            interval_s=resolved.health.probe_interval_s,
            timeout_s=resolved.health.probe_timeout_s,
        )
    executor = ResilientExecutor(
        router=router,
        proxy=proxy,
        breaker=breaker,
        health=health_registry,
        retry=RetryPolicy(
            max_attempts=resolved.resilience.retry.max_attempts,
            backoff_base_ms=resolved.resilience.retry.backoff_base_ms,
            backoff_max_ms=resolved.resilience.retry.backoff_max_ms,
            max_elapsed_ms=resolved.resilience.retry.max_elapsed_ms,
        ),
        metrics=metrics,
        tracer=tracer,
        latency_tracker=latency_tracker,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client = httpx.AsyncClient(timeout=10.0)
        app.state.http_client = client
        log.info(
            "gateway.startup",
            providers=sorted(providers.keys()),
            default_provider=resolved.routing.default_provider,
        )
        if prober is not None:
            prober.start()
            log.info(
                "gateway.health_prober_started",
                interval_s=resolved.health.probe_interval_s,
                timeout_s=resolved.health.probe_timeout_s,
            )
        try:
            yield
        finally:
            if prober is not None:
                await prober.stop()
            await client.aclose()

    app = FastAPI(title="LLM Gateway", version=__version__, lifespan=lifespan)
    app.add_middleware(RequestIdMiddleware)
    app.state.settings = resolved
    app.state.router = router
    app.state.proxy = proxy
    app.state.health = health_registry
    app.state.latency_tracker = latency_tracker
    app.state.cache = cache
    app.state.metrics = metrics
    app.state.tracer = tracer
    app.state.executor = executor
    app.state.health_prober = prober

    def _envelope(
        *, code: str, message: str, provider: str | None, retryable: bool, request_id: str
    ) -> dict[str, Any]:
        payload = GatewayErrorPayload(
            code=code,
            message=message,
            provider=provider,
            retryable=retryable,
            request_id=request_id,
        )
        return GatewayErrorEnvelope(error=payload).model_dump()

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(request: Request, exc: GatewayError) -> JSONResponse:
        request_id = exc.request_id or _request_id_of(request)
        structlog.get_logger("gateway").warning(
            "gateway.error",
            request_id=request_id,
            code=exc.code,
            provider=exc.provider,
            status_code=exc.status_code,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(
                code=exc.code,
                message=exc.message,
                provider=exc.provider,
                retryable=exc.retryable,
                request_id=request_id,
            ),
            headers={REQUEST_ID_HEADER: request_id},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = _request_id_of(request)
        return JSONResponse(
            status_code=422,
            content=_envelope(
                code="INVALID_REQUEST",
                message="request validation failed",
                provider=None,
                retryable=False,
                request_id=request_id,
            ),
            headers={REQUEST_ID_HEADER: request_id},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = _request_id_of(request)
        structlog.get_logger("gateway").error(
            "gateway.unhandled", request_id=request_id, error=str(exc)
        )
        return JSONResponse(
            status_code=500,
            content=_envelope(
                code="INTERNAL",
                message="internal gateway error",
                provider=None,
                retryable=False,
                request_id=request_id,
            ),
            headers={REQUEST_ID_HEADER: request_id},
        )

    @app.get("/health", response_model=HealthResponse)
    async def health_endpoint() -> HealthResponse:
        return HealthResponse(version=__version__, providers=sorted(providers.keys()))

    @app.get("/ready", response_model=ReadyResponse)
    async def ready_endpoint() -> ReadyResponse:
        snapshot = health_registry.readiness()
        return ReadyResponse(
            ready=all(snapshot.values()) if snapshot else False, providers=snapshot
        )

    async def _aclose_iterator(it: AsyncIterator[str]) -> None:
        """Best-effort close of an async iterator (async generators expose aclose)."""
        aclose = getattr(it, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
            except (StopAsyncIteration, StopIteration, RuntimeError, GeneratorExit):
                pass

    def _record_pre_first_byte_failure(name: str, exc: GatewayError) -> bool:
        """Apply M2 policy for a pre-first-byte streaming failure.

        Returns True if fallback to the next candidate is allowed.
        """
        category = classify(exc)
        metrics.increment("gateway_provider_attempts_total", provider=name, code=exc.code)
        if affects_health(category):
            health_registry.mark_unhealthy(name, exc.code)
        elif counts_toward_breaker(category):
            was_open = breaker.state_of(name)
            breaker.record_failure(name)
            if breaker.state_of(name) is CircuitState.OPEN and (
                was_open is not CircuitState.OPEN
            ):
                metrics.increment(
                    "gateway_breaker_opens_total", provider=name, code=exc.code
                )
        return allows_fallback(category)

    def _record_post_first_byte_failure(name: str, exc: GatewayError) -> None:
        """Breaker/health accounting for mid-stream failures (no reroute)."""
        category = classify(exc)
        metrics.increment("gateway_provider_attempts_total", provider=name, code=exc.code)
        metrics.increment("gateway_stream_errors_total", provider=name, code=exc.code)
        if affects_health(category):
            health_registry.mark_unhealthy(name, exc.code)
        elif counts_toward_breaker(category):
            was_open = breaker.state_of(name)
            breaker.record_failure(name)
            if breaker.state_of(name) is CircuitState.OPEN and (
                was_open is not CircuitState.OPEN
            ):
                metrics.increment(
                    "gateway_breaker_opens_total", provider=name, code=exc.code
                )

    async def _handle_streaming(
        body: ChatRequest, request: Request, request_id: str, timer: Timer
    ) -> StreamingResponse:
        """M3 SSE streaming: pre-first-byte fallback, pinned post-first-byte stream."""
        req_log = structlog.get_logger("gateway")
        start_ns = time.perf_counter_ns()
        candidates = router.plan(model=body.model)
        if not candidates:
            raise GatewayError(
                code="NO_HEALTHY_PROVIDER",
                message="no healthy providers available",
                status_code=503,
                provider=None,
                retryable=True,
                request_id=request_id,
            )
        primary = candidates[0]
        base = NormalizedChatRequest(
            request_id=request_id,
            model=body.model,
            provider=primary,
            messages=list(body.messages),
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            user=body.user,
        )
        pinned_provider: str | None = None
        pinned_stream: AsyncIterator[str] | None = None
        pinned_iterator: AsyncIterator[str] | None = None
        first_text: str | None = None
        first_ns: int | None = None
        empty_stream_provider: str | None = None
        last_error: GatewayError | None = None
        fallback_counted = False

        for name in candidates:
            if not health_registry.is_healthy(name):
                continue
            if not breaker.can_execute(name):
                continue
            if name != primary and not fallback_counted:
                metrics.increment("gateway_fallbacks_total", provider=name)
                fallback_counted = True
            scoped = base if base.provider == name else base.model_copy(update={"provider": name})
            route = RouteDecision(
                provider_name=name,
                reason="primary" if name == primary else "fallback",
                tried=list(candidates),
            )
            candidate_stream = proxy.stream_forward(route, scoped)
            candidate_iter = candidate_stream.__aiter__()
            try:
                with tracer.span("provider.stream", provider=name):
                    first = await candidate_iter.__anext__()
            except StopAsyncIteration:
                # Empty provider stream: pin provider, complete with DONE-only body.
                pinned_provider = name
                pinned_stream = candidate_stream
                pinned_iterator = candidate_iter
                empty_stream_provider = name
                first_ns = time.perf_counter_ns()
                breaker.record_success(name)
                health_registry.mark_healthy(name)
                metrics.increment(
                    "gateway_provider_attempts_total", provider=name, code="ok"
                )
                break
            except GatewayError as exc:
                if not exc.request_id:
                    exc.request_id = request_id
                metrics.increment(
                    "gateway_errors_total", provider=exc.provider or "", code=exc.code
                )
                can_fallback = _record_pre_first_byte_failure(name, exc)
                last_error = exc
                # Ensure abandoned generator is closed to release resources.
                await _aclose_iterator(candidate_stream)
                if not can_fallback:
                    raise
                continue
            else:
                pinned_provider = name
                pinned_stream = candidate_stream
                pinned_iterator = candidate_iter
                first_text = first
                first_ns = time.perf_counter_ns()
                break

        if pinned_provider is None or pinned_stream is None or pinned_iterator is None:
            if last_error is not None:
                raise last_error
            raise GatewayError(
                code="NO_HEALTHY_PROVIDER",
                message="no healthy providers available",
                status_code=503,
                provider=None,
                retryable=True,
                request_id=request_id,
            )

        final_provider: str = pinned_provider
        stream_obj: AsyncIterator[str] = pinned_stream
        stream_iter: AsyncIterator[str] = pinned_iterator
        initial_text: str | None = first_text
        initial_ns: int = first_ns if first_ns is not None else time.perf_counter_ns()
        ttft_ms: float = (initial_ns - start_ns) / 1e6
        chunk_id = f"chatcmpl-{request_id[:8]}"
        created = int(time.time())
        model_name = body.model
        max_tokens = body.max_tokens
        _ = timer
        _ = request

        async def _event_generator() -> AsyncIterator[str]:
            itl_samples: list[float] = []
            chunk_count = 0
            words_yielded = 0
            finish: Literal["stop", "length"] = "stop"
            truncated = False
            prev_ns = initial_ns
            first_to_yield: str | None = initial_text
            # Empty-stream case: no content chunks at all.
            is_empty = empty_stream_provider is not None

            metrics.increment("gateway_stream_requests_total", provider=final_provider)
            metrics.observe_latency(
                "gateway_stream_ttft_ms", ttft_ms, provider=final_provider
            )
            # Pre-first-byte success accounting for the pinned provider when
            # the stream had content (empty case already recorded above).
            if not is_empty:
                metrics.increment(
                    "gateway_provider_attempts_total", provider=final_provider, code="ok"
                )

            async def _aclose_pinned() -> None:
                await _aclose_iterator(stream_obj)

            def _split_truncate(text: str) -> tuple[str, bool]:
                """Apply max_tokens word budget. Returns (text_to_yield, exhausted)."""
                nonlocal words_yielded, finish, truncated
                if max_tokens is None:
                    words_yielded += len(text.split())
                    return text, False
                words = text.split()
                remaining = max_tokens - words_yielded
                if remaining <= 0:
                    finish = "length"
                    truncated = True
                    return "", True
                if len(words) > remaining:
                    finish = "length"
                    truncated = True
                    words_yielded += remaining
                    return " ".join(words[:remaining]), True
                words_yielded += len(words)
                return text, False

            try:
                # Yield the pre-fetched first chunk (if any) with role.
                if first_to_yield is not None:
                    text_out, exhausted = _split_truncate(first_to_yield)
                    if text_out:
                        yield format_chunk(
                            chunk_id=chunk_id,
                            created=created,
                            model=model_name,
                            provider=final_provider,
                            request_id=request_id,
                            content=text_out,
                            role="assistant",
                        )
                        chunk_count += 1
                        metrics.increment(
                            "gateway_stream_chunks_total", provider=final_provider
                        )
                    if exhausted:
                        await _aclose_pinned()
                        yield format_chunk(
                            chunk_id=chunk_id,
                            created=created,
                            model=model_name,
                            provider=final_provider,
                            request_id=request_id,
                            content=None,
                            role=None,
                            finish_reason=finish,
                        )
                        yield format_done()
                        breaker.record_success(final_provider)
                        health_registry.mark_healthy(final_provider)
                        metrics.increment(
                            "gateway_stream_completed_total", provider=final_provider
                        )
                        summary = _itl_summary(itl_samples)
                        timing = StreamingTiming(
                            ttft_ms=ttft_ms, itl_ms=summary["avg"]
                        )
                        req_log.info(
                            "gateway.stream",
                            request_id=request_id,
                            model=model_name,
                            provider=final_provider,
                            status="ok_truncated",
                            ttft_ms=round(ttft_ms, 3),
                            itl_ms=(
                                round(summary["avg"], 3)
                                if summary["avg"] is not None
                                else None
                            ),
                            chunks=chunk_count,
                            timing_ms={"ttft": timing.ttft_ms, "itl": timing.itl_ms},
                        )
                        return
                # Stream the remainder without buffering the full response.
                while True:
                    try:
                        nxt = await stream_iter.__anext__()
                    except StopAsyncIteration:
                        break
                    now_ns = time.perf_counter_ns()
                    itl_samples.append((now_ns - prev_ns) / 1e6)
                    prev_ns = now_ns
                    text_out, exhausted = _split_truncate(nxt)
                    if text_out:
                        yield format_chunk(
                            chunk_id=chunk_id,
                            created=created,
                            model=model_name,
                            provider=final_provider,
                            request_id=request_id,
                            content=text_out,
                            role=None,
                        )
                        chunk_count += 1
                        metrics.increment(
                            "gateway_stream_chunks_total", provider=final_provider
                        )
                    if exhausted:
                        truncated = True
                        break
                # Clean exhaustion: terminal finish chunk + DONE.
                yield format_chunk(
                    chunk_id=chunk_id,
                    created=created,
                    model=model_name,
                    provider=final_provider,
                    request_id=request_id,
                    content=None,
                    role=None,
                    finish_reason=finish,
                )
                yield format_done()
                breaker.record_success(final_provider)
                health_registry.mark_healthy(final_provider)
                metrics.increment(
                    "gateway_stream_completed_total", provider=final_provider
                )
                for sample in itl_samples:
                    metrics.observe_latency(
                        "gateway_stream_itl_ms", sample, provider=final_provider
                    )
                summary = _itl_summary(itl_samples)
                timing = StreamingTiming(ttft_ms=ttft_ms, itl_ms=summary["avg"])
                req_log.info(
                    "gateway.stream",
                    request_id=request_id,
                    model=model_name,
                    provider=final_provider,
                    status="ok",
                    ttft_ms=round(ttft_ms, 3),
                    itl_ms=(
                        round(summary["avg"], 3) if summary["avg"] is not None else None
                    ),
                    itl_p50=summary["p50"],
                    itl_p95=summary["p95"],
                    chunks=chunk_count,
                    truncated=truncated,
                    timing_ms={"ttft": timing.ttft_ms, "itl": timing.itl_ms},
                )
            except (GeneratorExit, asyncio.CancelledError):
                # Client disconnect: release upstream, no success/error accounting,
                # no error event (headers already sent, client is gone).
                metrics.increment(
                    "gateway_stream_interrupted_total", provider=final_provider
                )
                req_log.info(
                    "gateway.stream",
                    request_id=request_id,
                    provider=final_provider,
                    status="interrupted",
                    chunks=chunk_count,
                )
                await _aclose_pinned()
                raise
            except GatewayError as exc:
                if not exc.request_id:
                    exc.request_id = request_id
                _record_post_first_byte_failure(final_provider, exc)
                metrics.increment(
                    "gateway_errors_total", provider=exc.provider or "", code=exc.code
                )
                req_log.warning(
                    "gateway.stream_error",
                    request_id=request_id,
                    provider=final_provider,
                    code=exc.code,
                    chunks=chunk_count,
                )
                # Mid-stream failure: error event then close, never [DONE],
                # never fallback (bytes already sent).
                yield format_error(
                    code=exc.code,
                    message=exc.message,
                    provider=exc.provider,
                    retryable=exc.retryable,
                    request_id=request_id,
                )
                await _aclose_pinned()
                return
            except Exception as exc:  # pragma: no cover - defensive stream guard
                req_log.error(
                    "gateway.stream_unhandled", request_id=request_id, error=str(exc)
                )
                metrics.increment(
                    "gateway_stream_errors_total",
                    provider=final_provider,
                    code="INTERNAL",
                )
                yield format_error(
                    code="INTERNAL",
                    message="internal gateway error",
                    provider=final_provider,
                    retryable=False,
                    request_id=request_id,
                )
                await _aclose_pinned()
                return

        return StreamingResponse(
            _event_generator(),
            media_type="text/event-stream",
            headers={
                REQUEST_ID_HEADER: request_id,
                "x-provider": final_provider,
                "cache-control": "no-cache",
                "connection": "keep-alive",
            },
        )

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        body: ChatRequest, request: Request
    ) -> JSONResponse | StreamingResponse:
        request_id = _request_id_of(request)
        timer = Timer()
        timer.start()
        logger = structlog.get_logger("gateway")

        if body.stream:
            return await _handle_streaming(body, request, request_id, timer)

        route = router.select_provider(model=body.model, request_id=request_id)
        normalized = NormalizedChatRequest(
            request_id=request_id,
            model=body.model,
            provider=route.provider_name,
            messages=list(body.messages),
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            user=body.user,
        )
        prompt_text = build_prompt_text(body.messages)
        cache_key = build_prompt_key(
            model=body.model, temperature=body.temperature, prompt_text=prompt_text
        )
        active_cache: CacheManager = getattr(
            request.app.state, "cache", cache
        )
        with tracer.span("cache.lookup", provider=route.provider_name):
            cached_content = await active_cache.get(cache_key)
        if isinstance(cached_content, str):
            content = cached_content
            finish_reason: Literal["stop", "length"] = "stop"
            if body.max_tokens is not None:
                words = content.split()
                if len(words) > body.max_tokens:
                    content = " ".join(words[: body.max_tokens])
                    finish_reason = "length"
            latency_ms = timer.stop()
            metrics.observe_latency(
                "gateway_request_ms", latency_ms, provider=route.provider_name
            )
            metrics.increment("gateway_requests_total", provider=route.provider_name)
            metrics.increment("gateway_cache_hits_total", provider=route.provider_name)
            prompt_tokens = sum(len(m.content.split()) for m in body.messages)
            completion_tokens = len(content.split())
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
            body_out = ChatCompletionResponse(
                id=f"chatcmpl-{request_id[:8]}",
                created=int(time.time()),
                model=body.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatCompletionMessage(content=content),
                        finish_reason=finish_reason,
                    )
                ],
                usage=usage,
                provider=route.provider_name,
                request_id=request_id,
                latency_ms=latency_ms,
                cached=True,
                ttft_ms=None,
                itl_ms=None,
            )
            logger.info(
                "gateway.request",
                request_id=request_id,
                model=body.model,
                provider=route.provider_name,
                latency_ms=round(latency_ms, 3),
                status="ok",
                cached=True,
            )
            return JSONResponse(
                status_code=200,
                content=body_out.model_dump(),
                headers={
                    REQUEST_ID_HEADER: request_id,
                    "x-provider": route.provider_name,
                    "x-gateway-latency-ms": f"{latency_ms:.3f}",
                },
            )
        metrics.increment("gateway_cache_misses_total", provider=route.provider_name)
        try:
            executed = await executor.execute(normalized)
        except GatewayError as exc:
            if not exc.request_id:
                exc.request_id = request_id
            metrics.increment(
                "gateway_errors_total", provider=exc.provider or "", code=exc.code
            )
            raise
        result = executed.response
        request.state.execution_outcome = executed.outcome

        content = result.content
        finish_reason = "stop"
        if body.max_tokens is not None:
            words = content.split()
            if len(words) > body.max_tokens:
                content = " ".join(words[: body.max_tokens])
                finish_reason = "length"

        latency_ms = timer.stop()
        metrics.observe_latency("gateway_request_ms", latency_ms, provider=result.provider)
        metrics.increment("gateway_requests_total", provider=result.provider)
        await active_cache.put(cache_key, content)

        body_out = ChatCompletionResponse(
            id=f"chatcmpl-{request_id[:8]}",
            created=int(time.time()),
            model=body.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(content=content),
                    finish_reason=finish_reason,
                )
            ],
            usage=result.usage,
            provider=result.provider,
            request_id=request_id,
            latency_ms=latency_ms,
            cached=False,
            ttft_ms=None,
            itl_ms=None,
        )
        logger.info(
            "gateway.request",
            request_id=request_id,
            model=body.model,
            provider=result.provider,
            latency_ms=round(latency_ms, 3),
            status="ok",
        )
        return JSONResponse(
            status_code=200,
            content=body_out.model_dump(),
            headers={
                REQUEST_ID_HEADER: request_id,
                "x-provider": result.provider,
                "x-gateway-latency-ms": f"{latency_ms:.3f}",
            },
        )

    return app


try:
    app = create_app()
except Exception:  # pragma: no cover - startup fallback keeps process importable
    app = create_app(AppSettings())
