"""FastAPI ingress: request IDs, timing, routing, proxy, normalized errors."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import httpx
import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src import __version__
from src.cache.manager import NoOpCacheManager
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
)
from src.providers.base import ProviderAdapter
from src.providers.mock import MockProvider
from src.proxy.client import ProxyClient
from src.resilience.executor import ResilientExecutor
from src.resilience.retry import RetryPolicy
from src.router.circuit_breaker import ResilientCircuitBreaker
from src.router.engine import RouterEngine
from src.router.health import HealthRegistry
from src.telemetry.latency import Timer
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


def build_providers(settings: AppSettings) -> dict[str, ProviderAdapter]:
    providers: dict[str, ProviderAdapter] = {}
    for entry in settings.providers:
        if entry.type == "mock":
            providers[entry.name] = MockProvider(entry.name, entry.mock)
    return providers


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
    router = RouterEngine(
        priority=priority,
        default_provider=resolved.routing.default_provider,
        model_aliases=dict(resolved.routing.model_aliases),
        health=health_registry,
        breaker=breaker,
    )
    proxy = ProxyClient(
        providers,
        {p.name: p.timeout_s for p in enabled_ordered},
        default_timeout_s=5.0,
    )
    cache = NoOpCacheManager()
    metrics = NoOpMetricsRecorder()
    tracer = NoOpTracer()
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
        try:
            yield
        finally:
            await client.aclose()

    app = FastAPI(title="LLM Gateway", version=__version__, lifespan=lifespan)
    app.add_middleware(RequestIdMiddleware)
    app.state.settings = resolved
    app.state.router = router
    app.state.proxy = proxy
    app.state.health = health_registry
    app.state.cache = cache
    app.state.metrics = metrics
    app.state.tracer = tracer
    app.state.executor = executor

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

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(body: ChatRequest, request: Request) -> JSONResponse:
        request_id = _request_id_of(request)
        timer = Timer()
        timer.start()
        logger = structlog.get_logger("gateway")

        if body.stream:
            raise GatewayError(
                code="STREAMING_NOT_SUPPORTED",
                message="stream=true is planned for M3; retry with stream=false",
                status_code=422,
                provider=None,
                retryable=False,
                request_id=request_id,
            )

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
        with tracer.span("cache.lookup", provider=route.provider_name):
            _ = await cache.get(request_id)
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
        finish_reason: Literal["stop", "length"] = "stop"
        if body.max_tokens is not None:
            words = content.split()
            if len(words) > body.max_tokens:
                content = " ".join(words[: body.max_tokens])
                finish_reason = "length"

        latency_ms = timer.stop()
        metrics.observe_latency("gateway_request_ms", latency_ms, provider=result.provider)
        metrics.increment("gateway_requests_total", provider=result.provider)
        await cache.put(request_id, content)

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
