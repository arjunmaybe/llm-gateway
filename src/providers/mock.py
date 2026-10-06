"""Deterministic mock provider for offline development and tests.

Supports configurable latency, failure injection, and response size so future
resilience (M2), streaming (M3), and cache (M4) work can be exercised without
paid APIs. Deterministic: outcomes derive from ``request_id`` hash, so tests
with a fixed request ID are reproducible.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from collections.abc import AsyncIterator
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.models import NormalizedChatRequest, Usage
from src.providers.base import HealthStatus, ProviderAdapter, ProviderError, ProviderResponse

FailureMode = Literal["timeout", "unavailable", "rate_limited", "connection", "permanent"]


class MockProviderSettings(BaseModel):
    """Knobs for latency / failure / size experiments.

    ``failure_script`` is an M2 per-invocation fault script: entry ``None``
    succeeds, otherwise the entry names the failure mode to raise. Consumed
    in invocation order; once exhausted, the legacy ``fail_rate`` path
    applies. Empty script preserves exact M1 behavior.

    M3 streaming knobs: ``stream_chunk_words`` controls deterministic word
    chunking; ``stream_failure_after`` (chunk index, 0-based) injects a
    mid-stream ``ProviderError`` after that many chunks have been yielded
    (``0`` fails before the first chunk); ``stream_failure_mode`` overrides
    ``failure_mode`` for the mid-stream fault.
    """

    model_config = ConfigDict(frozen=True)

    latency_ms: float = Field(default=0.0, ge=0.0)
    fail_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    failure_mode: FailureMode = "unavailable"
    response_tokens: int = Field(default=24, ge=1, le=4096)
    failure_script: tuple[FailureMode | None, ...] = ()
    stream_chunk_words: int = Field(default=2, ge=1, le=64)
    stream_failure_after: int | None = Field(default=None, ge=0)
    stream_failure_mode: FailureMode | None = None


def _deterministic_rng(*parts: str) -> random.Random:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _count_tokens(text: str) -> int:
    return len(text.split())


class MockProvider(ProviderAdapter):
    """In-process fake LLM. No network, no credentials, fully offline."""

    def __init__(self, name: str, settings: MockProviderSettings) -> None:
        self._name = name
        self._settings = settings
        self._calls = 0
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def settings(self) -> MockProviderSettings:
        return self._settings

    async def chat(self, request: NormalizedChatRequest) -> ProviderResponse:
        start_ns = time.perf_counter_ns()
        async with self._lock:
            call_index = self._calls
            self._calls += 1
        if self._settings.latency_ms > 0:
            await asyncio.sleep(self._settings.latency_ms / 1000.0)

        script = self._settings.failure_script
        if call_index < len(script):
            scripted = script[call_index]
            if scripted is not None:
                raise self._failure(mode=scripted, request_id=request.request_id)
        else:
            rng = _deterministic_rng(self._name, request.request_id)
            if rng.random() < self._settings.fail_rate:
                raise self._failure(mode=self._settings.failure_mode, request_id=request.request_id)

        content = self._generate(request)
        prompt_tokens = sum(_count_tokens(m.content) for m in request.messages)
        completion_tokens = _count_tokens(content)
        latency_ms = (time.perf_counter_ns() - start_ns) / 1e6
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return ProviderResponse(
            provider=self._name,
            model=request.model,
            content=content,
            usage=usage,
            latency_ms=latency_ms,
        )

    async def chat_stream(self, request: NormalizedChatRequest) -> AsyncIterator[str]:
        """Yield deterministic word chunks. Failures mirror ``chat`` pre-first-byte.

        Pre-first-byte: ``failure_script`` / ``fail_rate`` raise before any
        yield (enables pre-first-byte fallback). Mid-stream:
        ``stream_failure_after=N`` raises after N chunks yielded.
        """
        async with self._lock:
            call_index = self._calls
            self._calls += 1
        if self._settings.latency_ms > 0:
            await asyncio.sleep(self._settings.latency_ms / 1000.0)

        script = self._settings.failure_script
        if call_index < len(script):
            scripted = script[call_index]
            if scripted is not None:
                raise self._failure(mode=scripted, request_id=request.request_id)
        else:
            rng = _deterministic_rng(self._name, request.request_id)
            if rng.random() < self._settings.fail_rate:
                raise self._failure(mode=self._settings.failure_mode, request_id=request.request_id)

        content = self._generate(request)
        words = content.split()
        size = self._settings.stream_chunk_words
        chunks = [" ".join(words[i : i + size]) for i in range(0, len(words), size)]
        failure_after = self._settings.stream_failure_after
        failure_mode = self._settings.stream_failure_mode or self._settings.failure_mode
        yielded = 0
        for chunk in chunks:
            if failure_after is not None and yielded == failure_after:
                raise self._failure(mode=failure_mode, request_id=request.request_id)
            yield chunk
            yielded += 1
            # Cooperative yield point so cancellation/disconnect is observed promptly.
            await asyncio.sleep(0)
        if failure_after is not None and yielded == failure_after:
            raise self._failure(mode=failure_mode, request_id=request.request_id)

    async def health_check(self) -> HealthStatus:
        healthy = self._settings.fail_rate < 1.0
        return HealthStatus(
            provider=self._name,
            healthy=healthy,
            latency_ms=0.0,
            error=None if healthy else "mock fail_rate=1.0",
        )

    def _failure(self, *, mode: FailureMode, request_id: str) -> ProviderError:
        _ = request_id
        if mode == "timeout":
            return ProviderError(
                code="PROVIDER_TIMEOUT",
                message=f"mock provider '{self._name}' timed out",
                provider=self._name,
                retryable=True,
                status_code=504,
            )
        if mode == "rate_limited":
            return ProviderError(
                code="PROVIDER_RATE_LIMITED",
                message=f"mock provider '{self._name}' rate limited",
                provider=self._name,
                retryable=True,
                status_code=429,
            )
        if mode == "connection":
            return ProviderError(
                code="PROVIDER_CONNECTION_FAILED",
                message=f"mock provider '{self._name}' connection failed",
                provider=self._name,
                retryable=True,
                status_code=502,
            )
        if mode == "permanent":
            return ProviderError(
                code="PROVIDER_REJECTED",
                message=f"mock provider '{self._name}' rejected the request",
                provider=self._name,
                retryable=False,
                status_code=502,
            )
        return ProviderError(
            code="PROVIDER_UNAVAILABLE",
            message=f"mock provider '{self._name}' unavailable (injected failure)",
            provider=self._name,
            retryable=True,
            status_code=502,
        )

    def _generate(self, request: NormalizedChatRequest) -> str:
        last_user = next(
            (m.content for m in reversed(request.messages) if m.role == "user"),
            request.messages[-1].content if request.messages else "",
        )
        base = f"[mock:{self._name}] echo: {last_user}"
        words = base.split()
        target = self._settings.response_tokens
        if len(words) >= target:
            return " ".join(words[:target])
        filler = ["lorem", "ipsum", "dolor", "sit", "amet"]
        idx = 0
        while len(words) < target:
            words.append(filler[idx % len(filler)])
            idx += 1
        return " ".join(words)
