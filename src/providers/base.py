"""Stable provider abstraction. Routing must only depend on this module.

Future real providers (OpenAI, Anthropic) implement :class:`ProviderAdapter`
without changing routing, proxy, or API code (M2+).
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.models import NormalizedChatRequest, Usage

ProviderFailureMode = Literal["timeout", "unavailable", "rate_limited", "connection", "permanent"]


class ProviderResponse(BaseModel):
    """Normalized provider result."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    content: str
    usage: Usage
    latency_ms: float


class HealthStatus(BaseModel):
    """Point-in-time provider health snapshot."""

    model_config = ConfigDict(frozen=True)

    provider: str
    healthy: bool
    latency_ms: float | None = None
    error: str | None = None


class ProviderError(Exception):
    """Raw provider-side failure. Mapped to GatewayError at the proxy boundary."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        provider: str,
        retryable: bool,
        status_code: int = 502,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.provider = provider
        self.retryable = retryable
        self.status_code = status_code


class ProviderAdapter(abc.ABC):
    """Interface every provider (mock now, real APIs later) must implement."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable provider name matching config (e.g. ``mock-a``)."""
        raise NotImplementedError

    @abc.abstractmethod
    async def chat(self, request: NormalizedChatRequest) -> ProviderResponse:
        """Non-streaming completion."""
        raise NotImplementedError

    @abc.abstractmethod
    def chat_stream(self, request: NormalizedChatRequest) -> AsyncIterator[str]:
        """Streaming completion yielding normalized content chunks.

        Transport-agnostic: providers yield plain text chunks, never SSE.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def health_check(self) -> HealthStatus:
        """Lightweight health probe. Must be cheap and offline-safe for mocks."""
        raise NotImplementedError
