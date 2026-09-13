"""Placeholder for a future real Anthropic adapter (M5+). Not implemented in M1."""

from __future__ import annotations

from src.models import NormalizedChatRequest
from src.providers.base import HealthStatus, ProviderAdapter, ProviderError
from src.providers.base import ProviderResponse as _ProviderResponse


class AnthropicProvider(ProviderAdapter):
    """Reserved for M5+. Instantiating now always fails fast."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def chat(self, request: NormalizedChatRequest) -> _ProviderResponse:
        raise ProviderError(
            code="PROVIDER_NOT_CONFIGURED",
            message="Anthropic provider is planned but not implemented in M1",
            provider=self._name,
            retryable=False,
            status_code=501,
        )

    async def health_check(self) -> HealthStatus:
        return HealthStatus(provider=self._name, healthy=False, error="not implemented in M1")
