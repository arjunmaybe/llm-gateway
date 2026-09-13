"""Non-streaming proxy fan-out. Owns timeouts and error normalization.

Provider-specific failures (:class:`ProviderError`) are translated here into
gateway-normalized :class:`GatewayError` so clients never see raw payloads.
``provider`` / ``retryable`` / ``code`` are always preserved.
"""

from __future__ import annotations

import asyncio

from src.errors import GatewayError
from src.models import NormalizedChatRequest
from src.providers.base import ProviderAdapter, ProviderError, ProviderResponse
from src.router.engine import RouteDecision


class ProxyClient:
    """Forwards a normalized request to the routed provider with a timeout."""

    def __init__(
        self,
        providers: dict[str, ProviderAdapter],
        timeouts_s: dict[str, float],
        default_timeout_s: float = 5.0,
    ) -> None:
        self._providers = dict(providers)
        self._timeouts = dict(timeouts_s)
        self._default_timeout = default_timeout_s

    def provider_names(self) -> list[str]:
        return sorted(self._providers.keys())

    async def forward(
        self, route: RouteDecision, request: NormalizedChatRequest
    ) -> ProviderResponse:
        provider = self._providers.get(route.provider_name)
        if provider is None:
            raise GatewayError(
                code="UNKNOWN_PROVIDER",
                message=f"unknown provider '{route.provider_name}'",
                status_code=500,
                provider=route.provider_name,
                retryable=False,
                request_id=request.request_id,
            )
        timeout_s = self._timeouts.get(provider.name, self._default_timeout)
        try:
            return await asyncio.wait_for(provider.chat(request), timeout=timeout_s)
        except TimeoutError as exc:
            raise GatewayError(
                code="UPSTREAM_TIMEOUT",
                message=f"provider '{provider.name}' timed out after {timeout_s}s",
                status_code=504,
                provider=provider.name,
                retryable=True,
                request_id=request.request_id,
            ) from exc
        except ProviderError as exc:
            raise GatewayError(
                code=exc.code,
                message=exc.message,
                status_code=exc.status_code,
                provider=exc.provider,
                retryable=exc.retryable,
                request_id=request.request_id,
            ) from exc
