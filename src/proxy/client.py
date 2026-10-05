"""Proxy fan-out. Owns timeouts and error normalization.

Provider-specific failures (:class:`ProviderError`) are translated here into
gateway-normalized :class:`GatewayError` so clients never see raw payloads.
``provider`` / ``retryable`` / ``code`` are always preserved.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

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
        except asyncio.TimeoutError as exc:
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

    async def stream_forward(
        self, route: RouteDecision, request: NormalizedChatRequest
    ) -> AsyncIterator[str]:
        """Yield content chunks without buffering the full response.

        The per-provider timeout protects stream establishment / first-byte
        acquisition only. Wrapping the whole long-lived stream in a single
        ``wait_for`` would kill healthy streams that last longer than
        ``timeout_s``; after the first chunk the lifecycle is managed
        separately by the caller.
        """
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
        stream = provider.chat_stream(request)
        iterator = stream.__aiter__()
        try:
            first = await asyncio.wait_for(iterator.__anext__(), timeout=timeout_s)
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError as exc:
            await _aclose_stream(stream)
            raise GatewayError(
                code="UPSTREAM_TIMEOUT",
                message=f"provider '{provider.name}' timed out after {timeout_s}s",
                status_code=504,
                provider=provider.name,
                retryable=True,
                request_id=request.request_id,
            ) from exc
        except ProviderError as exc:
            await _aclose_stream(stream)
            raise GatewayError(
                code=exc.code,
                message=exc.message,
                status_code=exc.status_code,
                provider=exc.provider,
                retryable=exc.retryable,
                request_id=request.request_id,
            ) from exc
        try:
            yield first
            while True:
                try:
                    chunk = await iterator.__anext__()
                except StopAsyncIteration:
                    break
                yield chunk
        except ProviderError as exc:
            raise GatewayError(
                code=exc.code,
                message=exc.message,
                status_code=exc.status_code,
                provider=exc.provider,
                retryable=exc.retryable,
                request_id=request.request_id,
            ) from exc
        finally:
            await _aclose_stream(stream)


async def _aclose_stream(stream: AsyncIterator[str]) -> None:
    """Best-effort close of an async iterator (async generators expose ``aclose``)."""
    aclose = getattr(stream, "aclose", None)
    if callable(aclose):
        try:
            await aclose()
        except (StopAsyncIteration, StopIteration):
            pass
        except RuntimeError:
            # Already closing / event loop shutting down; nothing to do.
            pass
