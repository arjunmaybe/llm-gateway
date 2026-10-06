"""Real OpenRouter adapter (OpenAI-compatible chat completions + SSE).

Secrets never live in YAML — the bearer token resolves at call time from
(1) explicit constructor arg, (2) ``OpenRouterSettings.api_key``,
(3) ``OPENROUTER_API_KEY`` env. See ``.env.example``.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from src.models import NormalizedChatRequest, Usage
from src.providers.base import HealthStatus, ProviderAdapter, ProviderError, ProviderResponse

DEFAULT_MODEL = "google/gemma-4-31b-it:free"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
CHAT_PATH = "/chat/completions"
MODELS_PATH = "/models"


class OpenRouterSettings(BaseModel):
    """Knobs for the OpenRouter adapter. Mirrors ``MockProviderSettings`` style."""

    model_config = ConfigDict(frozen=True)

    api_key: str = ""
    model: str = Field(default=DEFAULT_MODEL, min_length=1)
    base_url: str = Field(default=DEFAULT_BASE_URL, min_length=1)
    timeout_s: float = Field(default=30.0, gt=0.0)
    site_url: str = ""
    app_name: str = "llm-gateway"


def _resolve_api_key(settings: OpenRouterSettings, override: str | None) -> str:
    if override:
        return override
    if settings.api_key:
        return settings.api_key
    return os.getenv("OPENROUTER_API_KEY", "")


def _extract_error_detail(body_text: str) -> str:
    """Best-effort ``{"error": {"message": ...}}`` extraction, else raw snippet."""
    import json

    try:
        data: Any = json.loads(body_text)
    except Exception:
        return body_text[:300]
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            msg = err.get("message")
            if isinstance(msg, str) and msg:
                return msg[:300]
        msg2 = data.get("message")
        if isinstance(msg2, str) and msg2:
            return msg2[:300]
    return body_text[:300]


class OpenRouterProvider(ProviderAdapter):
    """OpenRouter chat provider over ``httpx``."""

    def __init__(
        self,
        name: str,
        settings: OpenRouterSettings | None = None,
        client: httpx.AsyncClient | None = None,
        api_key: str | None = None,
    ) -> None:
        self._name = name
        self._settings = settings if settings is not None else OpenRouterSettings()
        self._client = client
        self._api_key_override = api_key

    @property
    def name(self) -> str:
        return self._name

    @property
    def settings(self) -> OpenRouterSettings:
        return self._settings

    def _headers(self, api_key: str) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self._settings.site_url:
            headers["HTTP-Referer"] = self._settings.site_url
        if self._settings.app_name:
            headers["X-Title"] = self._settings.app_name
        return headers

    def _payload(self, request: NormalizedChatRequest, *, stream: bool) -> dict[str, Any]:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        payload: dict[str, Any] = {
            "model": self._settings.model or request.model,
            "messages": messages,
            "temperature": request.temperature,
            "stream": stream,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        return payload

    def _map_http_error(self, status: int, body_text: str) -> ProviderError:
        detail = _extract_error_detail(body_text)
        base = f"openrouter provider '{self._name}' failed (http {status}): {detail}"
        if status == 429:
            return ProviderError(
                code="PROVIDER_RATE_LIMITED",
                message=base,
                provider=self._name,
                retryable=True,
                status_code=429,
            )
        if status in (408, 504):
            return ProviderError(
                code="PROVIDER_TIMEOUT",
                message=base,
                provider=self._name,
                retryable=True,
                status_code=504,
            )
        if 500 <= status <= 599:
            return ProviderError(
                code="PROVIDER_UNAVAILABLE",
                message=base,
                provider=self._name,
                retryable=True,
                status_code=status,
            )
        # All other 4xx (400 auth/bad-request, 401, 403, 404, 422, ...): not retryable.
        return ProviderError(
            code="PROVIDER_REJECTED",
            message=base,
            provider=self._name,
            retryable=False,
            status_code=status,
        )

    async def _post(
        self, path: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> httpx.Response:
        url = self._settings.base_url.rstrip("/") + path
        timeout = self._settings.timeout_s
        if self._client is not None:
            return await self._client.post(url, json=payload, headers=headers, timeout=timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.post(url, json=payload, headers=headers, timeout=timeout)

    async def chat(self, request: NormalizedChatRequest) -> ProviderResponse:
        start_ns = time.perf_counter_ns()
        api_key = _resolve_api_key(self._settings, self._api_key_override)
        if not api_key:
            raise ProviderError(
                code="PROVIDER_NOT_CONFIGURED",
                message=f"openrouter provider '{self._name}' has no api key "
                "(set OPENROUTER_API_KEY)",
                provider=self._name,
                retryable=False,
                status_code=501,
            )
        try:
            resp = await self._post(
                CHAT_PATH, self._payload(request, stream=False), self._headers(api_key)
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(
                code="PROVIDER_TIMEOUT",
                message=f"openrouter provider '{self._name}' timed out: {exc}",
                provider=self._name,
                retryable=True,
                status_code=504,
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderError(
                code="PROVIDER_CONNECTION_FAILED",
                message=f"openrouter provider '{self._name}' connection failed: {exc}",
                provider=self._name,
                retryable=True,
                status_code=502,
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderError(
                code="PROVIDER_CONNECTION_FAILED",
                message=f"openrouter provider '{self._name}' transport error: {exc}",
                provider=self._name,
                retryable=True,
                status_code=502,
            ) from exc
        if resp.status_code != 200:
            raise self._map_http_error(resp.status_code, resp.text)
        try:
            data: Any = resp.json()
        except Exception as exc:
            raise ProviderError(
                code="PROVIDER_UNAVAILABLE",
                message=f"openrouter provider '{self._name}' returned invalid JSON",
                provider=self._name,
                retryable=True,
                status_code=502,
            ) from exc
        content = self._extract_content(data)
        usage = self._extract_usage(data, request, content)
        latency_ms = (time.perf_counter_ns() - start_ns) / 1e6
        return ProviderResponse(
            provider=self._name,
            model=request.model,
            content=content,
            usage=usage,
            latency_ms=latency_ms,
        )

    def _extract_content(self, data: Any) -> str:
        try:
            choices = data["choices"]
            text = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                code="PROVIDER_UNAVAILABLE",
                message="openrouter returned an unexpected response shape",
                provider=self._name,
                retryable=True,
                status_code=502,
            ) from exc
        if not isinstance(text, str):
            raise ProviderError(
                code="PROVIDER_UNAVAILABLE",
                message="openrouter returned a non-string completion",
                provider=self._name,
                retryable=True,
                status_code=502,
            )
        return text

    def _extract_usage(
        self, data: Any, request: NormalizedChatRequest, content: str
    ) -> Usage:
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        total_tokens: int | None = None
        raw = data.get("usage") if isinstance(data, dict) else None
        if isinstance(raw, dict):
            pt = raw.get("prompt_tokens")
            ct = raw.get("completion_tokens")
            tt = raw.get("total_tokens")
            if isinstance(pt, int) and pt >= 0:
                prompt_tokens = pt
            if isinstance(ct, int) and ct >= 0:
                completion_tokens = ct
            if isinstance(tt, int) and tt >= 0:
                total_tokens = tt
        if prompt_tokens is None:
            prompt_tokens = sum(len(m.content.split()) for m in request.messages)
        if completion_tokens is None:
            completion_tokens = len(content.split())
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    async def chat_stream(self, request: NormalizedChatRequest) -> AsyncIterator[str]:
        """Yield plain-text chunks from OpenRouter SSE (parsed via sse_parser)."""
        from src.proxy.sse_parser import SseIncrementalDecoder

        api_key = _resolve_api_key(self._settings, self._api_key_override)
        if not api_key:
            raise ProviderError(
                code="PROVIDER_NOT_CONFIGURED",
                message=f"openrouter provider '{self._name}' has no api key "
                "(set OPENROUTER_API_KEY)",
                provider=self._name,
                retryable=False,
                status_code=501,
            )
        payload = self._payload(request, stream=True)
        headers = dict(self._headers(api_key))
        headers["Accept"] = "text/event-stream"
        url = self._settings.base_url.rstrip("/") + CHAT_PATH
        timeout = self._settings.timeout_s
        decoder = SseIncrementalDecoder()
        try:
            if self._client is not None:
                async with self._client.stream(
                    "POST", url, json=payload, headers=headers, timeout=timeout
                ) as resp:
                    async for chunk in self._stream_chunks(resp, decoder):
                        yield chunk
            else:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "POST", url, json=payload, headers=headers, timeout=timeout
                    ) as resp:
                        async for chunk in self._stream_chunks(resp, decoder):
                            yield chunk
        except ProviderError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderError(
                code="PROVIDER_TIMEOUT",
                message=f"openrouter provider '{self._name}' timed out: {exc}",
                provider=self._name,
                retryable=True,
                status_code=504,
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderError(
                code="PROVIDER_CONNECTION_FAILED",
                message=f"openrouter provider '{self._name}' connection failed: {exc}",
                provider=self._name,
                retryable=True,
                status_code=502,
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderError(
                code="PROVIDER_CONNECTION_FAILED",
                message=f"openrouter provider '{self._name}' transport error: {exc}",
                provider=self._name,
                retryable=True,
                status_code=502,
            ) from exc

    async def _stream_chunks(
        self, resp: httpx.Response, decoder: Any
    ) -> AsyncIterator[str]:
        if resp.status_code != 200:
            try:
                body = await resp.aread()
                text = body.decode("utf-8", errors="replace")
            except Exception:
                text = ""
            raise self._map_http_error(resp.status_code, text)
        try:
            async for text_piece in resp.aiter_text(chunk_size=4096):
                for event in decoder.feed(text_piece):
                    chunk = self._extract_stream_delta(event)
                    if chunk is not None:
                        yield chunk
            for event in decoder.flush():
                chunk = self._extract_stream_delta(event)
                if chunk is not None:
                    yield chunk
        except ProviderError:
            raise
        except ValueError as exc:
            raise ProviderError(
                code="PROVIDER_UNAVAILABLE",
                message=f"openrouter provider '{self._name}' sent malformed SSE: {exc}",
                provider=self._name,
                retryable=True,
                status_code=502,
            ) from exc

    def _extract_stream_delta(self, event: Any) -> str | None:
        if event == "DONE" or event is None:
            return None
        if not isinstance(event, dict):
            return None
        if "error" in event:
            detail = str(event.get("error"))[:300]
            raise ProviderError(
                code="PROVIDER_UNAVAILABLE",
                message=f"openrouter provider '{self._name}' stream error: {detail}",
                provider=self._name,
                retryable=True,
                status_code=502,
            )
        try:
            choices = event.get("choices")
            if not choices:
                return None
            delta = choices[0].get("delta", {})
            text = delta.get("content")
        except (AttributeError, IndexError, TypeError):
            return None
        if not isinstance(text, str) or not text:
            return None
        return text

    async def health_check(self) -> HealthStatus:
        """Cheap probe: ``GET /models`` lists models without spending tokens."""
        start_ns = time.perf_counter_ns()
        api_key = _resolve_api_key(self._settings, self._api_key_override)
        if not api_key:
            return HealthStatus(
                provider=self._name, healthy=False, error="missing OPENROUTER_API_KEY"
            )
        url = self._settings.base_url.rstrip("/") + MODELS_PATH
        timeout = min(self._settings.timeout_s, 10.0)
        try:
            if self._client is not None:
                resp = await self._client.get(
                    url, headers=self._headers(api_key), timeout=timeout
                )
            else:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.get(
                        url, headers=self._headers(api_key), timeout=timeout
                    )
        except httpx.TimeoutException as exc:
            return HealthStatus(provider=self._name, healthy=False, error=f"timeout: {exc}")
        except httpx.TransportError as exc:
            return HealthStatus(
                provider=self._name, healthy=False, error=f"connection failed: {exc}"
            )
        latency_ms = (time.perf_counter_ns() - start_ns) / 1e6
        if resp.status_code == 200:
            return HealthStatus(provider=self._name, healthy=True, latency_ms=latency_ms)
        if resp.status_code in (401, 403):
            return HealthStatus(
                provider=self._name,
                healthy=False,
                latency_ms=latency_ms,
                error=f"unauthorized (http {resp.status_code})",
            )
        return HealthStatus(
            provider=self._name,
            healthy=False,
            latency_ms=latency_ms,
            error=f"unhealthy (http {resp.status_code})",
        )
