"""Shared offline fixtures. No network, no credentials."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from src.config import AppSettings, ProviderEntry, RoutingSettings
from src.main import create_app
from src.providers.mock import MockProviderSettings


def make_settings(
    *,
    fail_rate_a: float = 0.0,
    latency_ms_a: float = 0.0,
    response_tokens_a: int = 16,
    failure_mode_a: str = "unavailable",
    fail_rate_b: float = 0.0,
    failure_mode_b: str = "unavailable",
) -> AppSettings:
    return AppSettings(
        routing=RoutingSettings(
            default_provider="mock-a",
            model_aliases={"mock": "mock-a", "mock-a": "mock-a", "mock-b": "mock-b"},
        ),
        providers=[
            ProviderEntry(
                name="mock-a",
                type="mock",
                enabled=True,
                priority=10,
                timeout_s=5.0,
                mock=MockProviderSettings(
                    latency_ms=latency_ms_a,
                    fail_rate=fail_rate_a,  # type: ignore[arg-type]
                    failure_mode=failure_mode_a,  # type: ignore[arg-type]
                    response_tokens=response_tokens_a,
                ),
            ),
            ProviderEntry(
                name="mock-b",
                type="mock",
                enabled=True,
                priority=20,
                timeout_s=5.0,
                mock=MockProviderSettings(
                    fail_rate=fail_rate_b,  # type: ignore[arg-type]
                    failure_mode=failure_mode_b,  # type: ignore[arg-type]
                ),
            ),
        ],
    )


@pytest.fixture()
def settings() -> AppSettings:
    return make_settings()


@pytest.fixture()
def app(settings: AppSettings) -> FastAPI:
    return create_app(settings)


@pytest.fixture()
def failing_app() -> FastAPI:
    return create_app(make_settings(fail_rate_a=1.0))


@pytest.fixture()
def all_failing_app() -> FastAPI:
    return create_app(make_settings(fail_rate_a=1.0, fail_rate_b=1.0))


def make_client(app: FastAPI) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


def chat_body(model: str = "mock-a") -> dict[str, object]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hello gateway"}],
        "temperature": 0.0,
        "stream": False,
    }
