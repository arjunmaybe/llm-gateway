"""YAML + environment configuration. Env vars take precedence over YAML.

Secrets must never live in YAML — only in the environment (see .env.example).
M1 env overrides are intentionally small; the surface grows with M2+.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.providers.mock import MockProviderSettings


class ServerSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)


class LoggingSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class RoutingSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_provider: str = "mock-a"
    model_aliases: dict[str, str] = Field(default_factory=dict)


class ProviderEntry(BaseModel):
    """One provider block from gateway.yaml."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    type: Literal["mock"] = "mock"
    enabled: bool = True
    priority: int = 10
    timeout_s: float = Field(default=5.0, gt=0.0)
    mock: MockProviderSettings = Field(default_factory=MockProviderSettings)


class RetrySettings(BaseModel):
    """M2 retry policy. First attempt counts toward ``max_attempts``."""

    model_config = ConfigDict(frozen=True)

    max_attempts: int = Field(default=2, ge=1)
    backoff_base_ms: float = Field(default=50.0, gt=0.0)
    backoff_max_ms: float = Field(default=1000.0, gt=0.0)
    max_elapsed_ms: float = Field(default=8000.0, ge=0.0)


class CircuitBreakerSettings(BaseModel):
    """M2 per-provider breaker thresholds."""

    model_config = ConfigDict(frozen=True)

    failure_threshold: int = Field(default=5, ge=1)
    recovery_timeout_s: float = Field(default=30.0, gt=0.0)
    half_open_max_inflight: int = Field(default=1, ge=1)


class ResilienceSettings(BaseModel):
    """M2 fault-tolerance tuning. Env vars (``GATEWAY_RETRY_*`` /
    ``GATEWAY_CIRCUIT_*``) take precedence over these YAML values."""

    model_config = ConfigDict(frozen=True)

    retry: RetrySettings = Field(default_factory=RetrySettings)
    circuit_breaker: CircuitBreakerSettings = Field(default_factory=CircuitBreakerSettings)


class AppSettings(BaseModel):
    """Validated, fully-resolved gateway configuration."""

    model_config = ConfigDict(frozen=True)

    server: ServerSettings = Field(default_factory=ServerSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    routing: RoutingSettings = Field(default_factory=RoutingSettings)
    providers: list[ProviderEntry] = Field(default_factory=list)
    resilience: ResilienceSettings = Field(default_factory=ResilienceSettings)

    @field_validator("providers")
    @classmethod
    def _names_unique(cls, providers: list[ProviderEntry]) -> list[ProviderEntry]:
        names = [p.name for p in providers]
        if len(set(names)) != len(names):
            raise ValueError("provider names must be unique")
        return providers

    def enabled_providers_in_priority_order(self) -> list[ProviderEntry]:
        return sorted(
            [p for p in self.providers if p.enabled], key=lambda p: (p.priority, p.name)
        )


def default_config_path() -> Path:
    """Resolve YAML path. ``GATEWAY_CONFIG_PATH`` env wins over the default."""
    raw = os.getenv("GATEWAY_CONFIG_PATH", "configs/gateway.yaml")
    return Path(raw)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data: Any = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must be a mapping at top level")
    return dict(data)


def _env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    return float(raw)


def _env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    return int(raw)


def load_settings(path: Path | None = None) -> AppSettings:
    """Load YAML then overlay env vars (env wins). No secrets are read from YAML."""
    cfg_path = path if path is not None else default_config_path()
    data = _read_yaml(cfg_path)
    settings = AppSettings.model_validate(data)

    host = os.getenv("GATEWAY_HOST")
    port_raw = os.getenv("GATEWAY_PORT")
    level = os.getenv("GATEWAY_LOG_LEVEL")
    default_provider = os.getenv("GATEWAY_DEFAULT_PROVIDER")
    mock_latency = _env_float("GATEWAY_MOCK_LATENCY_MS")
    mock_fail = _env_float("GATEWAY_MOCK_FAIL_RATE")

    server = settings.server
    if host is not None and host != "":
        server = ServerSettings(host=host, port=server.port)
    if port_raw is not None and port_raw != "":
        server = ServerSettings(host=server.host, port=int(port_raw))

    logging_cfg = settings.logging
    if level is not None and level != "":
        logging_cfg = LoggingSettings(level=level.upper())  # type: ignore[arg-type]

    routing = settings.routing
    if default_provider is not None and default_provider != "":
        routing = RoutingSettings(
            default_provider=default_provider, model_aliases=routing.model_aliases
        )

    providers = list(settings.providers)
    if mock_latency is not None or mock_fail is not None:
        rebuilt: list[ProviderEntry] = []
        for entry in providers:
            mock = entry.mock
            rebuilt.append(
                ProviderEntry(
                    name=entry.name,
                    type=entry.type,
                    enabled=entry.enabled,
                    priority=entry.priority,
                    timeout_s=entry.timeout_s,
                    mock=MockProviderSettings(
                        latency_ms=mock_latency if mock_latency is not None else mock.latency_ms,
                        fail_rate=mock_fail if mock_fail is not None else mock.fail_rate,
                        failure_mode=mock.failure_mode,
                        response_tokens=mock.response_tokens,
                    ),
                )
            )
        providers = rebuilt

    return AppSettings(
        server=server,
        logging=logging_cfg,
        routing=routing,
        providers=providers,
        resilience=_resolve_resilience(settings.resilience),
    )


def _resolve_resilience(base: ResilienceSettings) -> ResilienceSettings:
    """Overlay M2 resilience env vars on top of YAML values (env wins)."""
    retry_attempts = _env_int("GATEWAY_RETRY_MAX_ATTEMPTS")
    retry_base = _env_float("GATEWAY_RETRY_BACKOFF_BASE_MS")
    retry_max = _env_float("GATEWAY_RETRY_BACKOFF_MAX_MS")
    retry_elapsed = _env_float("GATEWAY_RETRY_MAX_ELAPSED_MS")
    cb_threshold = _env_int("GATEWAY_CIRCUIT_FAILURE_THRESHOLD")
    cb_recovery = _env_float("GATEWAY_CIRCUIT_RECOVERY_TIMEOUT_S")
    cb_inflight = _env_int("GATEWAY_CIRCUIT_HALF_OPEN_INFLIGHT")

    retry = base.retry
    if (
        retry_attempts is not None
        or retry_base is not None
        or retry_max is not None
        or retry_elapsed is not None
    ):
        retry = RetrySettings(
            max_attempts=retry_attempts if retry_attempts is not None else retry.max_attempts,
            backoff_base_ms=retry_base if retry_base is not None else retry.backoff_base_ms,
            backoff_max_ms=retry_max if retry_max is not None else retry.backoff_max_ms,
            max_elapsed_ms=retry_elapsed if retry_elapsed is not None else retry.max_elapsed_ms,
        )

    breaker = base.circuit_breaker
    if cb_threshold is not None or cb_recovery is not None or cb_inflight is not None:
        breaker = CircuitBreakerSettings(
            failure_threshold=(
                cb_threshold if cb_threshold is not None else breaker.failure_threshold
            ),
            recovery_timeout_s=(
                cb_recovery if cb_recovery is not None else breaker.recovery_timeout_s
            ),
            half_open_max_inflight=(
                cb_inflight if cb_inflight is not None else breaker.half_open_max_inflight
            ),
        )

    if retry is base.retry and breaker is base.circuit_breaker:
        return base
    return ResilienceSettings(retry=retry, circuit_breaker=breaker)
