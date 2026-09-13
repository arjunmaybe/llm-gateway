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


class AppSettings(BaseModel):
    """Validated, fully-resolved gateway configuration."""

    model_config = ConfigDict(frozen=True)

    server: ServerSettings = Field(default_factory=ServerSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    routing: RoutingSettings = Field(default_factory=RoutingSettings)
    providers: list[ProviderEntry] = Field(default_factory=list)

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
        server=server, logging=logging_cfg, routing=routing, providers=providers
    )
