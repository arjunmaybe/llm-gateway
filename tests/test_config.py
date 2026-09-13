"""Config tests: YAML load + env precedence."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import AppSettings, load_settings


def test_load_bundled_yaml() -> None:
    settings = load_settings(Path("configs/gateway.yaml"))
    assert settings.routing.default_provider == "mock-a"
    assert {p.name for p in settings.providers} >= {"mock-a", "mock-b"}


def test_env_overrides_yaml(tmp_path: Path, monkeypatch: object) -> None:
    import os

    yaml_text = (
        "server:\n  host: 127.0.0.1\n  port: 8000\n"
        "routing:\n  default_provider: mock-a\n"
        "providers:\n"
        "  - name: mock-a\n    type: mock\n    enabled: true\n    priority: 10\n"
    )
    cfg = tmp_path / "gateway.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    os.environ["GATEWAY_DEFAULT_PROVIDER"] = "mock-b"
    try:
        settings = load_settings(cfg)
    finally:
        del os.environ["GATEWAY_DEFAULT_PROVIDER"]
    assert settings.routing.default_provider == "mock-b"


def test_duplicate_provider_names_rejected() -> None:
    from src.config import ProviderEntry

    with pytest.raises(ValueError):
        AppSettings(
            providers=[
                ProviderEntry(name="dup", type="mock"),
                ProviderEntry(name="dup", type="mock"),
            ]
        )


def test_resilience_defaults_from_bundled_yaml() -> None:
    settings = load_settings(Path("configs/gateway.yaml"))
    assert settings.resilience.retry.max_attempts == 2
    assert settings.resilience.retry.backoff_base_ms == 50.0
    assert settings.resilience.retry.backoff_max_ms == 1000.0
    assert settings.resilience.retry.max_elapsed_ms == 8000.0
    assert settings.resilience.circuit_breaker.failure_threshold == 5
    assert settings.resilience.circuit_breaker.recovery_timeout_s == 30.0
    assert settings.resilience.circuit_breaker.half_open_max_inflight == 1


def test_resilience_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_RETRY_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("GATEWAY_RETRY_BACKOFF_BASE_MS", "25")
    monkeypatch.setenv("GATEWAY_CIRCUIT_FAILURE_THRESHOLD", "2")
    monkeypatch.setenv("GATEWAY_CIRCUIT_RECOVERY_TIMEOUT_S", "5")
    settings = load_settings(Path("configs/gateway.yaml"))
    assert settings.resilience.retry.max_attempts == 4
    assert settings.resilience.retry.backoff_base_ms == 25.0
    assert settings.resilience.retry.backoff_max_ms == 1000.0  # untouched YAML value
    assert settings.resilience.circuit_breaker.failure_threshold == 2
    assert settings.resilience.circuit_breaker.recovery_timeout_s == 5.0
