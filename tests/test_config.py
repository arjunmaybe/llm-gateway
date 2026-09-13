"""Config tests: YAML load + env precedence."""

from __future__ import annotations

from pathlib import Path

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
    import pytest

    from src.config import ProviderEntry

    with pytest.raises(ValueError):
        AppSettings(
            providers=[
                ProviderEntry(name="dup", type="mock"),
                ProviderEntry(name="dup", type="mock"),
            ]
        )
