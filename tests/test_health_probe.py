"""Active health probing: probe_once recovery, isolation, lifecycle, lockout."""

from __future__ import annotations

import asyncio
from pathlib import Path

from structlog.testing import capture_logs

from src.config import AppSettings, HealthSettings, ProviderEntry, RoutingSettings, load_settings
from src.main import create_app
from src.models import ChatMessage, NormalizedChatRequest
from src.providers.base import HealthStatus
from src.providers.mock import MockProvider, MockProviderSettings
from src.router.health import HealthRegistry
from src.router.prober import HealthProber


def healthy_provider(name: str = "mock-a") -> MockProvider:
    return MockProvider(name, MockProviderSettings())


def failing_provider(name: str = "mock-a") -> MockProvider:
    return MockProvider(name, MockProviderSettings(fail_rate=1.0))


def make_prober(
    providers: dict[str, MockProvider],
    health: HealthRegistry,
    *,
    interval_s: float = 0.01,
    timeout_s: float = 1.0,
) -> HealthProber:
    names = sorted(providers)
    return HealthProber(
        providers=dict(providers),
        enabled=names,
        health=health,
        interval_s=interval_s,
        timeout_s=timeout_s,
    )


def probe_events(logs: list[dict]) -> list[dict]:
    return [e for e in logs if e.get("event") == "gateway.health_probe"]


async def test_failing_probe_marks_provider_unhealthy() -> None:
    health = HealthRegistry(["mock-a"])
    prober = make_prober({"mock-a": failing_provider()}, health)
    with capture_logs() as logs:
        await prober.probe_once()
    assert health.is_healthy("mock-a") is False
    assert health.snapshot()["mock-a"].error == "mock fail_rate=1.0"
    events = probe_events(logs)
    assert len(events) == 1
    assert events[0]["provider"] == "mock-a"
    assert events[0]["healthy"] is False


async def test_successful_probe_restores_provider() -> None:
    health = HealthRegistry(["mock-a"])
    health.mark_unhealthy("mock-a", "PROVIDER_REJECTED")
    prober = make_prober({"mock-a": healthy_provider()}, health)
    with capture_logs() as logs:
        await prober.probe_once()
    assert health.is_healthy("mock-a") is True
    events = probe_events(logs)
    assert len(events) == 1
    assert events[0]["healthy"] is True


async def test_steady_state_probes_log_no_transition() -> None:
    health = HealthRegistry(["mock-a"])
    prober = make_prober({"mock-a": healthy_provider()}, health)
    with capture_logs() as logs:
        await prober.probe_once()
    assert health.is_healthy("mock-a") is True
    assert probe_events(logs) == []


class ExplodingProvider(MockProvider):
    async def health_check(self) -> HealthStatus:
        raise RuntimeError("boom")


class HangingProvider(MockProvider):
    async def health_check(self) -> HealthStatus:
        await asyncio.sleep(5.0)
        return HealthStatus(provider=self.name, healthy=True)


async def test_exception_in_one_probe_does_not_break_others() -> None:
    health = HealthRegistry(["mock-a", "mock-b"])
    prober = make_prober(
        {
            "mock-a": ExplodingProvider("mock-a", MockProviderSettings()),
            "mock-b": healthy_provider("mock-b"),
        },
        health,
    )
    await prober.probe_once()
    assert health.is_healthy("mock-a") is False
    assert "probe error: boom" in (health.snapshot()["mock-a"].error or "")
    assert health.is_healthy("mock-b") is True


async def test_probe_timeout_marks_unhealthy() -> None:
    health = HealthRegistry(["mock-a"])
    prober = make_prober(
        {"mock-a": HangingProvider("mock-a", MockProviderSettings())},
        health,
        timeout_s=0.05,
    )
    await prober.probe_once()
    assert health.is_healthy("mock-a") is False
    assert "probe timeout" in (health.snapshot()["mock-a"].error or "")


async def test_probe_loop_cancels_cleanly() -> None:
    health = HealthRegistry(["mock-a"])
    prober = make_prober({"mock-a": healthy_provider()}, health, interval_s=0.01)
    task = prober.start()
    await asyncio.sleep(0.05)
    assert health.is_healthy("mock-a") is True
    await prober.stop()
    assert prober.task is None
    assert task.done()
    assert task.cancelled() or task.exception() is None


async def test_probing_disabled_means_no_task_starts() -> None:
    settings = AppSettings(
        routing=RoutingSettings(default_provider="mock-a"),
        providers=[
            ProviderEntry(name="mock-a", type="mock", enabled=True, priority=10),
        ],
    )
    assert settings.health.probe_enabled is False
    app = create_app(settings)
    assert app.state.health_prober is None


async def test_enabled_app_starts_and_stops_prober_in_lifespan() -> None:
    settings = AppSettings(
        routing=RoutingSettings(default_provider="mock-a"),
        providers=[
            ProviderEntry(name="mock-a", type="mock", enabled=True, priority=10),
        ],
        health=HealthSettings(probe_enabled=True, probe_interval_s=30.0, probe_timeout_s=5.0),
    )
    app = create_app(settings)
    prober = app.state.health_prober
    assert prober is not None
    assert prober.task is None
    async with app.router.lifespan_context(app):
        assert prober.task is not None
        assert not prober.task.done()
    assert prober.task is None


def lockout_request(request_id: str) -> NormalizedChatRequest:
    return NormalizedChatRequest(
        request_id=request_id,
        model="mock-a",
        provider="mock-a",
        messages=[ChatMessage(role="user", content="hello gateway")],
        temperature=0.0,
        max_tokens=None,
    )


async def test_permanent_lockout_without_probing_and_recovery_with_probe() -> None:
    """Pin the Step-1 lockout: permanent failure skips the provider until a probe restores it."""
    settings = AppSettings(
        routing=RoutingSettings(
            default_provider="mock-a",
            model_aliases={"mock-a": "mock-a", "mock-b": "mock-b"},
        ),
        providers=[
            ProviderEntry(
                name="mock-a",
                type="mock",
                enabled=True,
                priority=10,
                mock=MockProviderSettings(failure_script=("permanent",)),  # type: ignore[arg-type]
            ),
            ProviderEntry(name="mock-b", type="mock", enabled=True, priority=20),
        ],
    )
    app = create_app(settings)
    assert app.state.health_prober is None  # code default: passive behavior
    health: HealthRegistry = app.state.health
    executor = app.state.executor

    first = await executor.execute(lockout_request("lockout-1"))
    assert first.response.provider == "mock-b"
    assert health.is_healthy("mock-a") is False

    second = await executor.execute(lockout_request("lockout-2"))
    assert second.response.provider == "mock-b"
    skips = [a for a in second.outcome.attempts if a.outcome == "skipped"]
    assert any(a.provider == "mock-a" and a.skip_reason == "unhealthy" for a in skips)

    prober = make_prober(
        {"mock-a": healthy_provider(), "mock-b": healthy_provider("mock-b")}, health
    )
    await prober.probe_once()
    assert health.is_healthy("mock-a") is True

    third = await executor.execute(lockout_request("lockout-3"))
    assert third.response.provider == "mock-a"


def test_health_defaults_and_bundled_yaml() -> None:
    defaults = AppSettings()
    assert defaults.health.probe_enabled is False
    assert defaults.health.probe_interval_s == 30.0
    assert defaults.health.probe_timeout_s == 5.0
    shipped = load_settings(Path("configs/gateway.yaml"))
    assert shipped.health.probe_enabled is True
    assert shipped.health.probe_interval_s == 30.0
    assert shipped.health.probe_timeout_s == 5.0
