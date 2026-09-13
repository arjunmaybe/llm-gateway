"""M1 passive health registry. Active probing arrives in M2."""

from __future__ import annotations

from src.providers.base import HealthStatus


class HealthRegistry:
    """Static health view derived from config + runtime marks.

    M1: initialized healthy for every enabled provider; ``mark_*`` hooks exist
    so M2 background probing / failure tracking can plug in without API change.
    """

    def __init__(self, providers: list[str]) -> None:
        self._status: dict[str, HealthStatus] = {
            name: HealthStatus(provider=name, healthy=True) for name in providers
        }

    def mark_healthy(self, provider: str) -> None:
        self._status[provider] = HealthStatus(provider=provider, healthy=True)

    def mark_unhealthy(self, provider: str, error: str) -> None:
        self._status[provider] = HealthStatus(provider=provider, healthy=False, error=error)

    def is_healthy(self, provider: str) -> bool:
        status = self._status.get(provider)
        return status.healthy if status is not None else False

    def snapshot(self) -> dict[str, HealthStatus]:
        return dict(self._status)

    def readiness(self) -> dict[str, bool]:
        return {name: status.healthy for name, status in self._status.items()}
