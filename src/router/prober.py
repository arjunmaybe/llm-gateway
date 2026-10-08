"""Active provider health probing.

Complements the passive M2 reporting in :mod:`src.router.health`: the
:class:`HealthProber` periodically calls every enabled provider's
``health_check()`` and reconciles the :class:`HealthRegistry`, so a provider
marked unhealthy by a permanent failure re-enters rotation once it recovers.
Without probing, that mark sticks until a success re-marks it — which can
never happen while the provider is skipped.
"""

from __future__ import annotations

import asyncio

import structlog

from src.providers.base import ProviderAdapter
from src.router.health import HealthRegistry


class HealthProber:
    """Probes enabled providers and reconciles the health registry.

    :meth:`probe_once` calls each enabled provider's ``health_check()`` under
    a timeout: a healthy status marks the provider healthy, an unhealthy
    status / timeout / exception marks it unhealthy with the error. One
    provider's exception never breaks the other providers' probes. Only
    actual healthy/unhealthy transitions are logged.
    """

    def __init__(
        self,
        *,
        providers: dict[str, ProviderAdapter],
        enabled: list[str],
        health: HealthRegistry,
        interval_s: float,
        timeout_s: float,
    ) -> None:
        self._providers = dict(providers)
        self._enabled = list(enabled)
        self._health = health
        self._interval_s = interval_s
        self._timeout_s = timeout_s
        self._task: asyncio.Task[None] | None = None
        # Cooperative stop flag. Plain bool on purpose: on Python < 3.12,
        # ``asyncio.wait_for`` swallows an outer cancellation when its inner
        # future is already done (returns the inner result instead of
        # raising), which is the common case here since adapter
        # ``health_check()`` calls usually complete instantly. A swallowed
        # cancel must still stop the loop, so ``stop()`` sets this flag and
        # ``_run`` checks it every iteration; ``task.cancel()`` stays as the
        # prompt path for when the task is parked in ``asyncio.sleep``.
        self._stopping = False
        self._log = structlog.get_logger("gateway")

    @property
    def task(self) -> asyncio.Task[None] | None:
        """The background loop task, or ``None`` when not running."""
        return self._task

    async def probe_once(self) -> None:
        """Probe every enabled provider once and update the registry."""
        for name in self._enabled:
            provider = self._providers.get(name)
            if provider is None:
                continue
            try:
                status = await asyncio.wait_for(
                    provider.health_check(), timeout=self._timeout_s
                )
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:  # noqa: UP041 - distinct from builtin on py<3.11
                self._reconcile(
                    name, healthy=False, error=f"probe timeout after {self._timeout_s}s"
                )
            except Exception as exc:
                self._reconcile(name, healthy=False, error=f"probe error: {exc}")
            else:
                if status.healthy:
                    self._reconcile(name, healthy=True, error=None)
                else:
                    self._reconcile(
                        name, healthy=False, error=status.error or "probe unhealthy"
                    )

    def start(self) -> asyncio.Task[None]:
        """Spawn the background probe loop. Idempotent while running."""
        if self._task is not None and not self._task.done():
            return self._task
        self._stopping = False
        self._task = asyncio.get_running_loop().create_task(
            self._run(), name="health-prober"
        )
        return self._task

    async def stop(self) -> None:
        """Signal the background loop to exit, then wait for it to finish."""
        self._stopping = True
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        """Probe immediately, then on every interval, until stopped."""
        while not self._stopping:
            try:
                await self.probe_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # never let the loop die on a probe bug
                self._log.warning("gateway.health_probe_loop_error", error=str(exc))
            if self._stopping:
                break
            try:
                await asyncio.sleep(self._interval_s)
            except asyncio.CancelledError:
                raise

    def _reconcile(self, name: str, *, healthy: bool, error: str | None) -> None:
        """Apply one probe result, logging only actual transitions."""
        was_healthy = self._health.is_healthy(name)
        if healthy:
            self._health.mark_healthy(name)
        else:
            assert error is not None
            self._health.mark_unhealthy(name, error)
        if was_healthy != healthy:
            self._log.info(
                "gateway.health_probe",
                provider=name,
                healthy=healthy,
                error=None if healthy else error,
            )
