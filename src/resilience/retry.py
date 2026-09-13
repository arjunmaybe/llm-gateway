"""Bounded retry policy: attempts, exponential backoff, jitter, elapsed budget.

The policy is pure math (no sleeping inside): the executor asks
: meth:`should_retry` whether another attempt is allowed, computes
: meth:`backoff_ms`, applies jitter, and awaits the injected sleeper.
Injectable sleeper + jitter source keep tests fast and deterministic.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from src.resilience.failures import FailureCategory, is_retryable_category

Sleeper = Callable[[float], Awaitable[None]]
"""Awaitable sleep taking milliseconds. Injected so tests never really wait."""


async def _default_sleeper(delay_ms: float) -> None:
    await asyncio.sleep(delay_ms / 1000.0)


@dataclass(frozen=True)
class RetryPolicy:
    """How one provider candidate may be retried before falling back.

    ``max_attempts`` counts every attempt including the first, so the
    default of 2 means initial try + at most one retry. ``max_elapsed_ms``
    caps total backoff sleeping per request (anti retry-storm budget).
    """

    max_attempts: int = 2
    backoff_base_ms: float = 50.0
    backoff_max_ms: float = 1000.0
    max_elapsed_ms: float = 8000.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.backoff_base_ms <= 0:
            raise ValueError("backoff_base_ms must be > 0")
        if self.backoff_max_ms < self.backoff_base_ms:
            raise ValueError("backoff_max_ms must be >= backoff_base_ms")
        if self.max_elapsed_ms < 0:
            raise ValueError("max_elapsed_ms must be >= 0")

    def should_retry(
        self, *, category: FailureCategory, attempt_index: int, elapsed_ms: float
    ) -> bool:
        """Decide after a failed attempt (1-based ``attempt_index``)."""
        if not is_retryable_category(category):
            return False
        if attempt_index >= self.max_attempts:
            return False
        return elapsed_ms < self.max_elapsed_ms

    def backoff_ms(self, *, attempt_index: int) -> float:
        """Exponential backoff ceiling for the attempt that just failed."""
        return min(self.backoff_base_ms * (2.0 ** (attempt_index - 1)), self.backoff_max_ms)


@dataclass(frozen=True)
class BackoffSleeper:
    """Computes full-jitter delays (``uniform(0, backoff)``) and sleeps."""

    sleeper: Sleeper | None = field(default=None)
    jitter: random.Random | None = field(default=None)

    async def sleep_before_retry(self, *, backoff_ceiling_ms: float) -> float:
        """Sleep a jittered delay; returns the milliseconds actually slept."""
        rng = self.jitter if self.jitter is not None else random
        delay_ms = rng.uniform(0.0, backoff_ceiling_ms)
        await (self.sleeper if self.sleeper is not None else _default_sleeper)(delay_ms)
        return delay_ms
