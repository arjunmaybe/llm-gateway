"""Singleflight stub. Stampede protection lands with caching in M4."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class Singleflight:
    """M1: no dedup — just executes. M4 will coalesce concurrent same-key calls."""

    async def run(self, key: str, fn: Callable[[], Awaitable[T]]) -> T:
        _ = key
        return await fn()
