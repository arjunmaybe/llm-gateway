"""Singleflight stampede protection. Coalesces concurrent same-key calls."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Singleflight(Generic[T]):
    """One in-flight ``fn()`` per key; concurrent waiters share the result."""

    def __init__(self) -> None:
        self._inflight: dict[str, asyncio.Future[T]] = {}

    async def run(self, key: str, fn: Callable[[], Awaitable[T]]) -> T:
        existing = self._inflight.get(key)
        if existing is not None:
            return await existing
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[T] = loop.create_future()
        self._inflight[key] = fut
        try:
            result = await fn()
        except BaseException as exc:
            if not fut.done():
                fut.set_exception(exc)
            raise
        else:
            if not fut.done():
                fut.set_result(result)
            return result
        finally:
            self._inflight.pop(key, None)
