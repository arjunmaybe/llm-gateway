"""Cache interfaces. M1 ships NoOps; exact + semantic caching land in M4."""

from __future__ import annotations

import abc
from typing import Any


class CacheManager(abc.ABC):
    """Lookup seam. Routing/proxy code must not change when Redis lands."""

    @abc.abstractmethod
    async def get(self, key: str) -> Any | None:
        raise NotImplementedError

    @abc.abstractmethod
    async def put(self, key: str, value: Any) -> None:
        raise NotImplementedError


class NoOpCacheManager(CacheManager):
    """M1 default: always miss, never store."""

    async def get(self, key: str) -> Any | None:
        _ = key
        return None

    async def put(self, key: str, value: Any) -> None:
        _ = (key, value)


def exact_cache_key(*, model: str, messages_hash: str, temperature: float) -> str:
    """Reserved key scheme for the M4 exact cache (sha256 documented in plan)."""
    return f"exact:{model}:{messages_hash}:{temperature}"
