"""Cache interfaces. M4 ships in-memory exact + semantic caching."""

from __future__ import annotations

import abc
import inspect
from collections.abc import Callable
from typing import Any

from src.cache.index import VectorIndex

EmbedFn = Callable[[str], Any]

DEFAULT_SEMANTIC_THRESHOLD = 0.85


class CacheManager(abc.ABC):
    """Lookup seam. Routing/proxy code must not change when Redis lands."""

    @abc.abstractmethod
    async def get(self, key: str) -> Any | None:
        raise NotImplementedError

    @abc.abstractmethod
    async def put(self, key: str, value: Any) -> None:
        raise NotImplementedError


class NoOpCacheManager(CacheManager):
    """Backend ``noop``: always miss, never store."""

    async def get(self, key: str) -> Any | None:
        _ = key
        return None

    async def put(self, key: str, value: Any) -> None:
        _ = (key, value)


class SemanticCacheManager(CacheManager):
    """In-memory exact + semantic cache.

    ``key`` is the caller-scoped prompt string (e.g. ``model:temp:prompt``).
    Exact duplicates hit via dict lookup; paraphrases hit via cosine search
    over ``embed_fn(key)`` vectors when similarity >= ``threshold``.
    """

    def __init__(
        self,
        *,
        embed_fn: EmbedFn,
        threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
        index: VectorIndex | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self._embed_fn = embed_fn
        self._threshold = threshold
        self._index = index if index is not None else VectorIndex()
        self._exact: dict[str, Any] = {}

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def index(self) -> VectorIndex:
        return self._index

    async def _embed(self, key: str) -> list[float] | None:
        try:
            result = self._embed_fn(key)
            if inspect.isawaitable(result):
                result = await result
            return [float(x) for x in result]
        except Exception:
            return None

    async def get(self, key: str) -> Any | None:
        if key in self._exact:
            return self._exact[key]
        vector = await self._embed(key)
        if vector is None:
            return None
        hit = self._index.search(vector, threshold=self._threshold)
        if hit is None:
            return None
        _, entry, _ = hit
        return entry

    async def put(self, key: str, value: Any) -> None:
        self._exact[key] = value
        vector = await self._embed(key)
        if vector is None:
            return
        try:
            self._index.add(key, vector, value)
        except Exception:
            return


def exact_cache_key(*, model: str, messages_hash: str, temperature: float) -> str:
    """Key scheme for the exact cache (sha256 documented in plan)."""
    return f"exact:{model}:{messages_hash}:{temperature}"


def build_prompt_key(*, model: str, temperature: float, prompt_text: str) -> str:
    """Scoped semantic key: model + temperature + raw prompt text."""
    return f"semantic:{model}:{temperature}:{prompt_text}"


def build_prompt_text(messages: Any) -> str:
    """Join chat messages into a single embeddable string."""
    parts: list[str] = []
    for m in messages:
        role = getattr(m, "role", "")
        content = getattr(m, "content", "")
        if isinstance(m, dict):
            role = m.get("role", role)
            content = m.get("content", content)
        parts.append(f"{role}:{content}")
    return "\n".join(parts)
