"""In-memory vector index. Brute-force cosine search (M4, no Redis required)."""

from __future__ import annotations

import math
from typing import Any


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]. Zero-norm vectors score 0.0."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} != {len(b)}")
    if not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class VectorIndex:
    """Dict-backed ``key -> (vector, entry)`` with brute-force search."""

    def __init__(self) -> None:
        self._vectors: dict[str, list[float]] = {}
        self._entries: dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self._vectors)

    def clear(self) -> None:
        self._vectors.clear()
        self._entries.clear()

    def add(self, key: str, vector: list[float], entry: Any) -> None:
        self._vectors[key] = list(vector)
        self._entries[key] = entry

    def get(self, key: str) -> Any | None:
        return self._entries.get(key)

    def search(
        self, query: list[float], threshold: float = 0.85
    ) -> tuple[str, Any, float] | None:
        """Return ``(key, entry, score)`` for best match with score >= threshold."""
        best_key: str | None = None
        best_score = threshold
        best_entry: Any = None
        found = False
        for key, vec in self._vectors.items():
            if len(vec) != len(query):
                continue
            try:
                score = cosine_similarity(query, vec)
            except ValueError:
                continue
            # Strictly greater wins; exact ties keep first-inserted.
            if score >= threshold and (not found or score > best_score):
                best_key = key
                best_entry = self._entries.get(key)
                best_score = score
                found = True
        if not found or best_key is None:
            return None
        return (best_key, best_entry, best_score)
