"""Embedding hook. Semantic caching / sentence-transformers land in M4."""

from __future__ import annotations


def embed(text: str) -> list[float]:
    """Reserved for M4. Never called in M1."""
    raise NotImplementedError("embeddings are planned for M4")
