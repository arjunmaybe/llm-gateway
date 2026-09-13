"""Scoring placeholder. Dynamic model routing is M2+ (planned, not implemented)."""

from __future__ import annotations


def score_provider(provider: str) -> float:
    """Reserved scoring hook. M1 always defers to static priority."""
    raise NotImplementedError("dynamic scoring is planned for M2")
