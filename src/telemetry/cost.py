"""Cost tracking seam. Token price tables land in M5."""

from __future__ import annotations

from src.models import Usage


class CostTracker:
    """M1: usage passthrough with no pricing. M5 adds per-model price tables."""

    def total_tokens(self, usage: Usage) -> int:
        return usage.total_tokens
