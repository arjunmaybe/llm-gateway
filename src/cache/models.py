"""Cache entry models. Reserved for M4 (planned, not used in M1)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CacheEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    content: str
    hits: int = Field(default=0, ge=0)
