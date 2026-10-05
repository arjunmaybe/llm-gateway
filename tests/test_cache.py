"""Semantic cache tests. Offline: fake embed_fn, no model download.

The real ``all-MiniLM-L6-v2`` scores the France paraphrase pair ~0.9
(above the 0.85 default); here we inject deterministic vectors so CI stays
offline and the threshold logic is verified exactly.
"""

from __future__ import annotations

import math

from fastapi import FastAPI

from src.cache.index import VectorIndex
from src.cache.manager import SemanticCacheManager, build_prompt_key, build_prompt_text
from src.models import ChatMessage
from tests.conftest import make_client

PROMPT_A = "What is the capital of France?"
PROMPT_PARAPHRASE = "Which city is France's capital?"
PROMPT_UNRELATED = "Write a quicksort in Python"


def _paraphrase_embed(text: str) -> list[float]:
    """Fake: paraphrases near-identical, unrelated orthogonal."""
    low = text.lower()
    if "quicksort" in low:
        return [0.0, 1.0, 0.0]
    if "capital" in low or "france" in low:
        if "which city" in low:
            # ~0.95 cosine to PROMPT_A vector.
            return [0.95, 0.3122498999199199, 0.0]
        return [1.0, 0.0, 0.0]
    return [0.0, 0.0, 1.0]


async def test_exact_duplicate_hit() -> None:
    cache = SemanticCacheManager(embed_fn=_paraphrase_embed, threshold=0.85)
    await cache.put(PROMPT_A, "Paris")
    assert await cache.get(PROMPT_A) == "Paris"


async def test_paraphrase_hit() -> None:
    cache = SemanticCacheManager(embed_fn=_paraphrase_embed, threshold=0.85)
    await cache.put(PROMPT_A, "Paris")
    assert await cache.get(PROMPT_PARAPHRASE) == "Paris"


async def test_unrelated_miss() -> None:
    cache = SemanticCacheManager(embed_fn=_paraphrase_embed, threshold=0.85)
    await cache.put(PROMPT_A, "Paris")
    assert await cache.get(PROMPT_UNRELATED) is None


async def test_threshold_boundary() -> None:
    """Just above threshold hits; just below misses (threshold=0.85)."""

    def embed(text: str) -> list[float]:
        if text == "stored":
            return [1.0, 0.0]
        if text == "just_above":  # cosine 0.86
            return [0.86, math.sqrt(1.0 - 0.86**2)]
        if text == "just_below":  # cosine 0.84
            return [0.84, math.sqrt(1.0 - 0.84**2)]
        raise AssertionError(f"unexpected key {text!r}")

    cache = SemanticCacheManager(embed_fn=embed, threshold=0.85)
    await cache.put("stored", "value")
    assert await cache.get("just_above") == "value"
    assert await cache.get("just_below") is None


async def test_api_cached_flag_reflects_hit_miss(app: FastAPI) -> None:
    """End-to-end: second identical prompt returns cached=True without upstream."""
    cache = SemanticCacheManager(embed_fn=_paraphrase_embed, threshold=0.85)
    app.state.cache = cache

    def body(prompt: str) -> dict[str, object]:
        return {
            "model": "mock-a",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "stream": False,
        }

    async with make_client(app) as client:
        first = await client.post("/v1/chat/completions", json=body(PROMPT_A))
        assert first.status_code == 200
        assert first.json()["cached"] is False

        second = await client.post("/v1/chat/completions", json=body(PROMPT_A))
        assert second.status_code == 200
        assert second.json()["cached"] is True

        paraphrase = await client.post(
            "/v1/chat/completions", json=body(PROMPT_PARAPHRASE)
        )
        assert paraphrase.status_code == 200
        assert paraphrase.json()["cached"] is True

        unrelated = await client.post(
            "/v1/chat/completions", json=body(PROMPT_UNRELATED)
        )
        assert unrelated.status_code == 200
        assert unrelated.json()["cached"] is False


def test_prompt_key_helpers_scope_model() -> None:
    msgs = [ChatMessage(role="user", content=PROMPT_A)]
    text = build_prompt_text(msgs)
    assert PROMPT_A in text
    key_a = build_prompt_key(model="mock-a", temperature=0.0, prompt_text=text)
    key_b = build_prompt_key(model="mock-b", temperature=0.0, prompt_text=text)
    assert key_a != key_b


def test_vector_index_brute_force_threshold() -> None:
    idx = VectorIndex()
    idx.add("a", [1.0, 0.0], "A")
    assert idx.search([1.0, 0.0], threshold=0.85) == ("a", "A", 1.0)
    assert idx.search([0.0, 1.0], threshold=0.85) is None


async def test_singleflight_dedups_concurrent_same_key() -> None:
    import asyncio

    from src.cache.singleflight import Singleflight

    sf = Singleflight()
    calls = 0

    async def fn() -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return "ok"

    results = await asyncio.gather(*(sf.run("k", fn) for _ in range(5)))
    assert results == ["ok"] * 5
    assert calls == 1

    # Different keys do not coalesce.
    calls = 0
    out = await asyncio.gather(sf.run("a", fn), sf.run("b", fn))
    assert out == ["ok", "ok"]
    assert calls == 2
