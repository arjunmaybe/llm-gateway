"""Embedding hook. Semantic caching via sentence-transformers (M4).

The heavy ``sentence-transformers`` dependency is imported lazily inside
:class:`TextEmbedder.__init__` so unit tests and ``backend=noop`` deployments
never pay the import / model-download cost. The model is loaded once at
init — never per-call.
"""

from __future__ import annotations

from threading import Lock
from typing import Any

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_singleton: TextEmbedder | None = None
_singleton_lock = Lock()


class TextEmbedder:
    """Loads ``all-MiniLM-L6-v2`` once; :meth:`embed` reuses it per call."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        self._model_name = model_name
        self._model: Any = SentenceTransformer(model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text)
        return [float(x) for x in vec]

    __call__ = embed


def embed(text: str) -> list[float]:
    """Embed with the process-wide singleton (loaded once, reused)."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = TextEmbedder()
    return _singleton.embed(text)


def _reset_default_embedder_for_tests() -> None:
    """Test hook: drop the singleton so tests can inject fakes."""
    global _singleton
    with _singleton_lock:
        _singleton = None
