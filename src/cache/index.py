"""Vector-index hook. Redis Stack vector search lands in M4."""

from __future__ import annotations


class VectorIndex:
    """Reserved for M4. Instantiation fails fast in M1."""

    def __init__(self) -> None:
        raise NotImplementedError("vector index is planned for M4")
