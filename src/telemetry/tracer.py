"""Tracing seam. OpenTelemetry spans land in M5."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


class NoOpTracer:
    @contextmanager
    def span(self, name: str, **attrs: str) -> Iterator[None]:
        _ = (name, attrs)
        yield
