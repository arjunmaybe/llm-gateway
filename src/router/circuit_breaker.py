"""Circuit breaker interface. M1 ships a NoOp; real state machine lands in M2."""

from __future__ import annotations

import abc
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker(abc.ABC):
    """Stability seam so routing code never changes when M2 lands."""

    @abc.abstractmethod
    def can_execute(self, provider: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def record_success(self, provider: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def record_failure(self, provider: str) -> None:
        raise NotImplementedError


class NoOpCircuitBreaker(CircuitBreaker):
    """M1 default: always closed, records nothing."""

    def can_execute(self, provider: str) -> bool:
        _ = provider
        return True

    def record_success(self, provider: str) -> None:
        _ = provider

    def record_failure(self, provider: str) -> None:
        _ = provider
