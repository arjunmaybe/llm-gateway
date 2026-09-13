"""Normalized failure taxonomy for M2 resilience.

Classification runs on the already-normalized :class:`GatewayError.code`
(produced at the proxy boundary), so future real providers are covered
without any routing/executor changes. Pure functions — no I/O, no state.
"""

from __future__ import annotations

from enum import Enum

from src.errors import GatewayError


class FailureCategory(str, Enum):
    """Stable failure classes driving retry / breaker / fallback decisions."""

    TIMEOUT = "timeout"
    CONNECTION = "connection"
    RATE_LIMITED = "rate_limited"
    UPSTREAM_TRANSIENT = "upstream_transient"
    PERMANENT = "permanent"
    CLIENT = "client"
    NO_CAPACITY = "no_capacity"
    INTERNAL = "internal"


_CODE_TO_CATEGORY: dict[str, FailureCategory] = {
    "UPSTREAM_TIMEOUT": FailureCategory.TIMEOUT,
    "PROVIDER_TIMEOUT": FailureCategory.TIMEOUT,
    "PROVIDER_CONNECTION_FAILED": FailureCategory.CONNECTION,
    "PROVIDER_RATE_LIMITED": FailureCategory.RATE_LIMITED,
    "PROVIDER_UNAVAILABLE": FailureCategory.UPSTREAM_TRANSIENT,
    "PROVIDER_REJECTED": FailureCategory.PERMANENT,
    "INVALID_REQUEST": FailureCategory.CLIENT,
    "STREAMING_NOT_SUPPORTED": FailureCategory.CLIENT,
    "NO_HEALTHY_PROVIDER": FailureCategory.NO_CAPACITY,
    "UNKNOWN_PROVIDER": FailureCategory.INTERNAL,
    "INTERNAL": FailureCategory.INTERNAL,
}

_RETRYABLE_CATEGORIES = frozenset(
    {
        FailureCategory.TIMEOUT,
        FailureCategory.CONNECTION,
        FailureCategory.RATE_LIMITED,
        FailureCategory.UPSTREAM_TRANSIENT,
    }
)


def classify(error: GatewayError) -> FailureCategory:
    """Map a normalized gateway error to its failure category.

    Unknown codes are treated as internal: never retried, never counted
    against provider health. Failing closed is safer than retrying blind.
    """
    return _CODE_TO_CATEGORY.get(error.code, FailureCategory.INTERNAL)


def is_retryable_category(category: FailureCategory) -> bool:
    """Whether a failed attempt in this category may be retried."""
    return category in _RETRYABLE_CATEGORIES


def counts_toward_breaker(category: FailureCategory) -> bool:
    """Whether a failure feeds the circuit-breaker consecutive counter.

    M2 rule: every retryable failure counts equally (rate limits included).
    Permanent failures skip the breaker — the provider is marked unhealthy
    directly instead. Client/internal errors never touch provider state.
    """
    return category in _RETRYABLE_CATEGORIES


def affects_health(category: FailureCategory) -> bool:
    """Whether a failure marks the provider unhealthy in the registry.

    Only permanent failures do. Transient failures are the breaker's job;
    marking a provider unhealthy on a single timeout would pin it out of
    rotation for one haircut.
    """
    return category is FailureCategory.PERMANENT


def allows_fallback(category: FailureCategory) -> bool:
    """Whether execution may continue with the next provider candidate."""
    return category in _RETRYABLE_CATEGORIES or category is FailureCategory.PERMANENT
