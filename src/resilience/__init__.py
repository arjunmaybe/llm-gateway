"""M2 resilience: failure taxonomy, retry policy, and fault-tolerant execution."""

from src.resilience.executor import (
    AttemptRecord,
    ExecuteResult,
    ExecutionOutcome,
    ResilientExecutor,
)
from src.resilience.failures import (
    FailureCategory,
    affects_health,
    allows_fallback,
    classify,
    counts_toward_breaker,
    is_retryable_category,
)
from src.resilience.retry import RetryPolicy

__all__ = [
    "AttemptRecord",
    "ExecuteResult",
    "ExecutionOutcome",
    "FailureCategory",
    "ResilientExecutor",
    "RetryPolicy",
    "affects_health",
    "allows_fallback",
    "classify",
    "counts_toward_breaker",
    "is_retryable_category",
]
