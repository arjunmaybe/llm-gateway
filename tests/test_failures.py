"""Failure taxonomy tests: code mapping and policy predicates."""

from __future__ import annotations

from src.errors import GatewayError
from src.resilience.failures import (
    FailureCategory,
    affects_health,
    allows_fallback,
    classify,
    counts_toward_breaker,
    is_retryable_category,
)


def make_error(code: str) -> GatewayError:
    return GatewayError(
        code=code, message="boom", status_code=500, provider="mock-a", request_id="r"
    )


def test_classify_known_codes() -> None:
    assert classify(make_error("UPSTREAM_TIMEOUT")) is FailureCategory.TIMEOUT
    assert classify(make_error("PROVIDER_TIMEOUT")) is FailureCategory.TIMEOUT
    assert classify(make_error("PROVIDER_CONNECTION_FAILED")) is FailureCategory.CONNECTION
    assert classify(make_error("PROVIDER_RATE_LIMITED")) is FailureCategory.RATE_LIMITED
    assert classify(make_error("PROVIDER_UNAVAILABLE")) is FailureCategory.UPSTREAM_TRANSIENT
    assert classify(make_error("PROVIDER_REJECTED")) is FailureCategory.PERMANENT
    assert classify(make_error("INVALID_REQUEST")) is FailureCategory.CLIENT
    assert classify(make_error("STREAMING_NOT_SUPPORTED")) is FailureCategory.CLIENT
    assert classify(make_error("NO_HEALTHY_PROVIDER")) is FailureCategory.NO_CAPACITY
    assert classify(make_error("UNKNOWN_PROVIDER")) is FailureCategory.INTERNAL
    assert classify(make_error("INTERNAL")) is FailureCategory.INTERNAL


def test_unknown_code_fails_closed_as_internal() -> None:
    category = classify(make_error("SOMETHING_NEW"))
    assert category is FailureCategory.INTERNAL
    assert is_retryable_category(category) is False
    assert counts_toward_breaker(category) is False
    assert affects_health(category) is False
    assert allows_fallback(category) is False


def test_retryable_set() -> None:
    for category in (
        FailureCategory.TIMEOUT,
        FailureCategory.CONNECTION,
        FailureCategory.RATE_LIMITED,
        FailureCategory.UPSTREAM_TRANSIENT,
    ):
        assert is_retryable_category(category) is True
        assert counts_toward_breaker(category) is True
        assert affects_health(category) is False
        assert allows_fallback(category) is True


def test_permanent_skips_breaker_but_marks_health_and_falls_back() -> None:
    assert is_retryable_category(FailureCategory.PERMANENT) is False
    assert counts_toward_breaker(FailureCategory.PERMANENT) is False
    assert affects_health(FailureCategory.PERMANENT) is True
    assert allows_fallback(FailureCategory.PERMANENT) is True
