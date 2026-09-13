"""Normalized gateway errors. Raw provider payloads must never reach clients."""

from __future__ import annotations


class GatewayError(Exception):
    """Gateway-level failure with a stable code and HTTP mapping."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int,
        provider: str | None = None,
        retryable: bool = False,
        request_id: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.provider = provider
        self.retryable = retryable
        self.request_id = request_id
