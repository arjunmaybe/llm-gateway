"""Gateway-level API data models.

OpenAI-compatible ``/v1/chat/completions`` strict subset for M1.
Deliberately NOT full OpenAI compatibility: unknown fields are rejected
(``extra="forbid"``) so clients fail fast instead of silently ignoring knobs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ChatRole = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """Single chat message."""

    model_config = ConfigDict(extra="forbid")

    role: ChatRole
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    """Ingress request body. Strict subset of OpenAI Chat Completion params."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False
    user: str | None = None


class NormalizedChatRequest(BaseModel):
    """Internal normalized request. Carries the correlation ID end to end."""

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    messages: list[ChatMessage]
    temperature: float
    max_tokens: int | None
    user: str | None = None


class Usage(BaseModel):
    """Token usage. M1 counts are naive whitespace estimates from the mock."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ChatCompletionMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int = 0
    message: ChatCompletionMessage
    finish_reason: Literal["stop", "length"] = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible response envelope plus gateway extensions."""

    model_config = ConfigDict(frozen=True)

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage
    # Gateway extensions (safe to ignore for OpenAI clients):
    provider: str
    request_id: str
    latency_ms: float
    cached: bool = False
    # Streaming placeholders — always null for non-streaming M1 responses.
    ttft_ms: float | None = None
    itl_ms: float | None = None


class GatewayErrorPayload(BaseModel):
    """Normalized error body. Never contains raw provider payloads."""

    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    provider: str | None
    retryable: bool
    request_id: str


class GatewayErrorEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    error: GatewayErrorPayload


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["ok"] = "ok"
    version: str
    providers: list[str]


class ReadyResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    ready: bool
    providers: dict[str, bool]
