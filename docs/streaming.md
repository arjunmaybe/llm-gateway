# Streaming (M3)

SSE streaming for `POST /v1/chat/completions` with `stream=true`, offline over mock providers. Non-streaming `stream=false` behavior is unchanged (M1/M2 JSON).

## Architecture

`ProviderAdapter.chat_stream()` yields normalized plain-text content chunks (transport-agnostic, never SSE) → `ProxyClient.stream_forward()` applies first-byte timeout and `ProviderError→GatewayError` normalization without buffering → `src/main.py` pre-first-byte selects a provider, pins it, and formats SSE via `src/proxy/sse_parser.py` into a `StreamingResponse`. Cache is bypassed for streams (M4 owns streamed caching).

## Wire format (locked)

Headers:

```text
HTTP 200
content-type: text/event-stream
cache-control: no-cache
connection: keep-alive
x-request-id: <echoed|minted>
x-provider: <pinned provider>
```

No `x-gateway-latency-ms`: total latency is unknowable before the stream finishes.

Normal chunks (OpenAI-style deltas):

```text
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{"role":"assistant","content":"..."},"finish_reason":null}],"provider":"...","request_id":"..."}
```

- First content chunk includes `"role":"assistant"`; subsequent content chunks contain content without repeating the role.
- Final terminal chunk carries `finish_reason` (`stop`, or `length` when `max_tokens` truncates) with an empty delta, then:

```text
data: [DONE]
```

Mid-stream failure after the first chunk reached the client:

```text
data: {"error":{"code":"...","message":"...","provider":"...","retryable":...,"request_id":"..."}}
```

Then the stream closes. No `[DONE]` after an error. No provider fallback after the first byte.

## Normalized chunks

Providers yield `str` word-chunks (mock: deterministic split of the same `_generate()` text used by `chat()`, `stream_chunk_words` default 2). SSE framing lives only in `sse_parser.format_chunk/format_done/format_error`; `models.ChatCompletionChunk*` types validate the shape (`delta` omits nulls, `finish_reason:null` preserved for content chunks).

## TTFT

Time from request start to the first actual upstream content chunk received (not provider invocation, not `StreamingResponse` creation, not SSE formatting). Observed as `gateway_stream_ttft_ms` and logged as `ttft_ms`. Uses `perf_counter_ns`.

## ITL

Time between consecutive content chunks received from the provider. Tracked per chunk; summary reports average, p50, p95, max (`gateway_stream_itl_ms`, log fields `itl_ms/itl_p50/itl_p95`). M3 measures chunk-level latency; the mock yields word chunks, not true token boundaries — do not claim token-level precision.

## Cancellation

Client disconnect surfaces as `asyncio.CancelledError` / `GeneratorExit` inside the response generator. Handling: close the upstream provider iterator (`aclose`), increment `gateway_stream_interrupted_total`, emit no error event, record neither success nor failure, release resources. Cancellation is never swallowed to continue consuming the provider.

## Backpressure

Chunks are yielded as they arrive (`async for` without buffering the full response; at most one chunk held for `max_tokens` accounting). Slow clients apply natural backpressure via the ASGI send queue; the gateway never pre-buffers the stream. Per-chunk cooperative `await asyncio.sleep(0)` keeps the loop responsive.

## Pre-first-byte fallback

Before headers/first chunk: `RouterEngine.plan()` ordered candidates → skip unhealthy (`HealthRegistry`) / circuit-open (`CircuitBreaker`) → `stream_forward` first-chunk acquisition (first-byte timeout inside proxy) → on `GatewayError` before first chunk, classify via `resilience/failures.py`, apply M2 breaker/health accounting (`counts_toward_breaker` → `record_failure`; `affects_health` → `mark_unhealthy`), and continue to the next candidate only if `allows_fallback`. First provider yielding a first chunk wins and pins `x-provider`. Exhaustion raises the last error (JSON, since SSE not yet committed) or `NO_HEALTHY_PROVIDER` 503.

## Post-first-byte failure semantics

After the first chunk: provider pinned, no retry, no fallback. Upstream `GatewayError` updates breaker/health identically (failure counted, health marked per M2 rules, `gateway_stream_errors_total` incremented) then yields a single error event and closes. Transparent fallback is impossible after bytes are sent: the client already received `x-provider`, chunk IDs, and partial content; switching providers mid-stream would corrupt ordering, duplicate/omit words, and violate `request_id`/usage accounting. M2 `ResilientExecutor` never grows streaming logic for the same reason.

## Telemetry

Reuses existing `MetricsRecorder`/`Tracer` interfaces (no Prometheus/OTel SDK):

- `gateway_stream_requests_total{provider}`, `gateway_stream_chunks_total{provider}`, `gateway_stream_completed_total{provider}`, `gateway_stream_errors_total{provider,code}`, `gateway_stream_interrupted_total{provider}`, `gateway_provider_attempts_total{provider,code}`, `gateway_fallbacks_total{provider}`, `gateway_breaker_opens_total{provider,code}`, `gateway_errors_total{provider,code}`
- `gateway_stream_ttft_ms`, `gateway_stream_itl_ms` latencies; `StreamingTiming(ttft_ms, itl_ms=avg)` logged per stream.
- `tracer.span("provider.stream", provider=...)` around first-byte acquisition.

## Mock knobs

`MockProviderSettings`: `stream_chunk_words` (words per chunk), `stream_failure_after` (chunk index to fail after, `0` = before first chunk), `stream_failure_mode` (defaults to `failure_mode`). `failure_script` / `fail_rate` still gate pre-first-byte success identically to `chat()`.
