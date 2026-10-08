# LLM Gateway

FastAPI gateway in front of chat-completion providers. It takes OpenAI-style
`POST /v1/chat/completions` requests (plus `GET /health` and `GET /ready`),
picks a provider through a scoring router, forwards with retries and fallback,
and returns either a JSON completion or an SSE stream. Request bodies are a
strict subset of the OpenAI shape: unknown fields are rejected (`extra="forbid"`
in `src/models.py`), so clients fail fast instead of silently ignoring knobs.
Every response carries `x-request-id` (the inbound value echoed, or a minted one).

See `docs/routing.md` for the retry / breaker / fallback design and
`docs/streaming.md` for the SSE streaming design.

## Architecture

- **Router** (`src/router/engine.py` + `src/router/scorer.py`): `plan()` orders
  the candidate chain for a model — an explicit model alias (or provider name
  used as model) pins its target first, the rest is score-ordered with static
  priority order as the tie-break. `select_provider()` drops unhealthy and
  breaker-open candidates, then keeps the alias pick when eligible, otherwise
  the best score. No healthy candidate → `NO_HEALTHY_PROVIDER` (503).
- **Resilience** (`src/resilience/`): `ResilientExecutor` walks the router's
  chain, skipping `unhealthy` / `circuit-open` candidates and recording a
  structured attempt log per request. Failure codes are classified in
  `src/resilience/failures.py` (timeout, connection, rate-limited,
  upstream-transient, permanent, client, no-capacity, internal).
- **Streaming proxy** (`src/proxy/client.py` + `src/proxy/sse_parser.py`):
  `ProxyClient` owns per-provider timeouts and translates provider errors into
  normalized `GatewayError`s (`{error: {code, message, provider, retryable,
  request_id}}`); `sse_parser` owns the `text/event-stream` framing and
  incremental parsing. Providers yield plain text chunks, never SSE.
- **Telemetry** (`src/telemetry/`): `LatencyTracker` (per-provider rolling
  mean), JSON structured logs via structlog, and no-op metrics/tracer seams
  (`NoOpMetricsRecorder`, `NoOpTracer` — Prometheus/OpenTelemetry export is
  future work).
- **Cache** (`src/cache/`): in-memory exact + semantic cache on the
  non-streaming path only (see below).
- **Providers** (`src/providers/`): `mock` (deterministic, offline),
  `openrouter` (real API), plus `openai` / `anthropic` placeholders (see below).

## Providers and the OpenRouter pair

`configs/gateway.yaml` wires two mock providers (`mock-a` priority 10,
`mock-b` priority 20) and a real OpenRouter primary/fallback pair:

- `openrouter-primary` (priority 30, timeout 30 s): `google/gemma-4-31b-it:free`,
  `cost_per_1k_tokens: 0.02`, `quality_weight: 0.7`.
- `openrouter-fallback` (priority 31, timeout 30 s):
  `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`,
  `cost_per_1k_tokens: 0.06`, `quality_weight: 0.85`.

The OpenRouter adapter (`src/providers/openrouter.py`) speaks OpenAI-compatible
chat completions over `httpx` (SSE parsed incrementally for `chat_stream`).
The bearer token resolves at call time from the constructor arg, then
`OpenRouterSettings.api_key`, then the `OPENROUTER_API_KEY` env var — secrets
never live in YAML. HTTP mapping: 429 → `PROVIDER_RATE_LIMITED` (retryable),
408/504 → `PROVIDER_TIMEOUT`, other 5xx → `PROVIDER_UNAVAILABLE`, other
4xx → `PROVIDER_REJECTED` (permanent, not retryable); no key →
`PROVIDER_NOT_CONFIGURED` (501). `health_check()` is a cheap `GET /models`
that spends no tokens. `GATEWAY_OPENROUTER_MODEL` overrides the model on every
OpenRouter provider (see `src/config.py`).

The `openai` and `anthropic` adapters are placeholders: `chat()` raises
`PROVIDER_NOT_CONFIGURED` ("planned but not implemented", 501) and
`health_check()` reports unhealthy ("not implemented"). They are not wired
into `build_providers`, which only constructs `mock` and `openrouter` entries.

## Scoring

Each eligible provider scores
`(w_lat·latency + w_cost·cost + w_qual·quality) / (w_lat + w_cost + w_qual)`
(`src/router/scorer.py`). Latency and cost are min-max normalized across the
eligible set and inverted (lower raw value is better; all-equal is a tie);
quality is the configured `quality_weight` used as-is (higher is better).
Highest score wins; ties keep static priority order.

- **Latency** is the `LatencyTracker` rolling mean over the last 20 *successful*
  attempts per provider — failures are never recorded, so a fast-failing
  provider never looks attractive; providers with no samples score neutrally.
- **Cost** (`cost_per_1k_tokens`) and **quality** (`quality_weight`) are the
  configured values from each provider entry in `configs/gateway.yaml`.
- Weights come from the `scoring:` block (`weight_latency`, `weight_cost`,
  `weight_quality`); all zeros fall back to equal weighting instead of
  dividing by zero. An explicit model alias still pins its target when
  eligible, regardless of score.

## Retries and the circuit breaker

- `RetryPolicy` (`src/resilience/retry.py`): `max_attempts` counts every
  attempt including the first (default 2 = try + at most one retry per
  provider). Only retryable categories (timeout, connection, rate-limited,
  upstream-transient) are retried, with exponential backoff
  (`backoff_base_ms` 50 → `backoff_max_ms` 1000 by default), full-jitter
  `uniform(0, backoff)`, and a `max_elapsed_ms` (8000) cap on total backoff
  sleep per request. Permanent, client, and internal errors are never retried.
- `ResilientCircuitBreaker` (`src/router/circuit_breaker.py`): counts
  consecutive retryable failures per provider; `failure_threshold` (default 5)
  opens the circuit, `recovery_timeout_s` (default 30) later admits
  `half_open_max_inflight` (default 1) probes — a probe success closes it, a
  probe failure reopens it. Any success resets the counter. Permanent failures
  skip the breaker and go straight to the health registry; client/internal
  errors touch neither. Tuning lives in the `resilience:` block of
  `configs/gateway.yaml`; `GATEWAY_RETRY_*` / `GATEWAY_CIRCUIT_*` env vars win.
- A failed provider falls through to the next candidate in the same request
  (retryable *and* permanent failures allow fallback); `x-provider` names the
  provider that actually served the request.

## Health: passive tracking plus the active prober

`HealthRegistry` (`src/router/health.py`) starts healthy for every enabled
provider. Request outcomes move it: a permanent failure (`PROVIDER_REJECTED`)
marks the provider unhealthy with the error code; any success marks it healthy
again. Retryable failures never touch the registry — that is the breaker's job.
Without further input, an unhealthy mark sticks, because skipped providers
can't succeed their way back (the executor records `skipped: unhealthy`).

The active `HealthProber` (`src/router/prober.py`) closes that loop: every
`probe_interval_s` it calls each enabled provider's `health_check()` under
`probe_timeout_s` and reconciles the registry (healthy → healthy, unhealthy /
timeout / exception → unhealthy with the error), logging each transition as a
`gateway.health_probe` event. One provider's exception never breaks the other
providers' probes, and the loop never dies on a probe bug. It runs as a
background task started and stopped in the app's lifespan; shutdown cancels it
cleanly. Config (`health:` block, defaults `probe_enabled: false`,
`probe_interval_s: 30`, `probe_timeout_s: 5`): probing is off unless enabled —
the shipped `configs/gateway.yaml` sets `probe_enabled: true` so deployments
recover automatically.

## Streaming

`POST /v1/chat/completions` with `stream: true` returns `text/event-stream`:
OpenAI-style `chat.completion.chunk` deltas (`role: "assistant"` on the first
content chunk only, `finish_reason: stop|length` on the terminal chunk), then
`data: [DONE]`. Headers include `x-request-id`, `x-provider`, and
`cache-control: no-cache`; the total-latency header (`x-gateway-latency-ms`)
is non-streaming only and is never set early.

- The per-provider timeout covers first-byte acquisition only — a healthy long
  stream is never cut off by it (`src/proxy/client.py`).
- Pre-first-byte failures follow the same retry/fallback policy as
  non-streaming and move to the next candidate; once bytes flow the provider is
  pinned — a mid-stream failure emits `data: {"error": {...}}` and closes with
  no `[DONE]` and no fallback.
- Client disconnect (`GeneratorExit`/`CancelledError`) closes the upstream
  iterator without an error event.

## TTFT / ITL telemetry

Time-to-first-byte and mean inter-chunk latency are measured with
`perf_counter_ns` on the streaming path and emitted **in the `gateway.stream`
structured log only** (`ttft_ms`, `itl_ms`, `timing_ms`, plus `itl_p50` /
`itl_p95` / `chunks` on clean completion). They appear in no response header
and no SSE payload, and the wired metrics recorder is a no-op. A single-chunk
stream has no inter-chunk gap, so `itl_ms` stays `None` rather than a
fabricated `0.0`. Non-streaming JSON responses always report `ttft_ms: null`
and `itl_ms: null`.

## Semantic cache

`SemanticCacheManager` is exact-match plus semantic: duplicates hit via dict
lookup, paraphrases hit via brute-force cosine search over `embed_fn(key)`
vectors in the in-memory `VectorIndex` when similarity meets the threshold.
Embeddings come from `sentence-transformers` (`TextEmbedder`, default model
`all-MiniLM-L6-v2`), imported lazily so `noop` deployments never pay the
import/model cost — if embedding fails, the app logs
`gateway.cache_fallback_noop` and runs uncached.

- Only the **non-streaming** path consults the cache (cache hits return
  `cached: true`); streaming never reads or writes it.
- **Off by default**: the code default, `configs/cache.yaml`, and the gateway
  YAML all resolve to backend `noop` (always miss, never store).
- Enable it with `GATEWAY_CACHE_BACKEND=semantic` (env wins over files);
  `GATEWAY_CACHE_THRESHOLD` / `GATEWAY_CACHE_MODEL` tune the cosine threshold
  and embedding model. Optionally pin values in `configs/cache.yaml` or an
  inline `cache:` section in `configs/gateway.yaml`.

## Setup and run

```bash
pip install -r requirements.txt
```

Python `>=3.11` is required (`pyproject.toml`). Config: `configs/gateway.yaml`;
copy `.env.example` to `.env` for local overrides (`GATEWAY_HOST`,
`GATEWAY_PORT`, `GATEWAY_LOG_LEVEL`, `GATEWAY_CONFIG_PATH`,
`GATEWAY_DEFAULT_PROVIDER`, …). Secrets belong in env, never in YAML.

```bash
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Endpoints: `GET /health` → `{status: "ok", version, providers}`;
`GET /ready` → `{ready, providers}` (ready only when every provider is
healthy); `POST /v1/chat/completions` with e.g.
`{"model":"mock-a","messages":[{"role":"user","content":"hello"}]}` →
`choices`, `usage`, `provider`, `request_id`, `latency_ms`, `cached: false`
plus `x-provider` / `x-gateway-latency-ms` headers. Errors use
`{error: {code, message, provider, retryable, request_id}}`.

```bash
curl -N -X POST 127.0.0.1:8000/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{"model":"mock-a","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

## API key, tests, live demo

Real OpenRouter calls need a key in the terminal (never in YAML):

```bash
set OPENROUTER_API_KEY=sk-or-...        (Windows cmd)
$env:OPENROUTER_API_KEY='sk-or-...'     (PowerShell)
```

Run the offline suite (no network, no credentials):

```bash
python -m pytest
```

`ruff check` and `mypy --strict src` are the remaining checks
(`py -m ruff check .`, `py -m mypy --strict src`).

`live_openrouter_test.py` exercises the real gateway stack
(`RouterEngine` + `ResilientExecutor` + `ProxyClient`, as wired in
`src/main.py:create_app`) against OpenRouter — not the provider's `chat()`
directly. Phase 1 sends one request through `openrouter-primary` and prints
the router plan plus the per-attempt log; Phase 2 rebuilds the primary with an
invalid model slug (mimics a bad slug in YAML: OpenRouter answers 400/404 →
`PROVIDER_REJECTED` → automatic fallback) and asserts the request lands on
`openrouter-fallback`:

```bash
python live_openrouter_test.py
```

It needs network access to `https://openrouter.ai`. Note the configured models
are free-tier slugs: they are rate-limited (429 → `PROVIDER_RATE_LIMITED`,
which retries with backoff and then falls back) and sometimes flaky — that is
exactly what retry and fallback are for, so re-run for a clean primary hit
rather than treating a fallback as a failure.
