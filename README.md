# LLM Gateway (M3: streaming offline gateway)

M1 foundation plus M2 resilience plus M3 SSE streaming: FastAPI ingress, static-priority routing over mock providers, normalized errors, `x-request-id` propagation, offline-friendly (`configs/gateway.yaml` + env overrides), bounded retries with backoff/jitter, per-provider circuit breaker, passive health tracking, fallback across providers, and OpenAI-style streaming chunks. No external LLM calls.

See `docs/routing.md` for the retry / breaker / fallback design. See `docs/streaming.md` for the SSE streaming design.

## Install (Python 3.11)

```bash
py -V:3.11 -m pip install -r requirements.txt
```

## Run

```bash
py -V:3.11 -m uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Config: `configs/gateway.yaml`. Copy `.env.example` to `.env` for local overrides (`GATEWAY_HOST`, `GATEWAY_PORT`, `GATEWAY_LOG_LEVEL`, `GATEWAY_CONFIG_PATH`, `GATEWAY_DEFAULT_PROVIDER`). Secrets belong in env, never in YAML.

## Endpoints

- `GET /health` → `{status, version, providers}`. Header: `x-request-id`.
- `GET /ready` → `{ready, providers}`. Header: `x-request-id`.
- `POST /v1/chat/completions` with `stream=false`, e.g. `{"model":"mock-a","messages":[{"role":"user","content":"hello"}]}` → OpenAI-compatible subset response (`choices`, `usage`, `provider`, `request_id`, `latency_ms`, `cached:false`, `ttft_ms:null`, `itl_ms:null`). Headers: `x-request-id` (echoes inbound value or mints one), `x-provider`, `x-gateway-latency-ms`. Errors use `{error: {code, message, provider, retryable, request_id}}`.
- `POST /v1/chat/completions` with `stream=true` → `text/event-stream` SSE with OpenAI-style `chat.completion.chunk` deltas (`role:assistant` on first chunk only, `finish_reason:stop|length` on terminal chunk), then `data: [DONE]`. Mid-stream failures emit `data: {"error": {...}}` then close (no `[DONE]`, no fallback). Headers: `x-request-id`, `x-provider`, `cache-control:no-cache`. Details: `docs/streaming.md`.

```bash
curl -N -X POST 127.0.0.1:8000/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{"model":"mock-a","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

## Checks

```bash
py -V:3.11 -m pytest -q
py -V:3.11 -m ruff check .
py -V:3.11 -m mypy --strict src
```

## Resilience (M2)

Primary failure falls back to the next healthy provider (`x-provider` names the final provider). Tuning lives in the `resilience:` block of `configs/gateway.yaml` (`GATEWAY_RETRY_*` / `GATEWAY_CIRCUIT_*` env vars win). Details: `docs/routing.md`.

## Not yet implemented

Scoring/adaptive routing, active health probing. M4: caching (exact/semantic). M5: real OpenAI/Anthropic adapters, pricing. Stubs raise `NotImplementedError` or return 422.
