# LLM Gateway (M2: resilient offline gateway)

M1 foundation plus M2 resilience: FastAPI ingress, static-priority routing over mock providers, normalized errors, `x-request-id` propagation, offline-friendly (`configs/gateway.yaml` + env overrides), bounded retries with backoff/jitter, per-provider circuit breaker, passive health tracking, and fallback across providers. No external LLM calls.

See `docs/routing.md` for the retry / breaker / fallback design.

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
- `POST /v1/chat/completions` with `stream=false`, e.g. `{"model":"mock-a","messages":[{"role":"user","content":"hello"}]}` → OpenAI-compatible subset response (`choices`, `usage`, `provider`, `request_id`, `latency_ms`, `cached:false`, `ttft_ms:null`, `itl_ms:null`). Headers: `x-request-id` (echoes inbound value or mints one), `x-provider`, `x-gateway-latency-ms`. `stream=true` is rejected (422, planned M3). Errors use `{error: {code, message, provider, retryable, request_id}}`.

## Checks

```bash
py -V:3.11 -m pytest -q
py -V:3.11 -m ruff check .
py -V:3.11 -m mypy --strict src
```

## Resilience (M2)

Primary failure falls back to the next healthy provider (`x-provider` names the final provider). Tuning lives in the `resilience:` block of `configs/gateway.yaml` (`GATEWAY_RETRY_*` / `GATEWAY_CIRCUIT_*` env vars win). Details: `docs/routing.md`.

## Not yet implemented

Scoring/adaptive routing, active health probing. M3: SSE streaming. M4: caching (exact/semantic). M5: real OpenAI/Anthropic adapters, pricing. Stubs raise `NotImplementedError` or return 422.
