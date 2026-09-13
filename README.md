# LLM Gateway (M1: minimal offline gateway)

M1 scope only: FastAPI ingress, static-priority routing over mock providers, normalized errors, `x-request-id` propagation, offline-friendly (`configs/gateway.yaml` + env overrides). No external LLM calls.

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

## Not in M1

M2: scoring/adaptive routing, active health probing, real circuit breaker. M3: SSE streaming. M4: caching (exact/semantic). M5: real OpenAI/Anthropic adapters, pricing. Stubs raise `NotImplementedError` or return 422.
