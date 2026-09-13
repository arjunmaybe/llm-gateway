# Routing & Resilience (M2)

How the gateway picks providers and survives their failures. M1 defined the
static-priority router; M2 adds failure classification, bounded retry, a
per-provider circuit breaker, passive health tracking, and fallback across
the candidate chain. The public API is unchanged.

## Pipeline

`RouterEngine.plan()` returns ordered candidate names (model alias first,
then `priority` order) → `ResilientExecutor` gates each candidate on health
and breaker state, attempts it via `ProxyClient.forward()`, retries
retryable failures with backoff, and falls back to the next candidate.
The router decides *which* providers may be tried; the executor handles
*how*; provider I/O stays in `ProxyClient` / `ProviderAdapter`.

## Failure taxonomy (`src/resilience/failures.py`)

Classification runs on the normalized `GatewayError.code`, never on raw
provider output:

| Category | Example codes | Retry? | Breaker? | Health? | Fallback? |
|---|---|---|---|---|---|
| `timeout` | `UPSTREAM_TIMEOUT`, `PROVIDER_TIMEOUT` (504) | yes | counts | no | yes |
| `connection` | `PROVIDER_CONNECTION_FAILED` (502) | yes | counts | no | yes |
| `rate_limited` | `PROVIDER_RATE_LIMITED` (429) | yes | counts, equal weight | no | yes |
| `upstream_transient` | `PROVIDER_UNAVAILABLE` (502) | yes | counts | no | yes |
| `permanent` | `PROVIDER_REJECTED` (502, non-retryable) | no | no | marks unhealthy | yes, next provider |
| `client` | `INVALID_REQUEST` (422) | no | no | no | no |
| `no_capacity` | `NO_HEALTHY_PROVIDER` (503) | n/a | no | no | n/a (terminal) |
| `internal` | `UNKNOWN_PROVIDER`, `INTERNAL` (500) | no | no | no | no (fail fast) |

Unknown codes classify as `internal` (fail closed). Only `permanent`
failures mark a provider unhealthy; transient failures feed the breaker
only, so one timeout never pins a provider out of rotation.

## Retry policy (`src/resilience/retry.py`)

- `max_attempts` (default 2): total attempts **per provider**, first attempt
  counts (default = try + one retry). Cross-provider fallback, not deep
  per-provider retry, is the primary mechanism.
- Backoff: `min(backoff_base_ms * 2^(n-1), backoff_max_ms)` (defaults
  50ms → cap 1000ms) with **full jitter** (`uniform(0, backoff)`).
- Budget: `max_elapsed_ms` (default 8000) caps total backoff sleeping per
  request — the anti-retry-storm guard together with jitter and the attempt
  cap. Per-attempt `timeout_s` still bounds each call.
- `request_id` is unchanged across attempts (only the per-attempt
  `provider` field is re-scoped). Sleeper and jitter source are injectable
  so tests never really wait.

## Circuit breaker (`src/router/circuit_breaker.py`)

Per-provider `CLOSED → OPEN → HALF_OPEN → CLOSED` state machine with
consecutive-failure counting (no EWMA by design — deterministic and
sufficient at this scale):

- `failure_threshold` (default 5) consecutive retryable failures open the
  circuit; any success resets the counter.
- `recovery_timeout_s` (default 30.0) after opening, the next `can_execute`
  transitions to `HALF_OPEN` (lazy — no background timers).
- `half_open_max_inflight` (default 1) bounds concurrent recovery probes;
  extra callers skip to fallback. Probe success closes; probe failure
  reopens with a fresh timestamp (stray reports while `OPEN` never extend
  the window).
- All methods are synchronous with no awaits, hence atomic on the event
  loop — no locks required. The monotonic clock is injectable for tests.

## Provider health (`src/router/health.py`)

M2 is **passive only**: success → `mark_healthy`; `permanent` failure →
`mark_unhealthy`; transient failures touch the breaker, not the registry.
There are no background tasks; `ProviderAdapter.health_check()` remains the
seam a future active prober will call. Known limitation: an unhealthy mark
sticks until an external signal re-marks healthy (operator, config reload,
future prober) — documented, not yet implemented.

## Fallback chain

- Healthy primary succeeds → response, `fallback_used=False`.
- Primary fails retryably → retry per policy → next candidate.
- Primary circuit-open/unhealthy → skipped directly to fallback.
- `permanent` failure → no retry, immediate fallback.
- All candidates exhausted → the **last** normalized error is raised
  (code/status/provider preserved, e.g. 502 `PROVIDER_UNAVAILABLE`);
  nothing executable at all → 503 `NO_HEALTHY_PROVIDER`.
- `x-provider` always names the final successful provider; error bodies
  stay in the normalized `{error: {code, message, provider, retryable,
  request_id}}` envelope with no raw upstream payloads.

## Concurrency behavior

Single event loop: breaker transitions and health writes are synchronous
and atomic. Half-open herd bounded by the probe permit count. Mock fault
scripts use one `asyncio.Lock`-guarded invocation counter. The executor
holds no shared mutable state per request.

## Execution metadata

Success returns `ExecuteResult(response, outcome)` with `ExecutionOutcome`:
`request_id`, `primary`, `final_provider`, per-attempt `AttemptRecord`s
(provider, 1-based attempt index, outcome, failure category, latency),
`retry_count`, `fallback_used`, per-candidate `circuit_states`, and total
duration. The outcome is attached to `request.state.execution_outcome` for
logs/future telemetry. M2 emits counter/latency signals through the
existing metrics interface (`gateway_retries_total`,
`gateway_fallbacks_total`, `gateway_breaker_opens_total`,
`gateway_provider_attempts_total`) and per-attempt tracer span attributes.
No Prometheus/OpenTelemetry SDK. No new client response fields.

## Pre-first-byte limitation

Retry and fallback complete **before** response serialization. Once
response bytes reach the client, transparent provider replacement is
impossible — M2 never attempts it, and M3 streaming must only use
single-attempt execution after the first byte.

## Configuration

```yaml
resilience:
  retry: { max_attempts: 2, backoff_base_ms: 50.0, backoff_max_ms: 1000.0, max_elapsed_ms: 8000.0 }
  circuit_breaker: { failure_threshold: 5, recovery_timeout_s: 30.0, half_open_max_inflight: 1 }
```

`GATEWAY_RETRY_MAX_ATTEMPTS`, `GATEWAY_RETRY_BACKOFF_BASE_MS`,
`GATEWAY_RETRY_BACKOFF_MAX_MS`, `GATEWAY_RETRY_MAX_ELAPSED_MS`,
`GATEWAY_CIRCUIT_FAILURE_THRESHOLD`, `GATEWAY_CIRCUIT_RECOVERY_TIMEOUT_S`,
`GATEWAY_CIRCUIT_HALF_OPEN_INFLIGHT` override YAML. Priority, `timeout_s`,
and `enabled` keep their M1 meaning (fallback order, per-attempt timeout,
chain membership).
