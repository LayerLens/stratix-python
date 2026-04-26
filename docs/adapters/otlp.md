# OTLP/HTTP Sink (`OTLPHttpSink`)

`OTLPHttpSink` is an `EventSink` that ships LayerLens Instrument
telemetry to any OTLP/HTTP-compatible backend — the OpenTelemetry
Collector, Jaeger, Honeycomb, Datadog, Grafana Tempo / Loki / Mimir,
New Relic, or atlas-app's `/v1/{traces,logs,metrics}` endpoints.

It mirrors the same contract as `HttpEventSink` (sync, batched, retried,
daemon-flushed, `stats()`-instrumented) but emits OTLP protobuf or JSON
on the wire instead of LayerLens's native span shape.

## When to use it

| Sink | Use when… |
|------|-----------|
| `HttpEventSink` | Shipping directly to atlas-app's `/api/v1/telemetry/spans`. Smallest payload, native LayerLens schema, control-plane features (replay, attestation chain, Kraken evaluations) automatically apply. |
| `OTLPHttpSink` | Shipping to an OTel Collector you already operate, or a third-party OTLP-aware backend (Jaeger, Honeycomb, etc.), or atlas-app's OTLP-compatible endpoints (`/v1/traces`, `/v1/logs`, `/v1/metrics`). |

If you already have an OTel Collector deployed, `OTLPHttpSink` lets
`layerlens` slot into your existing telemetry pipeline with no
infrastructure change. The Collector can then forward to atlas-app, your
existing observability vendor, or both.

## Install

```bash
pip install 'layerlens[otlp]'
```

The `[otlp]` extra adds a single dependency: `opentelemetry-proto>=1.27`.
The sink hand-rolls protobuf marshalling against the proto definitions —
it does **not** require `opentelemetry-sdk` or any provider, so default
install size stays small.

## Quickstart — local OTel Collector

```python
from layerlens.instrument.transport.sink_otlp import OTLPHttpSink
from layerlens.instrument.adapters.providers.openai_adapter import OpenAIAdapter
from layerlens.instrument.adapters._base import CaptureConfig

sink = OTLPHttpSink(
    adapter_name="openai",
    service_name="my-agent-service",
    service_version="1.4.2",
    endpoint="http://localhost:4318",  # base URL only
)

adapter = OpenAIAdapter(capture_config=CaptureConfig.standard())
adapter.add_sink(sink)
adapter.connect()
# ... run your workload ...
sink.close()
```

The sink derives per-signal endpoints automatically:

* Traces  → `http://localhost:4318/v1/traces`
* Logs    → `http://localhost:4318/v1/logs`
* Metrics → `http://localhost:4318/v1/metrics`

These match the [OTLP/HTTP spec](https://opentelemetry.io/docs/specs/otlp/#otlphttp).

## Quickstart — atlas-app OTLP endpoints

atlas-app exposes OTLP-compatible endpoints at `/v1/{traces,logs,metrics}`
that accept the standard `application/x-protobuf` payload **plus** a
LayerLens API key in `X-API-Key`:

```python
sink = OTLPHttpSink(
    adapter_name="openai",
    service_name="my-agent-service",
    endpoint="https://api.layerlens.ai",
    api_key=os.environ["LAYERLENS_STRATIX_API_KEY"],
)
```

Same wire format, same retry semantics — just a different endpoint and
the `X-API-Key` header.

## Quickstart — Honeycomb (or any third-party OTLP backend)

```python
sink = OTLPHttpSink(
    adapter_name="openai",
    service_name="my-agent-service",
    endpoint="https://api.honeycomb.io",
    bearer_token=os.environ["HONEYCOMB_API_KEY"],
    headers={"x-honeycomb-team": os.environ["HONEYCOMB_TEAM"]},
)
```

`headers=` adds vendor-specific headers on top of the standard auth
headers; the sink merges them all on every POST.

## Endpoint configuration

Three options, in order of precedence:

1. **Per-signal full URL** (highest precedence):

   ```python
   OTLPHttpSink(
       adapter_name="openai",
       traces_endpoint="https://traces.example.com/api/spans",
       logs_endpoint="https://logs.example.com/api/logs",
       metrics_endpoint="https://metrics.example.com/api/metrics",
   )
   ```

2. **Single base URL** (most common):

   ```python
   OTLPHttpSink(adapter_name="openai", endpoint="http://collector:4318")
   # → http://collector:4318/v1/traces
   # → http://collector:4318/v1/logs
   # → http://collector:4318/v1/metrics
   ```

3. **Environment variable** `LAYERLENS_OTLP_ENDPOINT`:

   ```bash
   export LAYERLENS_OTLP_ENDPOINT=http://collector:4318
   ```

## Content-Type

Two wire formats are supported:

| Content-Type | Default? | When to use |
|--------------|----------|-------------|
| `application/x-protobuf` | ✅ | Default. Smallest payload, what every OTLP receiver speaks natively. |
| `application/json` | | Easier to inspect during pipeline debugging; slower and ~5x larger. |

```python
from layerlens.instrument.transport.sink_otlp import (
    CONTENT_TYPE_JSON,
    OTLPHttpSink,
)

sink = OTLPHttpSink(
    adapter_name="openai",
    endpoint="http://collector:4318",
    content_type=CONTENT_TYPE_JSON,  # human-readable wire bytes
)
```

## Event → OTLP signal mapping

| Event type | OTLP signal | Notes |
|------------|-------------|-------|
| `model.invoke`, `model.response`, `tool.call`, `tool.result`, `agent.act`, `agent.start`, `agent.end`, `span.*` | **Traces** | One `Span` per event. Payload keys become span attributes (prefixed `layerlens.*` for adapter / event-type metadata). |
| `cost.record`, `metric.*` | **Metrics** | One monotonic `Sum` data point per *numeric* payload key. The metric name is `layerlens.<event_type>.<key>` (e.g. `layerlens.cost.record.cost_usd`). String fields are dropped silently — they're not metrics. |
| `log.<level>` (e.g. `log.info`, `log.error`) | **Logs** | One `LogRecord` per event. `payload["message"]` becomes the log body; severity is derived from the `<level>` suffix (`info`, `warn`, `error`, `fatal`, etc.). |
| Anything else | **Traces** (default) | Unknown events route to traces so nothing is silently dropped. |

Every export carries the standard OTel resource attributes:

* `service.name` — defaults to `adapter_name`, override with `service_name=`.
* `service.version` — defaults to the SDK version, override with `service_version=`.
* `telemetry.sdk.name = "layerlens"`, `telemetry.sdk.language = "python"`, `telemetry.sdk.version = <SDK version>`.

Pass `resource_attrs={"deployment.environment": "prod", "k8s.pod.name": "..."}`
to merge in your own resource attributes.

## Retry, backoff, and durability

`OTLPHttpSink` matches `HttpEventSink` exactly:

* On `429` / `500` / `502` / `503` / `504`: retry up to 2 times with
  exponential backoff (0.5s → 1s → 2s, capped at 8s).
* `Retry-After` header is honored when present.
* On `4xx` (except `429`): drop the batch — the body is malformed, no
  amount of retrying will fix it.
* After 3 consecutive batch drops on a given signal, the sink logs at
  `WARN` once with code `layerlens.sink.batch_dropped` so log-based
  alerting can pick it up. Subsequent drops in the same window stay at
  `DEBUG`.

The daemon flush thread (configured via `flush_interval_s`, default
1.0s) ensures sporadic adapters don't leave events buffered until
process exit. Set `background_flush=False` for deterministic
single-thread tests.

## `stats()` schema

`OTLPHttpSink.stats()` returns per-signal counters plus totals:

```python
{
    "batches_sent_traces":    int,
    "batches_sent_logs":      int,
    "batches_sent_metrics":   int,
    "batches_dropped_traces": int,
    "batches_dropped_logs":   int,
    "batches_dropped_metrics": int,
    "buffer_size_traces":     int,
    "buffer_size_logs":       int,
    "buffer_size_metrics":    int,
    "consecutive_drops_traces":  int,
    "consecutive_drops_logs":    int,
    "consecutive_drops_metrics": int,
    "batches_sent":      int,  # total across all signals
    "batches_dropped":   int,
    "buffer_size":       int,
    "consecutive_drops": int,  # max across signals
}
```

Surface these in your service health endpoints to alert on telemetry
pipeline degradation.

## Sample

End-to-end runnable sample:
[`samples/instrument/otlp_collector/main.py`](../../samples/instrument/otlp_collector/main.py).
