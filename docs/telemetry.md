# Client telemetry — opt-in

The `layerlens` SDK can optionally emit a small set of usage counters back
to LayerLens so we can compute SDK adoption + diagnose breakages. **It is
off by default and never sends customer payloads.**

## Quick start

Enable per-process:

```bash
export LAYERLENS_TELEMETRY=on
python my_agent.py
```

Or in code:

```python
import os
os.environ["LAYERLENS_TELEMETRY"] = "on"
from layerlens import Stratix
client = Stratix(api_key="...")  # emits one `init` event
```

Disable (default — no telemetry, no network calls):

```bash
unset LAYERLENS_TELEMETRY
```

## What is sent

When telemetry is on, the SDK emits the following metrics over OTLP/gRPC
to the LayerLens collector (`https://otel.layerlens.ai:4317`):

| Metric | Type | Labels |
|---|---|---|
| `atlas_sdk_events_total` | counter | `surface`, `event` (and optionally `command`, `outcome`, `status_code`, `resource`) |
| `atlas_sdk_request_duration_seconds` | histogram | `surface`, `event` |

Concrete events emitted by the SDK today:

| Where | `surface` | `event` |
|---|---|---|
| `Stratix(...)` constructor | `sdk_python` | `init` |
| `layerlens` CLI top-level invocation | `cli` | `cmd_run` (with `command` attribute = subcommand name) |

## What is NOT sent

- Your API key, your prompts, your traces, your evaluation data — none of
  this is touched by the telemetry path.
- Any attribute key not on the allowlist (`command`, `resource`,
  `outcome`, `status_code`) is silently dropped. PII never leaves the
  client.

## How it fails

The SDK treats telemetry as best-effort. If any of these happen, the SDK
silently disables telemetry for the rest of the process and continues
serving customer requests:

- The OpenTelemetry SDK is not installed (it's not a hard dep).
- The collector endpoint is unreachable.
- The exporter raises during init or send.

In other words: **enabling telemetry can never break your application.**

## Configuration

| Env var | Purpose | Default |
|---|---|---|
| `LAYERLENS_TELEMETRY` | Master switch (`on` / `true` / `1` / `yes` to enable) | unset (off) |
| `LAYERLENS_OTLP_ENDPOINT` | Collector endpoint | `https://otel.layerlens.ai:4317` |
| `LAYERLENS_OTLP_INSECURE` | Use plaintext gRPC (for local dev) | `false` |

## How it relates to the atlas-app server-side

This counter is the SDK-side mirror of the `atlas_sdk_events_total`
metric the atlas-app backend exports
(`apps/shared/observability/metrics.go`). Both surface to the same
metric name in Grafana so SDK-emitted and server-emitted events line up
on the same dashboards.
