"""Sample: OpenAI adapter → OTLPHttpSink → OTel Collector.

Demonstrates how to wire the LayerLens Instrument layer to any OTLP/HTTP
backend — the OpenTelemetry Collector, Jaeger, Honeycomb, atlas-app's
``/v1/{traces,logs,metrics}`` endpoints, etc.

Differences from ``samples/instrument/openai/main.py``:

* Uses :class:`OTLPHttpSink` instead of :class:`HttpEventSink`.
* Targets a local Collector at ``http://localhost:4318`` (override via
  ``LAYERLENS_OTLP_ENDPOINT``).
* Emits three calls (one ``model.invoke``, one ``log.info``, one
  ``cost.record``) so all three OTLP signals (traces, logs, metrics)
  exercise the wire.
* Mocks the OpenAI client so the sample runs with no live API key.

Required environment:

* ``LAYERLENS_OTLP_ENDPOINT`` — base URL of your OTLP collector
  (default: ``http://localhost:4318``).
* ``LAYERLENS_OTLP_BEARER`` — optional bearer token for the collector.
* ``LAYERLENS_OTLP_API_KEY`` — optional ``X-API-Key`` for atlas-app.

Run::

    pip install 'layerlens[otlp,providers-openai]'
    python -m samples.instrument.otlp_collector.main
"""

from __future__ import annotations

import os
import sys
import time

from layerlens.instrument.transport.sink_otlp import OTLPHttpSink


def main() -> int:
    endpoint = os.environ.get("LAYERLENS_OTLP_ENDPOINT", "http://localhost:4318")
    bearer = os.environ.get("LAYERLENS_OTLP_BEARER") or None
    api_key = os.environ.get("LAYERLENS_OTLP_API_KEY") or None

    print(f"Wiring OTLPHttpSink to {endpoint}")
    sink = OTLPHttpSink(
        adapter_name="openai",
        service_name="layerlens-sample",
        service_version="0.1.0",
        endpoint=endpoint,
        bearer_token=bearer,
        api_key=api_key,
        max_batch=10,
        flush_interval_s=1.0,
        resource_attrs={
            "deployment.environment": "demo",
        },
    )

    # Emit 3 events covering all three OTLP signals so the collector
    # sees traces + logs + metrics from a single sink.
    now_ns = time.time_ns()
    sink.send(
        "model.invoke",
        {
            "model": "gpt-4o-mini",
            "prompt_tokens": 17,
            "completion_tokens": 4,
            "trace_id": "0123456789abcdef0123456789abcdef",
        },
        now_ns,
    )
    sink.send(
        "log.info",
        {"message": "Sample agent run completed", "step": 1},
        now_ns + 1_000_000,
    )
    sink.send(
        "cost.record",
        {"cost_usd": 0.000125, "tokens": 21},
        now_ns + 2_000_000,
    )

    sink.flush()
    sink.close()

    stats = sink.stats()
    print("OTLP sink stats after flush:")
    for key in sorted(stats.keys()):
        print(f"  {key:32s} {stats[key]}")

    if stats["batches_sent"] == 0:
        print(
            "WARNING: no batches were accepted by the collector. "
            "Is it running and listening on the configured endpoint?",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
