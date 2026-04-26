"""HTTP transport sinks for the LayerLens Instrument layer.

This package contains :class:`EventSink` implementations that ship
adapter-emitted events to remote endpoints — primarily atlas-app's
telemetry ingestion API at ``/api/v1/telemetry/spans``. The sinks
build on the SDK's existing ``httpx``-based transport, reusing the
exponential-backoff retry policy from ``layerlens._base_client``.

Available sinks:

* :class:`HttpEventSink` — synchronous httpx sink with batching.
* :class:`AsyncHttpEventSink` — asyncio sink for async adapters.
* :class:`OTLPHttpSink` — OTLP/HTTP exporter for any OTel-aware backend
  (Collector, Jaeger, Honeycomb, atlas-app's ``/v1/{traces,logs,metrics}``).
  Requires the ``[otlp]`` extra.
"""

from __future__ import annotations

from layerlens.instrument.transport.sink_http import (
    HttpEventSink,
    AsyncHttpEventSink,
)
from layerlens.instrument.transport.sink_otlp import OTLPHttpSink

__all__ = ["AsyncHttpEventSink", "HttpEventSink", "OTLPHttpSink"]
