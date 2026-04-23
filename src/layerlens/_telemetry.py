"""Opt-in client telemetry for the layerlens SDK.

Emits the same `atlas_sdk_events_total{surface, event}` and
`atlas_sdk_request_duration_seconds` shapes that the atlas-app server
side records (see `apps/shared/observability/metrics.go` in the
metrics-analytics-dashboard branch of LayerLens/atlas-app).

The contract:

* Telemetry is **off by default**. The customer must set the env var
  ``LAYERLENS_TELEMETRY=on`` (or pass ``telemetry_enabled=True``
  to ``Stratix(...)``) to opt in.
* When OFF, every helper here is a no-op — zero overhead, zero network.
* When ON, events are buffered and flushed via the OTel Collector
  endpoint at ``LAYERLENS_OTLP_ENDPOINT`` (default
  ``https://otel.layerlens.ai:4317``). If that's unreachable, events
  are silently dropped — telemetry MUST NOT block customer work.
* No PII is ever included. Events carry only the surface name
  (sdk_python / cli / vscode) and the event name (init / cmd_run /
  trace_emit / etc.). Customer payloads are not transmitted.

Usage from internal SDK code::

    from . import _telemetry
    _telemetry.event("sdk_python", "init")
    with _telemetry.timed("sdk_python", "trace_emit"):
        ...

Usage from the CLI::

    from layerlens._telemetry import event
    event("cli", "cmd_run", attributes={"command": "trace ls"})
"""
from __future__ import annotations

import contextlib
import os
import time
from typing import Any, Iterator, Mapping, Optional

# Module-level singletons populated lazily on first use.
_initialized: bool = False
_meter: Any = None
_counter_events: Any = None
_hist_request_duration: Any = None


def _enabled() -> bool:
    raw = os.environ.get("LAYERLENS_TELEMETRY", "").strip().lower()
    return raw in ("on", "true", "1", "yes")


def _try_init() -> bool:
    """Lazily build OTel meter + instruments. Returns True on success.

    Failures (missing OTel SDK, network) silently disable telemetry for
    the lifetime of the process — never raise into customer code.
    """
    global _initialized, _meter, _counter_events, _hist_request_duration
    if _initialized:
        return _meter is not None
    _initialized = True

    if not _enabled():
        return False

    try:
        import socket
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
    except ImportError:
        # OTel SDK not present — silent no-op.
        return False

    try:
        from ._version import __version__
    except ImportError:
        __version__ = "unknown"

    endpoint = os.environ.get(
        "LAYERLENS_OTLP_ENDPOINT", "https://otel.layerlens.ai:4317"
    )
    insecure = (
        os.environ.get("LAYERLENS_OTLP_INSECURE", "false").lower() == "true"
    )

    resource = Resource.create({
        "service.name": "atlas-sdk-python",
        "service.version": __version__,
        "host.name": socket.gethostname() or "unknown",
    })

    try:
        exporter = OTLPMetricExporter(endpoint=endpoint, insecure=insecure)
        reader = PeriodicExportingMetricReader(
            exporter, export_interval_millis=30_000
        )
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(provider)
        _meter = metrics.get_meter("layerlens.sdk")
        _counter_events = _meter.create_counter(
            "atlas_sdk_events_total",
            description=(
                "SDK + CLI + IDE events emitted by layerlens; "
                "surface=sdk_python|cli|vscode|web, event=init|cmd_run|trace_emit|..."
            ),
        )
        _hist_request_duration = _meter.create_histogram(
            "atlas_sdk_request_duration_seconds",
            description="HTTP request duration emitted by layerlens SDK clients.",
            unit="s",
        )
    except Exception:
        # Any failure during init means telemetry is disabled for this process.
        _meter = None
        _counter_events = None
        _hist_request_duration = None
        return False

    return True


def event(
    surface: str,
    event_name: str,
    *,
    attributes: Optional[Mapping[str, str]] = None,
) -> None:
    """Increment ``atlas_sdk_events_total{surface, event}`` by 1.

    No-op when telemetry is disabled or OTel SDK is absent.
    """
    if not _try_init() or _counter_events is None:
        return
    attrs: dict[str, Any] = {"surface": surface, "event": event_name}
    if attributes:
        # Only allow a small allowlist of attribute keys — no PII.
        allow = {"command", "resource", "outcome", "status_code"}
        for k, v in attributes.items():
            if k in allow and isinstance(v, (str, int, bool, float)):
                attrs[k] = str(v)
    try:
        _counter_events.add(1, attributes=attrs)
    except Exception:
        # Never raise into customer code.
        pass


@contextlib.contextmanager
def timed(surface: str, event_name: str) -> Iterator[None]:
    """Context manager: emit one ``event`` on entry/exit and observe duration."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        event(surface, event_name)
        if _hist_request_duration is not None:
            try:
                _hist_request_duration.record(
                    elapsed,
                    attributes={"surface": surface, "event": event_name},
                )
            except Exception:
                pass


def shutdown() -> None:
    """Flush + shutdown any active providers. CLI calls this in atexit."""
    try:
        from opentelemetry import metrics
        provider = metrics.get_meter_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
    except Exception:
        pass
