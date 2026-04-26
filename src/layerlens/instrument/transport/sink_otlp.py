"""OTLP HTTP transport sink.

Bridges the LayerLens Instrument layer to any OTLP/HTTP-aware backend —
the OpenTelemetry Collector, Jaeger, Honeycomb, Datadog, Grafana Tempo,
New Relic, atlas-app's own ``/v1/{traces,logs,metrics}`` endpoints, etc.

This sink mirrors the contract of :class:`HttpEventSink`:

* Synchronous ``send`` / ``flush`` / ``close``.
* Per-signal batching with size + time thresholds.
* Exponential backoff (0.5s → 8s) on 429 / 5xx, honors ``Retry-After``.
* WARN-after-3-consecutive-drops with code ``layerlens.sink.batch_dropped``.
* Daemon timer thread bounds idle-flush latency.
* :meth:`stats` exposes per-signal ``batches_sent`` / ``batches_dropped``.

Event-to-OTLP translation:

* ``model.invoke`` / ``tool.call`` / ``agent.act`` → OTLP **traces**
  (one ``Span`` per event, attributes lifted from payload).
* ``cost.record`` → OTLP **metrics** (one ``Sum`` data point per record).
* ``log.*`` (any event whose type starts with ``log.``) → OTLP **logs**
  (one ``LogRecord`` per event).
* All other events default to traces so nothing is silently dropped.

Endpoint resolution (per signal):

1. Explicit per-signal kwarg (``traces_endpoint`` / ``logs_endpoint`` /
   ``metrics_endpoint``) wins.
2. Otherwise ``{base_url}/v1/traces``, ``{base_url}/v1/logs``,
   ``{base_url}/v1/metrics`` — matches the OTel spec for OTLP/HTTP.

Auth: emits both ``Authorization: Bearer <token>`` and ``X-API-Key`` when
the corresponding kwargs are set, so the same sink works against the
public OTLP collectors AND atlas-app's signed ingest endpoints.

Content-Type:

* ``application/x-protobuf`` (default) — wire-efficient, what every OTLP
  receiver speaks natively.
* ``application/json`` — handy for human inspection of pipeline issues.

Dependencies: requires the optional ``[otlp]`` extra
(``opentelemetry-proto>=1.27``). The sink is a hand-rolled HTTP POST of
``ExportTraceServiceRequest`` / ``ExportLogsServiceRequest`` /
``ExportMetricsServiceRequest`` protobuf messages, so it does NOT depend
on ``opentelemetry-sdk`` or any provider — installing the SDK is optional.
"""

from __future__ import annotations

import os
import json
import time
import logging
import threading
from typing import Any, Dict, List, Tuple, Union, Mapping, Optional, Sequence
from urllib.parse import urlparse

import httpx

from layerlens._version import __version__
from layerlens.instrument.adapters._base.sinks import EventSink

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables (mirror sink_http.py)
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL = os.environ.get(
    "LAYERLENS_OTLP_ENDPOINT",
    "http://localhost:4318",
)
_DEFAULT_TIMEOUT_S = 10.0
_DEFAULT_MAX_BATCH = 50
_DEFAULT_FLUSH_INTERVAL_S = 1.0

_MAX_RETRIES = 2
_INITIAL_RETRY_DELAY_S = 0.5
_MAX_RETRY_DELAY_S = 8.0
_RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

_CONSECUTIVE_DROP_WARN_THRESHOLD = 3
_DROP_WARN_LOG_CODE = "layerlens.sink.batch_dropped"

CONTENT_TYPE_PROTOBUF = "application/x-protobuf"
CONTENT_TYPE_JSON = "application/json"


# Event-type → OTLP signal classifier.
_TRACE_EVENT_PREFIXES: Tuple[str, ...] = (
    "model.invoke",
    "model.response",
    "tool.call",
    "tool.result",
    "agent.act",
    "agent.start",
    "agent.end",
    "span.",
)
_METRIC_EVENT_PREFIXES: Tuple[str, ...] = ("cost.record", "metric.",)
_LOG_EVENT_PREFIXES: Tuple[str, ...] = ("log.",)


def _classify_event(event_type: str) -> str:
    """Return ``'traces'``, ``'logs'``, or ``'metrics'`` for an event type."""
    if any(event_type.startswith(p) for p in _LOG_EVENT_PREFIXES):
        return "logs"
    if any(event_type.startswith(p) for p in _METRIC_EVENT_PREFIXES):
        return "metrics"
    if any(event_type.startswith(p) for p in _TRACE_EVENT_PREFIXES):
        return "traces"
    # Default unknown events to traces — never silently drop.
    return "traces"


# ---------------------------------------------------------------------------
# Protobuf marshalling — hand-rolled via opentelemetry-proto
# ---------------------------------------------------------------------------


def _import_proto() -> Dict[str, Any]:
    """Return the proto modules we need, raising a clear ImportError if missing."""
    try:
        from opentelemetry.proto.logs.v1 import logs_pb2
        from opentelemetry.proto.trace.v1 import trace_pb2
        from opentelemetry.proto.common.v1 import common_pb2
        from opentelemetry.proto.metrics.v1 import metrics_pb2
        from opentelemetry.proto.resource.v1 import resource_pb2
        from opentelemetry.proto.collector.logs.v1 import logs_service_pb2
        from opentelemetry.proto.collector.trace.v1 import trace_service_pb2
        from opentelemetry.proto.collector.metrics.v1 import metrics_service_pb2
    except ImportError as exc:  # pragma: no cover - exercised in env without extra
        raise ImportError(
            "OTLPHttpSink requires the optional 'otlp' extra. "
            "Install with: pip install 'layerlens[otlp]'"
        ) from exc

    return {
        "common": common_pb2,
        "logs": logs_pb2,
        "trace": trace_pb2,
        "metrics": metrics_pb2,
        "resource": resource_pb2,
        "logs_service": logs_service_pb2,
        "trace_service": trace_service_pb2,
        "metrics_service": metrics_service_pb2,
    }


def _to_any_value(value: Any, common_pb2: Any) -> Any:
    """Coerce a Python value into an OTLP ``AnyValue``."""
    av = common_pb2.AnyValue()
    if value is None:
        return av  # empty AnyValue
    if isinstance(value, bool):
        av.bool_value = value
        return av
    if isinstance(value, int):
        av.int_value = value
        return av
    if isinstance(value, float):
        av.double_value = value
        return av
    if isinstance(value, str):
        av.string_value = value
        return av
    if isinstance(value, (list, tuple)):
        for item in value:
            av.array_value.values.append(_to_any_value(item, common_pb2))
        return av
    if isinstance(value, dict):
        for k, v in value.items():
            kv = av.kvlist_value.values.add()
            kv.key = str(k)
            kv.value.CopyFrom(_to_any_value(v, common_pb2))
        return av
    # Fallback: stringify.
    av.string_value = repr(value)
    return av


def _attributes(
    items: Mapping[str, Any],
    common_pb2: Any,
) -> List[Any]:
    """Build a list of ``KeyValue`` from a Python mapping."""
    out: List[Any] = []
    for key, value in items.items():
        kv = common_pb2.KeyValue()
        kv.key = key
        kv.value.CopyFrom(_to_any_value(value, common_pb2))
        out.append(kv)
    return out


def _resource(
    service_name: str,
    service_version: str,
    extra: Optional[Mapping[str, Any]],
    proto: Mapping[str, Any],
) -> Any:
    res = proto["resource"].Resource()
    base: Dict[str, Any] = {
        "service.name": service_name,
        "service.version": service_version,
        "telemetry.sdk.language": "python",
        "telemetry.sdk.name": "layerlens",
        "telemetry.sdk.version": __version__,
    }
    if extra:
        for k, v in extra.items():
            base[k] = v
    res.attributes.extend(_attributes(base, proto["common"]))
    return res


def _trace_id_bytes(trace_id_hex: Optional[str], counter: int) -> bytes:
    """16 bytes for an OTLP trace ID; derive from hex string or counter."""
    if trace_id_hex:
        try:
            raw = bytes.fromhex(trace_id_hex.replace("-", "").rjust(32, "0")[:32])
            if len(raw) == 16:
                return raw
        except ValueError:
            pass
    # Deterministic synthetic trace id from counter (8 zero bytes + 8-byte counter).
    return b"\x00" * 8 + counter.to_bytes(8, "big", signed=False)


def _span_id_bytes(counter: int) -> bytes:
    """8 bytes for an OTLP span ID derived from the in-batch counter."""
    return counter.to_bytes(8, "big", signed=False)


def _build_traces_request(
    events: Sequence[Dict[str, Any]],
    service_name: str,
    service_version: str,
    resource_attrs: Optional[Mapping[str, Any]],
    proto: Mapping[str, Any],
) -> Any:
    """Compose an ``ExportTraceServiceRequest`` from buffered events."""
    request = proto["trace_service"].ExportTraceServiceRequest()
    rs = request.resource_spans.add()
    rs.resource.CopyFrom(
        _resource(service_name, service_version, resource_attrs, proto)
    )
    ss = rs.scope_spans.add()
    ss.scope.name = "layerlens.instrument"
    ss.scope.version = __version__

    for idx, evt in enumerate(events, start=1):
        span = ss.spans.add()
        span.trace_id = _trace_id_bytes(evt.get("trace_id"), idx)
        span.span_id = _span_id_bytes(idx)
        span.name = str(evt.get("event_type", "unknown"))
        ts_ns = int(evt.get("timestamp_ns", time.time_ns()))
        span.start_time_unix_nano = ts_ns
        end_ns = evt.get("end_time_unix_nano")
        span.end_time_unix_nano = int(end_ns) if end_ns is not None else ts_ns
        # Span kind: INTERNAL by default — adapter callers can override
        # via payload["span.kind"] in the future.
        span.kind = proto["trace"].Span.SpanKind.SPAN_KIND_INTERNAL

        flat_attrs: Dict[str, Any] = {
            "layerlens.adapter": evt.get("adapter", ""),
            "layerlens.event_type": evt.get("event_type", ""),
        }
        payload = evt.get("payload") or {}
        if isinstance(payload, dict):
            for k, v in payload.items():
                flat_attrs[str(k)] = v
        span.attributes.extend(_attributes(flat_attrs, proto["common"]))

    return request


def _build_logs_request(
    events: Sequence[Dict[str, Any]],
    service_name: str,
    service_version: str,
    resource_attrs: Optional[Mapping[str, Any]],
    proto: Mapping[str, Any],
) -> Any:
    """Compose an ``ExportLogsServiceRequest`` from buffered events."""
    request = proto["logs_service"].ExportLogsServiceRequest()
    rl = request.resource_logs.add()
    rl.resource.CopyFrom(
        _resource(service_name, service_version, resource_attrs, proto)
    )
    sl = rl.scope_logs.add()
    sl.scope.name = "layerlens.instrument"
    sl.scope.version = __version__

    for idx, evt in enumerate(events, start=1):
        rec = sl.log_records.add()
        ts_ns = int(evt.get("timestamp_ns", time.time_ns()))
        rec.time_unix_nano = ts_ns
        rec.observed_time_unix_nano = ts_ns
        # Map log.<level> → OTLP severity.
        evt_type = str(evt.get("event_type", "log.info"))
        level_token = evt_type.split(".", 1)[1] if "." in evt_type else "info"
        sev_number, sev_text = _severity_for_level(level_token, proto)
        rec.severity_number = sev_number
        rec.severity_text = sev_text

        payload = evt.get("payload") or {}
        body_str: Optional[str] = None
        attr_payload: Dict[str, Any] = {}
        if isinstance(payload, dict):
            body_str = str(payload.get("message", "")) if "message" in payload else None
            for k, v in payload.items():
                if k == "message":
                    continue
                attr_payload[str(k)] = v
        if body_str is None:
            body_str = json.dumps(payload, default=str) if payload else ""
        rec.body.CopyFrom(_to_any_value(body_str, proto["common"]))

        attrs = {
            "layerlens.adapter": evt.get("adapter", ""),
            "layerlens.event_type": evt_type,
        }
        attrs.update(attr_payload)
        rec.attributes.extend(_attributes(attrs, proto["common"]))
        rec.trace_id = _trace_id_bytes(evt.get("trace_id"), idx)
        rec.span_id = _span_id_bytes(idx)

    return request


def _severity_for_level(level: str, proto: Mapping[str, Any]) -> Tuple[int, str]:
    """Translate ``log.<level>`` into ``(severity_number, severity_text)``."""
    sev_enum = proto["logs"].SeverityNumber
    table: Dict[str, Tuple[int, str]] = {
        "trace": (sev_enum.SEVERITY_NUMBER_TRACE, "TRACE"),
        "debug": (sev_enum.SEVERITY_NUMBER_DEBUG, "DEBUG"),
        "info": (sev_enum.SEVERITY_NUMBER_INFO, "INFO"),
        "warn": (sev_enum.SEVERITY_NUMBER_WARN, "WARN"),
        "warning": (sev_enum.SEVERITY_NUMBER_WARN, "WARN"),
        "error": (sev_enum.SEVERITY_NUMBER_ERROR, "ERROR"),
        "fatal": (sev_enum.SEVERITY_NUMBER_FATAL, "FATAL"),
        "critical": (sev_enum.SEVERITY_NUMBER_FATAL, "FATAL"),
    }
    return table.get(level.lower(), (sev_enum.SEVERITY_NUMBER_INFO, "INFO"))


def _build_metrics_request(
    events: Sequence[Dict[str, Any]],
    service_name: str,
    service_version: str,
    resource_attrs: Optional[Mapping[str, Any]],
    proto: Mapping[str, Any],
) -> Any:
    """Compose an ``ExportMetricsServiceRequest`` from buffered events.

    Each ``cost.record`` event becomes a metric. Numeric scalar payload
    fields produce a ``Sum`` data point; the metric name is derived from
    the payload key (``cost_usd`` → ``layerlens.cost.cost_usd``).
    """
    request = proto["metrics_service"].ExportMetricsServiceRequest()
    rm = request.resource_metrics.add()
    rm.resource.CopyFrom(
        _resource(service_name, service_version, resource_attrs, proto)
    )
    sm = rm.scope_metrics.add()
    sm.scope.name = "layerlens.instrument"
    sm.scope.version = __version__

    aggregation_temporality_cumulative = proto[
        "metrics"
    ].AggregationTemporality.AGGREGATION_TEMPORALITY_CUMULATIVE

    for evt in events:
        ts_ns = int(evt.get("timestamp_ns", time.time_ns()))
        adapter = str(evt.get("adapter", ""))
        event_type = str(evt.get("event_type", ""))
        payload = evt.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            metric = sm.metrics.add()
            metric.name = f"layerlens.{event_type}.{key}".replace("..", ".")
            metric.unit = ""
            metric.description = f"{event_type} {key}"
            data_point = metric.sum.data_points.add()
            metric.sum.aggregation_temporality = aggregation_temporality_cumulative
            metric.sum.is_monotonic = True
            data_point.time_unix_nano = ts_ns
            data_point.start_time_unix_nano = ts_ns
            if isinstance(value, int):
                data_point.as_int = value
            else:
                data_point.as_double = float(value)
            data_point.attributes.extend(
                _attributes(
                    {
                        "layerlens.adapter": adapter,
                        "layerlens.event_type": event_type,
                    },
                    proto["common"],
                )
            )

    return request


def _serialize_request(request: Any, content_type: str) -> bytes:
    """Serialize an export-service-request proto to wire bytes."""
    if content_type == CONTENT_TYPE_JSON:
        try:
            from google.protobuf.json_format import MessageToJson  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "JSON content-type requires google.protobuf "
                "(pulled in by opentelemetry-proto)."
            ) from exc
        json_str: str = MessageToJson(request, preserving_proto_field_name=True)
        return json_str.encode("utf-8")
    raw: Any = request.SerializeToString()
    return raw if isinstance(raw, bytes) else bytes(raw)


# ---------------------------------------------------------------------------
# HTTP send-with-retry (mirrors sink_http.py)
# ---------------------------------------------------------------------------


def _post_with_retry(
    client: httpx.Client,
    url: str,
    body: bytes,
    headers: Mapping[str, str],
) -> bool:
    """POST ``body`` to ``url`` with backoff on 429 / 5xx (honors Retry-After)."""
    delay = _INITIAL_RETRY_DELAY_S
    retries_left = _MAX_RETRIES

    while True:
        try:
            resp = client.post(url, content=body, headers=dict(headers))
        except httpx.HTTPError as exc:
            if retries_left > 0:
                logger.debug(
                    "OTLPHttpSink transport error: %s (retries left: %d)",
                    exc,
                    retries_left,
                )
                time.sleep(delay)
                delay = min(delay * 2, _MAX_RETRY_DELAY_S)
                retries_left -= 1
                continue
            logger.debug("OTLPHttpSink giving up after transport errors", exc_info=True)
            return False

        if resp.status_code in _RETRY_STATUS_CODES and retries_left > 0:
            retry_after = resp.headers.get("retry-after")
            try:
                sleep_for = float(retry_after) if retry_after else delay
            except ValueError:
                sleep_for = delay
            sleep_for = min(sleep_for, _MAX_RETRY_DELAY_S)
            time.sleep(sleep_for)
            delay = min(delay * 2, _MAX_RETRY_DELAY_S)
            retries_left -= 1
            continue

        if 200 <= resp.status_code < 300:
            return True

        logger.debug(
            "OTLPHttpSink got non-retriable status %d body=%s",
            resp.status_code,
            resp.text[:500],
        )
        return False


# ---------------------------------------------------------------------------
# OTLPHttpSink
# ---------------------------------------------------------------------------


class OTLPHttpSink(EventSink):
    """Synchronous OTLP/HTTP sink for the LayerLens Instrument layer.

    Args:
        adapter_name: Tag inserted into every event so the receiver can
            attribute the source adapter. Required.
        service_name: ``service.name`` resource attribute (OTel spec).
            Defaults to ``adapter_name``.
        service_version: ``service.version`` resource attribute.
            Defaults to the SDK version.
        endpoint: Base OTLP endpoint, e.g. ``http://collector:4318``.
            Per-signal paths are appended as ``/v1/traces`` etc. Falls
            back to ``$LAYERLENS_OTLP_ENDPOINT`` or
            ``http://localhost:4318``.
        traces_endpoint: Full URL override for traces.
        logs_endpoint: Full URL override for logs.
        metrics_endpoint: Full URL override for metrics.
        bearer_token: Optional bearer token sent as
            ``Authorization: Bearer <token>``.
        api_key: Optional ``X-API-Key`` header value.
        headers: Extra static headers merged into every request.
        content_type: Wire format. Default ``application/x-protobuf``;
            ``application/json`` is supported for human-readable
            inspection.
        resource_attrs: Extra OTel resource attributes (e.g. ``deployment.environment``).
        max_batch: Max events per signal before forced flush.
        flush_interval_s: Wall-clock idle-flush interval.
        timeout_s: Per-request HTTP timeout.
        client: Optional pre-built ``httpx.Client``.
        background_flush: If True, spawn a daemon thread that flushes
            partial buffers every ``flush_interval_s``.

    Per-signal batching keeps each OTLP export self-contained: traces,
    logs, and metrics are buffered, flushed, and stat-tracked
    independently.
    """

    def __init__(
        self,
        adapter_name: str,
        *,
        service_name: Optional[str] = None,
        service_version: Optional[str] = None,
        endpoint: Optional[str] = None,
        traces_endpoint: Optional[str] = None,
        logs_endpoint: Optional[str] = None,
        metrics_endpoint: Optional[str] = None,
        bearer_token: Optional[str] = None,
        api_key: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
        content_type: str = CONTENT_TYPE_PROTOBUF,
        resource_attrs: Optional[Mapping[str, Any]] = None,
        max_batch: int = _DEFAULT_MAX_BATCH,
        flush_interval_s: float = _DEFAULT_FLUSH_INTERVAL_S,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        client: Optional[httpx.Client] = None,
        background_flush: bool = True,
    ) -> None:
        if content_type not in (CONTENT_TYPE_PROTOBUF, CONTENT_TYPE_JSON):
            raise ValueError(
                f"content_type must be {CONTENT_TYPE_PROTOBUF!r} or "
                f"{CONTENT_TYPE_JSON!r}, got {content_type!r}"
            )

        self._adapter_name = adapter_name
        self._service_name = service_name or adapter_name
        self._service_version = service_version or __version__
        self._content_type = content_type
        self._resource_attrs = dict(resource_attrs) if resource_attrs else None
        self._max_batch = max_batch
        self._flush_interval_s = flush_interval_s

        # Endpoint resolution.
        base = endpoint or _DEFAULT_BASE_URL
        self._endpoints: Dict[str, str] = {
            "traces": traces_endpoint or _join_signal_path(base, "traces"),
            "logs": logs_endpoint or _join_signal_path(base, "logs"),
            "metrics": metrics_endpoint or _join_signal_path(base, "metrics"),
        }

        # Auth + extra headers.
        merged_headers: Dict[str, str] = {
            "Content-Type": content_type,
            "Accept": content_type,
        }
        if bearer_token:
            merged_headers["Authorization"] = f"Bearer {bearer_token}"
        if api_key:
            merged_headers["X-API-Key"] = api_key
        if headers:
            for k, v in headers.items():
                merged_headers[k] = v
        self._headers = merged_headers

        # Per-signal buffer + counters.
        self._buffers: Dict[str, List[Dict[str, Any]]] = {
            "traces": [],
            "logs": [],
            "metrics": [],
        }
        self._batches_sent: Dict[str, int] = {"traces": 0, "logs": 0, "metrics": 0}
        self._batches_dropped: Dict[str, int] = {"traces": 0, "logs": 0, "metrics": 0}
        self._consecutive_drops: Dict[str, int] = {"traces": 0, "logs": 0, "metrics": 0}
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self._closed = False

        # Lazy proto import — only when we need to build a request.
        self._proto: Optional[Dict[str, Any]] = None

        # HTTP client.
        self._owns_client = client is None
        if client is not None:
            self._client = client
        else:
            self._client = httpx.Client(timeout=timeout_s)

        # Daemon flush timer.
        self._stop_event = threading.Event()
        self._timer_thread: Optional[threading.Thread] = None
        if background_flush and flush_interval_s > 0:
            self._timer_thread = threading.Thread(
                target=self._timer_loop,
                name=f"layerlens-otlp-sink-{adapter_name}",
                daemon=True,
            )
            self._timer_thread.start()

    # -- public API ---------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        """Snapshot of per-signal counters and total buffer size.

        Returns a flat dict with keys:

        * ``batches_sent_<signal>`` / ``batches_dropped_<signal>``
        * ``buffer_size_<signal>``
        * ``consecutive_drops_<signal>``
        * ``batches_sent`` / ``batches_dropped`` / ``buffer_size`` /
          ``consecutive_drops`` (totals across all signals).
        """
        with self._lock:
            out: Dict[str, int] = {}
            total_sent = 0
            total_dropped = 0
            total_buffer = 0
            total_consec = 0
            for signal in ("traces", "logs", "metrics"):
                sent = self._batches_sent[signal]
                dropped = self._batches_dropped[signal]
                buf = len(self._buffers[signal])
                consec = self._consecutive_drops[signal]
                out[f"batches_sent_{signal}"] = sent
                out[f"batches_dropped_{signal}"] = dropped
                out[f"buffer_size_{signal}"] = buf
                out[f"consecutive_drops_{signal}"] = consec
                total_sent += sent
                total_dropped += dropped
                total_buffer += buf
                total_consec = max(total_consec, consec)
            out["batches_sent"] = total_sent
            out["batches_dropped"] = total_dropped
            out["buffer_size"] = total_buffer
            out["consecutive_drops"] = total_consec
            return out

    def send(self, event_type: str, payload: Dict[str, Any], timestamp_ns: int) -> None:
        if self._closed:
            return

        signal = _classify_event(event_type)
        evt: Dict[str, Any] = {
            "event_type": event_type,
            "payload": payload,
            "timestamp_ns": timestamp_ns,
            "adapter": self._adapter_name,
            "trace_id": payload.get("trace_id") if isinstance(payload, dict) else None,
        }

        should_flush = False
        with self._lock:
            self._buffers[signal].append(evt)
            if len(self._buffers[signal]) >= self._max_batch:
                should_flush = True

        if should_flush:
            self._flush_signal(signal)

    def flush(self) -> None:
        for signal in ("traces", "logs", "metrics"):
            self._flush_signal(signal)
        with self._lock:
            self._last_flush = time.monotonic()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._timer_thread is not None:
            self._stop_event.set()
            self._timer_thread.join(timeout=max(self._flush_interval_s * 2, 1.0))
        try:
            self.flush()
        finally:
            if self._owns_client:
                try:
                    self._client.close()
                except Exception:
                    logger.debug("OTLPHttpSink client.close() failed", exc_info=True)

    # -- internals ----------------------------------------------------------

    def _ensure_proto(self) -> Dict[str, Any]:
        if self._proto is None:
            self._proto = _import_proto()
        return self._proto

    def _build_request(self, signal: str, events: Sequence[Dict[str, Any]]) -> Any:
        proto = self._ensure_proto()
        if signal == "traces":
            return _build_traces_request(
                events,
                self._service_name,
                self._service_version,
                self._resource_attrs,
                proto,
            )
        if signal == "logs":
            return _build_logs_request(
                events,
                self._service_name,
                self._service_version,
                self._resource_attrs,
                proto,
            )
        if signal == "metrics":
            return _build_metrics_request(
                events,
                self._service_name,
                self._service_version,
                self._resource_attrs,
                proto,
            )
        raise ValueError(f"Unknown signal: {signal}")

    def _flush_signal(self, signal: str) -> None:
        with self._lock:
            buf = self._buffers[signal]
            if not buf:
                return
            batch = list(buf)
            buf.clear()

        try:
            request = self._build_request(signal, batch)
            body = _serialize_request(request, self._content_type)
        except Exception:
            logger.debug("OTLPHttpSink failed to build %s request", signal, exc_info=True)
            with self._lock:
                self._batches_dropped[signal] += 1
                self._consecutive_drops[signal] += 1
            return

        url = self._endpoints[signal]
        ok = _post_with_retry(self._client, url, body, self._headers)
        if ok:
            with self._lock:
                self._batches_sent[signal] += 1
                self._consecutive_drops[signal] = 0
            return

        with self._lock:
            self._batches_dropped[signal] += 1
            self._consecutive_drops[signal] += 1
            consecutive = self._consecutive_drops[signal]
        if consecutive == _CONSECUTIVE_DROP_WARN_THRESHOLD:
            logger.warning(
                "%s: OTLPHttpSink (%s) for adapter %s dropped %d consecutive %s "
                "batches (latest had %d events). OTLP pipeline may be degraded.",
                _DROP_WARN_LOG_CODE,
                signal,
                self._adapter_name,
                consecutive,
                signal,
                len(batch),
            )
        else:
            logger.debug(
                "OTLPHttpSink dropped %s batch of %d events (consecutive=%d)",
                signal,
                len(batch),
                consecutive,
            )

    def _timer_loop(self) -> None:
        while not self._stop_event.wait(self._flush_interval_s):
            if self._closed:
                return
            try:
                with self._lock:
                    has_data = any(self._buffers[s] for s in ("traces", "logs", "metrics"))
                if has_data:
                    self.flush()
            except Exception:
                logger.debug("OTLPHttpSink timer flush failed", exc_info=True)


# ---------------------------------------------------------------------------
# Endpoint helpers
# ---------------------------------------------------------------------------


def _join_signal_path(base: str, signal: str) -> str:
    """Append ``/v1/<signal>`` to ``base``, idempotent for already-pathed URLs."""
    parsed = urlparse(base)
    path = parsed.path or ""
    target = f"/v1/{signal}"
    # If base already ends with /v1, just add the signal name. If it
    # already contains /v1/<other_signal>, replace the last segment.
    if path.rstrip("/").endswith(f"/v1/{signal}"):
        return base
    if path.rstrip("/").endswith("/v1"):
        new_path = path.rstrip("/") + f"/{signal}"
        return parsed._replace(path=new_path).geturl()
    if "/v1/" in path:
        # Replace the trailing signal segment.
        head, _, _ = path.rpartition("/v1/")
        new_path = f"{head}/v1/{signal}".replace("//", "/")
        return parsed._replace(path=new_path).geturl()
    new_path = path.rstrip("/") + target
    return parsed._replace(path=new_path).geturl()


__all__ = [
    "OTLPHttpSink",
    "CONTENT_TYPE_JSON",
    "CONTENT_TYPE_PROTOBUF",
]


# Type alias for re-exports — keeps mypy strict happy on Union of dicts.
_AnyDict = Union[Dict[str, Any], Mapping[str, Any]]
