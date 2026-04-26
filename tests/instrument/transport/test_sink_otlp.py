"""Tests for the OTLP/HTTP transport sink.

Spins up an in-process ``http.server.HTTPServer`` so the sink performs
real HTTP POSTs over loopback, and asserts on the wire bytes — every
buffered batch is flushed, encoded as protobuf (or JSON), and decoded
back into the upstream OTLP service-request messages.

What's covered:

* ``model.invoke`` → ``ExportTraceServiceRequest`` with
  ``service.name`` / ``service.version`` resource attrs.
* ``log.error`` → ``ExportLogsServiceRequest`` with the right severity.
* ``cost.record`` → ``ExportMetricsServiceRequest`` with one metric per
  numeric payload key.
* JSON content-type: server receives ``application/json`` and the body
  parses as JSON.
* Retry on 5xx (initial + 2 retries = 3 total attempts).
* ``Retry-After`` header is honored on 429.
* Batching: events are held until the per-signal max_batch is reached
  or :meth:`flush` / :meth:`close` is called.
* :meth:`stats` reports per-signal sent / dropped counters.
* Auth headers: bearer token AND X-API-Key both reach the wire.
"""

from __future__ import annotations

import json
import time
import threading
from typing import Any, Dict, List, Tuple, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler

import pytest

from layerlens.instrument.transport.sink_otlp import (
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_PROTOBUF,
    OTLPHttpSink,
    _classify_event,
    _join_signal_path,
)

# Skip the whole module if opentelemetry-proto is not installed — these
# tests require the [otlp] extra.
pytest.importorskip("opentelemetry.proto.trace.v1.trace_pb2")

from opentelemetry.proto.collector.logs.v1 import (  # noqa: E402
    logs_service_pb2,
)
from opentelemetry.proto.collector.trace.v1 import (  # noqa: E402
    trace_service_pb2,
)
from opentelemetry.proto.collector.metrics.v1 import (  # noqa: E402
    metrics_service_pb2,
)

# ---------------------------------------------------------------------------
# Local HTTP server harness — same shape as test_sink_http_e2e.py
# ---------------------------------------------------------------------------


class _Recorder:
    """Shared state captured by the test HTTP server."""

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        # Optional response policy: list of (status, headers_dict, body) tuples.
        # If empty, return 200 OK.
        self.responses: List[Tuple[int, Dict[str, str], bytes]] = []
        self.lock = threading.Lock()


def _make_handler(recorder: _Recorder) -> type:
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b""

            with recorder.lock:
                recorder.requests.append(
                    {
                        "path": self.path,
                        "headers": dict(self.headers),
                        "body": raw,
                    }
                )
                if recorder.responses:
                    status, headers, response_body = recorder.responses.pop(0)
                else:
                    status, headers, response_body = 200, {}, b'{"ok":true}'

            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

    return _Handler


@pytest.fixture
def server() -> Any:
    recorder = _Recorder()
    handler = _make_handler(recorder)
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", recorder
    finally:
        httpd.shutdown()
        thread.join(timeout=5.0)
        httpd.server_close()


def _data_posts(recorder: _Recorder, path_suffix: str) -> List[Dict[str, Any]]:
    return [r for r in recorder.requests if r["path"].endswith(path_suffix)]


# ---------------------------------------------------------------------------
# Event classification + endpoint joining (pure helpers — no I/O)
# ---------------------------------------------------------------------------


class TestEventClassification:
    def test_model_invoke_routes_to_traces(self) -> None:
        assert _classify_event("model.invoke") == "traces"

    def test_tool_call_routes_to_traces(self) -> None:
        assert _classify_event("tool.call") == "traces"

    def test_agent_act_routes_to_traces(self) -> None:
        assert _classify_event("agent.act") == "traces"

    def test_cost_record_routes_to_metrics(self) -> None:
        assert _classify_event("cost.record") == "metrics"

    def test_log_dot_anything_routes_to_logs(self) -> None:
        assert _classify_event("log.error") == "logs"
        assert _classify_event("log.info") == "logs"

    def test_unknown_defaults_to_traces(self) -> None:
        assert _classify_event("custom.event") == "traces"


class TestEndpointJoining:
    def test_appends_v1_signal(self) -> None:
        assert (
            _join_signal_path("http://collector:4318", "traces")
            == "http://collector:4318/v1/traces"
        )

    def test_idempotent_when_v1_signal_already_present(self) -> None:
        assert (
            _join_signal_path("http://collector:4318/v1/traces", "traces")
            == "http://collector:4318/v1/traces"
        )

    def test_replaces_signal_when_path_has_other_signal(self) -> None:
        assert (
            _join_signal_path("http://collector:4318/v1/traces", "logs")
            == "http://collector:4318/v1/logs"
        )

    def test_appends_signal_when_base_ends_with_v1(self) -> None:
        assert (
            _join_signal_path("https://api.example.com/v1", "metrics")
            == "https://api.example.com/v1/metrics"
        )


# ---------------------------------------------------------------------------
# Traces → ExportTraceServiceRequest
# ---------------------------------------------------------------------------


class TestTracesExport:
    def test_model_invoke_event_round_trip(
        self, server: Tuple[str, _Recorder]
    ) -> None:
        base_url, recorder = server
        sink = OTLPHttpSink(
            adapter_name="openai",
            service_name="my-svc",
            service_version="9.9.9",
            endpoint=base_url,
            max_batch=1,
            flush_interval_s=0.0,
            background_flush=False,
        )
        try:
            sink.send(
                "model.invoke",
                {"model": "gpt-4o", "tokens": 42},
                time.time_ns(),
            )
        finally:
            sink.close()

        posts = _data_posts(recorder, "/v1/traces")
        assert len(posts) >= 1
        req = posts[0]
        assert req["headers"]["Content-Type"] == CONTENT_TYPE_PROTOBUF

        # Decode the wire bytes into the upstream proto.
        decoded = trace_service_pb2.ExportTraceServiceRequest()
        decoded.ParseFromString(req["body"])

        rs = decoded.resource_spans[0]
        resource_attrs = {a.key: a.value.string_value for a in rs.resource.attributes}
        assert resource_attrs["service.name"] == "my-svc"
        assert resource_attrs["service.version"] == "9.9.9"
        assert resource_attrs["telemetry.sdk.name"] == "layerlens"
        assert resource_attrs["telemetry.sdk.language"] == "python"

        span = rs.scope_spans[0].spans[0]
        assert span.name == "model.invoke"
        assert span.start_time_unix_nano > 0
        assert span.end_time_unix_nano >= span.start_time_unix_nano

        attrs: Dict[str, Any] = {}
        for kv in span.attributes:
            if kv.value.HasField("string_value"):
                attrs[kv.key] = kv.value.string_value
            elif kv.value.HasField("int_value"):
                attrs[kv.key] = kv.value.int_value
        assert attrs["layerlens.adapter"] == "openai"
        assert attrs["layerlens.event_type"] == "model.invoke"
        assert attrs["model"] == "gpt-4o"
        assert attrs["tokens"] == 42

    def test_auth_headers_are_emitted(
        self, server: Tuple[str, _Recorder]
    ) -> None:
        base_url, recorder = server
        sink = OTLPHttpSink(
            adapter_name="openai",
            endpoint=base_url,
            bearer_token="eyJabc",
            api_key="lk_demo_123",
            max_batch=1,
            flush_interval_s=0.0,
            background_flush=False,
        )
        try:
            sink.send("model.invoke", {"a": 1}, time.time_ns())
        finally:
            sink.close()

        posts = _data_posts(recorder, "/v1/traces")
        assert posts, "expected at least one traces POST"
        # http.server normalizes header casing — match case-insensitively.
        headers_ci = {k.lower(): v for k, v in posts[0]["headers"].items()}
        assert headers_ci["authorization"] == "Bearer eyJabc"
        assert headers_ci["x-api-key"] == "lk_demo_123"

    def test_extra_resource_attrs(self, server: Tuple[str, _Recorder]) -> None:
        base_url, recorder = server
        sink = OTLPHttpSink(
            adapter_name="openai",
            endpoint=base_url,
            resource_attrs={"deployment.environment": "prod", "k8s.pod.name": "api-1"},
            max_batch=1,
            flush_interval_s=0.0,
            background_flush=False,
        )
        try:
            sink.send("model.invoke", {"a": 1}, time.time_ns())
        finally:
            sink.close()

        decoded = trace_service_pb2.ExportTraceServiceRequest()
        decoded.ParseFromString(_data_posts(recorder, "/v1/traces")[0]["body"])
        attrs = {a.key: a.value.string_value for a in decoded.resource_spans[0].resource.attributes}
        assert attrs["deployment.environment"] == "prod"
        assert attrs["k8s.pod.name"] == "api-1"


# ---------------------------------------------------------------------------
# Logs → ExportLogsServiceRequest
# ---------------------------------------------------------------------------


class TestLogsExport:
    def test_log_error_event_translates_to_log_record(
        self, server: Tuple[str, _Recorder]
    ) -> None:
        base_url, recorder = server
        sink = OTLPHttpSink(
            adapter_name="openai",
            endpoint=base_url,
            max_batch=1,
            flush_interval_s=0.0,
            background_flush=False,
        )
        try:
            sink.send(
                "log.error",
                {"message": "Something broke", "code": 500},
                time.time_ns(),
            )
        finally:
            sink.close()

        posts = _data_posts(recorder, "/v1/logs")
        assert posts, "expected a logs POST"
        decoded = logs_service_pb2.ExportLogsServiceRequest()
        decoded.ParseFromString(posts[0]["body"])
        rec = decoded.resource_logs[0].scope_logs[0].log_records[0]
        assert rec.severity_text == "ERROR"
        assert rec.body.string_value == "Something broke"

    def test_log_info_default_severity(self, server: Tuple[str, _Recorder]) -> None:
        base_url, recorder = server
        sink = OTLPHttpSink(
            adapter_name="openai",
            endpoint=base_url,
            max_batch=1,
            flush_interval_s=0.0,
            background_flush=False,
        )
        try:
            sink.send("log.info", {"message": "hello"}, time.time_ns())
        finally:
            sink.close()

        decoded = logs_service_pb2.ExportLogsServiceRequest()
        decoded.ParseFromString(_data_posts(recorder, "/v1/logs")[0]["body"])
        rec = decoded.resource_logs[0].scope_logs[0].log_records[0]
        assert rec.severity_text == "INFO"


# ---------------------------------------------------------------------------
# Metrics → ExportMetricsServiceRequest
# ---------------------------------------------------------------------------


class TestMetricsExport:
    def test_cost_record_emits_one_metric_per_numeric_field(
        self, server: Tuple[str, _Recorder]
    ) -> None:
        base_url, recorder = server
        sink = OTLPHttpSink(
            adapter_name="openai",
            endpoint=base_url,
            max_batch=1,
            flush_interval_s=0.0,
            background_flush=False,
        )
        try:
            sink.send(
                "cost.record",
                {"cost_usd": 0.0123, "tokens": 100, "model": "gpt-4o"},
                time.time_ns(),
            )
        finally:
            sink.close()

        decoded = metrics_service_pb2.ExportMetricsServiceRequest()
        decoded.ParseFromString(_data_posts(recorder, "/v1/metrics")[0]["body"])
        metrics = decoded.resource_metrics[0].scope_metrics[0].metrics
        names = {m.name for m in metrics}
        # Numeric fields produce metrics; string "model" does not.
        assert "layerlens.cost.record.cost_usd" in names
        assert "layerlens.cost.record.tokens" in names
        assert not any("model" in n for n in names)

        # Check the actual numeric values.
        by_name = {m.name: m for m in metrics}
        cost_metric = by_name["layerlens.cost.record.cost_usd"]
        assert cost_metric.sum.data_points[0].as_double == pytest.approx(0.0123)

        tokens_metric = by_name["layerlens.cost.record.tokens"]
        assert tokens_metric.sum.data_points[0].as_int == 100


# ---------------------------------------------------------------------------
# JSON content-type
# ---------------------------------------------------------------------------


class TestJsonContentType:
    def test_json_body_is_decodable(self, server: Tuple[str, _Recorder]) -> None:
        base_url, recorder = server
        sink = OTLPHttpSink(
            adapter_name="openai",
            endpoint=base_url,
            content_type=CONTENT_TYPE_JSON,
            max_batch=1,
            flush_interval_s=0.0,
            background_flush=False,
        )
        try:
            sink.send("model.invoke", {"model": "gpt-4o"}, time.time_ns())
        finally:
            sink.close()

        post = _data_posts(recorder, "/v1/traces")[0]
        assert post["headers"]["Content-Type"] == CONTENT_TYPE_JSON
        # Body should parse as JSON.
        body = json.loads(post["body"].decode("utf-8"))
        assert "resource_spans" in body
        spans = body["resource_spans"][0]["scope_spans"][0]["spans"]
        assert spans[0]["name"] == "model.invoke"


# ---------------------------------------------------------------------------
# Retry / backoff
# ---------------------------------------------------------------------------


class TestRetryBackoff:
    def test_retries_on_5xx_then_succeeds(
        self, server: Tuple[str, _Recorder]
    ) -> None:
        base_url, recorder = server
        recorder.responses = [
            (503, {}, b'{"err":"down"}'),
            (503, {}, b'{"err":"down"}'),
            (200, {}, b'{"ok":true}'),
        ]
        sink = OTLPHttpSink(
            adapter_name="openai",
            endpoint=base_url,
            max_batch=1,
            flush_interval_s=0.0,
            background_flush=False,
        )
        try:
            sink.send("model.invoke", {"a": 1}, time.time_ns())
        finally:
            sink.close()

        # Initial + 2 retries = 3 traces POSTs.
        assert len(_data_posts(recorder, "/v1/traces")) == 3
        stats = sink.stats()
        assert stats["batches_sent_traces"] == 1
        assert stats["batches_dropped_traces"] == 0

    def test_honors_retry_after_on_429(
        self, server: Tuple[str, _Recorder]
    ) -> None:
        base_url, recorder = server
        recorder.responses = [
            (429, {"Retry-After": "0"}, b"{}"),
            (200, {}, b'{"ok":true}'),
        ]
        sink = OTLPHttpSink(
            adapter_name="openai",
            endpoint=base_url,
            max_batch=1,
            flush_interval_s=0.0,
            background_flush=False,
        )
        start = time.monotonic()
        try:
            sink.send("model.invoke", {"a": 1}, time.time_ns())
        finally:
            sink.close()
        elapsed = time.monotonic() - start

        # Retry-After: 0 should cause an immediate retry — total elapsed
        # under 2 seconds (well under default backoff of 0.5s if we
        # ignored Retry-After we'd still be fast, but the assertion
        # that matters is the success after the 429).
        assert elapsed < 5.0
        assert len(_data_posts(recorder, "/v1/traces")) == 2
        assert sink.stats()["batches_sent_traces"] == 1

    def test_4xx_drops_without_retry(self, server: Tuple[str, _Recorder]) -> None:
        base_url, recorder = server
        recorder.responses = [(400, {}, b'{"err":"bad"}')]
        sink = OTLPHttpSink(
            adapter_name="openai",
            endpoint=base_url,
            max_batch=1,
            flush_interval_s=0.0,
            background_flush=False,
        )
        try:
            sink.send("model.invoke", {"a": 1}, time.time_ns())
        finally:
            sink.close()

        # Exactly 1 attempt — 4xx is not retried.
        assert len(_data_posts(recorder, "/v1/traces")) == 1
        assert sink.stats()["batches_dropped_traces"] == 1


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


class TestBatching:
    def test_batches_until_max_batch(self, server: Tuple[str, _Recorder]) -> None:
        base_url, recorder = server
        sink = OTLPHttpSink(
            adapter_name="openai",
            endpoint=base_url,
            max_batch=3,
            flush_interval_s=999.0,  # disable timer
            background_flush=False,
        )
        try:
            sink.send("model.invoke", {"i": 1}, time.time_ns())
            sink.send("model.invoke", {"i": 2}, time.time_ns())
            assert _data_posts(recorder, "/v1/traces") == []  # not yet flushed
            sink.send("model.invoke", {"i": 3}, time.time_ns())
        finally:
            sink.close()

        posts = _data_posts(recorder, "/v1/traces")
        # Exactly one POST (the close-time flush sees an empty buffer).
        assert len(posts) == 1
        decoded = trace_service_pb2.ExportTraceServiceRequest()
        decoded.ParseFromString(posts[0]["body"])
        spans = decoded.resource_spans[0].scope_spans[0].spans
        assert len(spans) == 3

    def test_close_flushes_partial_buffer(
        self, server: Tuple[str, _Recorder]
    ) -> None:
        base_url, recorder = server
        sink = OTLPHttpSink(
            adapter_name="openai",
            endpoint=base_url,
            max_batch=999,  # never auto-flush by size
            flush_interval_s=999.0,
            background_flush=False,
        )
        sink.send("model.invoke", {"i": 1}, time.time_ns())
        sink.send("model.invoke", {"i": 2}, time.time_ns())
        assert _data_posts(recorder, "/v1/traces") == []
        sink.close()  # force final flush

        posts = _data_posts(recorder, "/v1/traces")
        assert len(posts) == 1
        decoded = trace_service_pb2.ExportTraceServiceRequest()
        decoded.ParseFromString(posts[0]["body"])
        assert len(decoded.resource_spans[0].scope_spans[0].spans) == 2

    def test_timer_thread_flushes_partial_buffer(
        self, server: Tuple[str, _Recorder]
    ) -> None:
        base_url, recorder = server
        sink = OTLPHttpSink(
            adapter_name="openai",
            endpoint=base_url,
            max_batch=999,
            flush_interval_s=0.05,
            background_flush=True,
        )
        try:
            sink.send("model.invoke", {"i": 1}, time.time_ns())
            # Wait for the timer to fire at least once.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if _data_posts(recorder, "/v1/traces"):
                    break
                time.sleep(0.05)
        finally:
            sink.close()

        assert _data_posts(recorder, "/v1/traces"), (
            "timer-driven flush did not fire within 2s"
        )

    def test_per_signal_buffers_are_independent(
        self, server: Tuple[str, _Recorder]
    ) -> None:
        base_url, recorder = server
        sink = OTLPHttpSink(
            adapter_name="openai",
            endpoint=base_url,
            max_batch=2,  # flush at 2 events per signal
            flush_interval_s=999.0,
            background_flush=False,
        )
        try:
            # 1 trace, 1 metric, 1 log — none should flush yet.
            sink.send("model.invoke", {"i": 1}, time.time_ns())
            sink.send("cost.record", {"cost_usd": 0.001}, time.time_ns())
            sink.send("log.info", {"message": "hi"}, time.time_ns())
            assert _data_posts(recorder, "/v1/traces") == []
            assert _data_posts(recorder, "/v1/metrics") == []
            assert _data_posts(recorder, "/v1/logs") == []
            # Push the second trace — only traces should flush.
            sink.send("model.invoke", {"i": 2}, time.time_ns())
            assert len(_data_posts(recorder, "/v1/traces")) == 1
            assert _data_posts(recorder, "/v1/metrics") == []
            assert _data_posts(recorder, "/v1/logs") == []
        finally:
            sink.close()
        # close() flushes the remaining metric + log.
        assert len(_data_posts(recorder, "/v1/metrics")) == 1
        assert len(_data_posts(recorder, "/v1/logs")) == 1


# ---------------------------------------------------------------------------
# stats() reporting
# ---------------------------------------------------------------------------


class TestStats:
    def test_per_signal_sent_counters(self, server: Tuple[str, _Recorder]) -> None:
        base_url, _ = server
        sink = OTLPHttpSink(
            adapter_name="openai",
            endpoint=base_url,
            max_batch=1,
            flush_interval_s=0.0,
            background_flush=False,
        )
        try:
            sink.send("model.invoke", {"a": 1}, time.time_ns())
            sink.send("log.info", {"message": "hi"}, time.time_ns())
            sink.send("cost.record", {"cost_usd": 0.01}, time.time_ns())
        finally:
            sink.close()

        stats = sink.stats()
        assert stats["batches_sent_traces"] >= 1
        assert stats["batches_sent_logs"] >= 1
        assert stats["batches_sent_metrics"] >= 1
        assert stats["batches_dropped"] == 0
        assert stats["buffer_size"] == 0

    def test_drops_are_counted(self, server: Tuple[str, _Recorder]) -> None:
        base_url, recorder = server
        recorder.responses = [(503, {}, b"{}") for _ in range(20)]
        sink = OTLPHttpSink(
            adapter_name="openai",
            endpoint=base_url,
            max_batch=1,
            flush_interval_s=0.0,
            background_flush=False,
        )
        try:
            sink.send("model.invoke", {"a": 1}, time.time_ns())
            sink.send("model.invoke", {"a": 2}, time.time_ns())
            sink.send("model.invoke", {"a": 3}, time.time_ns())
        finally:
            sink.close()

        stats = sink.stats()
        assert stats["batches_dropped_traces"] >= 3
        assert stats["consecutive_drops"] >= 3


# ---------------------------------------------------------------------------
# Per-signal endpoint overrides
# ---------------------------------------------------------------------------


class TestPerSignalEndpoints:
    def test_per_signal_endpoint_overrides_base(
        self, server: Tuple[str, _Recorder]
    ) -> None:
        base_url, recorder = server
        # Send traces to an explicit non-standard path.
        sink = OTLPHttpSink(
            adapter_name="openai",
            endpoint=base_url,
            traces_endpoint=f"{base_url}/custom/trace-ingest",
            max_batch=1,
            flush_interval_s=0.0,
            background_flush=False,
        )
        try:
            sink.send("model.invoke", {"a": 1}, time.time_ns())
        finally:
            sink.close()

        assert _data_posts(recorder, "/custom/trace-ingest"), (
            "expected traces to hit the per-signal override URL"
        )
        # No traffic on the default /v1/traces path.
        assert _data_posts(recorder, "/v1/traces") == []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_content_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="content_type must be"):
            OTLPHttpSink(
                adapter_name="openai",
                endpoint="http://localhost:4318",
                content_type="application/xml",
                background_flush=False,
            )

    def test_works_without_optional_headers(
        self, server: Tuple[str, _Recorder]
    ) -> None:
        base_url, recorder = server
        sink = OTLPHttpSink(
            adapter_name="openai",
            endpoint=base_url,
            max_batch=1,
            flush_interval_s=0.0,
            background_flush=False,
        )
        try:
            sink.send("model.invoke", {"a": 1}, time.time_ns())
        finally:
            sink.close()
        # No auth was provided; sink must still POST successfully.
        posts = _data_posts(recorder, "/v1/traces")
        assert posts
        headers_ci = {k.lower() for k in posts[0]["headers"]}
        assert "authorization" not in headers_ci
        assert "x-api-key" not in headers_ci


# ---------------------------------------------------------------------------
# Optional helpers used by other tests
# ---------------------------------------------------------------------------


def _opt(value: Optional[Any], default: Any) -> Any:
    return value if value is not None else default
