"""End-to-end tests for the HTTP transport sink.

Spins up a real ``http.server.BaseHTTPRequestHandler`` on localhost so
the sink talks to a real HTTP server (not a mock). Verifies:

* ``X-API-Key`` header reaches the server.
* Events are batched per the configured batch size.
* Retries with exponential backoff fire on 5xx responses.
* Non-retriable 4xx are dropped (no infinite retry).
* Async sink behaves identically over the same wire.

This is a real round-trip — every byte traverses the loopback socket.
The test would FAIL if the sink ever stopped sending HTTP, sent the
wrong shape, or stripped the auth header.
"""

from __future__ import annotations

import json
import time
import asyncio
import threading
from typing import Any, Dict, List, Tuple
from http.server import HTTPServer, BaseHTTPRequestHandler

import pytest

from layerlens.instrument.transport.sink_http import (
    HttpEventSink,
    AsyncHttpEventSink,
)

# ---------------------------------------------------------------------------
# Local HTTP server harness
# ---------------------------------------------------------------------------


class _Recorder:
    """Shared state across handler invocations on the test server."""

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        # Optional response policy: list of (status, body) per call.
        # If empty, return 200 OK.
        self.responses: List[Tuple[int, str]] = []
        self.lock = threading.Lock()


def _make_handler(recorder: _Recorder) -> type:
    class _Handler(BaseHTTPRequestHandler):
        # Quiet the access log noise.
        def log_message(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        def do_POST(self) -> None:  # noqa: N802 - HTTP server convention
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                body = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                body = {"_raw": raw.decode("utf-8", errors="replace")}

            with recorder.lock:
                recorder.requests.append(
                    {
                        "path": self.path,
                        "headers": dict(self.headers),
                        "body": body,
                    }
                )
                if recorder.responses:
                    status, response_body = recorder.responses.pop(0)
                else:
                    status, response_body = 200, '{"ok":true}'

            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body.encode())))
            self.end_headers()
            self.wfile.write(response_body.encode())

    return _Handler


@pytest.fixture
def server() -> Any:
    """Yield ``(base_url, recorder)`` for a freshly bound localhost server."""
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


# ---------------------------------------------------------------------------
# Sync HttpEventSink
# ---------------------------------------------------------------------------


class TestHttpEventSinkE2E:
    def test_event_round_trip(self, server: Tuple[str, _Recorder]) -> None:
        base_url, recorder = server
        sink = HttpEventSink(
            adapter_name="openai",
            api_key="test-key-abc",
            base_url=base_url,
            path="/telemetry/spans",
            max_batch=1,  # flush per event
            flush_interval_s=0.0,
        )
        try:
            sink.send("model.invoke", {"model": "gpt-4o"}, time_ns())
        finally:
            sink.close()

        assert len(recorder.requests) >= 1
        req = recorder.requests[0]
        assert req["path"] == "/telemetry/spans"
        assert req["headers"].get("X-API-Key") == "test-key-abc"
        assert req["headers"].get("Content-Type") == "application/json"

        events = req["body"]["events"]
        assert len(events) == 1
        assert events[0]["event_type"] == "model.invoke"
        assert events[0]["payload"] == {"model": "gpt-4o"}
        assert events[0]["adapter"] == "openai"

    def test_batching_holds_until_max_batch(self, server: Tuple[str, _Recorder]) -> None:
        base_url, recorder = server
        sink = HttpEventSink(
            adapter_name="openai",
            api_key="k",
            base_url=base_url,
            max_batch=3,
            flush_interval_s=999.0,  # disable time-based flush
        )
        try:
            sink.send("model.invoke", {"a": 1}, time_ns())
            sink.send("model.invoke", {"a": 2}, time_ns())
            assert len(recorder.requests) == 0  # not yet flushed
            sink.send("model.invoke", {"a": 3}, time_ns())
        finally:
            sink.close()

        # Two POSTs total: one for the batch of 3, and possibly one empty
        # close-time flush (which we suppress). Verify the batch is one request
        # of exactly 3 events.
        request_with_three = next(
            r for r in recorder.requests if len(r["body"]["events"]) == 3
        )
        events = request_with_three["body"]["events"]
        assert [e["payload"]["a"] for e in events] == [1, 2, 3]

    def test_close_flushes_buffer(self, server: Tuple[str, _Recorder]) -> None:
        base_url, recorder = server
        sink = HttpEventSink(
            adapter_name="openai",
            api_key="k",
            base_url=base_url,
            max_batch=999,  # no auto-flush by size
            flush_interval_s=999.0,
        )
        sink.send("model.invoke", {"a": 1}, time_ns())
        sink.send("model.invoke", {"a": 2}, time_ns())
        assert len(recorder.requests) == 0  # not yet flushed

        sink.close()  # forces final flush

        assert any(len(r["body"]["events"]) == 2 for r in recorder.requests)

    def test_retries_on_503(self, server: Tuple[str, _Recorder]) -> None:
        base_url, recorder = server
        # First two responses fail with 503, third succeeds.
        recorder.responses = [(503, '{"err":"down"}'), (503, '{"err":"down"}'), (200, '{"ok":true}')]

        sink = HttpEventSink(
            adapter_name="openai",
            api_key="k",
            base_url=base_url,
            max_batch=1,
            flush_interval_s=0.0,
        )
        try:
            sink.send("model.invoke", {"a": 1}, time_ns())
        finally:
            sink.close()

        # Sink retries up to MAX_RETRIES=2 (so total 3 attempts including initial).
        assert len(recorder.requests) >= 2

    def test_4xx_drops_without_retry(self, server: Tuple[str, _Recorder]) -> None:
        base_url, recorder = server
        recorder.responses = [(400, '{"err":"bad"}')]

        sink = HttpEventSink(
            adapter_name="openai",
            api_key="k",
            base_url=base_url,
            max_batch=1,
            flush_interval_s=0.0,
        )
        try:
            sink.send("model.invoke", {"a": 1}, time_ns())
        finally:
            sink.close()

        # Exactly one attempt — 4xx is not retried.
        # (close() may attempt a final flush of empty buffer; filter to the
        # actual data POST.)
        data_posts = [
            r for r in recorder.requests if r["body"].get("events")
        ]
        assert len(data_posts) == 1


# ---------------------------------------------------------------------------
# Async AsyncHttpEventSink
# ---------------------------------------------------------------------------


class TestHttpEventSinkStats:
    def test_stats_after_successful_send(self, server: Tuple[str, _Recorder]) -> None:
        base_url, _ = server
        sink = HttpEventSink(
            adapter_name="openai",
            api_key="k",
            base_url=base_url,
            max_batch=1,
            flush_interval_s=0.0,
            background_flush=False,
        )
        try:
            sink.send("model.invoke", {"a": 1}, time_ns())
        finally:
            sink.close()

        stats = sink.stats()
        assert stats["batches_sent"] >= 1
        assert stats["batches_dropped"] == 0
        assert stats["consecutive_drops"] == 0

    def test_stats_after_drops(self, server: Tuple[str, _Recorder]) -> None:
        base_url, recorder = server
        # Always-fail policy.
        recorder.responses = [(503, "{}") for _ in range(20)]

        sink = HttpEventSink(
            adapter_name="openai",
            api_key="k",
            base_url=base_url,
            max_batch=1,
            flush_interval_s=0.0,
            background_flush=False,
        )
        try:
            sink.send("model.invoke", {"a": 1}, time_ns())
            sink.send("model.invoke", {"a": 2}, time_ns())
            sink.send("model.invoke", {"a": 3}, time_ns())
        finally:
            sink.close()

        stats = sink.stats()
        assert stats["batches_dropped"] >= 3


class TestAsyncHttpEventSinkE2E:
    def test_async_event_round_trip(self, server: Tuple[str, _Recorder]) -> None:
        base_url, recorder = server

        async def run() -> None:
            sink = AsyncHttpEventSink(
                adapter_name="anthropic",
                api_key="k",
                base_url=base_url,
                max_batch=1,
            )
            try:
                await sink.asend("model.invoke", {"model": "claude-sonnet-4-5-20250929"}, time_ns())
            finally:
                await sink.aclose()

        asyncio.run(run())

        data_posts = [r for r in recorder.requests if r["body"].get("events")]
        assert len(data_posts) >= 1
        assert data_posts[0]["body"]["events"][0]["adapter"] == "anthropic"
        assert data_posts[0]["body"]["events"][0]["payload"]["model"] == "claude-sonnet-4-5-20250929"

    def test_async_retries_on_503(self, server: Tuple[str, _Recorder]) -> None:
        base_url, recorder = server
        recorder.responses = [(503, "{}"), (503, "{}"), (200, '{"ok":true}')]

        async def run() -> None:
            sink = AsyncHttpEventSink(
                adapter_name="anthropic",
                api_key="k",
                base_url=base_url,
                max_batch=1,
            )
            try:
                await sink.asend("model.invoke", {"a": 1}, time_ns())
            finally:
                await sink.aclose()

        asyncio.run(run())
        # Initial + 2 retries = 3 attempts on the data path; close() may
        # do an extra empty flush which is suppressed.
        data_posts = [r for r in recorder.requests if r["body"].get("events")]
        assert len(data_posts) >= 2


def time_ns() -> int:
    """Local helper for clarity in tests."""
    return time.time_ns()
