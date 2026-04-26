"""Live Anthropic integration tests for ``AnthropicAdapter``.

Gated by ``@pytest.mark.live`` AND the presence of ``ANTHROPIC_API_KEY``.
These make REAL calls and incur small cost (single-token completions).

Same testing strategy as the OpenAI live tests: a real Anthropic call
flows through the adapter into a real ``HttpEventSink`` pointed at a
localhost ingest server that mirrors atlas-app's wire contract.
Structural-invariant assertions only.
"""

from __future__ import annotations

import os
import json
import time
import threading
from typing import Any, Dict, List, Tuple
from http.server import HTTPServer, BaseHTTPRequestHandler

import pytest

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.providers.anthropic_adapter import AnthropicAdapter

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set; skipping live Anthropic tests",
    ),
]


@pytest.fixture
def live_anthropic_client() -> Any:
    try:
        from anthropic import Anthropic
    except ImportError:
        pytest.skip("anthropic package not installed")
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


class _IngestRecorder:
    def __init__(self) -> None:
        self.batches: List[Dict[str, Any]] = []
        self.lock = threading.Lock()


def _make_ingest_handler(recorder: _IngestRecorder) -> type:
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = {"_raw": raw.decode("utf-8", "replace")}
            with recorder.lock:
                recorder.batches.append(body)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

    return _Handler


@pytest.fixture
def ingest_server() -> Any:
    recorder = _IngestRecorder()
    httpd = HTTPServer(("127.0.0.1", 0), _make_ingest_handler(recorder))
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", recorder
    finally:
        httpd.shutdown()
        thread.join(timeout=5.0)
        httpd.server_close()


class TestAnthropicAdapterLive:
    def test_real_messages_create_emits_full_event_set(
        self,
        live_anthropic_client: Any,
        ingest_server: Tuple[str, _IngestRecorder],
    ) -> None:
        base_url, recorder = ingest_server

        sink = HttpEventSink(
            adapter_name="anthropic",
            api_key="test-org-key",
            base_url=base_url,
            path="/telemetry/spans",
            max_batch=1,
            flush_interval_s=0.0,
        )

        adapter = AnthropicAdapter(capture_config=CaptureConfig.standard())
        adapter.add_sink(sink)
        adapter.connect()
        adapter.connect_client(live_anthropic_client)

        try:
            response = live_anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=10,
                messages=[{"role": "user", "content": "Say hi in one word."}],
            )
        finally:
            sink.close()
            adapter.disconnect()

        assert response.content
        assert response.usage is not None

        time.sleep(0.5)
        with recorder.lock:
            batches = list(recorder.batches)
        assert batches, "no events reached the ingest server"

        all_events: List[Dict[str, Any]] = []
        for batch in batches:
            all_events.extend(batch.get("events", []))

        types = [e["event_type"] for e in all_events]
        assert "model.invoke" in types
        assert "cost.record" in types

        invoke = next(e for e in all_events if e["event_type"] == "model.invoke")
        # Real provider field — would FAIL if Anthropic SDK renamed `usage.input_tokens`.
        assert invoke["payload"]["prompt_tokens"] == response.usage.input_tokens
        assert invoke["payload"]["completion_tokens"] == response.usage.output_tokens
        assert invoke["payload"]["latency_ms"] > 0

        cost = next(e for e in all_events if e["event_type"] == "cost.record")
        assert cost["payload"]["api_cost_usd"] is not None
        assert cost["payload"]["api_cost_usd"] >= 0
