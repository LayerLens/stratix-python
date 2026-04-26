"""Live OpenAI integration tests for ``OpenAIAdapter``.

Tests in this module are gated by the ``@pytest.mark.live`` marker
(registered in ``tests/conftest.py``) AND by the presence of an
``OPENAI_API_KEY`` env var. They make REAL calls to the OpenAI API and
incur real cost (single-token completions, chosen to be < $0.0001 per
test). Skip on PR CI; run nightly or on demand:

::

    OPENAI_API_KEY=sk-... pytest tests/instrument/adapters/providers/test_openai_adapter_live.py -m live

These tests exist to catch:

1. **OpenAI SDK schema drift** — if the SDK renames ``usage.prompt_tokens``
   or removes ``response.system_fingerprint`` or changes the shape of
   ``tool_calls``, the mocked tests pass but these will fail.
2. **End-to-end transport** — events flow from the real SDK call through
   the adapter into the real HTTP transport sink and reach a live
   localhost endpoint that mirrors the atlas-app ingest contract.
3. **Streaming behavior** — the streaming wrapper is exercised against
   real chunk sequences, not synthesized ones.

The tests assert on **structural invariants** (event types fired,
required fields present, costs computed) rather than exact byte values,
so they remain stable as model outputs change.
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
from layerlens.instrument.adapters.providers.openai_adapter import OpenAIAdapter

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set; skipping live OpenAI tests",
    ),
]


@pytest.fixture
def live_openai_client() -> Any:
    """Build a real ``openai.OpenAI`` client.

    Skips the test cleanly if the openai package isn't installed.
    """
    try:
        from openai import OpenAI
    except ImportError:
        pytest.skip("openai package not installed")

    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ---------------------------------------------------------------------------
# Shared local HTTP capture server (mirrors atlas-app ingest contract)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Live tests — real OpenAI, real transport, real localhost ingest server
# ---------------------------------------------------------------------------


class TestOpenAIAdapterLive:
    def test_real_chat_completion_emits_full_event_set(
        self,
        live_openai_client: Any,
        ingest_server: Tuple[str, _IngestRecorder],
    ) -> None:
        """A single real ``chat.completions.create`` call must:

        * Reach OpenAI and return a valid response.
        * Emit ``model.invoke`` and ``cost.record`` events.
        * Route those events through HttpEventSink to the local server.
        * Carry usage tokens that match the real response.
        """
        base_url, recorder = ingest_server

        sink = HttpEventSink(
            adapter_name="openai",
            api_key="test-org-key",
            base_url=base_url,
            path="/telemetry/spans",
            max_batch=1,
            flush_interval_s=0.0,
        )

        adapter = OpenAIAdapter(capture_config=CaptureConfig.standard())
        adapter.add_sink(sink)
        adapter.connect()
        adapter.connect_client(live_openai_client)

        try:
            response = live_openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say hi in one word."}],
                max_tokens=5,
            )
        finally:
            sink.close()
            adapter.disconnect()

        # Real OpenAI response shape.
        assert response.choices, "OpenAI returned no choices"
        assert response.usage is not None, "OpenAI returned no usage"

        # Adapter routed events through the sink to our localhost server.
        time.sleep(0.5)  # give close() a moment if needed
        with recorder.lock:
            batches = list(recorder.batches)
        assert batches, "no events reached the ingest server"

        all_events: List[Dict[str, Any]] = []
        for batch in batches:
            all_events.extend(batch.get("events", []))

        types = [e["event_type"] for e in all_events]
        assert "model.invoke" in types, f"missing model.invoke in {types}"
        assert "cost.record" in types, f"missing cost.record in {types}"

        invoke = next(e for e in all_events if e["event_type"] == "model.invoke")
        # Real provider field — would FAIL if SDK renamed `usage.prompt_tokens`.
        assert invoke["payload"]["prompt_tokens"] == response.usage.prompt_tokens
        assert invoke["payload"]["completion_tokens"] == response.usage.completion_tokens
        assert invoke["payload"]["total_tokens"] == response.usage.total_tokens
        assert invoke["payload"]["latency_ms"] > 0
        assert invoke["payload"]["model"] == "gpt-4o-mini"

        cost = next(e for e in all_events if e["event_type"] == "cost.record")
        # gpt-4o-mini IS in the pricing table — must compute a cost.
        assert cost["payload"]["api_cost_usd"] is not None
        assert cost["payload"]["api_cost_usd"] >= 0

    def test_real_streaming_emits_consolidated_event(
        self,
        live_openai_client: Any,
        ingest_server: Tuple[str, _IngestRecorder],
    ) -> None:
        """Streaming consumption must emit exactly one ``model.invoke``
        on stream completion (not one per chunk)."""
        base_url, recorder = ingest_server

        sink = HttpEventSink(
            adapter_name="openai",
            api_key="test-org-key",
            base_url=base_url,
            max_batch=1,
            flush_interval_s=0.0,
        )

        adapter = OpenAIAdapter(capture_config=CaptureConfig.standard())
        adapter.add_sink(sink)
        adapter.connect()
        adapter.connect_client(live_openai_client)

        try:
            stream = live_openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Count to three."}],
                max_tokens=20,
                stream=True,
                stream_options={"include_usage": True},
            )
            chunks_seen = 0
            for _chunk in stream:
                chunks_seen += 1
            assert chunks_seen > 0, "stream produced no chunks"
        finally:
            sink.close()
            adapter.disconnect()

        time.sleep(0.5)
        with recorder.lock:
            batches = list(recorder.batches)

        all_events: List[Dict[str, Any]] = []
        for batch in batches:
            all_events.extend(batch.get("events", []))

        invoke_events = [e for e in all_events if e["event_type"] == "model.invoke"]
        # Exactly one model.invoke per LLM call, regardless of chunk count.
        assert len(invoke_events) == 1
        # The streaming flag is captured in metadata.
        assert invoke_events[0]["payload"].get("streaming") is True

    def test_real_error_path_emits_policy_violation(
        self,
        live_openai_client: Any,
        ingest_server: Tuple[str, _IngestRecorder],
    ) -> None:
        """An invalid model name produces a real OpenAI error which the
        adapter must convert into a ``policy.violation`` event."""
        base_url, recorder = ingest_server

        sink = HttpEventSink(
            adapter_name="openai",
            api_key="test-org-key",
            base_url=base_url,
            max_batch=1,
            flush_interval_s=0.0,
        )

        adapter = OpenAIAdapter(capture_config=CaptureConfig.standard())
        adapter.add_sink(sink)
        adapter.connect()
        adapter.connect_client(live_openai_client)

        try:
            with pytest.raises(Exception):  # noqa: B017 - OpenAI raises one of several SDK error types
                live_openai_client.chat.completions.create(
                    model="this-model-definitely-does-not-exist-xyz123",
                    messages=[{"role": "user", "content": "x"}],
                    max_tokens=1,
                )
        finally:
            sink.close()
            adapter.disconnect()

        time.sleep(0.5)
        with recorder.lock:
            batches = list(recorder.batches)

        all_events: List[Dict[str, Any]] = []
        for batch in batches:
            all_events.extend(batch.get("events", []))

        types = [e["event_type"] for e in all_events]
        assert "model.invoke" in types  # error variant
        assert "policy.violation" in types

        invoke = next(e for e in all_events if e["event_type"] == "model.invoke")
        assert "error" in invoke["payload"]
