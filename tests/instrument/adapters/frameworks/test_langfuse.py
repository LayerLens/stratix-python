"""Tests for the Langfuse bidirectional batch sync adapter.

The adapter connects to Langfuse via HTTP API and supports:
- Import: pull traces from Langfuse, convert observations to flat LayerLens events
- Export: convert LayerLens events to Langfuse ingestion format

httpx is NOT imported; we set _HAS_HTTPX = True and mock all HTTP interactions.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

import layerlens.instrument.adapters.frameworks.langfuse as _mod

_mod._HAS_HTTPX = True

from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.frameworks._utils import truncate as _truncate
from layerlens.instrument.adapters.frameworks.langfuse import (
    LangfuseAdapter,
    _safe_dict,
)

from .conftest import find_event, find_events, capture_framework_trace

# ---------------------------------------------------------------------------
# Helpers: mock HTTP plumbing
# ---------------------------------------------------------------------------


def _make_mock_http():
    """Create a mock httpx.Client that returns controlled responses."""
    http = Mock(spec=[])
    http.get = Mock()
    http.post = Mock()
    http.close = Mock()
    return http


def _make_response(json_data=None, status_code=200):
    resp = Mock(spec=[])
    resp.status_code = status_code
    resp.json = Mock(return_value=json_data or {})
    resp.raise_for_status = Mock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


# ---------------------------------------------------------------------------
# Helpers: fake Langfuse data
# ---------------------------------------------------------------------------


def _make_langfuse_trace(trace_id="lf-trace-001", observations=None):
    return {
        "id": trace_id,
        "name": "test-trace",
        "input": "Hello, world!",
        "output": "Hi there!",
        "metadata": {"key": "value"},
        "observations": observations or [],
    }


def _make_generation(
    obs_id="gen-001",
    model="gpt-4",
    prompt_tokens=100,
    completion_tokens=50,
):
    return {
        "id": obs_id,
        "type": "GENERATION",
        "name": "llm-call",
        "model": model,
        "input": "What is AI?",
        "output": "AI is...",
        "usage": {
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "totalTokens": prompt_tokens + completion_tokens,
        },
        "calculatedTotalCost": 0.005,
    }


def _make_span(obs_id="span-001", name="retriever"):
    return {
        "id": obs_id,
        "type": "SPAN",
        "name": name,
        "input": "search query",
        "output": "search results",
    }


def _make_event(obs_id="evt-001", name="status-update"):
    return {
        "id": obs_id,
        "type": "EVENT",
        "name": name,
        "statusMessage": "Processing complete",
        "input": "some data",
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def connected_adapter(mock_client):
    """Return a pre-connected adapter, the uploaded-data dict, and the mock HTTP client."""
    uploaded = capture_framework_trace(mock_client)
    adapter = LangfuseAdapter(mock_client)
    mock_http = _make_mock_http()
    adapter._http = mock_http
    adapter._connected = True
    adapter._host = "https://test.langfuse.com"
    adapter._public_key = "pk-test"
    adapter._secret_key = "sk-test"
    adapter._metadata["host"] = "https://test.langfuse.com"
    return adapter, uploaded, mock_http


# ===================================================================
# Connect / Disconnect
# ===================================================================


class TestConnect:
    def test_connect_raises_import_error_when_httpx_missing(self, mock_client):
        adapter = LangfuseAdapter(mock_client)
        saved = _mod._HAS_HTTPX
        try:
            _mod._HAS_HTTPX = False
            with pytest.raises(ImportError, match="httpx"):
                adapter.connect(public_key="pk", secret_key="sk")
        finally:
            _mod._HAS_HTTPX = saved

    def test_connect_raises_value_error_when_keys_missing(self, mock_client):
        adapter = LangfuseAdapter(mock_client)
        with pytest.raises(ValueError, match="public_key.*secret_key"):
            adapter.connect(public_key=None, secret_key=None)

    def test_connect_raises_value_error_when_only_public_key(self, mock_client):
        adapter = LangfuseAdapter(mock_client)
        with pytest.raises(ValueError):
            adapter.connect(public_key="pk", secret_key=None)

    @patch("layerlens.instrument.adapters.frameworks.langfuse.httpx")
    def test_connect_validates_connectivity(self, mock_httpx, mock_client):
        mock_http = _make_mock_http()
        mock_httpx.Client.return_value = mock_http
        mock_http.get.return_value = _make_response({"data": []})

        adapter = LangfuseAdapter(mock_client)
        adapter.connect(public_key="pk-lf-test", secret_key="sk-lf-test")

        assert adapter._connected is True
        mock_http.get.assert_called_once_with("/api/public/traces", params={"limit": 1})

    @patch("layerlens.instrument.adapters.frameworks.langfuse.httpx")
    def test_connect_sets_default_host(self, mock_httpx, mock_client):
        mock_http = _make_mock_http()
        mock_httpx.Client.return_value = mock_http
        mock_http.get.return_value = _make_response({"data": []})

        adapter = LangfuseAdapter(mock_client)
        adapter.connect(public_key="pk", secret_key="sk")

        assert adapter._host == "https://cloud.langfuse.com"

    @patch("layerlens.instrument.adapters.frameworks.langfuse.httpx")
    def test_connect_failure_cleans_up(self, mock_httpx, mock_client):
        mock_http = _make_mock_http()
        mock_httpx.Client.return_value = mock_http
        mock_http.get.return_value = _make_response(status_code=401)

        adapter = LangfuseAdapter(mock_client)
        with pytest.raises(ConnectionError, match="Failed to connect"):
            adapter.connect(public_key="pk", secret_key="sk")

        assert adapter._connected is False
        assert adapter._http is None
        mock_http.close.assert_called_once()


class TestDisconnect:
    def test_disconnect_sets_connected_false(self, connected_adapter):
        adapter, _, _ = connected_adapter
        assert adapter._connected is True
        adapter.disconnect()
        assert adapter._connected is False

    def test_disconnect_closes_http_client(self, connected_adapter):
        adapter, _, mock_http = connected_adapter
        adapter.disconnect()
        mock_http.close.assert_called_once()

    def test_disconnect_clears_state(self, connected_adapter):
        adapter, _, _ = connected_adapter
        adapter._last_cursor = "2026-01-01T00:00:00Z"
        adapter.disconnect()
        assert adapter._http is None
        assert adapter._public_key is None
        assert adapter._secret_key is None
        assert adapter._host is None
        assert adapter._last_cursor is None


class TestAdapterInfo:
    def test_adapter_info_returns_correct_metadata(self, connected_adapter):
        adapter, _, _ = connected_adapter
        info = adapter.adapter_info()
        assert info.name == "langfuse"
        assert info.adapter_type == "framework"
        assert info.connected is True
        # The Langfuse-specific ``host`` metadata must be present; resilience
        # health metadata is added by FrameworkAdapter.adapter_info() to
        # every framework adapter — assert presence of both surfaces.
        assert info.metadata["host"] == "https://test.langfuse.com"
        assert info.metadata["resilience_status"] == "healthy"

    def test_adapter_info_disconnected(self, mock_client):
        adapter = LangfuseAdapter(mock_client)
        info = adapter.adapter_info()
        assert info.connected is False
        # Disconnected adapters expose only the resilience health surface
        # (no per-adapter metadata since connect() never populated it).
        assert info.metadata.get("host") is None
        assert info.metadata["resilience_status"] == "healthy"
        assert info.metadata["resilience_failures_total"] == 0


# ===================================================================
# Import: traces
# ===================================================================


class TestImportTraces:
    def test_import_traces_fetches_and_returns_count(self, connected_adapter):
        adapter, uploaded, mock_http = connected_adapter
        mock_http.get.side_effect = [
            _make_response({"data": [{"id": "t1", "updatedAt": "2026-01-01T00:00:00Z"}]}),
            _make_response(_make_langfuse_trace("t1", observations=[_make_generation()])),
        ]

        count = adapter.import_traces()
        assert count == 1

    def test_import_traces_no_results_returns_zero(self, connected_adapter):
        adapter, _, mock_http = connected_adapter
        mock_http.get.return_value = _make_response({"data": []})

        count = adapter.import_traces()
        assert count == 0

    def test_import_traces_respects_since_parameter(self, connected_adapter):
        adapter, _, mock_http = connected_adapter
        mock_http.get.side_effect = [
            _make_response({"data": []}),
        ]

        adapter.import_traces(since="2026-01-15T00:00:00Z")
        call_args = mock_http.get.call_args_list[0]
        params = (
            call_args[1].get("params") or call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("params", {})
        )
        assert params.get("fromTimestamp") == "2026-01-15T00:00:00Z"

    def test_import_traces_respects_limit_parameter(self, connected_adapter):
        adapter, _, mock_http = connected_adapter
        mock_http.get.side_effect = [
            _make_response({"data": []}),
        ]

        adapter.import_traces(limit=10)
        call_args = mock_http.get.call_args_list[0]
        params = call_args[1].get("params") or {}
        assert params.get("limit") == 10

    def test_import_traces_updates_cursor(self, connected_adapter):
        adapter, _, mock_http = connected_adapter
        mock_http.get.side_effect = [
            _make_response({"data": [{"id": "t1", "updatedAt": "2026-03-15T12:00:00Z"}]}),
            _make_response(_make_langfuse_trace("t1")),
        ]

        adapter.import_traces()
        assert adapter._last_cursor == "2026-03-15T12:00:00Z"

    def test_import_traces_raises_when_not_connected(self, mock_client):
        adapter = LangfuseAdapter(mock_client)
        with pytest.raises(RuntimeError, match="not connected"):
            adapter.import_traces()


# ===================================================================
# Import: observations
# ===================================================================


class TestImportObservations:
    def test_generation_emits_model_invoke(self, connected_adapter):
        adapter, uploaded, mock_http = connected_adapter
        gen = _make_generation()
        mock_http.get.side_effect = [
            _make_response({"data": [{"id": "t1", "updatedAt": "2026-01-01T00:00:00Z"}]}),
            _make_response(_make_langfuse_trace("t1", observations=[gen])),
        ]

        adapter.import_traces()
        events = uploaded["events"]
        model_events = find_events(events, "model.invoke")
        assert len(model_events) == 1
        assert model_events[0]["payload"]["model"] == "gpt-4"

    def test_generation_emits_cost_record_with_tokens(self, connected_adapter):
        adapter, uploaded, mock_http = connected_adapter
        gen = _make_generation(prompt_tokens=200, completion_tokens=80)
        mock_http.get.side_effect = [
            _make_response({"data": [{"id": "t1", "updatedAt": "2026-01-01T00:00:00Z"}]}),
            _make_response(_make_langfuse_trace("t1", observations=[gen])),
        ]

        adapter.import_traces()
        events = uploaded["events"]
        cost = find_event(events, "cost.record")
        assert cost["payload"]["tokens_prompt"] == 200
        assert cost["payload"]["tokens_completion"] == 80
        assert cost["payload"]["tokens_total"] == 280

    def test_generation_includes_cost_usd(self, connected_adapter):
        adapter, uploaded, mock_http = connected_adapter
        gen = _make_generation()
        mock_http.get.side_effect = [
            _make_response({"data": [{"id": "t1", "updatedAt": "2026-01-01T00:00:00Z"}]}),
            _make_response(_make_langfuse_trace("t1", observations=[gen])),
        ]

        adapter.import_traces()
        events = uploaded["events"]
        cost = find_event(events, "cost.record")
        assert cost["payload"]["cost_usd"] == 0.005

    def test_span_emits_tool_call(self, connected_adapter):
        adapter, uploaded, mock_http = connected_adapter
        span = _make_span(name="retriever")
        mock_http.get.side_effect = [
            _make_response({"data": [{"id": "t1", "updatedAt": "2026-01-01T00:00:00Z"}]}),
            _make_response(_make_langfuse_trace("t1", observations=[span])),
        ]

        adapter.import_traces()
        events = uploaded["events"]
        tool_events = find_events(events, "tool.call")
        assert len(tool_events) == 1
        assert tool_events[0]["payload"]["name"] == "retriever"

    def test_span_with_code_in_name_emits_agent_code(self, mock_client):
        # agent.code requires l2_agent_code=True (CaptureConfig.full())
        uploaded = capture_framework_trace(mock_client)
        adapter = LangfuseAdapter(mock_client, capture_config=CaptureConfig.full())
        mock_http = _make_mock_http()
        adapter._http = mock_http
        adapter._connected = True
        adapter._host = "https://test.langfuse.com"

        span = _make_span(name="code-executor")
        mock_http.get.side_effect = [
            _make_response({"data": [{"id": "t1", "updatedAt": "2026-01-01T00:00:00Z"}]}),
            _make_response(_make_langfuse_trace("t1", observations=[span])),
        ]

        adapter.import_traces()
        events = uploaded["events"]
        code_events = find_events(events, "agent.code")
        assert len(code_events) == 1

    def test_span_with_exec_in_name_emits_agent_code(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = LangfuseAdapter(mock_client, capture_config=CaptureConfig.full())
        mock_http = _make_mock_http()
        adapter._http = mock_http
        adapter._connected = True
        adapter._host = "https://test.langfuse.com"

        span = _make_span(name="python-exec-tool")
        mock_http.get.side_effect = [
            _make_response({"data": [{"id": "t1", "updatedAt": "2026-01-01T00:00:00Z"}]}),
            _make_response(_make_langfuse_trace("t1", observations=[span])),
        ]

        adapter.import_traces()
        events = uploaded["events"]
        code_events = find_events(events, "agent.code")
        assert len(code_events) == 1

    def test_span_with_sandbox_in_name_emits_agent_code(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = LangfuseAdapter(mock_client, capture_config=CaptureConfig.full())
        mock_http = _make_mock_http()
        adapter._http = mock_http
        adapter._connected = True
        adapter._host = "https://test.langfuse.com"

        span = _make_span(name="sandbox-runner")
        mock_http.get.side_effect = [
            _make_response({"data": [{"id": "t1", "updatedAt": "2026-01-01T00:00:00Z"}]}),
            _make_response(_make_langfuse_trace("t1", observations=[span])),
        ]

        adapter.import_traces()
        events = uploaded["events"]
        code_events = find_events(events, "agent.code")
        assert len(code_events) == 1

    def test_event_observation_emits_agent_state_change(self, connected_adapter):
        adapter, uploaded, mock_http = connected_adapter
        evt = _make_event()
        mock_http.get.side_effect = [
            _make_response({"data": [{"id": "t1", "updatedAt": "2026-01-01T00:00:00Z"}]}),
            _make_response(_make_langfuse_trace("t1", observations=[evt])),
        ]

        adapter.import_traces()
        events = uploaded["events"]
        state_events = find_events(events, "agent.state.change")
        assert len(state_events) == 1
        assert state_events[0]["payload"]["status_message"] == "Processing complete"

    def test_trace_input_output_emit_agent_events(self, connected_adapter):
        adapter, uploaded, mock_http = connected_adapter
        mock_http.get.side_effect = [
            _make_response({"data": [{"id": "t1", "updatedAt": "2026-01-01T00:00:00Z"}]}),
            _make_response(_make_langfuse_trace("t1")),
        ]

        adapter.import_traces()
        events = uploaded["events"]
        input_evt = find_event(events, "agent.input")
        assert input_evt["payload"]["content"] == "Hello, world!"
        output_evt = find_event(events, "agent.output")
        assert output_evt["payload"]["content"] == "Hi there!"


# ===================================================================
# Export: traces
# ===================================================================


class TestExportTraces:
    def _make_ll_events(self):
        """Create a set of LayerLens events for export testing."""
        return [
            {
                "event_type": "agent.input",
                "span_id": "s1",
                "span_name": "my-agent",
                "payload": {"content": "Hello from LL", "name": "my-agent"},
            },
            {
                "event_type": "model.invoke",
                "span_id": "s2",
                "span_name": "llm-call",
                "payload": {
                    "model": "gpt-4",
                    "messages": "What is AI?",
                    "output_message": "AI is...",
                    "tokens_prompt": 50,
                    "tokens_completion": 30,
                    "tokens_total": 80,
                },
            },
            {
                "event_type": "tool.call",
                "span_id": "s3",
                "span_name": "search",
                "payload": {"input": "query", "output": "results"},
            },
            {
                "event_type": "agent.state.change",
                "span_id": "s4",
                "span_name": "status",
                "payload": {"status": "done"},
            },
            {
                "event_type": "agent.output",
                "span_id": "s5",
                "span_name": "my-agent",
                "payload": {"content": "Goodbye from LL"},
            },
        ]

    def test_export_traces_converts_events_to_batch(self, connected_adapter):
        adapter, _, mock_http = connected_adapter
        mock_http.post.return_value = _make_response({})

        events = self._make_ll_events()
        count = adapter.export_traces(events_by_trace={"trace-1": events})

        assert count == 1
        mock_http.post.assert_called_once()
        call_kwargs = mock_http.post.call_args
        batch = call_kwargs[1]["json"]["batch"]
        assert len(batch) > 0

    def test_export_creates_trace_envelope(self, connected_adapter):
        adapter, _, mock_http = connected_adapter
        mock_http.post.return_value = _make_response({})

        events = self._make_ll_events()
        adapter.export_traces(events_by_trace={"trace-1": events})

        batch = mock_http.post.call_args[1]["json"]["batch"]
        trace_items = [b for b in batch if b["type"] == "trace-create"]
        assert len(trace_items) == 1
        body = trace_items[0]["body"]
        assert body["input"] == "Hello from LL"
        assert body["output"] == "Goodbye from LL"
        assert body["name"] == "my-agent"

    def test_model_invoke_becomes_generation_create(self, connected_adapter):
        adapter, _, mock_http = connected_adapter
        mock_http.post.return_value = _make_response({})

        events = self._make_ll_events()
        adapter.export_traces(events_by_trace={"trace-1": events})

        batch = mock_http.post.call_args[1]["json"]["batch"]
        gen_items = [b for b in batch if b["type"] == "generation-create"]
        assert len(gen_items) == 1
        body = gen_items[0]["body"]
        assert body["model"] == "gpt-4"
        assert body["input"] == "What is AI?"
        assert body["output"] == "AI is..."
        assert body["usage"]["promptTokens"] == 50
        assert body["usage"]["completionTokens"] == 30
        assert body["usage"]["totalTokens"] == 80

    def test_tool_call_becomes_span_create(self, connected_adapter):
        adapter, _, mock_http = connected_adapter
        mock_http.post.return_value = _make_response({})

        events = self._make_ll_events()
        adapter.export_traces(events_by_trace={"trace-1": events})

        batch = mock_http.post.call_args[1]["json"]["batch"]
        span_items = [b for b in batch if b["type"] == "span-create"]
        assert len(span_items) == 1
        body = span_items[0]["body"]
        assert body["input"] == "query"
        assert body["output"] == "results"

    def test_other_events_become_event_create(self, connected_adapter):
        adapter, _, mock_http = connected_adapter
        mock_http.post.return_value = _make_response({})

        events = self._make_ll_events()
        adapter.export_traces(events_by_trace={"trace-1": events})

        batch = mock_http.post.call_args[1]["json"]["batch"]
        event_items = [b for b in batch if b["type"] == "event-create"]
        assert len(event_items) == 1
        body = event_items[0]["body"]
        assert body["name"] == "status"

    def test_export_returns_count(self, connected_adapter):
        adapter, _, mock_http = connected_adapter
        mock_http.post.return_value = _make_response({})

        events = self._make_ll_events()
        count = adapter.export_traces(
            events_by_trace={
                "trace-1": events,
                "trace-2": events,
            }
        )
        assert count == 2

    def test_export_empty_returns_zero(self, connected_adapter):
        adapter, _, _ = connected_adapter
        count = adapter.export_traces(events_by_trace={})
        assert count == 0

    def test_export_raises_when_not_connected(self, mock_client):
        adapter = LangfuseAdapter(mock_client)
        with pytest.raises(RuntimeError, match="not connected"):
            adapter.export_traces(events_by_trace={"t": []})


# ===================================================================
# CaptureConfig gating
# ===================================================================


class TestCaptureConfigGating:
    def test_minimal_config_suppresses_model_invoke(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        config = CaptureConfig.minimal()  # l3_model_metadata=False
        adapter = LangfuseAdapter(mock_client, capture_config=config)
        mock_http = _make_mock_http()
        adapter._http = mock_http
        adapter._connected = True
        adapter._host = "https://test.langfuse.com"

        gen = _make_generation()
        mock_http.get.side_effect = [
            _make_response({"data": [{"id": "t1", "updatedAt": "2026-01-01T00:00:00Z"}]}),
            _make_response(_make_langfuse_trace("t1", observations=[gen])),
        ]

        adapter.import_traces()
        events = uploaded["events"]
        model_events = find_events(events, "model.invoke")
        assert len(model_events) == 0

    def test_cost_record_always_emitted(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        config = CaptureConfig.minimal()
        adapter = LangfuseAdapter(mock_client, capture_config=config)
        mock_http = _make_mock_http()
        adapter._http = mock_http
        adapter._connected = True
        adapter._host = "https://test.langfuse.com"

        gen = _make_generation()
        mock_http.get.side_effect = [
            _make_response({"data": [{"id": "t1", "updatedAt": "2026-01-01T00:00:00Z"}]}),
            _make_response(_make_langfuse_trace("t1", observations=[gen])),
        ]

        adapter.import_traces()
        events = uploaded["events"]
        cost_events = find_events(events, "cost.record")
        assert len(cost_events) == 1

    def test_agent_state_change_always_emitted(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        config = CaptureConfig.minimal()
        adapter = LangfuseAdapter(mock_client, capture_config=config)
        mock_http = _make_mock_http()
        adapter._http = mock_http
        adapter._connected = True
        adapter._host = "https://test.langfuse.com"

        evt = _make_event()
        mock_http.get.side_effect = [
            _make_response({"data": [{"id": "t1", "updatedAt": "2026-01-01T00:00:00Z"}]}),
            _make_response(_make_langfuse_trace("t1", observations=[evt])),
        ]

        adapter.import_traces()
        events = uploaded["events"]
        state_events = find_events(events, "agent.state.change")
        assert len(state_events) == 1


# ===================================================================
# Error isolation
# ===================================================================


class TestErrorIsolation:
    def test_import_failure_for_single_trace_doesnt_stop_others(self, connected_adapter):
        adapter, uploaded, mock_http = connected_adapter
        mock_http.get.side_effect = [
            # List traces returns 2
            _make_response(
                {
                    "data": [
                        {"id": "t1", "updatedAt": "2026-01-01T00:00:00Z"},
                        {"id": "t2", "updatedAt": "2026-01-02T00:00:00Z"},
                    ],
                }
            ),
            # Fetch t1 fails
            _make_response(status_code=500),
            # Fetch t2 succeeds
            _make_response(_make_langfuse_trace("t2")),
        ]

        count = adapter.import_traces()
        assert count == 1

    def test_export_failure_for_single_trace_doesnt_stop_others(self, connected_adapter):
        adapter, _, mock_http = connected_adapter

        call_count = {"n": 0}

        def _post_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise Exception("network error")
            return _make_response({})

        mock_http.post.side_effect = _post_side_effect

        events = [
            {"event_type": "agent.input", "span_id": "s1", "payload": {"content": "hi"}},
        ]
        count = adapter.export_traces(
            events_by_trace={
                "trace-fail": events,
                "trace-ok": events,
            }
        )
        assert count == 1


# ===================================================================
# Helper functions
# ===================================================================


class TestHelpers:
    def test_truncate_short_string_unchanged(self):
        assert _truncate("hello") == "hello"

    def test_truncate_long_string(self):
        long_str = "x" * 5000
        result = _truncate(long_str)
        assert len(result) == 2003  # 2000 + "..."
        assert result.endswith("...")
        assert result.startswith("x" * 100)

    def test_truncate_custom_max_len(self):
        result = _truncate("abcdefghij", max_len=5)
        assert result == "abcde..."

    def test_safe_dict_with_dict(self):
        d = {"a": 1}
        assert _safe_dict(d) == {"a": 1}

    def test_safe_dict_with_none(self):
        assert _safe_dict(None) == {}

    def test_safe_dict_with_string(self):
        assert _safe_dict("not a dict") == {}

    def test_safe_dict_with_list(self):
        assert _safe_dict([1, 2, 3]) == {}
