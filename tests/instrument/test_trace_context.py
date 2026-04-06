"""Tests for trace context: shared collectors, context propagation,
callback scope, and upload circuit breaker.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import pytest

from layerlens.instrument import (
    trace,
    trace_context,
    emit,
    span,
    get_trace_context,
    CaptureConfig,
)
from layerlens.instrument._context import _current_collector, _current_span_id
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument import _upload
from layerlens.instrument.adapters.frameworks._base_framework import FrameworkAdapter

from .conftest import find_event, find_events


# ---------------------------------------------------------------------------
# Minimal concrete adapter for testing
# ---------------------------------------------------------------------------

class StubAdapter(FrameworkAdapter):
    name = "stub"

    def fire_event(self, event_type: str, payload: Dict[str, Any],
                   span_id: Optional[str] = None,
                   parent_span_id: Optional[str] = None) -> None:
        kwargs: Dict[str, Any] = {"span_name": event_type}
        if span_id is not None:
            kwargs["span_id"] = span_id
        if parent_span_id is not None:
            kwargs["parent_span_id"] = parent_span_id
        self._emit(event_type, payload, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_client():
    client = Mock()
    client.traces = Mock()
    client.traces.upload = Mock()
    return client


@pytest.fixture
def capture_trace(mock_client):
    """Capture uploaded trace payloads. Supports multiple uploads."""
    uploads: List[Dict[str, Any]] = []

    def _capture(path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        uploads.append(data[0])

    mock_client.traces.upload.side_effect = _capture
    return uploads


@pytest.fixture(autouse=True)
def reset_upload_channels():
    """Clear all upload channels between tests."""
    _upload._channels.clear()
    yield
    _upload._channels.clear()


# ===================================================================
# 1. Shared trace_id via @trace
# ===================================================================

class TestSharedCollectorViaTrace:

    def test_framework_adapter_shares_trace_id_with_trace_decorator(
        self, mock_client, capture_trace,
    ):
        adapter = StubAdapter(mock_client)
        adapter.connect()

        @trace(mock_client)
        def agent_run():
            adapter.fire_event("agent.lifecycle", {"action": "crew.start"})
            return "done"

        agent_run()

        assert len(capture_trace) == 1
        events = capture_trace[0]["events"]
        lifecycle = find_event(events, "agent.lifecycle")
        agent_input = find_event(events, "agent.input")
        assert lifecycle["trace_id"] == agent_input["trace_id"]

    def test_multiple_adapters_share_same_trace(
        self, mock_client, capture_trace,
    ):
        adapter_a = StubAdapter(mock_client)
        adapter_b = StubAdapter(mock_client)
        adapter_a.connect()
        adapter_b.connect()

        @trace(mock_client)
        def agent_run():
            adapter_a.fire_event("agent.lifecycle", {"source": "A"})
            adapter_b.fire_event("agent.lifecycle", {"source": "B"})
            return "done"

        agent_run()

        assert len(capture_trace) == 1
        events = capture_trace[0]["events"]
        lifecycles = find_events(events, "agent.lifecycle")
        assert len(lifecycles) == 2
        assert lifecycles[0]["trace_id"] == lifecycles[1]["trace_id"]

    def test_framework_adapter_standalone_creates_own_trace(
        self, mock_client, capture_trace,
    ):
        adapter = StubAdapter(mock_client)
        adapter.connect()
        adapter._begin_run()
        adapter.fire_event("agent.lifecycle", {"action": "standalone"})
        adapter._end_run()
        adapter.disconnect()

        assert len(capture_trace) == 1
        events = capture_trace[0]["events"]
        assert len(events) == 1
        assert events[0]["event_type"] == "agent.lifecycle"


# ===================================================================
# 2. Cross-adapter parent-child spans
# ===================================================================

class TestCrossAdapterSpanHierarchy:

    def test_framework_events_parent_to_trace_root_span(
        self, mock_client, capture_trace,
    ):
        adapter = StubAdapter(mock_client)
        adapter.connect()

        @trace(mock_client)
        def agent_run():
            adapter.fire_event("agent.lifecycle", {"action": "start"})
            return "done"

        agent_run()

        events = capture_trace[0]["events"]
        agent_input = find_event(events, "agent.input")
        lifecycle = find_event(events, "agent.lifecycle")
        root_span = agent_input["span_id"]
        assert lifecycle["parent_span_id"] == root_span

    def test_framework_events_parent_to_active_span(
        self, mock_client, capture_trace,
    ):
        adapter = StubAdapter(mock_client)
        adapter.connect()

        @trace(mock_client)
        def agent_run():
            with span("retrieval"):
                adapter.fire_event("tool.call", {"name": "search", "input": "q"})
            return "done"

        agent_run()

        events = capture_trace[0]["events"]
        agent_input = find_event(events, "agent.input")
        tool_call = find_event(events, "tool.call")
        assert tool_call["parent_span_id"] is not None
        assert tool_call["trace_id"] == agent_input["trace_id"]

    def test_adapter_with_explicit_parent_overrides_default(
        self, mock_client, capture_trace,
    ):
        adapter = StubAdapter(mock_client)
        adapter.connect()
        explicit_parent = "custom_parent_id"

        @trace(mock_client)
        def agent_run():
            adapter.fire_event(
                "agent.lifecycle", {"action": "step"},
                parent_span_id=explicit_parent,
            )
            return "done"

        agent_run()

        events = capture_trace[0]["events"]
        lifecycle = find_event(events, "agent.lifecycle")
        assert lifecycle["parent_span_id"] == explicit_parent


# ===================================================================
# 3. trace_context()
# ===================================================================

class TestTraceContext:

    def test_creates_shared_collector(self, mock_client, capture_trace):
        adapter_a = StubAdapter(mock_client)
        adapter_b = StubAdapter(mock_client)
        adapter_a.connect()
        adapter_b.connect()

        with trace_context(mock_client):
            adapter_a.fire_event("agent.lifecycle", {"source": "A"})
            adapter_b.fire_event("agent.lifecycle", {"source": "B"})

        assert len(capture_trace) == 1
        events = capture_trace[0]["events"]
        assert len(events) == 2
        assert events[0]["trace_id"] == events[1]["trace_id"]

    def test_flushes_on_exit(self, mock_client, capture_trace):
        with trace_context(mock_client):
            emit("tool.call", {"name": "test", "input": "x"})

        assert len(capture_trace) == 1

    def test_cleans_up_on_exit(self, mock_client):
        with trace_context(mock_client):
            assert _current_collector.get() is not None

        assert _current_collector.get() is None
        assert _current_span_id.get() is None

    def test_cleans_up_on_error(self, mock_client):
        with pytest.raises(RuntimeError):
            with trace_context(mock_client):
                raise RuntimeError("boom")

        assert _current_collector.get() is None
        assert _current_span_id.get() is None

    def test_yields_collector(self, mock_client):
        with trace_context(mock_client) as collector:
            assert isinstance(collector, TraceCollector)
            assert len(collector.trace_id) == 16

    def test_with_custom_capture_config(self, mock_client, capture_trace):
        config = CaptureConfig.standard()

        with trace_context(mock_client, capture_config=config):
            emit("tool.call", {"name": "test", "input": "x"})

        assert len(capture_trace) == 1
        assert capture_trace[0]["capture_config"] == config.to_dict()


# ===================================================================
# 4. Context serialisation (get_trace_context / from_context)
# ===================================================================

class TestGetTraceContext:

    def test_returns_none_outside_trace(self):
        assert get_trace_context() is None

    def test_returns_dict_inside_trace(self, mock_client, capture_trace):
        @trace(mock_client)
        def run():
            ctx = get_trace_context()
            assert ctx is not None
            assert "trace_id" in ctx
            assert "span_id" in ctx
            assert "parent_span_id" in ctx
            assert ctx["version"] == 1
            return ctx

        ctx = run()
        assert len(ctx["trace_id"]) == 16
        assert len(ctx["span_id"]) == 16

    def test_returns_dict_inside_trace_context(self, mock_client, capture_trace):
        with trace_context(mock_client):
            ctx = get_trace_context()
            assert ctx is not None
            assert len(ctx["trace_id"]) == 16

    def test_span_id_updates_inside_child_span(self, mock_client, capture_trace):
        @trace(mock_client)
        def run():
            ctx_outer = get_trace_context()
            with span("inner"):
                ctx_inner = get_trace_context()
            return ctx_outer, ctx_inner

        outer, inner = run()
        assert outer["trace_id"] == inner["trace_id"]
        assert outer["span_id"] != inner["span_id"]


class TestTraceContextFromContext:

    def test_restores_trace_id(self, mock_client, capture_trace):
        with trace_context(mock_client):
            original_ctx = get_trace_context()
            emit("tool.call", {"name": "origin", "input": "x"})

        original_trace_id = original_ctx["trace_id"]

        with trace_context(mock_client, from_context=original_ctx) as restored:
            assert restored.trace_id == original_trace_id
            emit("tool.call", {"name": "remote", "input": "y"})

        assert len(capture_trace) == 2
        assert capture_trace[0]["trace_id"] == capture_trace[1]["trace_id"]

    def test_creates_child_span(self, mock_client, capture_trace):
        with trace_context(mock_client):
            original_ctx = get_trace_context()
            emit("tool.call", {"name": "origin", "input": "x"})

        with trace_context(mock_client, from_context=original_ctx):
            ctx_inside = get_trace_context()

        assert ctx_inside["span_id"] != original_ctx["span_id"]
        assert ctx_inside["trace_id"] == original_ctx["trace_id"]


# ===================================================================
# 5. Flush semantics
# ===================================================================

class TestFlushSemantics:

    def test_adapter_disconnect_does_not_flush_shared_collector(
        self, mock_client, capture_trace,
    ):
        adapter = StubAdapter(mock_client)
        adapter.connect()

        @trace(mock_client)
        def agent_run():
            adapter.fire_event("agent.lifecycle", {"action": "start"})
            adapter.disconnect()
            emit("tool.call", {"name": "post_disconnect", "input": "x"})
            return "done"

        agent_run()

        assert len(capture_trace) == 1
        events = capture_trace[0]["events"]
        types = [e["event_type"] for e in events]
        assert "agent.lifecycle" in types
        assert "tool.call" in types
        assert "agent.output" in types

    def test_adapter_begin_end_run_flushes_collector(
        self, mock_client, capture_trace,
    ):
        adapter = StubAdapter(mock_client)
        adapter.connect()
        adapter._begin_run()
        adapter.fire_event("agent.lifecycle", {"action": "standalone"})
        adapter._end_run()
        adapter.disconnect()

        assert len(capture_trace) == 1

    def test_multiple_adapters_disconnect_independently_under_shared_context(
        self, mock_client, capture_trace,
    ):
        adapter_a = StubAdapter(mock_client)
        adapter_b = StubAdapter(mock_client)
        adapter_a.connect()
        adapter_b.connect()

        with trace_context(mock_client):
            adapter_a.fire_event("agent.lifecycle", {"source": "A"})
            adapter_a.disconnect()
            adapter_b.fire_event("agent.lifecycle", {"source": "B"})
            adapter_b.disconnect()

        assert len(capture_trace) == 1
        events = capture_trace[0]["events"]
        sources = [e["payload"]["source"] for e in events]
        assert "A" in sources
        assert "B" in sources


# ===================================================================
# 6. Run lifecycle (_begin_run / _end_run)
# ===================================================================

class TestRunLifecycle:

    def test_begin_run_pushes_collector_standalone(self, mock_client, capture_trace):
        adapter = StubAdapter(mock_client)
        adapter.connect()

        assert _current_collector.get() is None
        run = adapter._begin_run()
        assert _current_collector.get() is not None
        assert _current_span_id.get() == run.root_span_id
        emit("tool.call", {"name": "test", "input": "x"})
        adapter._end_run()

        assert len(capture_trace) == 1

    def test_begin_run_preserves_shared_collector(self, mock_client, capture_trace):
        adapter = StubAdapter(mock_client)
        adapter.connect()

        @trace(mock_client)
        def run():
            shared_collector = _current_collector.get()
            adapter_run = adapter._begin_run()
            assert adapter_run.collector is shared_collector
            emit("tool.call", {"name": "inner_tool", "input": "x"})
            adapter._end_run()
            return "done"

        run()

        assert len(capture_trace) == 1
        events = capture_trace[0]["events"]
        tool_call = find_event(events, "tool.call")
        assert tool_call["payload"]["name"] == "inner_tool"

    def test_end_run_cleans_up_on_error(self, mock_client):
        adapter = StubAdapter(mock_client)
        adapter.connect()

        adapter._begin_run()
        assert _current_collector.get() is not None
        adapter._end_run()
        assert _current_collector.get() is None

    def test_begin_run_makes_providers_visible(self, mock_client, capture_trace):
        adapter = StubAdapter(mock_client)
        adapter.connect()

        def fake_agent_run(prompt):
            assert _current_collector.get() is not None
            emit("model.invoke", {"model": "gpt-4", "input": prompt})
            return "result"

        assert _current_collector.get() is None
        adapter._begin_run()
        result = fake_agent_run("hello")
        adapter._end_run()
        assert result == "result"
        assert _current_collector.get() is None

        assert len(capture_trace) == 1
        events = capture_trace[0]["events"]
        model_event = find_event(events, "model.invoke")
        assert model_event["payload"]["model"] == "gpt-4"

    def test_begin_run_under_shared_context(self, mock_client, capture_trace):
        adapter = StubAdapter(mock_client)
        adapter.connect()

        @trace(mock_client)
        def run():
            adapter._begin_run()
            emit("model.invoke", {"model": "gpt-4", "input": "hello"})
            adapter._end_run()
            return "done"

        run()
        assert len(capture_trace) == 1
        events = capture_trace[0]["events"]
        assert find_event(events, "model.invoke")
        assert find_event(events, "agent.input")


# ===================================================================
# 7. Upload circuit breaker
# ===================================================================

class TestUploadCircuitBreaker:

    def _channel(self, mock_client):
        """Get or create the upload channel for mock_client."""
        return _upload._get_channel(mock_client)

    def test_successful_upload(self, mock_client, capture_trace):
        with trace_context(mock_client):
            emit("tool.call", {"name": "test", "input": "x"})

        assert len(capture_trace) == 1
        assert self._channel(mock_client)._error_count == 0

    def test_upload_failure_records_error(self, mock_client):
        mock_client.traces.upload.side_effect = RuntimeError("network error")

        with trace_context(mock_client):
            emit("tool.call", {"name": "test", "input": "x"})

        ch = self._channel(mock_client)
        assert ch._error_count == 1
        assert not ch._circuit_open

    def test_circuit_opens_after_threshold(self, mock_client):
        mock_client.traces.upload.side_effect = RuntimeError("network error")

        for _ in range(_upload.UploadChannel._THRESHOLD):
            with trace_context(mock_client):
                emit("tool.call", {"name": "test", "input": "x"})

        ch = self._channel(mock_client)
        assert ch._circuit_open
        assert ch._error_count == _upload.UploadChannel._THRESHOLD

    def test_open_circuit_skips_upload(self, mock_client):
        ch = self._channel(mock_client)
        ch._circuit_open = True
        ch._opened_at = __import__("time").monotonic()

        with trace_context(mock_client):
            emit("tool.call", {"name": "test", "input": "x"})

        mock_client.traces.upload.assert_not_called()

    def test_circuit_resets_after_cooldown(self, mock_client, capture_trace):
        ch = self._channel(mock_client)
        ch._circuit_open = True
        ch._error_count = _upload.UploadChannel._THRESHOLD
        ch._opened_at = (
            __import__("time").monotonic() - _upload.UploadChannel._COOLDOWN_S - 1
        )

        with trace_context(mock_client):
            emit("tool.call", {"name": "test", "input": "x"})

        assert len(capture_trace) == 1
        assert not ch._circuit_open
        assert ch._error_count == 0

    def test_success_after_failures_resets_count(self, mock_client, capture_trace):
        ch = self._channel(mock_client)
        ch._error_count = 5

        with trace_context(mock_client):
            emit("tool.call", {"name": "test", "input": "x"})

        assert ch._error_count == 0

    def test_protects_trace_decorator(self, mock_client):
        ch = self._channel(mock_client)
        ch._circuit_open = True
        ch._opened_at = __import__("time").monotonic()

        @trace(mock_client)
        def run():
            emit("tool.call", {"name": "test", "input": "x"})
            return "done"

        run()
        mock_client.traces.upload.assert_not_called()

    def test_protects_framework_adapter(self, mock_client):
        adapter = StubAdapter(mock_client)
        adapter.connect()

        ch = self._channel(mock_client)
        ch._circuit_open = True
        ch._opened_at = __import__("time").monotonic()

        with trace_context(mock_client):
            adapter.fire_event("tool.call", {"name": "test", "input": "x"})

        mock_client.traces.upload.assert_not_called()


# ===================================================================
# 8. Edge cases
# ===================================================================

class TestEdgeCases:

    def test_adapter_used_across_multiple_traces(
        self, mock_client, capture_trace,
    ):
        adapter = StubAdapter(mock_client)
        adapter.connect()

        @trace(mock_client)
        def run_1():
            adapter.fire_event("agent.lifecycle", {"run": 1})
            return "done"

        @trace(mock_client)
        def run_2():
            adapter.fire_event("agent.lifecycle", {"run": 2})
            return "done"

        run_1()
        run_2()

        assert len(capture_trace) == 2
        assert capture_trace[0]["trace_id"] != capture_trace[1]["trace_id"]

    def test_no_events_means_no_upload(self, mock_client):
        with trace_context(mock_client):
            pass

        mock_client.traces.upload.assert_not_called()

    def test_standalone_adapter_unaffected_by_previous_shared_context(
        self, mock_client, capture_trace,
    ):
        adapter = StubAdapter(mock_client)
        adapter.connect()

        with trace_context(mock_client):
            adapter.fire_event("agent.lifecycle", {"phase": "shared"})

        adapter.disconnect()

        adapter = StubAdapter(mock_client)
        adapter.connect()
        adapter._begin_run()
        adapter.fire_event("agent.lifecycle", {"phase": "standalone"})
        adapter._end_run()
        adapter.disconnect()

        assert len(capture_trace) == 2
        assert capture_trace[0]["trace_id"] != capture_trace[1]["trace_id"]
