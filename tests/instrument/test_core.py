from __future__ import annotations

import os

import pytest

from layerlens.instrument import span, emit, trace
from layerlens.instrument._context import _current_collector, _current_span_id
from .conftest import find_events, find_event


class TestTraceDecorator:
    def test_basic_trace(self, mock_client):
        @trace(mock_client)
        def my_func(x):
            return x * 2

        result = my_func(5)
        assert result == 10
        mock_client.traces.upload.assert_called_once()

    def test_trace_with_custom_name(self, mock_client, capture_trace):
        @trace(mock_client, name="custom_name")
        def my_func():
            return "ok"

        my_func()
        events = capture_trace["events"]
        agent_input = find_event(events, "agent.input")
        assert agent_input["payload"]["name"] == "custom_name"

    def test_trace_captures_input(self, mock_client, capture_trace):
        @trace(mock_client)
        def my_func(query):
            return "result"

        my_func("hello")
        events = capture_trace["events"]
        agent_input = find_event(events, "agent.input")
        assert agent_input["payload"]["input"] == "hello"

    def test_trace_captures_output(self, mock_client, capture_trace):
        @trace(mock_client)
        def my_func():
            return {"answer": 42}

        my_func()
        events = capture_trace["events"]
        agent_output = find_event(events, "agent.output")
        assert agent_output["payload"]["output"] == {"answer": 42}

    def test_trace_on_error(self, mock_client, capture_trace):
        @trace(mock_client)
        def my_func():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            my_func()

        events = capture_trace["events"]
        error = find_event(events, "agent.error")
        assert error["payload"]["error"] == "boom"
        assert error["payload"]["status"] == "error"

    def test_trace_cleans_up_context(self, mock_client):
        @trace(mock_client)
        def my_func():
            return "ok"

        my_func()
        assert _current_collector.get() is None
        assert _current_span_id.get() is None

    def test_trace_cleans_up_context_on_error(self, mock_client):
        @trace(mock_client)
        def my_func():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            my_func()

        assert _current_collector.get() is None
        assert _current_span_id.get() is None

    def test_events_have_trace_id(self, mock_client, capture_trace):
        @trace(mock_client)
        def my_func():
            return "ok"

        my_func()
        trace_id = capture_trace["trace_id"]
        assert trace_id is not None
        assert len(trace_id) == 16
        for event in capture_trace["events"]:
            assert event["trace_id"] == trace_id

    def test_events_have_sequence_ids(self, mock_client, capture_trace):
        @trace(mock_client)
        def my_func():
            return "ok"

        my_func()
        events = capture_trace["events"]
        seq_ids = [e["sequence_id"] for e in events]
        assert seq_ids == sorted(seq_ids)
        assert seq_ids[0] == 1


class TestSpanContextManager:
    def test_span_creates_child_events(self, mock_client, capture_trace):
        @trace(mock_client)
        def my_func():
            with span("child_span") as span_id:
                emit("tool.call", {"name": "search", "input": "query"})
            return "done"

        my_func()
        events = capture_trace["events"]
        tool_call = find_event(events, "tool.call")
        assert tool_call["payload"]["name"] == "search"
        # tool.call should have a different span_id than root
        agent_input = find_event(events, "agent.input")
        assert tool_call["span_id"] != agent_input["span_id"]
        # tool.call parent should be root span
        assert tool_call["parent_span_id"] == agent_input["span_id"]

    def test_nested_spans(self, mock_client, capture_trace):
        @trace(mock_client)
        def my_func():
            with span("outer") as outer_id:
                emit("agent.input", {"name": "outer"})
                with span("inner") as inner_id:
                    emit("tool.call", {"name": "inner_tool", "input": "x"})
            return "done"

        my_func()
        events = capture_trace["events"]
        # Find the events emitted inside spans
        inner_tool = [e for e in events if e["event_type"] == "tool.call"][0]
        outer_input = [e for e in events if e["event_type"] == "agent.input" and e["payload"].get("name") == "outer"][0]
        # inner_tool's parent should be the outer span
        assert inner_tool["parent_span_id"] == outer_input["span_id"]

    def test_span_without_trace_noops(self):
        with span("orphan") as span_id:
            assert isinstance(span_id, str)
            assert len(span_id) == 16

    def test_multiple_sibling_spans(self, mock_client, capture_trace):
        @trace(mock_client)
        def my_func():
            with span("retrieve"):
                emit("tool.call", {"name": "retriever", "input": "q"})
            with span("generate"):
                emit("model.invoke", {"name": "gpt-4"})
            return "done"

        my_func()
        events = capture_trace["events"]
        tool_call = find_event(events, "tool.call")
        model_invoke = find_event(events, "model.invoke")
        root_input = find_event(events, "agent.input")
        # Both siblings should have root as parent
        assert tool_call["parent_span_id"] == root_input["span_id"]
        assert model_invoke["parent_span_id"] == root_input["span_id"]
        # But different span_ids
        assert tool_call["span_id"] != model_invoke["span_id"]


class TestEmitFunction:
    def test_emit_outside_trace_noops(self):
        # Should not raise
        emit("tool.call", {"name": "test"})

    def test_emit_inside_trace(self, mock_client, capture_trace):
        @trace(mock_client)
        def my_func():
            emit("tool.call", {"name": "search", "input": "query"})
            return "ok"

        my_func()
        events = capture_trace["events"]
        tool_call = find_event(events, "tool.call")
        assert tool_call["payload"]["name"] == "search"


class TestAttestationIntegration:
    def test_attestation_present(self, mock_client, capture_trace):
        @trace(mock_client)
        def my_func():
            return "ok"

        my_func()
        attestation = capture_trace["attestation"]
        assert "root_hash" in attestation
        assert "chain" in attestation
        assert attestation["schema_version"] == "1.0"
