"""Tests for the OpenAI Agents SDK adapter using real SDK types.

Uses real TracingProcessor, SpanImpl, Trace, and span data types.
No mocking of Agents SDK internals — only our mock_client for upload capture.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

import sys
if sys.version_info < (3, 10):
    pytest.skip("openai-agents requires Python >= 3.10", allow_module_level=True)
try:
    import agents  # noqa: F401
except (ImportError, Exception):
    pytest.skip("openai-agents not installed or incompatible", allow_module_level=True)

from agents.tracing import TracingProcessor, set_trace_processors  # noqa: E402
from agents.tracing.spans import SpanImpl  # noqa: E402
from agents.tracing.traces import TraceImpl  # noqa: E402
from agents.tracing.span_data import (  # noqa: E402
    AgentSpanData,
    HandoffSpanData,
    FunctionSpanData,
    GuardrailSpanData,
    GenerationSpanData,
)

from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.openai_agents import OpenAIAgentsAdapter  # noqa: E402

from .conftest import capture_framework_trace, find_event, find_events  # noqa: E402

# -- Helpers --


class _NoOpProcessor(TracingProcessor):
    """Minimal processor that does nothing — used to reset global state."""

    def on_trace_start(self, trace):
        pass

    def on_trace_end(self, trace):
        pass

    def on_span_start(self, span):
        pass

    def on_span_end(self, span):
        pass

    def shutdown(self):
        pass

    def force_flush(self):
        pass


_noop = _NoOpProcessor()


def _make_span(
    _adapter: Any,
    trace_id: str,
    span_id: str,
    span_data: Any,
    parent_id: str | None = None,
) -> SpanImpl:
    """Create a real SpanImpl for testing.

    Uses a NoOpProcessor internally so span.start()/finish() don't
    double-trigger our adapter. Tests call adapter.on_span_end() manually.
    The _adapter param is accepted for call-site readability but unused.
    """
    return SpanImpl(
        trace_id=trace_id,
        span_id=span_id,
        parent_id=parent_id,
        processor=_noop,
        span_data=span_data,
        tracing_api_key=None,
    )


def _make_trace(name: str = "test_trace", trace_id: str = "trace_001", processor: Any = None) -> TraceImpl:
    """Create a real TraceImpl for testing.

    If processor is None, uses a no-op processor. In actual tests,
    pass the adapter's processor so trace lifecycle events route correctly.
    """
    proc = processor or _NoOpProcessor()
    return TraceImpl(name=name, trace_id=trace_id, group_id=None, metadata=None, processor=proc)


# -- Fixtures --


@pytest.fixture
def adapter_and_trace(mock_client):
    """Create adapter, connect, yield (adapter, uploaded_dict), then clean up.

    The adapter IS the TracingProcessor, so tests call adapter.on_span_end() etc.
    directly — no separate processor object.
    """
    uploaded = capture_framework_trace(mock_client)
    adapter = OpenAIAgentsAdapter(mock_client)
    adapter.connect()
    yield adapter, uploaded
    adapter.disconnect()
    set_trace_processors([])  # ensure clean slate


@pytest.fixture(autouse=True)
def clean_processors():
    """Reset global trace processors after each test."""
    yield
    set_trace_processors([])


# -- Tests --


class TestOpenAIAgentsAdapterLifecycle:
    def test_connect_sets_connected(self, mock_client):
        adapter = OpenAIAgentsAdapter(mock_client)
        adapter.connect()
        assert adapter.is_connected
        info = adapter.adapter_info()
        assert info.name == "openai-agents"
        assert info.adapter_type == "framework"
        adapter.disconnect()

    def test_disconnect_clears_state(self, mock_client):
        adapter = OpenAIAgentsAdapter(mock_client)
        adapter.connect()
        adapter.disconnect()
        assert not adapter.is_connected

    def test_connect_without_agents_raises(self, mock_client, monkeypatch):
        import layerlens.instrument.adapters.frameworks.openai_agents as mod

        monkeypatch.setattr(mod, "_HAS_OPENAI_AGENTS", False)
        adapter = OpenAIAgentsAdapter(mock_client)
        with pytest.raises(ImportError, match="openai-agents"):
            adapter.connect()


class TestAgentSpans:
    """Test agent span handling with real AgentSpanData."""

    def test_agent_span_emits_input_and_output(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace

        trace = _make_trace(trace_id="t1")

        # Simulate trace + agent span lifecycle
        adapter.on_trace_start(trace)

        span = _make_span(
            adapter,"t1", "s_agent",
            AgentSpanData(name="research_agent", tools=["search", "browse"], handoffs=["writer"]),
        )
        span.start()
        adapter.on_span_start(span)
        span.finish()
        adapter.on_span_end(span)

        adapter.on_trace_end(trace)

        events = uploaded["events"]
        assert len(events) >= 2

        inp = find_event(events, "agent.input")
        assert inp["payload"]["agent_name"] == "research_agent"
        assert inp["payload"]["tools"] == ["search", "browse"]
        assert inp["payload"]["handoffs"] == ["writer"]
        assert inp["payload"]["framework"] == "openai-agents"
        assert inp["span_id"] == "s_agent"

        out = find_event(events, "agent.output")
        assert out["payload"]["agent_name"] == "research_agent"
        assert out["payload"]["status"] == "ok"
        assert out["span_id"] == "s_agent"

    def test_agent_span_with_error(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace

        trace = _make_trace(trace_id="t_err")

        adapter.on_trace_start(trace)

        span = _make_span(adapter,"t_err", "s_err", AgentSpanData(name="buggy_agent"))
        span.start()
        adapter.on_span_start(span)
        span.set_error({"message": "Agent crashed", "data": {"step": 3}})
        span.finish()
        adapter.on_span_end(span)

        adapter.on_trace_end(trace)

        events = uploaded["events"]
        err = find_event(events, "agent.error")
        assert err["payload"]["agent_name"] == "buggy_agent"
        assert err["payload"]["status"] == "error"
        assert "Agent crashed" in str(err["payload"]["error"])

    def test_nested_agent_spans(self, adapter_and_trace):
        """Multi-agent: parent agent delegates to child agent."""
        adapter, uploaded = adapter_and_trace

        trace = _make_trace(trace_id="t_nested")

        adapter.on_trace_start(trace)

        # Parent agent
        parent = _make_span(adapter,"t_nested", "s_parent", AgentSpanData(name="orchestrator"))
        parent.start()
        adapter.on_span_start(parent)

        # Child agent
        child = _make_span(adapter,"t_nested", "s_child", AgentSpanData(name="researcher"), parent_id="s_parent")
        child.start()
        adapter.on_span_start(child)
        child.finish()
        adapter.on_span_end(child)

        parent.finish()
        adapter.on_span_end(parent)

        adapter.on_trace_end(trace)

        events = uploaded["events"]
        agent_inputs = find_events(events, "agent.input")
        assert len(agent_inputs) == 2

        # Child should have parent_span_id pointing to parent
        child_input = [e for e in agent_inputs if e["payload"]["agent_name"] == "researcher"][0]
        assert child_input["parent_span_id"] == "s_parent"


class TestGenerationSpans:
    """Test LLM generation span handling."""

    def test_generation_emits_model_invoke(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace

        trace = _make_trace(trace_id="t_gen")

        adapter.on_trace_start(trace)

        span = _make_span(
            adapter,"t_gen", "s_gen",
            GenerationSpanData(
                input=[{"role": "user", "content": "What is 2+2?"}],
                output=[{"role": "assistant", "content": "4"}],
                model="gpt-4o",
                model_config={"temperature": 0.7},
                usage={"input_tokens": 50, "output_tokens": 10},
            ),
            parent_id="s_agent",
        )
        span.start()
        adapter.on_span_start(span)
        span.finish()
        adapter.on_span_end(span)

        adapter.on_trace_end(trace)

        events = uploaded["events"]
        me = find_event(events, "model.invoke")
        assert me["payload"]["model"] == "gpt-4o"
        assert me["payload"]["tokens_prompt"] == 50
        assert me["payload"]["tokens_completion"] == 10
        assert me["payload"]["tokens_total"] == 60
        assert me["payload"]["latency_ms"] >= 0
        assert me["payload"]["messages"] == [{"role": "user", "content": "What is 2+2?"}]
        assert me["payload"]["output_message"] == [{"role": "assistant", "content": "4"}]
        assert me["span_id"] == "s_gen"
        assert me["parent_span_id"] == "s_agent"

    def test_generation_emits_cost_record(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace

        trace = _make_trace(trace_id="t_cost")

        adapter.on_trace_start(trace)

        span = _make_span(
            adapter,"t_cost", "s_cost",
            GenerationSpanData(
                input=[], output=[], model="gpt-4o-mini",
                model_config={},
                usage={"input_tokens": 100, "output_tokens": 25},
            ),
        )
        span.start()
        span.finish()
        adapter.on_span_end(span)
        adapter.on_trace_end(trace)

        events = uploaded["events"]
        cost = find_event(events, "cost.record")
        assert cost["payload"]["model"] == "gpt-4o-mini"
        assert cost["payload"]["tokens_prompt"] == 100
        assert cost["payload"]["tokens_completion"] == 25
        assert cost["payload"]["tokens_total"] == 125

    def test_generation_error(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace

        trace = _make_trace(trace_id="t_gen_err")

        adapter.on_trace_start(trace)

        span = _make_span(
            adapter,"t_gen_err", "s_gen_err",
            GenerationSpanData(
                input=[{"role": "user", "content": "fail"}],
                output=[], model="gpt-4o",
                model_config={}, usage={},
            ),
        )
        span.start()
        span.set_error({"message": "Rate limit exceeded"})
        span.finish()
        adapter.on_span_end(span)
        adapter.on_trace_end(trace)

        events = uploaded["events"]
        err = find_event(events, "agent.error")
        assert "Rate limit" in str(err["payload"]["error"])

    def test_multiple_generations(self, adapter_and_trace):
        """Agent makes multiple LLM calls (e.g. tool use loop)."""
        adapter, uploaded = adapter_and_trace

        trace = _make_trace(trace_id="t_multi_gen")

        adapter.on_trace_start(trace)

        for i, (inp_tok, out_tok) in enumerate([(50, 15), (80, 20)]):
            span = _make_span(
                adapter,"t_multi_gen", f"s_gen_{i}",
                GenerationSpanData(
                    input=[], output=[], model="gpt-4o",
                    model_config={},
                    usage={"input_tokens": inp_tok, "output_tokens": out_tok},
                ),
                parent_id="s_agent",
            )
            span.start()
            span.finish()
            adapter.on_span_end(span)

        adapter.on_trace_end(trace)

        events = uploaded["events"]
        gens = find_events(events, "model.invoke")
        assert len(gens) == 2
        assert gens[0]["span_id"] != gens[1]["span_id"]


class TestFunctionSpans:
    """Test tool/function span handling."""

    def test_function_span_emits_tool_call(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace

        trace = _make_trace(trace_id="t_func")

        adapter.on_trace_start(trace)

        span = _make_span(
            adapter,"t_func", "s_func",
            FunctionSpanData(name="get_weather", input='{"city":"NYC"}', output='{"temp":72}'),
            parent_id="s_agent",
        )
        span.start()
        span.finish()
        adapter.on_span_end(span)
        adapter.on_trace_end(trace)

        events = uploaded["events"]
        tc = find_event(events, "tool.call")
        assert tc["payload"]["tool_name"] == "get_weather"
        assert tc["payload"]["input"] == '{"city":"NYC"}'
        assert tc["parent_span_id"] == "s_agent"

        tr = find_event(events, "tool.result")
        assert tr["payload"]["tool_name"] == "get_weather"
        assert tr["payload"]["output"] == '{"temp":72}'
        assert tr["payload"]["latency_ms"] >= 0
        assert tr["parent_span_id"] == "s_agent"

    def test_function_span_with_error(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace

        trace = _make_trace(trace_id="t_func_err")

        adapter.on_trace_start(trace)

        span = _make_span(
            adapter,"t_func_err", "s_func_err",
            FunctionSpanData(name="dangerous_tool", input="delete all", output=None),
        )
        span.start()
        span.set_error({"message": "Permission denied"})
        span.finish()
        adapter.on_span_end(span)
        adapter.on_trace_end(trace)

        events = uploaded["events"]
        err = find_event(events, "agent.error")
        assert err["payload"]["tool_name"] == "dangerous_tool"
        assert "Permission denied" in str(err["payload"]["error"])

    def test_function_span_with_mcp(self, adapter_and_trace):
        """Function spans can include MCP data."""
        adapter, uploaded = adapter_and_trace

        trace = _make_trace(trace_id="t_mcp")

        adapter.on_trace_start(trace)

        span = _make_span(
            adapter,"t_mcp", "s_mcp",
            FunctionSpanData(name="mcp_tool", input="query", output="result"),
        )
        # Set mcp_data manually
        span.span_data.mcp_data = {"server": "my-mcp-server", "tool": "query_db"}
        span.start()
        span.finish()
        adapter.on_span_end(span)
        adapter.on_trace_end(trace)

        events = uploaded["events"]
        tc = find_event(events, "tool.call")
        assert tc["payload"]["mcp_data"]["server"] == "my-mcp-server"


class TestHandoffSpans:
    """Test handoff span handling."""

    def test_handoff_emits_event(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace

        trace = _make_trace(trace_id="t_handoff")

        adapter.on_trace_start(trace)

        span = _make_span(
            adapter,"t_handoff", "s_handoff",
            HandoffSpanData(from_agent="triage", to_agent="specialist"),
            parent_id="s_agent",
        )
        span.start()
        span.finish()
        adapter.on_span_end(span)
        adapter.on_trace_end(trace)

        events = uploaded["events"]
        ho = find_event(events, "agent.handoff")
        assert ho["payload"]["from_agent"] == "triage"
        assert ho["payload"]["to_agent"] == "specialist"
        assert ho["parent_span_id"] == "s_agent"


class TestGuardrailSpans:
    """Test guardrail span handling."""

    def test_guardrail_emits_evaluation_result(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace

        trace = _make_trace(trace_id="t_guard")

        adapter.on_trace_start(trace)

        span = _make_span(
            adapter,"t_guard", "s_guard",
            GuardrailSpanData(name="content_filter", triggered=True),
        )
        span.start()
        span.finish()
        adapter.on_span_end(span)
        adapter.on_trace_end(trace)

        events = uploaded["events"]
        ev = find_event(events, "evaluation.result")
        assert ev["payload"]["guardrail_name"] == "content_filter"
        assert ev["payload"]["triggered"] is True

    def test_guardrail_not_triggered(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace

        trace = _make_trace(trace_id="t_guard2")

        adapter.on_trace_start(trace)

        span = _make_span(
            adapter,"t_guard2", "s_guard2",
            GuardrailSpanData(name="pii_detector", triggered=False),
        )
        span.start()
        span.finish()
        adapter.on_span_end(span)
        adapter.on_trace_end(trace)

        events = uploaded["events"]
        ev = find_event(events, "evaluation.result")
        assert ev["payload"]["triggered"] is False


class TestFullAgentFlow:
    """End-to-end test simulating a complete agent run with tools and handoff."""

    def test_complete_flow(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace

        trace = _make_trace(trace_id="t_flow", name="customer_support")

        adapter.on_trace_start(trace)

        # Agent span
        agent = _make_span(adapter,"t_flow", "s_agent", AgentSpanData(name="triage", tools=["classify"]))
        agent.start()
        adapter.on_span_start(agent)

        # LLM call
        gen = _make_span(
            adapter,"t_flow", "s_gen",
            GenerationSpanData(
                input=[{"role": "user", "content": "I need help"}],
                output=[{"role": "assistant", "content": "Let me classify this"}],
                model="gpt-4o-mini",
                model_config={},
                usage={"input_tokens": 30, "output_tokens": 10},
            ),
            parent_id="s_agent",
        )
        gen.start()
        gen.finish()
        adapter.on_span_end(gen)

        # Tool call
        tool = _make_span(
            adapter,"t_flow", "s_tool",
            FunctionSpanData(name="classify", input="I need help", output="billing"),
            parent_id="s_agent",
        )
        tool.start()
        tool.finish()
        adapter.on_span_end(tool)

        # Guardrail
        guard = _make_span(
            adapter,"t_flow", "s_guard",
            GuardrailSpanData(name="safety_check", triggered=False),
            parent_id="s_agent",
        )
        guard.start()
        guard.finish()
        adapter.on_span_end(guard)

        # Handoff
        handoff = _make_span(
            adapter,"t_flow", "s_handoff",
            HandoffSpanData(from_agent="triage", to_agent="billing_agent"),
            parent_id="s_agent",
        )
        handoff.start()
        handoff.finish()
        adapter.on_span_end(handoff)

        agent.finish()
        adapter.on_span_end(agent)

        adapter.on_trace_end(trace)

        events = uploaded["events"]
        types = [e["event_type"] for e in events]

        assert "agent.input" in types
        assert "agent.output" in types
        assert "model.invoke" in types
        assert "cost.record" in types
        assert "tool.call" in types
        assert "evaluation.result" in types
        assert "agent.handoff" in types

        # Verify ordering
        seq_ids = [e["sequence_id"] for e in events]
        assert seq_ids == sorted(seq_ids)
        assert len(set(seq_ids)) == len(seq_ids)

        # Verify parent-child relationships
        me = find_event(events, "model.invoke")
        assert me["parent_span_id"] == "s_agent"

        tc = find_event(events, "tool.call")
        assert tc["parent_span_id"] == "s_agent"


class TestCaptureConfigGating:
    """Test that CaptureConfig gates events properly."""

    def test_minimal_config(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        config = CaptureConfig.minimal()
        adapter = OpenAIAgentsAdapter(mock_client, capture_config=config)
        adapter.connect()


        trace = _make_trace(trace_id="t_min")

        adapter.on_trace_start(trace)

        # Agent span (L1 — should be captured)
        agent = _make_span(adapter,"t_min", "s_agent", AgentSpanData(name="test"))
        agent.start()
        agent.finish()
        adapter.on_span_end(agent)

        # Generation span (L3 — should be skipped)
        gen = _make_span(
            adapter,"t_min", "s_gen",
            GenerationSpanData(
                input=[], output=[], model="gpt-4o",
                model_config={}, usage={"input_tokens": 10, "output_tokens": 5},
            ),
        )
        gen.start()
        gen.finish()
        adapter.on_span_end(gen)

        # Tool span (L5a — should be skipped)
        tool = _make_span(
            adapter,"t_min", "s_tool",
            FunctionSpanData(name="search", input="q", output="r"),
        )
        tool.start()
        tool.finish()
        adapter.on_span_end(tool)

        adapter.on_trace_end(trace)

        events = uploaded.get("events", [])
        types = [e["event_type"] for e in events]

        assert "agent.input" in types
        assert "agent.output" in types
        assert "model.invoke" not in types
        assert "tool.call" not in types
        # cost.record is always enabled
        assert "cost.record" in types

        adapter.disconnect()


class TestConcurrentTraces:
    """Test that multiple concurrent traces are isolated."""

    def test_parallel_traces_isolated(self, mock_client):
        all_uploads: List[Dict[str, Any]] = []

        def _capture(path: str) -> None:
            with open(path) as f:
                data = json.load(f)
            all_uploads.append(data[0])

        mock_client.traces.upload = MagicMock(side_effect=_capture)

        adapter = OpenAIAgentsAdapter(mock_client)
        adapter.connect()


        # Two concurrent traces
        t1 = _make_trace(trace_id="t_par_1")
        t2 = _make_trace(trace_id="t_par_2")

        adapter.on_trace_start(t1)
        adapter.on_trace_start(t2)

        # Agent in trace 1
        s1 = _make_span(adapter,"t_par_1", "s1", AgentSpanData(name="agent_1"))
        s1.start()
        s1.finish()
        adapter.on_span_end(s1)

        # Agent in trace 2
        s2 = _make_span(adapter,"t_par_2", "s2", AgentSpanData(name="agent_2"))
        s2.start()
        s2.finish()
        adapter.on_span_end(s2)

        adapter.on_trace_end(t1)
        adapter.on_trace_end(t2)

        assert len(all_uploads) == 2

        # Each trace should have its own events
        names = set()
        for upload in all_uploads:
            for e in upload["events"]:
                if e["event_type"] == "agent.input":
                    names.add(e["payload"]["agent_name"])

        assert names == {"agent_1", "agent_2"}

        adapter.disconnect()


class TestErrorIsolation:
    """Verify hooks never crash the SDK."""

    def test_broken_collector_does_not_crash(self, mock_client):
        adapter = OpenAIAgentsAdapter(mock_client)
        adapter.connect()


        trace = _make_trace(trace_id="t_safe")
        adapter.on_trace_start(trace)

        # Break the run's collector
        adapter._trace_runs["t_safe"] = None  # type: ignore[assignment]

        # This should not raise
        span = _make_span(adapter,"t_safe", "s_safe", AgentSpanData(name="test"))
        span.start()
        span.finish()
        adapter.on_span_end(span)  # Should log warning, not crash

        # Trace end should not crash either
        adapter.on_trace_end(trace)

        adapter.disconnect()


class TestEdgeCases:
    def test_empty_usage(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace

        trace = _make_trace(trace_id="t_empty")

        adapter.on_trace_start(trace)

        span = _make_span(
            adapter,"t_empty", "s_empty",
            GenerationSpanData(input=[], output=[], model="gpt-4o", model_config={}, usage={}),
        )
        span.start()
        span.finish()
        adapter.on_span_end(span)
        adapter.on_trace_end(trace)

        events = uploaded["events"]
        me = find_event(events, "model.invoke")
        assert "tokens_prompt" not in me["payload"]
        assert "tokens_completion" not in me["payload"]

    def test_none_values_in_span_data(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace

        trace = _make_trace(trace_id="t_none")

        adapter.on_trace_start(trace)

        span = _make_span(
            adapter,"t_none", "s_none",
            AgentSpanData(name="minimal_agent"),  # no tools, no handoffs
        )
        span.start()
        span.finish()
        adapter.on_span_end(span)
        adapter.on_trace_end(trace)

        events = uploaded["events"]
        inp = find_event(events, "agent.input")
        assert inp["payload"]["agent_name"] == "minimal_agent"
        assert "tools" not in inp["payload"]
        assert "handoffs" not in inp["payload"]

    def test_function_span_with_none_output(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace

        trace = _make_trace(trace_id="t_none_out")

        adapter.on_trace_start(trace)

        span = _make_span(
            adapter,"t_none_out", "s_func",
            FunctionSpanData(name="void_tool", input="run", output=None),
        )
        span.start()
        span.finish()
        adapter.on_span_end(span)
        adapter.on_trace_end(trace)

        events = uploaded["events"]
        tc = find_event(events, "tool.call")
        assert tc["payload"]["tool_name"] == "void_tool"
        # output should not be in payload since it was None
        assert "output" not in tc["payload"]

    def test_span_duration_tracking(self, adapter_and_trace):
        """Verify duration_ms is computed from span timing."""
        import time as _time

        adapter, uploaded = adapter_and_trace

        trace = _make_trace(trace_id="t_dur")

        adapter.on_trace_start(trace)

        span = _make_span(adapter,"t_dur", "s_dur", AgentSpanData(name="slow_agent"))
        span.start()
        _time.sleep(0.02)  # 20ms
        span.finish()
        adapter.on_span_end(span)
        adapter.on_trace_end(trace)

        events = uploaded["events"]
        out = find_event(events, "agent.output")
        assert out["payload"]["duration_ms"] >= 15  # allow tolerance
