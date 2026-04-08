"""Tests for Google ADK adapter.

The adapter uses the ADK plugin system (BasePlugin) for observability.
Tests call the adapter's sync handler methods directly to verify event
emission without needing a real ADK Runner.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional
from unittest.mock import Mock

import pytest

pytest.importorskip("google.adk")

from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.google_adk import GoogleADKAdapter  # noqa: E402

from .conftest import capture_framework_trace, find_event, find_events  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_invocation_context(
    agent_name: str = "root_agent",
    user_content: Any = "Hello",
    invocation_id: str = "inv-001",
    session_id: str = "sess-001",
) -> Mock:
    ctx = Mock()
    agent = Mock()
    agent.name = agent_name
    ctx.agent = agent
    ctx.invocation_id = invocation_id
    ctx.user_content = user_content
    session = Mock()
    session.id = session_id
    ctx.session = session
    return ctx


def _make_agent(
    name: str = "test_agent",
    description: Optional[str] = None,
    instruction: Optional[str] = None,
    model: Optional[str] = None,
    tools: Optional[list] = None,
    sub_agents: Optional[list] = None,
) -> Mock:
    agent = Mock()
    agent.name = name
    type(agent).__name__ = "LlmAgent"
    agent.description = description
    agent.instruction = instruction
    agent.model = model
    agent.tools = tools or []
    agent.sub_agents = sub_agents or []
    return agent


def _make_callback_context(
    agent_name: str = "test_agent",
    user_content: Any = None,
    session_id: Optional[str] = None,
    function_call_id: Optional[str] = None,
) -> Mock:
    ctx = Mock()
    ctx.agent_name = agent_name
    ctx.user_content = user_content
    ctx.function_call_id = function_call_id
    if session_id:
        session = Mock()
        session.id = session_id
        ctx.session = session
    else:
        del ctx.session
    return ctx


def _make_llm_response(
    model_version: Optional[str] = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> Mock:
    resp = Mock()
    resp.model_version = model_version
    if prompt_tokens or completion_tokens:
        usage = Mock()
        usage.prompt_token_count = prompt_tokens
        usage.candidates_token_count = completion_tokens
        usage.total_token_count = prompt_tokens + completion_tokens
        resp.usage_metadata = usage
    else:
        resp.usage_metadata = None
    return resp


def _make_llm_request(model: Optional[str] = None) -> Mock:
    req = Mock()
    req.model = model
    return req


def _make_tool(name: str = "search") -> Mock:
    tool = Mock()
    tool.name = name
    return tool


def _make_tool_context(
    agent_name: str = "test_agent",
    function_call_id: str = "fc-001",
) -> Mock:
    ctx = Mock()
    ctx.agent_name = agent_name
    ctx.function_call_id = function_call_id
    return ctx


def _make_event(
    author: str = "root_agent",
    transfer_to_agent: Optional[str] = None,
) -> Mock:
    event = Mock()
    event.author = author
    if transfer_to_agent:
        actions = Mock()
        actions.transfer_to_agent = transfer_to_agent
        event.actions = actions
    else:
        event.actions = Mock(spec=[])  # no transfer_to_agent attr
    return event


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_connect_creates_plugin(self, mock_client):
        adapter = GoogleADKAdapter(mock_client)
        adapter.connect()
        assert adapter.plugin is not None
        assert adapter.plugin.name == "layerlens"
        adapter.disconnect()

    def test_disconnect_clears_plugin(self, mock_client):
        adapter = GoogleADKAdapter(mock_client)
        adapter.connect()
        adapter.disconnect()
        assert adapter.plugin is None
        assert adapter._collector is None
        assert adapter._run_span_id is None

    def test_adapter_info(self, mock_client):
        adapter = GoogleADKAdapter(mock_client)
        adapter.connect()
        info = adapter.adapter_info()
        assert info.name == "google_adk"
        assert info.adapter_type == "framework"
        assert info.connected is True
        assert "framework_version" in info.metadata
        adapter.disconnect()


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


class TestRunLifecycle:
    def test_before_run_creates_collector_and_emits_input(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = GoogleADKAdapter(mock_client)
        adapter.connect()

        inv_ctx = _make_invocation_context(agent_name="root", user_content="Hello world")
        adapter._on_before_run(inv_ctx)
        adapter._on_after_run(inv_ctx)

        events = uploaded["events"]
        agent_in = find_event(events, "agent.input")
        assert agent_in["payload"]["agent_name"] == "root"
        assert agent_in["payload"]["input"] == "Hello world"
        assert agent_in["payload"]["session_id"] == "sess-001"
        assert agent_in["payload"]["invocation_id"] == "inv-001"

        adapter.disconnect()

    def test_after_run_emits_output_with_duration(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = GoogleADKAdapter(mock_client)
        adapter.connect()

        inv_ctx = _make_invocation_context(agent_name="root")
        adapter._on_before_run(inv_ctx)
        time.sleep(0.01)
        adapter._on_after_run(inv_ctx)

        events = uploaded["events"]
        agent_out = find_event(events, "agent.output")
        assert agent_out["payload"]["agent_name"] == "root"
        assert agent_out["payload"]["duration_ns"] > 0

        adapter.disconnect()

    def test_after_run_flushes_trace(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = GoogleADKAdapter(mock_client)
        adapter.connect()

        inv_ctx = _make_invocation_context()
        adapter._on_before_run(inv_ctx)
        adapter._on_after_run(inv_ctx)

        assert uploaded.get("trace_id") is not None
        assert uploaded["attestation"].get("root_hash") is not None

        adapter.disconnect()


# ---------------------------------------------------------------------------
# Agent lifecycle
# ---------------------------------------------------------------------------


class TestAgentLifecycle:
    def test_before_agent_emits_input(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = GoogleADKAdapter(mock_client)
        adapter.connect()

        inv_ctx = _make_invocation_context()
        adapter._on_before_run(inv_ctx)

        agent = _make_agent(name="planner")
        cb_ctx = _make_callback_context("planner", user_content="Plan a trip")
        adapter._on_before_agent(agent, cb_ctx)
        adapter._on_after_agent(agent, cb_ctx)
        adapter._on_after_run(inv_ctx)

        events = uploaded["events"]
        # Find the agent-level input (not the run-level one)
        agent_inputs = find_events(events, "agent.input")
        agent_level = [e for e in agent_inputs if e.get("span_name") == "agent:planner"]
        assert len(agent_level) == 1
        assert agent_level[0]["payload"]["agent_name"] == "planner"
        assert agent_level[0]["payload"]["input"] == "Plan a trip"

        adapter.disconnect()

    def test_after_agent_emits_output_with_duration(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = GoogleADKAdapter(mock_client)
        adapter.connect()

        inv_ctx = _make_invocation_context()
        adapter._on_before_run(inv_ctx)

        agent = _make_agent(name="planner")
        cb_ctx = _make_callback_context("planner")
        adapter._on_before_agent(agent, cb_ctx)
        time.sleep(0.01)
        adapter._on_after_agent(agent, cb_ctx)
        adapter._on_after_run(inv_ctx)

        events = uploaded["events"]
        agent_outputs = find_events(events, "agent.output")
        agent_level = [e for e in agent_outputs if e.get("span_name") == "agent:planner"]
        assert len(agent_level) == 1
        assert agent_level[0]["payload"]["duration_ns"] > 0

        adapter.disconnect()

    def test_agent_config_emitted_once(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = GoogleADKAdapter(mock_client)
        adapter.connect()

        inv_ctx = _make_invocation_context()
        adapter._on_before_run(inv_ctx)

        agent = _make_agent(name="router")
        cb_ctx = _make_callback_context("router")
        adapter._on_before_agent(agent, cb_ctx)
        adapter._on_after_agent(agent, cb_ctx)
        adapter._on_before_agent(agent, cb_ctx)
        adapter._on_after_agent(agent, cb_ctx)
        adapter._on_after_run(inv_ctx)

        events = uploaded["events"]
        configs = find_events(events, "environment.config")
        assert len(configs) == 1
        assert configs[0]["payload"]["agent_name"] == "router"

        adapter.disconnect()

    def test_agent_config_captures_attributes(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = GoogleADKAdapter(mock_client)
        adapter.connect()

        inv_ctx = _make_invocation_context()
        adapter._on_before_run(inv_ctx)

        tool1 = _make_tool("search")
        tool2 = _make_tool("calculator")
        sub = Mock()
        sub.name = "sub_agent"
        agent = _make_agent(
            name="smart",
            description="A smart agent",
            instruction="Be helpful",
            model="gemini-2.0-flash",
            tools=[tool1, tool2],
            sub_agents=[sub],
        )
        cb_ctx = _make_callback_context("smart", session_id="sess-abc")
        adapter._on_before_agent(agent, cb_ctx)
        adapter._on_after_run(inv_ctx)

        events = uploaded["events"]
        config = find_event(events, "environment.config")
        p = config["payload"]
        assert p["description"] == "A smart agent"
        assert p["instruction"] == "Be helpful"
        assert p["model"] == "gemini-2.0-flash"
        assert p["tools"] == ["search", "calculator"]
        assert p["sub_agents"] == ["sub_agent"]
        assert p["session_id"] == "sess-abc"

        adapter.disconnect()


# ---------------------------------------------------------------------------
# Model callbacks
# ---------------------------------------------------------------------------


class TestModelCallbacks:
    def test_model_invoke_with_tokens(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = GoogleADKAdapter(mock_client)
        adapter.connect()

        inv_ctx = _make_invocation_context()
        adapter._on_before_run(inv_ctx)

        cb_ctx = _make_callback_context("agent1")
        llm_req = _make_llm_request(model="gemini-2.0-flash")
        llm_resp = _make_llm_response(model_version="gemini-2.0-flash", prompt_tokens=100, completion_tokens=50)

        adapter._on_before_model(cb_ctx, llm_req)
        time.sleep(0.01)
        adapter._on_after_model(cb_ctx, llm_resp)
        adapter._on_after_run(inv_ctx)

        events = uploaded["events"]
        model_evt = find_event(events, "model.invoke")
        assert model_evt["payload"]["model"] == "gemini-2.0-flash"
        assert model_evt["payload"]["provider"] == "google"
        assert model_evt["payload"]["tokens_prompt"] == 100
        assert model_evt["payload"]["tokens_completion"] == 50
        assert model_evt["payload"]["tokens_total"] == 150
        assert model_evt["payload"]["latency_ms"] >= 5

        adapter.disconnect()

    def test_model_invoke_emits_cost_record(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = GoogleADKAdapter(mock_client)
        adapter.connect()

        inv_ctx = _make_invocation_context()
        adapter._on_before_run(inv_ctx)

        cb_ctx = _make_callback_context("agent1")
        llm_resp = _make_llm_response(model_version="gemini-pro", prompt_tokens=200, completion_tokens=100)
        adapter._on_before_model(cb_ctx, Mock())
        adapter._on_after_model(cb_ctx, llm_resp)
        adapter._on_after_run(inv_ctx)

        events = uploaded["events"]
        cost = find_event(events, "cost.record")
        assert cost["payload"]["model"] == "gemini-pro"
        assert cost["payload"]["tokens_total"] == 300

        adapter.disconnect()

    def test_model_invoke_without_usage(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = GoogleADKAdapter(mock_client)
        adapter.connect()

        inv_ctx = _make_invocation_context()
        adapter._on_before_run(inv_ctx)

        cb_ctx = _make_callback_context("agent1")
        llm_resp = _make_llm_response()
        adapter._on_before_model(cb_ctx, Mock())
        adapter._on_after_model(cb_ctx, llm_resp)
        adapter._on_after_run(inv_ctx)

        events = uploaded["events"]
        model_evt = find_event(events, "model.invoke")
        assert "tokens_prompt" not in model_evt["payload"]
        assert len(find_events(events, "cost.record")) == 0

        adapter.disconnect()

    def test_model_error_emits_error_event(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = GoogleADKAdapter(mock_client)
        adapter.connect()

        inv_ctx = _make_invocation_context()
        adapter._on_before_run(inv_ctx)

        cb_ctx = _make_callback_context("agent1")
        llm_req = _make_llm_request(model="gemini-2.0-flash")
        adapter._on_before_model(cb_ctx, llm_req)
        adapter._on_model_error(cb_ctx, llm_req, RuntimeError("API timeout"))
        adapter._on_after_run(inv_ctx)

        events = uploaded["events"]
        error_evt = find_event(events, "agent.error")
        assert error_evt["payload"]["error"] == "API timeout"
        assert error_evt["payload"]["error_type"] == "RuntimeError"
        assert error_evt["payload"]["model"] == "gemini-2.0-flash"

        adapter.disconnect()


# ---------------------------------------------------------------------------
# Tool callbacks
# ---------------------------------------------------------------------------


class TestToolCallbacks:
    def test_tool_call_and_result(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = GoogleADKAdapter(mock_client)
        adapter.connect()

        inv_ctx = _make_invocation_context()
        adapter._on_before_run(inv_ctx)

        tool = _make_tool("weather_search")
        tool_args = {"query": "weather in NYC"}
        tool_ctx = _make_tool_context(function_call_id="fc-001")

        adapter._on_before_tool(tool, tool_args, tool_ctx)
        time.sleep(0.01)
        adapter._on_after_tool(tool, tool_args, tool_ctx, {"result": "Sunny, 72F"})
        adapter._on_after_run(inv_ctx)

        events = uploaded["events"]
        tool_call = find_event(events, "tool.call")
        assert tool_call["payload"]["tool_name"] == "weather_search"
        assert tool_call["payload"]["input"] == {"query": "weather in NYC"}
        assert tool_call["payload"]["latency_ms"] >= 5
        assert tool_call["span_name"] == "tool:weather_search"

        tool_result = find_event(events, "tool.result")
        assert tool_result["payload"]["tool_name"] == "weather_search"
        assert tool_result["payload"]["output"] == {"result": "Sunny, 72F"}

        adapter.disconnect()

    def test_tool_content_gated(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = GoogleADKAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        adapter.connect()

        inv_ctx = _make_invocation_context()
        adapter._on_before_run(inv_ctx)

        tool = _make_tool("search")
        tool_ctx = _make_tool_context()
        adapter._on_before_tool(tool, {"secret": "data"}, tool_ctx)
        adapter._on_after_tool(tool, {"secret": "data"}, tool_ctx, {"result": "secret"})
        adapter._on_after_run(inv_ctx)

        events = uploaded["events"]
        tool_call = find_event(events, "tool.call")
        assert "input" not in tool_call["payload"]
        tool_result = find_event(events, "tool.result")
        assert "output" not in tool_result["payload"]

        adapter.disconnect()

    def test_tool_error_emits_error_event(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = GoogleADKAdapter(mock_client)
        adapter.connect()

        inv_ctx = _make_invocation_context()
        adapter._on_before_run(inv_ctx)

        tool = _make_tool("broken_tool")
        tool_ctx = _make_tool_context(function_call_id="fc-err")
        adapter._on_before_tool(tool, {}, tool_ctx)
        adapter._on_tool_error(tool, {}, tool_ctx, ValueError("bad input"))
        adapter._on_after_run(inv_ctx)

        events = uploaded["events"]
        error_evt = find_event(events, "agent.error")
        assert error_evt["payload"]["tool_name"] == "broken_tool"
        assert error_evt["payload"]["error"] == "bad input"
        assert error_evt["payload"]["error_type"] == "ValueError"

        adapter.disconnect()


# ---------------------------------------------------------------------------
# Handoff detection
# ---------------------------------------------------------------------------


class TestHandoffDetection:
    def test_handoff_via_event_actions(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = GoogleADKAdapter(mock_client)
        adapter.connect()

        inv_ctx = _make_invocation_context()
        adapter._on_before_run(inv_ctx)

        event = _make_event(author="router", transfer_to_agent="billing_agent")
        adapter._on_event(inv_ctx, event)
        adapter._on_after_run(inv_ctx)

        events = uploaded["events"]
        handoff = find_event(events, "agent.handoff")
        assert handoff["payload"]["from_agent"] == "router"
        assert handoff["payload"]["to_agent"] == "billing_agent"
        assert handoff["span_name"] == "handoff:router->billing_agent"

        adapter.disconnect()

    def test_no_handoff_without_transfer(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = GoogleADKAdapter(mock_client)
        adapter.connect()

        inv_ctx = _make_invocation_context()
        adapter._on_before_run(inv_ctx)

        event = _make_event(author="agent1")
        adapter._on_event(inv_ctx, event)
        adapter._on_after_run(inv_ctx)

        events = uploaded["events"]
        assert len(find_events(events, "agent.handoff")) == 0

        adapter.disconnect()


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------


class TestErrorIsolation:
    def test_handlers_dont_crash_on_none_args(self, mock_client):
        adapter = GoogleADKAdapter(mock_client)
        adapter.connect()

        inv_ctx = _make_invocation_context()
        adapter._on_before_run(inv_ctx)

        # These should not raise
        adapter._on_before_agent(None, None)
        adapter._on_after_agent(None, None)
        adapter._on_before_model(None, None)
        adapter._on_after_model(None, Mock(model_version=None, usage_metadata=None))
        adapter._on_before_tool(None, None, Mock(function_call_id=None))
        adapter._on_event(None, Mock(actions=None))

        adapter._on_after_run(inv_ctx)
        adapter.disconnect()

    def test_no_events_when_no_collector(self, mock_client):
        """Calling handlers before _on_before_run should be safe."""
        adapter = GoogleADKAdapter(mock_client)
        adapter.connect()

        # No collector — fire() should silently no-op
        adapter._on_before_agent(_make_agent(), _make_callback_context())
        adapter._on_after_model(_make_callback_context(), _make_llm_response(prompt_tokens=10))

        adapter.disconnect()


# ---------------------------------------------------------------------------
# Trace integrity
# ---------------------------------------------------------------------------


class TestTraceIntegrity:
    def test_all_events_share_trace_id(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = GoogleADKAdapter(mock_client)
        adapter.connect()

        inv_ctx = _make_invocation_context()
        adapter._on_before_run(inv_ctx)

        agent = _make_agent(name="agent1")
        cb_ctx = _make_callback_context("agent1")
        adapter._on_before_agent(agent, cb_ctx)

        llm_resp = _make_llm_response(model_version="gemini", prompt_tokens=10, completion_tokens=5)
        adapter._on_before_model(cb_ctx, Mock())
        adapter._on_after_model(cb_ctx, llm_resp)

        tool = _make_tool("search")
        tool_ctx = _make_tool_context()
        adapter._on_before_tool(tool, {"q": "test"}, tool_ctx)
        adapter._on_after_tool(tool, {"q": "test"}, tool_ctx, {"r": "ok"})

        adapter._on_after_agent(agent, cb_ctx)
        adapter._on_after_run(inv_ctx)

        events = uploaded["events"]
        trace_ids = {e["trace_id"] for e in events}
        assert len(trace_ids) == 1

        adapter.disconnect()

    def test_sequence_ids_monotonic(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = GoogleADKAdapter(mock_client)
        adapter.connect()

        inv_ctx = _make_invocation_context()
        adapter._on_before_run(inv_ctx)

        agent = _make_agent(name="a")
        cb_ctx = _make_callback_context("a")
        adapter._on_before_agent(agent, cb_ctx)
        adapter._on_after_agent(agent, cb_ctx)
        adapter._on_after_run(inv_ctx)

        events = uploaded["events"]
        seq_ids = [e["sequence_id"] for e in events]
        assert seq_ids == sorted(seq_ids)

        adapter.disconnect()

    def test_span_hierarchy(self, mock_client):
        """Agent events are children of run span, model/tool events are children of agent span."""
        uploaded = capture_framework_trace(mock_client)
        adapter = GoogleADKAdapter(mock_client)
        adapter.connect()

        inv_ctx = _make_invocation_context()
        adapter._on_before_run(inv_ctx)

        agent = _make_agent(name="worker")
        cb_ctx = _make_callback_context("worker")
        adapter._on_before_agent(agent, cb_ctx)

        llm_resp = _make_llm_response(model_version="gemini", prompt_tokens=10, completion_tokens=5)
        adapter._on_before_model(cb_ctx, Mock())
        adapter._on_after_model(cb_ctx, llm_resp)

        tool = _make_tool("calc")
        tool_ctx = _make_tool_context()
        adapter._on_before_tool(tool, {}, tool_ctx)
        adapter._on_after_tool(tool, {}, tool_ctx, {})

        adapter._on_after_agent(agent, cb_ctx)
        adapter._on_after_run(inv_ctx)

        events = uploaded["events"]
        run_input = [e for e in find_events(events, "agent.input") if e["span_name"] == "root_agent"][0]
        run_span_id = run_input["span_id"]

        agent_input = [e for e in find_events(events, "agent.input") if e["span_name"] == "agent:worker"][0]
        assert agent_input["parent_span_id"] == run_span_id

        agent_span_id = agent_input["span_id"]
        model_evt = find_event(events, "model.invoke")
        assert model_evt["parent_span_id"] == agent_span_id

        tool_evt = find_event(events, "tool.call")
        assert tool_evt["parent_span_id"] == agent_span_id

        adapter.disconnect()


# ---------------------------------------------------------------------------
# Full lifecycle
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    def test_multi_agent_trace(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = GoogleADKAdapter(mock_client)
        adapter.connect()

        inv_ctx = _make_invocation_context(agent_name="root")
        adapter._on_before_run(inv_ctx)

        # Agent 1: router
        router = _make_agent(name="router")
        router_ctx = _make_callback_context("router", user_content="Book a flight")
        adapter._on_before_agent(router, router_ctx)

        llm_resp = _make_llm_response(model_version="gemini-2.0-flash", prompt_tokens=50, completion_tokens=20)
        adapter._on_before_model(router_ctx, Mock())
        adapter._on_after_model(router_ctx, llm_resp)

        adapter._on_after_agent(router, router_ctx)

        # Handoff event
        handoff_event = _make_event(author="router", transfer_to_agent="booking_agent")
        adapter._on_event(inv_ctx, handoff_event)

        # Agent 2: booking_agent
        booking = _make_agent(name="booking_agent")
        booking_ctx = _make_callback_context("booking_agent", user_content="Flight SFO->NYC")
        adapter._on_before_agent(booking, booking_ctx)

        tool = _make_tool("flight_search")
        tool_ctx = _make_tool_context(function_call_id="fc-flight")
        adapter._on_before_tool(tool, {"origin": "SFO", "dest": "NYC"}, tool_ctx)
        adapter._on_after_tool(tool, {"origin": "SFO", "dest": "NYC"}, tool_ctx, {"flights": 3})

        adapter._on_after_agent(booking, booking_ctx)
        adapter._on_after_run(inv_ctx)

        events = uploaded["events"]
        assert uploaded["trace_id"] is not None

        # Event counts
        assert len(find_events(events, "environment.config")) == 2  # router + booking
        assert len(find_events(events, "agent.input")) == 3  # run + router + booking
        assert len(find_events(events, "agent.output")) == 3  # run + router + booking
        assert len(find_events(events, "model.invoke")) == 1
        assert len(find_events(events, "cost.record")) == 1
        assert len(find_events(events, "tool.call")) == 1
        assert len(find_events(events, "tool.result")) == 1
        assert len(find_events(events, "agent.handoff")) == 1

        # Trace integrity
        trace_ids = {e["trace_id"] for e in events}
        assert len(trace_ids) == 1
        seqs = [e["sequence_id"] for e in events]
        assert seqs == sorted(seqs)
        assert uploaded["attestation"].get("root_hash") is not None

        adapter.disconnect()
