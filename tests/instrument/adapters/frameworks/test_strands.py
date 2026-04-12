"""Tests for Strands Agents adapter.

Uses real strands hook event types to test the hook-based adapter.
Tests call hook handler methods directly with properly constructed event objects.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import Mock

import pytest

strands_mod = pytest.importorskip("strands")
from strands.hooks import HookRegistry  # noqa: E402
from strands.hooks.events import (  # noqa: E402
    AfterToolCallEvent,
    AfterModelCallEvent,
    BeforeToolCallEvent,
    AfterInvocationEvent,
    BeforeModelCallEvent,
    BeforeInvocationEvent,
)

from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.strands import StrandsAdapter  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cycle(input_tokens: int = 0, output_tokens: int = 0) -> Mock:
    """Create a mock Strands cycle with per-call token usage."""
    cycle = Mock()
    cycle.usage = {
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
    }
    return cycle


def _make_agent(
    name: str = "TestAgent",
    model_id: str = "us.anthropic.claude-sonnet-4-20250514",
    tool_names: Optional[list] = None,
    system_prompt: Optional[str] = None,
) -> Mock:
    agent = Mock()
    agent.name = name
    type(agent).__name__ = "Agent"
    agent.model = Mock()
    agent.model.config = {"model_id": model_id}
    agent.tool_names = tool_names or []
    agent.system_prompt = system_prompt
    # event_loop_metrics — cycles populated by _simulate_invocation
    agent.event_loop_metrics = Mock()
    agent.event_loop_metrics.agent_invocations = []
    agent.hooks = HookRegistry()
    return agent


def _make_result(
    stop_reason: str = "end_turn",
    message: Any = "Final answer",
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> Mock:
    result = Mock()
    result.stop_reason = stop_reason
    result.message = message
    result.metrics = Mock()
    result.metrics.accumulated_usage = {
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "totalTokens": input_tokens + output_tokens,
    }
    return result


def _make_model_stop_response(stop_reason: str = "end_turn") -> Any:
    resp = AfterModelCallEvent.ModelStopResponse(
        message=Mock(),
        stop_reason=stop_reason,
    )
    return resp


def _simulate_invocation(
    adapter: StrandsAdapter,
    agent: Any,
    messages: Any = "Hello!",
    tool_calls: Optional[list] = None,
    model_tokens: Optional[Dict[str, int]] = None,
    result: Optional[Any] = None,
) -> None:
    """Simulate a full Strands agent invocation lifecycle via hook calls."""
    # BeforeInvocation
    before_inv = BeforeInvocationEvent(agent=agent, invocation_state={}, messages=messages)
    adapter._on_before_invocation(before_inv)

    # Model call
    before_model = BeforeModelCallEvent(agent=agent, invocation_state={})
    adapter._on_before_model(before_model)

    stop_reason = "end_turn" if not tool_calls else "tool_use"
    after_model = AfterModelCallEvent(
        agent=agent,
        invocation_state={},
        stop_response=_make_model_stop_response(stop_reason),
    )
    adapter._on_after_model(after_model)

    # Tool calls
    if tool_calls:
        for tc in tool_calls:
            tool_use = {"name": tc["name"], "toolUseId": tc.get("id", "tc-1"), "input": tc.get("input", {})}
            tool_result = tc.get(
                "result", {"toolUseId": tc.get("id", "tc-1"), "status": "success", "content": [{"text": "ok"}]}
            )
            before_tool = BeforeToolCallEvent(
                agent=agent,
                selected_tool=Mock(name=tc["name"]),
                tool_use=tool_use,
                invocation_state={},
            )
            adapter._on_before_tool(before_tool)

            after_tool = AfterToolCallEvent(
                agent=agent,
                selected_tool=Mock(name=tc["name"]),
                tool_use=tool_use,
                invocation_state={},
                result=tool_result,
                exception=tc.get("exception"),
            )
            adapter._on_after_tool(after_tool)

    # Set up per-cycle token data on agent (simulates what Strands does
    # AFTER AfterModelCallEvent but BEFORE AfterInvocationEvent)
    if model_tokens:
        invocation = Mock()
        invocation.cycles = [
            _make_cycle(
                input_tokens=model_tokens.get("input", 0),
                output_tokens=model_tokens.get("output", 0),
            )
        ]
        agent.event_loop_metrics.agent_invocations = [invocation]

    # AfterInvocation
    if result is None:
        result = _make_result(
            input_tokens=model_tokens.get("input", 0) if model_tokens else 0,
            output_tokens=model_tokens.get("output", 0) if model_tokens else 0,
        )
    after_inv = AfterInvocationEvent(agent=agent, invocation_state={}, result=result)
    adapter._on_after_invocation(after_inv)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_connect_sets_connected(self, mock_client):
        adapter = StrandsAdapter(mock_client)
        adapter.connect()
        assert adapter.is_connected
        adapter.disconnect()
        assert not adapter.is_connected

    def test_connect_with_target_registers_hooks(self, mock_client):
        adapter = StrandsAdapter(mock_client)
        agent = _make_agent()
        adapter.connect(target=agent)
        assert len(adapter._registered_callbacks) == 7
        adapter.disconnect()

    def test_disconnect_deregisters_hooks(self, mock_client):
        adapter = StrandsAdapter(mock_client)
        agent = _make_agent()
        adapter.connect(target=agent)
        adapter.disconnect()
        assert len(adapter._registered_callbacks) == 0

    def test_disconnect_clears_state(self, mock_client):
        adapter = StrandsAdapter(mock_client)
        adapter.connect()
        adapter.disconnect()
        assert adapter._collector is None
        assert adapter._run_span_id is None
        assert adapter._target is None

    def test_register_hooks_protocol(self, mock_client):
        """Adapter implements HookProvider protocol (register_hooks)."""
        adapter = StrandsAdapter(mock_client)
        adapter.connect()
        registry = HookRegistry()
        adapter.register_hooks(registry)
        assert registry.has_callbacks()
        adapter.disconnect()


# ---------------------------------------------------------------------------
# Invocation lifecycle
# ---------------------------------------------------------------------------


class TestInvocationLifecycle:
    def test_invocation_emits_input_and_output(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = StrandsAdapter(mock_client)
        adapter.connect()

        agent = _make_agent(name="MyAgent", model_id="claude-sonnet")
        _simulate_invocation(adapter, agent, messages="What is AI?")

        events = uploaded["events"]
        agent_in = find_event(events, "agent.input")
        assert agent_in["payload"]["agent_name"] == "MyAgent"
        assert agent_in["payload"]["model"] == "claude-sonnet"
        assert agent_in["payload"]["input"] == "What is AI?"

        agent_out = find_event(events, "agent.output")
        assert agent_out["payload"]["agent_name"] == "MyAgent"
        assert agent_out["payload"]["duration_ns"] > 0
        assert agent_out["payload"]["stop_reason"] == "end_turn"

        adapter.disconnect()

    def test_invocation_flushes_trace(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = StrandsAdapter(mock_client)
        adapter.connect()

        agent = _make_agent()
        _simulate_invocation(adapter, agent)

        assert uploaded.get("trace_id") is not None
        assert uploaded["attestation"].get("root_hash") is not None

        adapter.disconnect()

    def test_input_gated_by_capture_content(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = StrandsAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        adapter.connect()

        agent = _make_agent()
        _simulate_invocation(adapter, agent, messages="secret input")

        events = uploaded["events"]
        agent_in = find_event(events, "agent.input")
        assert "input" not in agent_in["payload"]

        adapter.disconnect()


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------


class TestModelCalls:
    def test_model_invoke_emits_timing_and_model(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = StrandsAdapter(mock_client)
        adapter.connect()

        agent = _make_agent(model_id="claude-sonnet")
        _simulate_invocation(adapter, agent, model_tokens={"input": 100, "output": 50})

        events = uploaded["events"]
        model_evt = find_event(events, "model.invoke")
        assert model_evt["payload"]["model"] == "claude-sonnet"
        # Tokens are NOT on model.invoke — they come via cost.record
        assert "tokens_prompt" not in model_evt["payload"]

        adapter.disconnect()

    def test_per_cycle_cost_record(self, mock_client):
        """Per-cycle cost.record events are emitted from _on_after_invocation."""
        uploaded = capture_framework_trace(mock_client)
        adapter = StrandsAdapter(mock_client)
        adapter.connect()

        agent = _make_agent(model_id="claude-sonnet")
        _simulate_invocation(adapter, agent, model_tokens={"input": 100, "output": 50})

        events = uploaded["events"]
        cost_evt = find_event(events, "cost.record")
        assert cost_evt["payload"]["tokens_prompt"] == 100
        assert cost_evt["payload"]["tokens_completion"] == 50
        assert cost_evt["payload"]["tokens_total"] == 150
        assert cost_evt["payload"]["model"] == "claude-sonnet"

        # cost.record should be parented to the model span
        model_evt = find_event(events, "model.invoke")
        assert cost_evt["parent_span_id"] == model_evt["span_id"]

        adapter.disconnect()

    def test_model_error(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = StrandsAdapter(mock_client)
        adapter.connect()

        agent = _make_agent()
        before_inv = BeforeInvocationEvent(agent=agent, invocation_state={}, messages="test")
        adapter._on_before_invocation(before_inv)

        before_model = BeforeModelCallEvent(agent=agent, invocation_state={})
        adapter._on_before_model(before_model)

        after_model = AfterModelCallEvent(
            agent=agent,
            invocation_state={},
            exception=RuntimeError("API timeout"),
        )
        adapter._on_after_model(after_model)

        after_inv = AfterInvocationEvent(agent=agent, invocation_state={})
        adapter._on_after_invocation(after_inv)

        events = uploaded["events"]
        model_evt = find_event(events, "model.invoke")
        assert model_evt["payload"]["error"] == "API timeout"
        assert model_evt["payload"]["error_type"] == "RuntimeError"

        adapter.disconnect()


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------


class TestToolCalls:
    def test_tool_call_and_result(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = StrandsAdapter(mock_client)
        adapter.connect()

        agent = _make_agent()
        _simulate_invocation(
            adapter,
            agent,
            tool_calls=[
                {
                    "name": "web_search",
                    "id": "tc-123",
                    "input": {"query": "AI safety"},
                    "result": {"toolUseId": "tc-123", "status": "success", "content": [{"text": "Found 5 results"}]},
                }
            ],
        )

        events = uploaded["events"]
        tool_call = find_event(events, "tool.call")
        assert tool_call["payload"]["tool_name"] == "web_search"
        assert tool_call["payload"]["input"] == {"query": "AI safety"}
        assert tool_call["span_name"] == "tool:web_search"

        tool_result = find_event(events, "tool.result")
        assert tool_result["payload"]["tool_name"] == "web_search"
        assert tool_result["payload"]["status"] == "success"

        adapter.disconnect()

    def test_tool_error(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = StrandsAdapter(mock_client)
        adapter.connect()

        agent = _make_agent()
        before_inv = BeforeInvocationEvent(agent=agent, invocation_state={}, messages="test")
        adapter._on_before_invocation(before_inv)

        before_model = BeforeModelCallEvent(agent=agent, invocation_state={})
        adapter._on_before_model(before_model)
        after_model = AfterModelCallEvent(
            agent=agent,
            invocation_state={},
            stop_response=_make_model_stop_response("tool_use"),
        )
        adapter._on_after_model(after_model)

        tool_use = {"name": "broken_tool", "toolUseId": "tc-err", "input": {}}
        before_tool = BeforeToolCallEvent(agent=agent, selected_tool=Mock(), tool_use=tool_use, invocation_state={})
        adapter._on_before_tool(before_tool)

        after_tool = AfterToolCallEvent(
            agent=agent,
            selected_tool=Mock(),
            tool_use=tool_use,
            invocation_state={},
            result={"toolUseId": "tc-err", "status": "error", "content": []},
            exception=ValueError("bad input"),
        )
        adapter._on_after_tool(after_tool)

        after_inv = AfterInvocationEvent(agent=agent, invocation_state={}, result=_make_result())
        adapter._on_after_invocation(after_inv)

        events = uploaded["events"]
        tool_result = find_event(events, "tool.result")
        assert tool_result["payload"]["error"] == "bad input"
        assert tool_result["payload"]["error_type"] == "ValueError"
        assert tool_result["payload"]["status"] == "error"

        adapter.disconnect()

    def test_tool_content_gated(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = StrandsAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        adapter.connect()

        agent = _make_agent()
        _simulate_invocation(
            adapter,
            agent,
            tool_calls=[
                {
                    "name": "search",
                    "id": "tc-1",
                    "input": {"secret": "data"},
                    "result": {"toolUseId": "tc-1", "status": "success", "content": [{"text": "secret result"}]},
                }
            ],
        )

        events = uploaded["events"]
        tool_call = find_event(events, "tool.call")
        assert "input" not in tool_call["payload"]
        tool_result = find_event(events, "tool.result")
        assert "output" not in tool_result["payload"]

        adapter.disconnect()

    def test_multiple_tool_calls(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = StrandsAdapter(mock_client)
        adapter.connect()

        agent = _make_agent()
        _simulate_invocation(
            adapter,
            agent,
            tool_calls=[
                {"name": "search", "id": "tc-1", "input": {"q": "a"}},
                {"name": "calculator", "id": "tc-2", "input": {"expr": "2+2"}},
            ],
        )

        events = uploaded["events"]
        tool_calls = find_events(events, "tool.call")
        assert len(tool_calls) == 2
        assert tool_calls[0]["payload"]["tool_name"] == "search"
        assert tool_calls[1]["payload"]["tool_name"] == "calculator"

        adapter.disconnect()


# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------


class TestAgentConfig:
    def test_config_emitted_on_invocation(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = StrandsAdapter(mock_client)
        adapter.connect()

        agent = _make_agent(
            name="SmartAgent",
            model_id="claude-sonnet",
            tool_names=["search", "calculator"],
            system_prompt="Be helpful",
        )
        _simulate_invocation(adapter, agent)

        events = uploaded["events"]
        config = find_event(events, "environment.config")
        assert config["payload"]["agent_name"] == "SmartAgent"
        assert config["payload"]["model"] == "claude-sonnet"
        assert config["payload"]["tools"] == ["search", "calculator"]
        assert config["payload"]["system_prompt"] == "Be helpful"

        adapter.disconnect()

    def test_config_emitted_once(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = StrandsAdapter(mock_client)
        adapter.connect()

        agent = _make_agent(name="Agent1")
        _simulate_invocation(adapter, agent)

        # Second invocation — config should not re-emit
        uploaded2 = capture_framework_trace(mock_client)
        _simulate_invocation(adapter, agent)

        events2 = uploaded2["events"]
        configs = find_events(events2, "environment.config")
        assert len(configs) == 0

        adapter.disconnect()

    def test_system_prompt_gated_by_capture_content(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = StrandsAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        adapter.connect()

        agent = _make_agent(system_prompt="Secret prompt")
        _simulate_invocation(adapter, agent)

        events = uploaded["events"]
        config = find_event(events, "environment.config")
        assert "system_prompt" not in config["payload"]

        adapter.disconnect()


# ---------------------------------------------------------------------------
# Trace integrity
# ---------------------------------------------------------------------------


class TestTraceIntegrity:
    def test_all_events_share_trace_id(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = StrandsAdapter(mock_client)
        adapter.connect()

        agent = _make_agent()
        _simulate_invocation(
            adapter,
            agent,
            model_tokens={"input": 100, "output": 50},
            tool_calls=[{"name": "search", "id": "tc-1", "input": {}}],
        )

        events = uploaded["events"]
        trace_ids = {e["trace_id"] for e in events}
        assert len(trace_ids) == 1

        adapter.disconnect()

    def test_sequence_ids_monotonic(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = StrandsAdapter(mock_client)
        adapter.connect()

        agent = _make_agent()
        _simulate_invocation(adapter, agent, model_tokens={"input": 10, "output": 5})

        events = uploaded["events"]
        seq_ids = [e["sequence_id"] for e in events]
        assert seq_ids == sorted(seq_ids)

        adapter.disconnect()

    def test_span_hierarchy(self, mock_client):
        """Model and tool events should be children of the run span."""
        uploaded = capture_framework_trace(mock_client)
        adapter = StrandsAdapter(mock_client)
        adapter.connect()

        agent = _make_agent()
        _simulate_invocation(
            adapter,
            agent,
            model_tokens={"input": 10, "output": 5},
            tool_calls=[{"name": "search", "id": "tc-1", "input": {}}],
        )

        events = uploaded["events"]
        run_input = find_event(events, "agent.input")
        run_span_id = run_input["span_id"]

        model_evt = find_event(events, "model.invoke")
        assert model_evt["parent_span_id"] == run_span_id

        tool_evt = find_event(events, "tool.call")
        assert tool_evt["parent_span_id"] == run_span_id

        adapter.disconnect()


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------


class TestErrorIsolation:
    def test_handlers_dont_crash_on_none_result(self, mock_client):
        adapter = StrandsAdapter(mock_client)
        adapter.connect()

        agent = _make_agent()
        before_inv = BeforeInvocationEvent(agent=agent, invocation_state={})
        adapter._on_before_invocation(before_inv)

        after_inv = AfterInvocationEvent(agent=agent, invocation_state={})
        adapter._on_after_invocation(after_inv)

        adapter.disconnect()

    def test_no_events_when_no_collector(self, mock_client):
        """Calling handlers without a collector should silently no-op."""
        adapter = StrandsAdapter(mock_client)
        adapter.connect()
        agent = _make_agent()
        after_model = AfterModelCallEvent(agent=agent, invocation_state={})
        adapter._on_after_model(after_model)
        adapter.disconnect()
