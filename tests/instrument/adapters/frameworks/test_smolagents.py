"""Tests for SmolAgents adapter.

Uses the real smolagents step types (ActionStep, PlanningStep, FinalAnswerStep)
to test the callback-based adapter. The adapter is tested by:
  1. Calling the run wrapper (which creates/flushes the collector)
  2. Directly invoking step callback methods with real step objects
"""

from __future__ import annotations

from typing import Any, List, Optional
from unittest.mock import Mock, MagicMock

import pytest

smolagents = pytest.importorskip("smolagents")
from smolagents import ActionStep, PlanningStep, FinalAnswerStep, ToolCall  # noqa: E402
from smolagents.memory import Timing, CallbackRegistry  # noqa: E402
from smolagents.monitoring import TokenUsage  # noqa: E402

from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.smolagents import SmolAgentsAdapter  # noqa: E402

from .conftest import capture_framework_trace, find_event, find_events  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_agent(
    name: str = "TestAgent",
    agent_type: str = "ToolCallingAgent",
    model_id: str = "gpt-4o",
    tools: Any = None,
    managed_agents: Any = None,
) -> MagicMock:
    """Create a mock agent with step_callbacks registry."""
    agent = MagicMock()
    agent.name = name
    type(agent).__name__ = agent_type
    agent.model = MagicMock()
    agent.model.model_id = model_id
    agent.tools = tools or {"search": Mock(), "calculator": Mock()}
    agent.managed_agents = managed_agents
    agent.step_callbacks = CallbackRegistry()
    agent.run = Mock(return_value="final result")
    return agent


def _make_action_step(
    step_number: int = 1,
    tool_calls: Optional[List[ToolCall]] = None,
    token_usage: Optional[TokenUsage] = None,
    model_output: str = "I'll search for that.",
    observations: str = "Search returned 5 results.",
    error: Any = None,
    is_final_answer: bool = False,
    code_action: Optional[str] = None,
    duration: float = 1.5,
) -> ActionStep:
    step = ActionStep(step_number=step_number, timing=Timing(start_time=100.0, end_time=100.0 + duration))
    step.tool_calls = tool_calls
    step.token_usage = token_usage or TokenUsage(input_tokens=100, output_tokens=50)
    step.model_output = model_output
    step.observations = observations
    step.error = error
    step.is_final_answer = is_final_answer
    step.code_action = code_action
    return step


def _make_planning_step(
    plan: str = "1. Search web\n2. Summarise results",
    token_usage: Optional[TokenUsage] = None,
    duration: float = 0.8,
) -> PlanningStep:
    step = PlanningStep(
        model_input_messages=[],
        model_output_message=MagicMock(),
        plan=plan,
        timing=Timing(start_time=100.0, end_time=100.0 + duration),
    )
    step.token_usage = token_usage or TokenUsage(input_tokens=200, output_tokens=100)
    return step


def _simulate_run(adapter: SmolAgentsAdapter, agent: Any, task: str = "test task", steps: Optional[list] = None) -> Any:
    """Call the traced run wrapper, firing step callbacks in between."""
    if steps is None:
        steps = [_make_action_step()]

    original_run = adapter._original_run

    def _fake_run(*args: Any, **kwargs: Any) -> str:
        for step in steps:
            agent.step_callbacks.callback(step, agent=agent)
        return "final result"

    adapter._original_run = _fake_run
    try:
        result = agent.run(task)
    finally:
        adapter._original_run = original_run
    return result


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_connect_raises_without_target(self, mock_client):
        adapter = SmolAgentsAdapter(mock_client)
        with pytest.raises(ValueError, match="requires a target"):
            adapter.connect()

    def test_connect_returns_same_agent(self, mock_client):
        adapter = SmolAgentsAdapter(mock_client)
        agent = _make_mock_agent()
        result = adapter.connect(target=agent)
        assert result is agent
        adapter.disconnect()

    def test_connect_wraps_run(self, mock_client):
        adapter = SmolAgentsAdapter(mock_client)
        agent = _make_mock_agent()
        original_run = agent.run
        adapter.connect(target=agent)
        assert agent.run is not original_run
        assert hasattr(agent.run, "_layerlens_original")
        adapter.disconnect()

    def test_disconnect_unwraps_run(self, mock_client):
        adapter = SmolAgentsAdapter(mock_client)
        agent = _make_mock_agent()
        original_run = agent.run
        adapter.connect(target=agent)
        adapter.disconnect()
        assert agent.run is original_run

    def test_disconnect_clears_state(self, mock_client):
        adapter = SmolAgentsAdapter(mock_client)
        agent = _make_mock_agent()
        adapter.connect(target=agent)
        adapter.disconnect()
        assert adapter._collector is None
        assert adapter._run_span_id is None
        assert adapter._step_count == 0
        assert adapter._target_agent is None

    def test_connect_registers_step_callbacks(self, mock_client):
        adapter = SmolAgentsAdapter(mock_client)
        agent = _make_mock_agent()
        adapter.connect(target=agent)
        registry = agent.step_callbacks
        assert ActionStep in registry._callbacks
        assert PlanningStep in registry._callbacks
        assert FinalAnswerStep in registry._callbacks
        adapter.disconnect()

    def test_disconnect_deregisters_step_callbacks(self, mock_client):
        adapter = SmolAgentsAdapter(mock_client)
        agent = _make_mock_agent()
        adapter.connect(target=agent)
        adapter.disconnect()
        registry = agent.step_callbacks
        for cbs in registry._callbacks.values():
            assert len(cbs) == 0


# ---------------------------------------------------------------------------
# Run wrapper
# ---------------------------------------------------------------------------


class TestRunWrapper:
    def test_successful_run_emits_input_and_output(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = SmolAgentsAdapter(mock_client)
        agent = _make_mock_agent(name="MyAgent", model_id="gpt-4o")
        adapter.connect(target=agent)

        _simulate_run(adapter, agent, task="Summarise this document.")

        events = uploaded["events"]
        agent_in = find_event(events, "agent.input")
        assert agent_in["payload"]["agent_name"] == "MyAgent"
        assert agent_in["payload"]["model"] == "gpt-4o"
        assert agent_in["payload"]["input"] == "Summarise this document."
        assert "tools" in agent_in["payload"]

        agent_out = find_event(events, "agent.output")
        assert agent_out["payload"]["agent_name"] == "MyAgent"
        assert agent_out["payload"]["output"] == "final result"
        assert agent_out["payload"]["duration_ns"] > 0
        assert "error" not in agent_out["payload"]

        adapter.disconnect()

    def test_run_error_emits_error_event(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = SmolAgentsAdapter(mock_client)
        agent = _make_mock_agent(name="FailAgent")
        adapter.connect(target=agent)

        adapter._original_run = Mock(side_effect=RuntimeError("LLM timeout"))
        with pytest.raises(RuntimeError, match="LLM timeout"):
            agent.run("do something")

        events = uploaded["events"]
        error_evt = find_event(events, "agent.error")
        assert error_evt["payload"]["error"] == "LLM timeout"
        assert error_evt["payload"]["error_type"] == "RuntimeError"

        agent_out = find_event(events, "agent.output")
        assert agent_out["payload"]["error"] == "LLM timeout"

        adapter.disconnect()

    def test_run_output_gated_by_capture_content(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        config = CaptureConfig(capture_content=False)
        adapter = SmolAgentsAdapter(mock_client, capture_config=config)
        agent = _make_mock_agent()
        adapter.connect(target=agent)

        _simulate_run(adapter, agent, task="secret task")

        events = uploaded["events"]
        agent_in = find_event(events, "agent.input")
        assert "input" not in agent_in["payload"]

        adapter.disconnect()


# ---------------------------------------------------------------------------
# ActionStep events
# ---------------------------------------------------------------------------


class TestActionStep:
    def test_action_step_emits_model_invoke(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = SmolAgentsAdapter(mock_client)
        agent = _make_mock_agent(model_id="gpt-4o")
        adapter.connect(target=agent)

        step = _make_action_step(token_usage=TokenUsage(input_tokens=100, output_tokens=50))
        _simulate_run(adapter, agent, steps=[step])

        events = uploaded["events"]
        model_evt = find_event(events, "model.invoke")
        assert model_evt["payload"]["model"] == "gpt-4o"
        assert model_evt["payload"]["tokens_prompt"] == 100
        assert model_evt["payload"]["tokens_completion"] == 50
        assert model_evt["payload"]["tokens_total"] == 150

        adapter.disconnect()

    def test_action_step_emits_cost_record(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = SmolAgentsAdapter(mock_client)
        agent = _make_mock_agent(model_id="gpt-4o")
        adapter.connect(target=agent)

        step = _make_action_step(token_usage=TokenUsage(input_tokens=100, output_tokens=50))
        _simulate_run(adapter, agent, steps=[step])

        events = uploaded["events"]
        cost_evt = find_event(events, "cost.record")
        assert cost_evt["payload"]["tokens_total"] == 150
        assert cost_evt["payload"]["model"] == "gpt-4o"

        adapter.disconnect()

    def test_action_step_emits_tool_events(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = SmolAgentsAdapter(mock_client)
        agent = _make_mock_agent()
        adapter.connect(target=agent)

        tool_calls = [
            ToolCall(name="web_search", arguments={"query": "AI safety"}, id="tc-1"),
            ToolCall(name="calculator", arguments={"expr": "2+2"}, id="tc-2"),
        ]
        step = _make_action_step(tool_calls=tool_calls, observations="Search found 5 results. 2+2=4.")
        _simulate_run(adapter, agent, steps=[step])

        events = uploaded["events"]
        tool_call_events = find_events(events, "tool.call")
        tool_result_events = find_events(events, "tool.result")
        assert len(tool_call_events) == 2
        assert len(tool_result_events) == 2
        assert tool_call_events[0]["payload"]["tool_name"] == "web_search"
        assert tool_call_events[1]["payload"]["tool_name"] == "calculator"

        adapter.disconnect()

    def test_final_answer_tool_call_is_skipped(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = SmolAgentsAdapter(mock_client)
        agent = _make_mock_agent()
        adapter.connect(target=agent)

        tool_calls = [
            ToolCall(name="web_search", arguments={"query": "test"}, id="tc-1"),
            ToolCall(name="final_answer", arguments={"answer": "done"}, id="tc-2"),
        ]
        step = _make_action_step(tool_calls=tool_calls, is_final_answer=True)
        _simulate_run(adapter, agent, steps=[step])

        events = uploaded["events"]
        tool_call_events = find_events(events, "tool.call")
        assert len(tool_call_events) == 1
        assert tool_call_events[0]["payload"]["tool_name"] == "web_search"

        adapter.disconnect()

    def test_action_step_emits_step_event(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = SmolAgentsAdapter(mock_client)
        agent = _make_mock_agent(model_id="gpt-4o")
        adapter.connect(target=agent)

        step = _make_action_step(step_number=3, duration=2.5)
        _simulate_run(adapter, agent, steps=[step])

        events = uploaded["events"]
        step_evt = find_event(events, "agent.step")
        assert step_evt["payload"]["step_number"] == 1  # adapter counts internally
        assert step_evt["payload"]["model"] == "gpt-4o"
        assert abs(step_evt["payload"]["duration_ns"] - 2_500_000_000) < 10

        adapter.disconnect()

    def test_action_step_with_error(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = SmolAgentsAdapter(mock_client)
        agent = _make_mock_agent()
        adapter.connect(target=agent)

        step = _make_action_step(error=MagicMock(__str__=lambda s: "tool failed"))
        _simulate_run(adapter, agent, steps=[step])

        events = uploaded["events"]
        step_evt = find_event(events, "agent.step")
        assert "tool failed" in step_evt["payload"]["error"]

        adapter.disconnect()

    def test_code_action_captured_with_content(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = SmolAgentsAdapter(mock_client, capture_config=CaptureConfig.full())
        agent = _make_mock_agent()
        adapter.connect(target=agent)

        step = _make_action_step(code_action="result = 2 + 2\nprint(result)")
        _simulate_run(adapter, agent, steps=[step])

        events = uploaded["events"]
        step_evt = find_event(events, "agent.step")
        assert step_evt["payload"]["code_action"] == "result = 2 + 2\nprint(result)"

        adapter.disconnect()

    def test_code_action_not_captured_without_content(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = SmolAgentsAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        agent = _make_mock_agent()
        adapter.connect(target=agent)

        step = _make_action_step(code_action="result = 2 + 2")
        _simulate_run(adapter, agent, steps=[step])

        events = uploaded["events"]
        step_evt = find_event(events, "agent.step")
        assert "code_action" not in step_evt["payload"]

        adapter.disconnect()

    def test_multiple_steps_counted(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = SmolAgentsAdapter(mock_client)
        agent = _make_mock_agent()
        adapter.connect(target=agent)

        steps = [_make_action_step(step_number=i) for i in range(1, 4)]
        _simulate_run(adapter, agent, steps=steps)

        events = uploaded["events"]
        step_events = find_events(events, "agent.step")
        assert len(step_events) == 3
        assert [e["payload"]["step_number"] for e in step_events] == [1, 2, 3]

        adapter.disconnect()


# ---------------------------------------------------------------------------
# PlanningStep events
# ---------------------------------------------------------------------------


class TestPlanningStep:
    def test_planning_step_emits_step_and_model(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = SmolAgentsAdapter(mock_client)
        agent = _make_mock_agent(model_id="gpt-4o")
        adapter.connect(target=agent)

        plan_step = _make_planning_step(plan="1. Search\n2. Summarise")
        action_step = _make_action_step()
        _simulate_run(adapter, agent, steps=[plan_step, action_step])

        events = uploaded["events"]
        step_events = find_events(events, "agent.step")
        plan_evt = step_events[0]
        assert plan_evt["payload"]["plan"] == "1. Search\n2. Summarise"
        assert abs(plan_evt["payload"]["duration_ns"] - 800_000_000) < 10

        model_events = find_events(events, "model.invoke")
        assert len(model_events) >= 2  # one for planning, one for action step

        adapter.disconnect()

    def test_planning_plan_gated_by_capture_content(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = SmolAgentsAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        agent = _make_mock_agent()
        adapter.connect(target=agent)

        plan_step = _make_planning_step(plan="secret plan")
        _simulate_run(adapter, agent, steps=[plan_step])

        events = uploaded["events"]
        step_evt = find_event(events, "agent.step")
        assert "plan" not in step_evt["payload"]

        adapter.disconnect()


# ---------------------------------------------------------------------------
# Trace integrity
# ---------------------------------------------------------------------------


class TestTraceIntegrity:
    def test_all_events_share_trace_id(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = SmolAgentsAdapter(mock_client)
        agent = _make_mock_agent()
        adapter.connect(target=agent)

        tool_calls = [ToolCall(name="search", arguments={}, id="tc-1")]
        step = _make_action_step(tool_calls=tool_calls)
        _simulate_run(adapter, agent, steps=[step])

        events = uploaded["events"]
        trace_ids = {e["trace_id"] for e in events}
        assert len(trace_ids) == 1

        adapter.disconnect()

    def test_sequence_ids_monotonic(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = SmolAgentsAdapter(mock_client)
        agent = _make_mock_agent()
        adapter.connect(target=agent)

        steps = [_make_action_step(step_number=i) for i in range(1, 4)]
        _simulate_run(adapter, agent, steps=steps)

        events = uploaded["events"]
        seq_ids = [e["sequence_id"] for e in events]
        assert seq_ids == sorted(seq_ids)

        adapter.disconnect()

    def test_attestation_present(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = SmolAgentsAdapter(mock_client)
        agent = _make_mock_agent()
        adapter.connect(target=agent)

        _simulate_run(adapter, agent)

        assert uploaded["attestation"].get("root_hash") is not None

        adapter.disconnect()

    def test_span_hierarchy(self, mock_client):
        """Step events should be children of the run span."""
        uploaded = capture_framework_trace(mock_client)
        adapter = SmolAgentsAdapter(mock_client)
        agent = _make_mock_agent()
        adapter.connect(target=agent)

        tool_calls = [ToolCall(name="search", arguments={}, id="tc-1")]
        step = _make_action_step(tool_calls=tool_calls)
        _simulate_run(adapter, agent, steps=[step])

        events = uploaded["events"]
        run_input = find_event(events, "agent.input")
        run_span_id = run_input["span_id"]

        step_evt = find_event(events, "agent.step")
        assert step_evt["parent_span_id"] == run_span_id

        model_evt = find_event(events, "model.invoke")
        step_span_id = step_evt["span_id"]
        assert model_evt["parent_span_id"] == step_span_id

        tool_evt = find_event(events, "tool.call")
        assert tool_evt["parent_span_id"] == step_span_id

        adapter.disconnect()


# ---------------------------------------------------------------------------
# Input config event
# ---------------------------------------------------------------------------


class TestInputConfig:
    def test_input_includes_tools(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = SmolAgentsAdapter(mock_client)
        agent = _make_mock_agent(tools={"web_search": Mock(), "calculator": Mock()})
        adapter.connect(target=agent)

        _simulate_run(adapter, agent)

        events = uploaded["events"]
        agent_in = find_event(events, "agent.input")
        assert set(agent_in["payload"]["tools"]) == {"web_search", "calculator"}

        adapter.disconnect()

    def test_input_includes_managed_agents(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        sub = MagicMock()
        sub.name = "SubAgent"
        sub.step_callbacks = CallbackRegistry()
        sub.run = Mock()
        adapter = SmolAgentsAdapter(mock_client)
        agent = _make_mock_agent(managed_agents={"sub": sub})
        adapter.connect(target=agent)

        _simulate_run(adapter, agent)

        events = uploaded["events"]
        agent_in = find_event(events, "agent.input")
        assert agent_in["payload"]["managed_agents"] == ["sub"]

        adapter.disconnect()

    def test_input_includes_model(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = SmolAgentsAdapter(mock_client)
        agent = _make_mock_agent(model_id="openai/gpt-4o-mini")
        adapter.connect(target=agent)

        _simulate_run(adapter, agent)

        events = uploaded["events"]
        agent_in = find_event(events, "agent.input")
        assert agent_in["payload"]["model"] == "openai/gpt-4o-mini"

        adapter.disconnect()
