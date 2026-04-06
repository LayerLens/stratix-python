"""Tests for CrewAI adapter using real CrewAI event bus.

These tests exercise the real crewai.events module — no mocking of CrewAI
internals. Events are constructed and emitted on the real event bus, and
we verify the correct layerlens events come out.

Requires crewai >= 1.0.0 (Python >= 3.10).
"""

from __future__ import annotations

import datetime

import pytest

from .conftest import capture_framework_trace, find_event, find_events

# Skip entire module if crewai is not importable (Python < 3.10 or not installed).
# crewai uses `type | None` syntax which causes TypeError on Python < 3.10,
# and importorskip only catches ImportError, so we guard explicitly.
import sys
if sys.version_info < (3, 10):
    pytest.skip("crewai requires Python >= 3.10", allow_module_level=True)
try:
    import crewai  # noqa: F401
except (ImportError, TypeError):
    pytest.skip("crewai not installed or incompatible", allow_module_level=True)

from crewai.events import (  # noqa: E402
    TaskFailedEvent,
    TaskStartedEvent,
    LLMCallFailedEvent,
    TaskCompletedEvent,
    ToolUsageErrorEvent,
    ToolUsageStartedEvent,
    LLMCallCompletedEvent,
    CrewKickoffFailedEvent,
    ToolUsageFinishedEvent,
    CrewKickoffStartedEvent,
    CrewKickoffCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
    AgentExecutionCompletedEvent,
    crewai_event_bus,  # noqa: E402
)
from crewai.tasks.task_output import TaskOutput  # noqa: E402

from layerlens.instrument.adapters.frameworks.crewai import CrewAIAdapter  # noqa: E402


@pytest.fixture
def adapter_and_trace(mock_client):
    """Create a connected CrewAI adapter with trace capture."""
    uploaded = capture_framework_trace(mock_client)
    adapter = CrewAIAdapter(mock_client)
    with crewai_event_bus.scoped_handlers():
        adapter.connect()
        yield adapter, uploaded
    adapter.disconnect()


class TestCrewAIAdapterLifecycle:
    def test_connect_sets_connected(self, mock_client):
        adapter = CrewAIAdapter(mock_client)
        assert not adapter.is_connected
        with crewai_event_bus.scoped_handlers():
            adapter.connect()
            assert adapter.is_connected
        adapter.disconnect()
        assert not adapter.is_connected

    def test_adapter_info(self, mock_client):
        adapter = CrewAIAdapter(mock_client)
        with crewai_event_bus.scoped_handlers():
            adapter.connect()
            info = adapter.adapter_info()
            assert info.name == "crewai"
            assert info.adapter_type == "framework"
            assert info.connected is True
        adapter.disconnect()

    def test_disconnect_clears_state(self, mock_client):
        adapter = CrewAIAdapter(mock_client)
        with crewai_event_bus.scoped_handlers():
            adapter.connect()
        adapter.disconnect()
        assert adapter._collector is None
        assert adapter._crew_span_id is None
        assert adapter._task_span_ids == {}


class TestCrewKickoff:
    def test_crew_start_emits_agent_input(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace
        evt = CrewKickoffStartedEvent(crew_name="Research Crew", inputs={"topic": "AI"})
        adapter._on_crew_started(None, evt)
        # Crew completed triggers flush
        to = TaskOutput(description="test", raw="done", agent="R")
        completed = CrewKickoffCompletedEvent(crew_name="Research Crew", output=to)
        adapter._on_crew_completed(None, completed)

        events = uploaded["events"]
        agent_in = find_event(events, "agent.input")
        assert agent_in["payload"]["crew_name"] == "Research Crew"
        assert agent_in["payload"]["input"] == {"topic": "AI"}
        assert agent_in["payload"]["framework"] == "crewai"

    def test_crew_completed_emits_agent_output(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace
        start = CrewKickoffStartedEvent(crew_name="MyCrew", inputs={})
        adapter._on_crew_started(None, start)

        to = TaskOutput(description="test", raw="final answer", agent="R")
        completed = CrewKickoffCompletedEvent(crew_name="MyCrew", output=to, total_tokens=500)
        adapter._on_crew_completed(None, completed)

        events = uploaded["events"]
        agent_out = find_event(events, "agent.output")
        assert agent_out["payload"]["crew_name"] == "MyCrew"
        assert agent_out["payload"]["duration_ns"] > 0
        assert agent_out["payload"]["tokens_total"] == 500

        # Should also emit cost.record for total_tokens
        cost = find_event(events, "cost.record")
        assert cost["payload"]["tokens_total"] == 500

    def test_crew_failed_emits_agent_error(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace
        start = CrewKickoffStartedEvent(crew_name="FailCrew", inputs={})
        adapter._on_crew_started(None, start)

        failed = CrewKickoffFailedEvent(crew_name="FailCrew", error="LLM rate limit exceeded")
        adapter._on_crew_failed(None, failed)

        events = uploaded["events"]
        error = find_event(events, "agent.error")
        assert error["payload"]["error"] == "LLM rate limit exceeded"
        assert error["payload"]["crew_name"] == "FailCrew"

    def test_crew_lifecycle_flushes_trace(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace
        start = CrewKickoffStartedEvent(crew_name="FlushCrew", inputs={})
        adapter._on_crew_started(None, start)

        to = TaskOutput(description="t", raw="ok", agent="R")
        completed = CrewKickoffCompletedEvent(crew_name="FlushCrew", output=to)
        adapter._on_crew_completed(None, completed)

        assert uploaded["trace_id"] is not None
        assert len(uploaded["events"]) >= 2
        assert uploaded["attestation"] is not None
        # Collector should be reset after flush
        assert adapter._collector is None


class TestTaskEvents:
    def test_task_start_and_complete(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace
        # Start crew
        adapter._on_crew_started(None, CrewKickoffStartedEvent(crew_name="C", inputs={}))

        # Task lifecycle
        adapter._on_task_started(
            None, TaskStartedEvent(context="research context", task_name="Research Task", agent_role="Researcher")
        )
        to = TaskOutput(description="Research Task", raw="found it", agent="Researcher")
        adapter._on_task_completed(None, TaskCompletedEvent(output=to, task_name="Research Task"))

        # Flush
        to2 = TaskOutput(description="t", raw="ok", agent="R")
        adapter._on_crew_completed(None, CrewKickoffCompletedEvent(crew_name="C", output=to2))

        events = uploaded["events"]
        # Should have crew agent.input, task agent.input, task agent.output, crew agent.output
        agent_inputs = find_events(events, "agent.input")
        assert len(agent_inputs) == 2  # crew + task
        task_input = [e for e in agent_inputs if e["payload"].get("task_name")]
        assert len(task_input) == 1
        assert task_input[0]["payload"]["task_name"] == "Research Task"
        assert task_input[0]["payload"]["agent_role"] == "Researcher"

        # Task events should be children of crew span
        crew_span_id = agent_inputs[0]["span_id"]
        assert task_input[0]["parent_span_id"] == crew_span_id

    def test_task_failed(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace
        adapter._on_crew_started(None, CrewKickoffStartedEvent(crew_name="C", inputs={}))

        adapter._on_task_started(None, TaskStartedEvent(context="ctx", task_name="Bad Task"))
        adapter._on_task_failed(None, TaskFailedEvent(error="task timeout", task_name="Bad Task"))

        adapter._on_crew_failed(None, CrewKickoffFailedEvent(crew_name="C", error="task failed"))

        events = uploaded["events"]
        errors = find_events(events, "agent.error")
        task_error = [e for e in errors if e["payload"].get("task_name")]
        assert len(task_error) == 1
        assert task_error[0]["payload"]["error"] == "task timeout"


class TestLLMEvents:
    def test_llm_completed_emits_model_invoke(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace
        adapter._on_crew_started(None, CrewKickoffStartedEvent(crew_name="C", inputs={}))

        # LLM call with token usage in response
        response = {"content": "hello", "usage": {"prompt_tokens": 100, "completion_tokens": 50}}
        evt = LLMCallCompletedEvent(model="gpt-4o", call_id="call_1", call_type="llm_call", response=response)
        adapter._on_llm_completed(None, evt)

        to = TaskOutput(description="t", raw="ok", agent="R")
        adapter._on_crew_completed(None, CrewKickoffCompletedEvent(crew_name="C", output=to))

        events = uploaded["events"]
        model_invoke = find_event(events, "model.invoke")
        assert model_invoke["payload"]["model"] == "gpt-4o"
        assert model_invoke["payload"]["tokens_prompt"] == 100
        assert model_invoke["payload"]["tokens_completion"] == 50
        assert model_invoke["payload"]["tokens_total"] == 150

        cost = find_event(events, "cost.record")
        assert cost["payload"]["tokens_total"] == 150

    def test_llm_failed_emits_agent_error(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace
        adapter._on_crew_started(None, CrewKickoffStartedEvent(crew_name="C", inputs={}))

        evt = LLMCallFailedEvent(model="gpt-4o", call_id="call_1", error="rate limit exceeded")
        adapter._on_llm_failed(None, evt)

        adapter._on_crew_failed(None, CrewKickoffFailedEvent(crew_name="C", error="llm fail"))

        events = uploaded["events"]
        errors = find_events(events, "agent.error")
        llm_error = [e for e in errors if e["payload"].get("model")]
        assert len(llm_error) == 1
        assert llm_error[0]["payload"]["error"] == "rate limit exceeded"
        assert llm_error[0]["payload"]["model"] == "gpt-4o"


class TestToolEvents:
    def test_tool_started_emits_tool_call(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace
        adapter._on_crew_started(None, CrewKickoffStartedEvent(crew_name="C", inputs={}))

        started_evt = ToolUsageStartedEvent(
            tool_name="web_search",
            tool_args="AI safety research",
            agent_key="researcher_1",
        )
        adapter._on_tool_started(None, started_evt)

        to = TaskOutput(description="t", raw="ok", agent="R")
        adapter._on_crew_completed(None, CrewKickoffCompletedEvent(crew_name="C", output=to))

        events = uploaded["events"]
        tool_call = find_event(events, "tool.call")
        assert tool_call["payload"]["tool_name"] == "web_search"
        assert tool_call["payload"]["input"] == "AI safety research"

    def test_tool_finished_emits_tool_result(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace
        adapter._on_crew_started(None, CrewKickoffStartedEvent(crew_name="C", inputs={}))

        now = datetime.datetime.now()
        later = now + datetime.timedelta(milliseconds=150)
        evt = ToolUsageFinishedEvent(
            tool_name="web_search",
            tool_args="AI safety research",
            started_at=now,
            finished_at=later,
            output="Found 10 results about AI safety",
        )
        adapter._on_tool_finished(None, evt)

        to = TaskOutput(description="t", raw="ok", agent="R")
        adapter._on_crew_completed(None, CrewKickoffCompletedEvent(crew_name="C", output=to))

        events = uploaded["events"]
        tool_result = find_event(events, "tool.result")
        assert tool_result["payload"]["tool_name"] == "web_search"
        assert tool_result["payload"]["output"] == "Found 10 results about AI safety"
        assert tool_result["payload"]["latency_ms"] == pytest.approx(150, abs=5)

    def test_tool_start_end_share_span_id(self, adapter_and_trace):
        """tool.call and tool.result for the same tool use share a span_id."""
        adapter, uploaded = adapter_and_trace
        adapter._on_crew_started(None, CrewKickoffStartedEvent(crew_name="C", inputs={}))

        started_evt = ToolUsageStartedEvent(
            tool_name="calculator",
            tool_args="2+2",
            agent_key="math_agent_1",
        )
        adapter._on_tool_started(None, started_evt)

        now = datetime.datetime.now()
        finished_evt = ToolUsageFinishedEvent(
            tool_name="calculator",
            tool_args="2+2",
            agent_key="math_agent_1",
            started_at=now,
            finished_at=now,
            output="4",
        )
        adapter._on_tool_finished(None, finished_evt)

        to = TaskOutput(description="t", raw="ok", agent="R")
        adapter._on_crew_completed(None, CrewKickoffCompletedEvent(crew_name="C", output=to))

        events = uploaded["events"]
        tool_call = find_event(events, "tool.call")
        tool_result = find_event(events, "tool.result")
        assert tool_call["span_id"] == tool_result["span_id"]

    def test_tool_from_cache(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace
        adapter._on_crew_started(None, CrewKickoffStartedEvent(crew_name="C", inputs={}))

        now = datetime.datetime.now()
        evt = ToolUsageFinishedEvent(
            tool_name="cached_tool",
            tool_args="query",
            started_at=now,
            finished_at=now,
            output="cached result",
            from_cache=True,
        )
        adapter._on_tool_finished(None, evt)

        to = TaskOutput(description="t", raw="ok", agent="R")
        adapter._on_crew_completed(None, CrewKickoffCompletedEvent(crew_name="C", output=to))

        events = uploaded["events"]
        tool_result = find_event(events, "tool.result")
        assert tool_result["payload"]["from_cache"] is True

    def test_tool_error_emits_agent_error(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace
        adapter._on_crew_started(None, CrewKickoffStartedEvent(crew_name="C", inputs={}))

        evt = ToolUsageErrorEvent(tool_name="calculator", tool_args="1/0", error="division by zero")
        adapter._on_tool_error(None, evt)

        adapter._on_crew_failed(None, CrewKickoffFailedEvent(crew_name="C", error="tool fail"))

        events = uploaded["events"]
        errors = find_events(events, "agent.error")
        tool_error = [e for e in errors if e["payload"].get("tool_name")]
        assert len(tool_error) == 1
        assert tool_error[0]["payload"]["tool_name"] == "calculator"
        assert tool_error[0]["payload"]["error"] == "division by zero"


class TestFullCrewLifecycle:
    """End-to-end test simulating a complete crew run with multiple tasks."""

    def test_full_crew_with_tasks_and_llm(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace

        # 1. Crew starts
        adapter._on_crew_started(
            None, CrewKickoffStartedEvent(crew_name="Analysis Crew", inputs={"topic": "quantum computing"})
        )

        # 2. Task 1: Research
        adapter._on_task_started(
            None, TaskStartedEvent(context="research quantum computing", task_name="Research", agent_role="Researcher")
        )

        # 2a. Agent execution starts within task 1
        adapter._on_agent_execution_started(
            None, AgentExecutionStartedEvent.model_construct(agent_role="Researcher", task_prompt="Research quantum computing")
        )

        # 3. LLM call within task 1
        response = {"content": "Quantum computing uses qubits...", "usage": {"prompt_tokens": 200, "completion_tokens": 100}}
        adapter._on_llm_completed(
            None, LLMCallCompletedEvent(model="claude-3-opus", call_id="c1", call_type="llm_call", response=response)
        )

        # 4. Tool use within task 1 (start + finish)
        now = datetime.datetime.now()
        adapter._on_tool_started(
            None,
            ToolUsageStartedEvent(tool_name="arxiv_search", tool_args="quantum computing 2024", agent_key="researcher_1"),
        )
        adapter._on_tool_finished(
            None,
            ToolUsageFinishedEvent(
                tool_name="arxiv_search",
                tool_args="quantum computing 2024",
                agent_key="researcher_1",
                started_at=now,
                finished_at=now,
                output="3 papers found",
            ),
        )

        # 4a. Agent execution completes
        adapter._on_agent_execution_completed(
            None, AgentExecutionCompletedEvent.model_construct(agent_role="Researcher", output="Research complete")
        )

        # 5. Task 1 completes
        to1 = TaskOutput(description="Research", raw="Research complete", agent="Researcher")
        adapter._on_task_completed(None, TaskCompletedEvent(output=to1, task_name="Research"))

        # 6. Task 2: Writing
        adapter._on_task_started(
            None, TaskStartedEvent(context="write about quantum computing", task_name="Write Report", agent_role="Writer")
        )

        # 6a. Agent execution starts within task 2
        adapter._on_agent_execution_started(
            None, AgentExecutionStartedEvent.model_construct(agent_role="Writer", task_prompt="Write the report")
        )

        # 7. Another LLM call
        response2 = {"content": "Final report..."}
        adapter._on_llm_completed(
            None, LLMCallCompletedEvent(model="gpt-4o", call_id="c2", call_type="llm_call", response=response2)
        )

        # 7a. Agent execution completes
        adapter._on_agent_execution_completed(
            None, AgentExecutionCompletedEvent.model_construct(agent_role="Writer", output="Report written")
        )

        # 8. Task 2 completes
        to2 = TaskOutput(description="Write Report", raw="Report written", agent="Writer")
        adapter._on_task_completed(None, TaskCompletedEvent(output=to2, task_name="Write Report"))

        # 9. Crew completes
        final = TaskOutput(description="final", raw="All done", agent="Writer")
        adapter._on_crew_completed(
            None, CrewKickoffCompletedEvent(crew_name="Analysis Crew", output=final, total_tokens=1500)
        )

        # Verify full event trace
        events = uploaded["events"]
        assert uploaded["trace_id"] is not None

        # Count event types
        agent_inputs = find_events(events, "agent.input")
        agent_outputs = find_events(events, "agent.output")
        model_invokes = find_events(events, "model.invoke")
        tool_calls = find_events(events, "tool.call")
        tool_results = find_events(events, "tool.result")
        cost_records = find_events(events, "cost.record")

        # crew + 2 tasks + 2 agent executions = 5 agent.input events
        assert len(agent_inputs) == 5
        # crew + 2 tasks + 2 agent executions = 5 agent.output events
        assert len(agent_outputs) == 5
        assert len(model_invokes) == 2  # 2 LLM calls
        assert len(tool_calls) == 1  # 1 tool.call (started)
        assert len(tool_results) == 1  # 1 tool.result (finished)
        assert len(cost_records) >= 1  # at least crew total_tokens

        # Verify span hierarchy: tasks are children of crew
        crew_span = agent_inputs[0]["span_id"]
        task_inputs = [e for e in agent_inputs if e["payload"].get("task_name")]
        for task_event in task_inputs:
            assert task_event["parent_span_id"] == crew_span

        # Verify all events share the same trace_id
        trace_ids = {e["trace_id"] for e in events}
        assert len(trace_ids) == 1

        # Verify sequence ordering
        sequence_ids = [e["sequence_id"] for e in events]
        assert sequence_ids == sorted(sequence_ids)

        # Verify attestation was built
        assert uploaded["attestation"].get("root_hash") is not None


class TestEventBusIntegration:
    """Test that the adapter actually receives events through the real CrewAI event bus."""

    def test_events_flow_through_bus(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = CrewAIAdapter(mock_client)

        with crewai_event_bus.scoped_handlers():
            adapter.connect()

            # Emit events on the real bus — adapter should pick them up.
            # Flush between events so the async started-handler completes
            # before completed-handler triggers _flush() (which resets state).
            crewai_event_bus.emit(None, event=CrewKickoffStartedEvent(crew_name="BusCrew", inputs={"x": 1}))
            crewai_event_bus.flush(timeout=5.0)

            to = TaskOutput(description="t", raw="bus result", agent="A")
            crewai_event_bus.emit(None, event=CrewKickoffCompletedEvent(crew_name="BusCrew", output=to))
            crewai_event_bus.flush(timeout=5.0)

        events = uploaded["events"]
        assert len(events) >= 2

        agent_in = find_event(events, "agent.input")
        assert agent_in["payload"]["crew_name"] == "BusCrew"

        agent_out = find_event(events, "agent.output")
        assert agent_out["payload"]["crew_name"] == "BusCrew"

    def test_scoped_handlers_cleanup(self, mock_client):
        """Verify that scoped_handlers prevents handler leaks between tests."""
        uploaded = capture_framework_trace(mock_client)
        adapter = CrewAIAdapter(mock_client)

        with crewai_event_bus.scoped_handlers():
            adapter.connect()

        # Events emitted AFTER scope should NOT be captured
        crewai_event_bus.emit(None, event=CrewKickoffStartedEvent(crew_name="Ghost", inputs={}))
        crewai_event_bus.flush(timeout=2.0)

        # Nothing should have been captured (no flush happened either)
        assert uploaded.get("events") is None or len(uploaded.get("events", [])) == 0


class TestCaptureConfigGating:
    """Verify CaptureConfig correctly gates event types."""

    def test_minimal_config_skips_model_and_tool(self, mock_client):
        from layerlens.instrument._capture_config import CaptureConfig

        uploaded = capture_framework_trace(mock_client)
        config = CaptureConfig.minimal()  # l3_model_metadata=False, l5a_tool_calls=False
        adapter = CrewAIAdapter(mock_client, capture_config=config)

        with crewai_event_bus.scoped_handlers():
            adapter.connect()
            adapter._on_crew_started(None, CrewKickoffStartedEvent(crew_name="C", inputs={}))

            # These should be filtered by CaptureConfig
            response = {"content": "hi", "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
            adapter._on_llm_completed(
                None, LLMCallCompletedEvent(model="gpt-4o", call_id="c1", call_type="llm_call", response=response)
            )
            now = datetime.datetime.now()
            adapter._on_tool_started(
                None, ToolUsageStartedEvent(tool_name="x", tool_args="y", agent_key="a1")
            )
            adapter._on_tool_finished(
                None, ToolUsageFinishedEvent(tool_name="x", tool_args="y", agent_key="a1", started_at=now, finished_at=now, output="z")
            )

            to = TaskOutput(description="t", raw="ok", agent="R")
            adapter._on_crew_completed(None, CrewKickoffCompletedEvent(crew_name="C", output=to))

        events = uploaded["events"]
        # model.invoke should be filtered out
        assert len(find_events(events, "model.invoke")) == 0
        # tool.call and tool.result should be filtered out
        assert len(find_events(events, "tool.call")) == 0
        assert len(find_events(events, "tool.result")) == 0
        # agent.input and agent.output should still be there (L1 is enabled)
        assert len(find_events(events, "agent.input")) >= 1
        assert len(find_events(events, "agent.output")) >= 1
        # cost.record IS always-enabled, so if tokens were extracted it should be there
        cost_events = find_events(events, "cost.record")
        assert len(cost_events) >= 1  # cost.record bypasses CaptureConfig


class TestFlowEvents:
    """Test CrewAI Flow lifecycle event handling."""

    def test_flow_start_and_finish(self, adapter_and_trace):
        from crewai.events import FlowStartedEvent, FlowFinishedEvent

        adapter, uploaded = adapter_and_trace
        adapter._on_flow_started(None, FlowStartedEvent(flow_name="AnalysisFlow", inputs={"topic": "AI"}))
        adapter._on_flow_finished(None, FlowFinishedEvent(flow_name="AnalysisFlow", result="done", state={}))

        events = uploaded["events"]
        flow_in = find_event(events, "agent.input")
        assert flow_in["payload"]["flow_name"] == "AnalysisFlow"
        assert flow_in["payload"]["input"] == {"topic": "AI"}
        assert flow_in["span_name"] == "flow:AnalysisFlow"

        flow_out = find_event(events, "agent.output")
        assert flow_out["payload"]["flow_name"] == "AnalysisFlow"
        assert flow_out["payload"]["duration_ns"] > 0


class TestMCPToolEvents:
    """Test MCP tool execution event handling."""

    def test_mcp_tool_completed(self, adapter_and_trace):
        from crewai.events import MCPToolExecutionCompletedEvent

        adapter, uploaded = adapter_and_trace
        adapter._on_crew_started(None, CrewKickoffStartedEvent(crew_name="C", inputs={}))

        now = datetime.datetime.now()
        adapter._on_mcp_tool_completed(
            None,
            MCPToolExecutionCompletedEvent(
                tool_name="read_file",
                tool_args={"path": "/etc/hosts"},
                server_name="filesystem",
                server_url="stdio://mcp-fs",
                transport_type="stdio",
                result="127.0.0.1 localhost",
                started_at=now,
                completed_at=now,
                execution_duration_ms=42,
            ),
        )

        to = TaskOutput(description="t", raw="ok", agent="R")
        adapter._on_crew_completed(None, CrewKickoffCompletedEvent(crew_name="C", output=to))

        events = uploaded["events"]
        tool_call = find_event(events, "tool.call")
        assert tool_call["payload"]["tool_name"] == "read_file"
        assert tool_call["payload"]["mcp_server"] == "filesystem"
        assert tool_call["payload"]["latency_ms"] == 42
        assert tool_call["payload"]["output"] == "127.0.0.1 localhost"

    def test_mcp_tool_failed(self, adapter_and_trace):
        from crewai.events import MCPToolExecutionFailedEvent

        adapter, uploaded = adapter_and_trace
        adapter._on_crew_started(None, CrewKickoffStartedEvent(crew_name="C", inputs={}))

        adapter._on_mcp_tool_failed(
            None,
            MCPToolExecutionFailedEvent(
                tool_name="exec_sql",
                tool_args={"query": "DROP TABLE users"},
                server_name="db-server",
                server_url="http://localhost:3000",
                transport_type="http",
                error="permission denied",
            ),
        )

        adapter._on_crew_failed(None, CrewKickoffFailedEvent(crew_name="C", error="mcp fail"))

        events = uploaded["events"]
        errors = find_events(events, "agent.error")
        mcp_error = [e for e in errors if e["payload"].get("mcp_server")]
        assert len(mcp_error) == 1
        assert mcp_error[0]["payload"]["tool_name"] == "exec_sql"
        assert mcp_error[0]["payload"]["mcp_server"] == "db-server"


class TestLLMLatencyTracking:
    """Test LLM call latency computation from start→complete events."""

    def test_latency_computed_from_started_event(self, adapter_and_trace):
        from crewai.events import LLMCallStartedEvent

        adapter, uploaded = adapter_and_trace
        adapter._on_crew_started(None, CrewKickoffStartedEvent(crew_name="C", inputs={}))

        # Start event stores timestamp
        adapter._on_llm_started(None, LLMCallStartedEvent(
            model="gpt-4o", call_id="latency_test", messages=[], call_type="llm_call",
        ))

        # Small delay to get measurable latency
        import time
        time.sleep(0.01)

        # Complete event computes latency
        response = {"content": "hi", "usage": {"prompt_tokens": 5, "completion_tokens": 3}}
        adapter._on_llm_completed(None, LLMCallCompletedEvent(
            model="gpt-4o", call_id="latency_test", call_type="llm_call", response=response,
        ))

        to = TaskOutput(description="t", raw="ok", agent="R")
        adapter._on_crew_completed(None, CrewKickoffCompletedEvent(crew_name="C", output=to))

        events = uploaded["events"]
        model_invoke = find_event(events, "model.invoke")
        assert "latency_ms" in model_invoke["payload"]
        assert model_invoke["payload"]["latency_ms"] >= 5  # at least 5ms from the sleep


class TestAgentExecutionLifecycle:
    """Test agent execution start/complete/error events."""

    def test_agent_execution_started(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace
        adapter._on_crew_started(None, CrewKickoffStartedEvent(crew_name="C", inputs={}))
        adapter._on_task_started(None, TaskStartedEvent(context="ctx", task_name="T", agent_role="Researcher"))

        adapter._on_agent_execution_started(
            None, AgentExecutionStartedEvent.model_construct(
                agent_role="Researcher", task_prompt="Find AI papers", tools=[]
            )
        )

        to = TaskOutput(description="t", raw="ok", agent="R")
        adapter._on_crew_completed(None, CrewKickoffCompletedEvent(crew_name="C", output=to))

        events = uploaded["events"]
        agent_inputs = find_events(events, "agent.input")
        # Filter for agent execution events (have agent_role but NOT task_name)
        agent_exec = [e for e in agent_inputs if e["payload"].get("agent_role") == "Researcher" and "task_name" not in e["payload"]]
        assert len(agent_exec) == 1
        assert agent_exec[0]["payload"]["framework"] == "crewai"
        assert agent_exec[0]["payload"]["task_prompt"] == "Find AI papers"

    def test_agent_execution_completed(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace
        adapter._on_crew_started(None, CrewKickoffStartedEvent(crew_name="C", inputs={}))

        adapter._on_agent_execution_started(
            None, AgentExecutionStartedEvent.model_construct(agent_role="Writer")
        )
        adapter._on_agent_execution_completed(
            None, AgentExecutionCompletedEvent.model_construct(agent_role="Writer", output="Final draft")
        )

        to = TaskOutput(description="t", raw="ok", agent="R")
        adapter._on_crew_completed(None, CrewKickoffCompletedEvent(crew_name="C", output=to))

        events = uploaded["events"]
        agent_outputs = find_events(events, "agent.output")
        agent_out = [e for e in agent_outputs if e["payload"].get("agent_role") == "Writer"]
        assert len(agent_out) == 1
        assert agent_out[0]["payload"]["status"] == "ok"
        assert agent_out[0]["payload"]["output"] == "Final draft"

    def test_agent_execution_error(self, adapter_and_trace):
        adapter, uploaded = adapter_and_trace
        adapter._on_crew_started(None, CrewKickoffStartedEvent(crew_name="C", inputs={}))

        adapter._on_agent_execution_started(
            None, AgentExecutionStartedEvent.model_construct(agent_role="Researcher")
        )
        adapter._on_agent_execution_error(
            None, AgentExecutionErrorEvent.model_construct(agent_role="Researcher", error="agent crashed")
        )

        adapter._on_crew_failed(None, CrewKickoffFailedEvent(crew_name="C", error="agent fail"))

        events = uploaded["events"]
        errors = find_events(events, "agent.error")
        agent_err = [e for e in errors if e["payload"].get("agent_role") == "Researcher"]
        assert len(agent_err) == 1
        assert agent_err[0]["payload"]["error"] == "agent crashed"

    def test_agent_span_hierarchy(self, adapter_and_trace):
        """Agent execution events are children of the current task span."""
        adapter, uploaded = adapter_and_trace
        adapter._on_crew_started(None, CrewKickoffStartedEvent(crew_name="C", inputs={}))
        adapter._on_task_started(None, TaskStartedEvent(context="ctx", task_name="T1", agent_role="R"))

        adapter._on_agent_execution_started(
            None, AgentExecutionStartedEvent.model_construct(agent_role="R")
        )
        adapter._on_agent_execution_completed(
            None, AgentExecutionCompletedEvent.model_construct(agent_role="R", output="done")
        )

        to = TaskOutput(description="t", raw="ok", agent="R")
        adapter._on_crew_completed(None, CrewKickoffCompletedEvent(crew_name="C", output=to))

        events = uploaded["events"]
        # Find the task span_id
        task_inputs = [e for e in find_events(events, "agent.input") if e["payload"].get("task_name") == "T1"]
        assert len(task_inputs) == 1
        task_span = task_inputs[0]["span_id"]

        # Agent execution should be parented to task (filter out task event which also has agent_role)
        agent_exec_inputs = [e for e in find_events(events, "agent.input") if e["payload"].get("agent_role") == "R" and "task_name" not in e["payload"]]
        assert len(agent_exec_inputs) == 1
        assert agent_exec_inputs[0]["parent_span_id"] == task_span

    def test_llm_parented_to_agent(self, adapter_and_trace):
        """LLM events should be children of the current agent execution span."""
        adapter, uploaded = adapter_and_trace
        adapter._on_crew_started(None, CrewKickoffStartedEvent(crew_name="C", inputs={}))
        adapter._on_task_started(None, TaskStartedEvent(context="ctx", task_name="T1", agent_role="R"))

        adapter._on_agent_execution_started(
            None, AgentExecutionStartedEvent.model_construct(agent_role="R")
        )

        response = {"content": "hi", "usage": {"prompt_tokens": 5, "completion_tokens": 3}}
        adapter._on_llm_completed(
            None, LLMCallCompletedEvent(model="gpt-4o", call_id="c1", call_type="llm_call", response=response)
        )

        adapter._on_agent_execution_completed(
            None, AgentExecutionCompletedEvent.model_construct(agent_role="R", output="done")
        )

        to = TaskOutput(description="t", raw="ok", agent="R")
        adapter._on_crew_completed(None, CrewKickoffCompletedEvent(crew_name="C", output=to))

        events = uploaded["events"]
        # Find the agent execution span_id (not the task event which also has agent_role)
        agent_exec_inputs = [e for e in find_events(events, "agent.input") if e["payload"].get("agent_role") == "R" and "task_name" not in e["payload"]]
        assert len(agent_exec_inputs) == 1
        agent_span = agent_exec_inputs[0]["span_id"]

        # LLM event should be parented to agent execution
        model_invoke = find_event(events, "model.invoke")
        assert model_invoke["parent_span_id"] == agent_span
