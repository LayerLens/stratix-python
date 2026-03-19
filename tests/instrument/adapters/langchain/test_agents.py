"""Tests for STRATIX LangChain Agent Instrumentation."""

import pytest

from layerlens.instrument.adapters.langchain.agents import (
    TracedAgent,
    instrument_agent,
    AgentTracer,
    AgentExecution,
    AgentStep,
)
from layerlens.instrument.adapters.langchain.callbacks import STRATIXCallbackHandler


class MockStratix:
    """Mock STRATIX instance for testing."""

    def __init__(self):
        self.events = []

    def emit(self, event_type: str, payload: dict):
        self.events.append({"type": event_type, "payload": payload})

    def get_events(self, event_type: str = None):
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


class MockAgent:
    """Mock LangChain agent executor."""

    def __init__(self, output=None):
        self._output = output or {"output": "Final answer"}
        self._invocations = []

    def invoke(self, input, config=None, **kwargs):
        self._invocations.append({
            "input": input,
            "config": config,
            "kwargs": kwargs,
        })
        return self._output

    async def ainvoke(self, input, config=None, **kwargs):
        self._invocations.append({
            "input": input,
            "config": config,
            "kwargs": kwargs,
        })
        return self._output

    def run(self, *args, **kwargs):
        self._invocations.append({
            "args": args,
            "kwargs": kwargs,
        })
        return "Final answer string"


class TestTracedAgent:
    """Tests for TracedAgent."""

    def test_initialization(self):
        """Test TracedAgent initializes correctly."""
        agent = MockAgent()
        traced = TracedAgent(agent)

        assert traced._agent is agent
        assert traced._agent_type == "MockAgent"
        assert isinstance(traced._handler, STRATIXCallbackHandler)

    def test_initialization_with_stratix(self):
        """Test initialization with STRATIX instance."""
        stratix = MockStratix()
        agent = MockAgent()
        traced = TracedAgent(agent, stratix)

        assert traced._stratix is stratix

    def test_invoke_executes_agent(self):
        """Test invoke executes underlying agent."""
        agent = MockAgent()
        traced = TracedAgent(agent)

        result = traced.invoke({"input": "What is the weather?"})

        assert result == {"output": "Final answer"}
        assert len(agent._invocations) == 1

    def test_invoke_injects_callback(self):
        """Test invoke injects callback handler."""
        agent = MockAgent()
        traced = TracedAgent(agent)

        traced.invoke({"input": "test"})

        kwargs = agent._invocations[0]["kwargs"]
        assert "callbacks" in kwargs
        assert traced._handler in kwargs["callbacks"]

    def test_invoke_emits_input_event(self):
        """Test invoke emits agent.input event."""
        stratix = MockStratix()
        agent = MockAgent()
        traced = TracedAgent(agent, stratix)

        traced.invoke({"input": "test query"})

        events = stratix.get_events("agent.input")
        assert len(events) == 1
        assert events[0]["payload"]["agent_type"] == "MockAgent"
        assert events[0]["payload"]["input"] == {"input": "test query"}

    def test_invoke_emits_output_event(self):
        """Test invoke emits agent.output event."""
        stratix = MockStratix()
        agent = MockAgent()
        traced = TracedAgent(agent, stratix)

        traced.invoke({"input": "test"})

        events = stratix.get_events("agent.output")
        assert len(events) == 1
        assert events[0]["payload"]["output"] == {"output": "Final answer"}

    def test_invoke_handles_exception(self):
        """Test invoke handles agent exceptions."""
        stratix = MockStratix()

        class FailingAgent:
            def invoke(self, input, config=None, **kwargs):
                raise ValueError("Agent failed")

        traced = TracedAgent(FailingAgent(), stratix)

        with pytest.raises(ValueError, match="Agent failed"):
            traced.invoke({"input": "test"})

        events = stratix.get_events("agent.output")
        assert events[0]["payload"]["error"] == "Agent failed"

    def test_invoke_tracks_execution(self):
        """Test invoke tracks execution."""
        agent = MockAgent()
        traced = TracedAgent(agent)

        traced.invoke({"input": "test1"})
        traced.invoke({"input": "test2"})

        assert len(traced.executions) == 2

    def test_invoke_with_string_input(self):
        """Test invoke handles string input."""
        stratix = MockStratix()
        agent = MockAgent()
        traced = TracedAgent(agent, stratix)

        traced.invoke("simple query")

        events = stratix.get_events("agent.input")
        assert events[0]["payload"]["input"] == "simple query"

    def test_run_method(self):
        """Test run method injects callback."""
        agent = MockAgent()
        traced = TracedAgent(agent)

        result = traced.run("test query")

        assert result == "Final answer string"
        kwargs = agent._invocations[0]["kwargs"]
        assert "callbacks" in kwargs

    def test_record_step(self):
        """Test record_step during execution."""
        agent = MockAgent()
        traced = TracedAgent(agent)

        # Start execution
        traced._current_execution = AgentExecution(
            agent_type="MockAgent",
            start_time_ns=1000,
        )
        traced._executions.append(traced._current_execution)

        traced.record_step(
            action="search",
            action_input="query",
            observation="Results found",
        )

        assert len(traced._current_execution.steps) == 1
        step = traced._current_execution.steps[0]
        assert step.action == "search"
        assert step.observation == "Results found"

    def test_callback_handler_property(self):
        """Test callback_handler property."""
        agent = MockAgent()
        traced = TracedAgent(agent)

        handler = traced.callback_handler

        assert isinstance(handler, STRATIXCallbackHandler)

    def test_attribute_proxying(self):
        """Test attribute access is proxied."""
        agent = MockAgent()
        agent.custom_attr = "value"
        traced = TracedAgent(agent)

        assert traced.custom_attr == "value"


@pytest.mark.asyncio
class TestTracedAgentAsync:
    """Async tests for TracedAgent."""

    async def test_ainvoke_executes_agent(self):
        """Test ainvoke executes underlying agent."""
        agent = MockAgent()
        traced = TracedAgent(agent)

        result = await traced.ainvoke({"input": "test"})

        assert result == {"output": "Final answer"}

    async def test_ainvoke_emits_events(self):
        """Test ainvoke emits input and output events."""
        stratix = MockStratix()
        agent = MockAgent()
        traced = TracedAgent(agent, stratix)

        await traced.ainvoke({"input": "test"})

        assert len(stratix.get_events("agent.input")) == 1
        assert len(stratix.get_events("agent.output")) == 1

    async def test_ainvoke_handles_exception(self):
        """Test ainvoke handles exceptions."""
        stratix = MockStratix()

        class FailingAsyncAgent:
            async def ainvoke(self, input, config=None, **kwargs):
                raise ValueError("Async agent failed")

        traced = TracedAgent(FailingAsyncAgent(), stratix)

        with pytest.raises(ValueError, match="Async agent failed"):
            await traced.ainvoke({"input": "test"})


class TestInstrumentAgent:
    """Tests for instrument_agent function."""

    def test_creates_traced_agent(self):
        """Test creates TracedAgent instance."""
        agent = MockAgent()
        traced = instrument_agent(agent)

        assert isinstance(traced, TracedAgent)

    def test_passes_stratix_instance(self):
        """Test passes STRATIX instance."""
        stratix = MockStratix()
        agent = MockAgent()
        traced = instrument_agent(agent, stratix)

        assert traced._stratix is stratix


class TestAgentTracer:
    """Tests for AgentTracer."""

    def test_initialization(self):
        """Test tracer initializes correctly."""
        tracer = AgentTracer()

        assert tracer._agents == {}

    def test_trace_creates_traced_agent(self):
        """Test trace creates traced agent."""
        tracer = AgentTracer()
        agent = MockAgent()

        traced = tracer.trace(agent)

        assert isinstance(traced, TracedAgent)

    def test_trace_with_custom_name(self):
        """Test trace with custom name."""
        tracer = AgentTracer()
        agent = MockAgent()

        tracer.trace(agent, name="my_agent")

        assert "my_agent" in tracer._agents

    def test_get_agent(self):
        """Test get_agent retrieves traced agent."""
        tracer = AgentTracer()
        agent = MockAgent()

        tracer.trace(agent, name="test")
        retrieved = tracer.get_agent("test")

        assert retrieved is not None

    def test_get_agent_not_found(self):
        """Test get_agent returns None for unknown agent."""
        tracer = AgentTracer()

        result = tracer.get_agent("unknown")

        assert result is None

    def test_get_all_executions(self):
        """Test get_all_executions returns all executions."""
        tracer = AgentTracer()
        agent1 = MockAgent()
        agent2 = MockAgent()

        traced1 = tracer.trace(agent1, name="agent1")
        traced2 = tracer.trace(agent2, name="agent2")

        traced1.invoke({"input": "test1"})
        traced2.invoke({"input": "test2"})

        all_execs = tracer.get_all_executions()

        assert len(all_execs) == 2

    def test_get_total_steps(self):
        """Test get_total_steps counts all steps."""
        tracer = AgentTracer()
        agent = MockAgent()

        traced = tracer.trace(agent)
        traced.invoke({"input": "test"})

        # No steps recorded via callback in mock, should be 0
        total = tracer.get_total_steps()
        assert total == 0


class TestAgentExecution:
    """Tests for AgentExecution dataclass."""

    def test_execution_creation(self):
        """Test execution creation."""
        execution = AgentExecution(
            agent_type="TestAgent",
            start_time_ns=1000,
            input="test query",
        )

        assert execution.agent_type == "TestAgent"
        assert execution.start_time_ns == 1000
        assert execution.input == "test query"
        assert execution.steps == []
        assert execution.error is None

    def test_execution_with_steps(self):
        """Test execution with steps."""
        step = AgentStep(
            step_number=1,
            action="search",
            action_input="query",
            observation="results",
        )

        execution = AgentExecution(
            agent_type="TestAgent",
            start_time_ns=1000,
            steps=[step],
        )

        assert len(execution.steps) == 1
        assert execution.steps[0].action == "search"


class TestAgentStep:
    """Tests for AgentStep dataclass."""

    def test_step_creation(self):
        """Test step creation."""
        step = AgentStep(
            step_number=1,
            action="search",
            action_input={"query": "test"},
            observation="Found 5 results",
            timestamp_ns=12345,
        )

        assert step.step_number == 1
        assert step.action == "search"
        assert step.action_input == {"query": "test"}
        assert step.observation == "Found 5 results"
        assert step.timestamp_ns == 12345

    def test_step_optional_fields(self):
        """Test step with optional fields."""
        step = AgentStep(step_number=1)

        assert step.action is None
        assert step.observation is None
