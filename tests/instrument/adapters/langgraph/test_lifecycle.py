"""Tests for STRATIX LangGraph Lifecycle Hooks."""

import pytest

from layerlens.instrument.adapters.langgraph.lifecycle import (
    STRATIXLangGraphAdapter,
    GraphExecution,
    _TracedGraph,
)
from layerlens.instrument.adapters.langgraph.state import LangGraphStateAdapter


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


class MockGraph:
    """Mock LangGraph compiled graph for testing."""

    def __init__(self, name: str = "test_graph"):
        self.name = name
        self._invocations = []

    def invoke(self, state: dict, config: dict = None):
        self._invocations.append({"state": state, "config": config})
        # Modify state to simulate processing
        return {**state, "processed": True}

    async def ainvoke(self, state: dict, config: dict = None):
        self._invocations.append({"state": state, "config": config})
        return {**state, "processed": True}


class TestSTRATIXLangGraphAdapter:
    """Tests for STRATIXLangGraphAdapter."""

    def test_adapter_initialization(self):
        """Test adapter initializes correctly."""
        stratix = MockStratix()
        adapter = STRATIXLangGraphAdapter(stratix)

        assert adapter._stratix is stratix
        assert adapter._state_adapter is not None
        assert adapter._emit_environment_config is True

    def test_adapter_with_custom_state_adapter(self):
        """Test adapter with custom state adapter."""
        stratix = MockStratix()
        state_adapter = LangGraphStateAdapter(exclude_keys=["secret"])
        adapter = STRATIXLangGraphAdapter(stratix, state_adapter=state_adapter)

        assert adapter._state_adapter is state_adapter

    def test_wrap_graph_returns_traced_graph(self):
        """Test wrap_graph returns a traced graph wrapper."""
        stratix = MockStratix()
        adapter = STRATIXLangGraphAdapter(stratix)
        graph = MockGraph()

        traced = adapter.wrap_graph(graph)

        assert isinstance(traced, _TracedGraph)

    def test_on_graph_start_creates_execution(self):
        """Test on_graph_start creates execution tracking."""
        stratix = MockStratix()
        adapter = STRATIXLangGraphAdapter(stratix)

        execution = adapter.on_graph_start(
            graph_id="test",
            execution_id="exec-1",
            initial_state={"input": "hello"},
        )

        assert isinstance(execution, GraphExecution)
        assert execution.graph_id == "test"
        assert execution.execution_id == "exec-1"
        assert execution.start_time_ns > 0

    def test_on_graph_start_emits_events(self):
        """Test on_graph_start emits expected events."""
        stratix = MockStratix()
        adapter = STRATIXLangGraphAdapter(stratix)

        adapter.on_graph_start(
            graph_id="test",
            execution_id="exec-1",
            initial_state={"input": "hello"},
            config={"recursion_limit": 10},
        )

        # Check environment.config was emitted
        env_events = stratix.get_events("environment.config")
        assert len(env_events) == 1
        assert env_events[0]["payload"]["framework"] == "langgraph"

        # Check agent.input was emitted
        input_events = stratix.get_events("agent.input")
        assert len(input_events) == 1
        assert input_events[0]["payload"]["graph_id"] == "test"

    def test_on_graph_end_completes_execution(self):
        """Test on_graph_end completes execution tracking."""
        stratix = MockStratix()
        adapter = STRATIXLangGraphAdapter(stratix)

        execution = adapter.on_graph_start(
            graph_id="test",
            execution_id="exec-1",
            initial_state={"input": "hello"},
        )

        adapter.on_graph_end(execution, {"output": "world"})

        assert execution.end_time_ns is not None
        assert isinstance(execution.end_time_ns, int)

    def test_on_graph_end_emits_output_event(self):
        """Test on_graph_end emits agent.output event."""
        stratix = MockStratix()
        adapter = STRATIXLangGraphAdapter(stratix)

        execution = adapter.on_graph_start(
            graph_id="test",
            execution_id="exec-1",
            initial_state={"input": "hello"},
        )

        adapter.on_graph_end(execution, {"output": "world"})

        output_events = stratix.get_events("agent.output")
        assert len(output_events) == 1
        assert output_events[0]["payload"]["graph_id"] == "test"

    def test_on_graph_end_emits_state_change_when_changed(self):
        """Test on_graph_end emits state change when state changed."""
        stratix = MockStratix()
        adapter = STRATIXLangGraphAdapter(stratix)

        execution = adapter.on_graph_start(
            graph_id="test",
            execution_id="exec-1",
            initial_state={"count": 0},
        )

        adapter.on_graph_end(execution, {"count": 1})  # Different state

        state_change_events = stratix.get_events("agent.state.change")
        assert len(state_change_events) == 1

    def test_on_graph_end_no_state_change_when_same(self):
        """Test on_graph_end doesn't emit state change when state unchanged."""
        stratix = MockStratix()
        adapter = STRATIXLangGraphAdapter(stratix)

        state = {"count": 0}
        execution = adapter.on_graph_start(
            graph_id="test",
            execution_id="exec-1",
            initial_state=state,
        )

        adapter.on_graph_end(execution, state)  # Same state

        state_change_events = stratix.get_events("agent.state.change")
        assert len(state_change_events) == 0

    def test_on_graph_end_handles_error(self):
        """Test on_graph_end handles execution errors."""
        stratix = MockStratix()
        adapter = STRATIXLangGraphAdapter(stratix)

        execution = adapter.on_graph_start(
            graph_id="test",
            execution_id="exec-1",
            initial_state={},
        )

        adapter.on_graph_end(execution, {}, error=ValueError("test error"))

        assert execution.error == "test error"

    def test_disable_environment_config(self):
        """Test disabling environment.config emission."""
        stratix = MockStratix()
        adapter = STRATIXLangGraphAdapter(stratix, emit_environment_config=False)

        adapter.on_graph_start(
            graph_id="test",
            execution_id="exec-1",
            initial_state={},
        )

        env_events = stratix.get_events("environment.config")
        assert len(env_events) == 0

    def test_node_tracking(self):
        """Test node execution tracking."""
        stratix = MockStratix()
        adapter = STRATIXLangGraphAdapter(stratix)

        execution = adapter.on_graph_start(
            graph_id="test",
            execution_id="exec-1",
            initial_state={"count": 0},
        )

        # Simulate node execution
        node_ctx = adapter.on_node_start(execution, "process_node", {"count": 0})
        adapter.on_node_end(execution, node_ctx, {"count": 1})

        assert len(execution.node_executions) == 1
        assert execution.node_executions[0]["node_name"] == "process_node"


class TestTracedGraph:
    """Tests for _TracedGraph wrapper."""

    def test_invoke_executes_graph(self):
        """Test invoke executes the underlying graph."""
        stratix = MockStratix()
        adapter = STRATIXLangGraphAdapter(stratix)
        graph = MockGraph()

        traced = adapter.wrap_graph(graph)
        result = traced.invoke({"input": "test"})

        assert result["processed"] is True
        assert len(graph._invocations) == 1

    def test_invoke_emits_lifecycle_events(self):
        """Test invoke emits start and end events."""
        stratix = MockStratix()
        adapter = STRATIXLangGraphAdapter(stratix)
        graph = MockGraph()

        traced = adapter.wrap_graph(graph)
        traced.invoke({"input": "test"})

        # Check events were emitted
        assert len(stratix.get_events("agent.input")) == 1
        assert len(stratix.get_events("agent.output")) == 1

    def test_invoke_handles_exception(self):
        """Test invoke handles graph exceptions."""
        stratix = MockStratix()
        adapter = STRATIXLangGraphAdapter(stratix)

        class FailingGraph:
            name = "failing"

            def invoke(self, state, config=None):
                raise ValueError("Graph failed")

        traced = adapter.wrap_graph(FailingGraph())

        with pytest.raises(ValueError, match="Graph failed"):
            traced.invoke({})

        # Should still emit output event with error
        output_events = stratix.get_events("agent.output")
        assert len(output_events) == 1
        assert output_events[0]["payload"]["error"] == "Graph failed"

    def test_attribute_proxying(self):
        """Test that attributes are proxied to underlying graph."""
        stratix = MockStratix()
        adapter = STRATIXLangGraphAdapter(stratix)

        class CustomGraph:
            name = "custom"
            custom_attr = "value"

            def invoke(self, state, config=None):
                return state

        traced = adapter.wrap_graph(CustomGraph())

        assert traced.name == "custom"
        assert traced.custom_attr == "value"

    def test_execution_counting(self):
        """Test that executions are counted."""
        stratix = MockStratix()
        adapter = STRATIXLangGraphAdapter(stratix)
        graph = MockGraph()

        traced = adapter.wrap_graph(graph)

        traced.invoke({})
        traced.invoke({})
        traced.invoke({})

        assert traced._execution_count == 3


@pytest.mark.asyncio
class TestTracedGraphAsync:
    """Async tests for _TracedGraph."""

    async def test_ainvoke_executes_graph(self):
        """Test ainvoke executes the underlying graph."""
        stratix = MockStratix()
        adapter = STRATIXLangGraphAdapter(stratix)
        graph = MockGraph()

        traced = adapter.wrap_graph(graph)
        result = await traced.ainvoke({"input": "test"})

        assert result["processed"] is True

    async def test_ainvoke_emits_events(self):
        """Test ainvoke emits lifecycle events."""
        stratix = MockStratix()
        adapter = STRATIXLangGraphAdapter(stratix)
        graph = MockGraph()

        traced = adapter.wrap_graph(graph)
        await traced.ainvoke({"input": "test"})

        assert len(stratix.get_events("agent.input")) == 1
        assert len(stratix.get_events("agent.output")) == 1
