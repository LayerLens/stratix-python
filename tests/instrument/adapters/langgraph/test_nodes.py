"""Tests for STRATIX LangGraph Node Tracing."""

import pytest

from layerlens.instrument.adapters.langgraph.nodes import (
    NodeTracer,
    trace_node,
    create_traced_node,
    NodeExecution,
)


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


class TestNodeTracer:
    """Tests for NodeTracer."""

    def test_tracer_initialization(self):
        """Test tracer initializes correctly."""
        stratix = MockStratix()
        tracer = NodeTracer(stratix)

        assert tracer._stratix is stratix
        assert tracer._executions == []

    def test_trace_node_context_manager(self):
        """Test trace_node works as context manager."""
        stratix = MockStratix()
        tracer = NodeTracer(stratix)

        state = {"count": 0}
        new_state = {"count": 1}

        with tracer.trace_node("test_node", state) as ctx:
            ctx.set_result(new_state)

        assert len(tracer._executions) == 1
        assert tracer._executions[0].node_name == "test_node"

    def test_trace_node_emits_state_change(self):
        """Test trace_node emits state change event."""
        stratix = MockStratix()
        tracer = NodeTracer(stratix)

        with tracer.trace_node("modifier", {"count": 0}) as ctx:
            ctx.set_result({"count": 1})

        state_events = stratix.get_events("agent.state.change")
        assert len(state_events) == 1
        assert state_events[0]["payload"]["node_name"] == "modifier"

    def test_trace_node_no_event_when_unchanged(self):
        """Test no event when state unchanged."""
        stratix = MockStratix()
        tracer = NodeTracer(stratix)

        state = {"count": 0}
        with tracer.trace_node("noop", state) as ctx:
            ctx.set_result(state)

        state_events = stratix.get_events("agent.state.change")
        assert len(state_events) == 0

    def test_decorate_wraps_function(self):
        """Test decorate wraps node function."""
        stratix = MockStratix()
        tracer = NodeTracer(stratix)

        @tracer.decorate
        def increment(state):
            return {"count": state["count"] + 1}

        result = increment({"count": 0})

        assert result == {"count": 1}
        assert len(tracer._executions) == 1

    def test_decorate_preserves_function_name(self):
        """Test decorate preserves original function name."""
        tracer = NodeTracer()

        @tracer.decorate
        def my_custom_node(state):
            return state

        assert my_custom_node.__name__ == "my_custom_node"

    def test_on_node_enter_creates_execution(self):
        """Test on_node_enter creates execution tracking."""
        tracer = NodeTracer()

        execution = tracer.on_node_enter("test", {"key": "value"})

        assert isinstance(execution, NodeExecution)
        assert execution.node_name == "test"
        assert execution.start_time_ns > 0
        assert execution.state_hash_before is not None

    def test_on_node_exit_completes_execution(self):
        """Test on_node_exit completes execution tracking."""
        tracer = NodeTracer()

        execution = tracer.on_node_enter("test", {"count": 0})
        tracer.on_node_exit(execution, {"count": 1})

        assert execution.end_time_ns is not None
        assert execution.state_hash_after is not None

    def test_on_node_exit_handles_error(self):
        """Test on_node_exit records errors."""
        tracer = NodeTracer()

        execution = tracer.on_node_enter("test", {})
        tracer.on_node_exit(execution, {}, error=ValueError("test error"))

        assert execution.error == "test error"

    def test_context_manager_handles_exception(self):
        """Test context manager handles exceptions."""
        stratix = MockStratix()
        tracer = NodeTracer(stratix)

        with pytest.raises(ValueError):
            with tracer.trace_node("failing", {}):
                raise ValueError("Node failed")

        assert len(tracer._executions) == 1


class TestTraceNodeDecorator:
    """Tests for trace_node decorator factory."""

    def test_trace_node_creates_traced_function(self):
        """Test trace_node creates traced function."""
        stratix = MockStratix()

        @trace_node(stratix)
        def my_node(state):
            return {"processed": True}

        result = my_node({})

        assert result == {"processed": True}

    def test_trace_node_emits_events(self):
        """Test trace_node emits state change events."""
        stratix = MockStratix()

        @trace_node(stratix)
        def modifier(state):
            return {"modified": True}

        modifier({"modified": False})

        # Should emit state change since state changed
        state_events = stratix.get_events("agent.state.change")
        assert len(state_events) == 1


class TestCreateTracedNode:
    """Tests for create_traced_node function."""

    def test_creates_traced_version(self):
        """Test creates traced version of function."""
        stratix = MockStratix()

        def original(state):
            return {"count": state.get("count", 0) + 1}

        traced = create_traced_node(original, stratix)
        result = traced({"count": 0})

        assert result == {"count": 1}

    def test_custom_node_name(self):
        """Test custom node name in tracing."""
        stratix = MockStratix()

        def original(state):
            return {"modified": True}

        traced = create_traced_node(original, stratix, node_name="custom_name")
        traced({"modified": False})

        state_events = stratix.get_events("agent.state.change")
        assert len(state_events) == 1
        assert state_events[0]["payload"]["node_name"] == "custom_name"

    def test_preserves_function_metadata(self):
        """Test preserves function metadata."""

        def original(state):
            """Original docstring."""
            return state

        traced = create_traced_node(original)

        assert traced.__name__ == "original"
        assert traced.__doc__ == "Original docstring."
