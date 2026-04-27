"""Tests for the LangGraph lifecycle hooks (LayerLensLangGraphAdapter).

Ported from ``ateam/tests/adapters/langgraph/test_lifecycle.py``.

Renames:
- ``STRATIXLangGraphAdapter`` → ``LayerLensLangGraphAdapter``
- import path ``stratix.sdk.python.adapters.langgraph.lifecycle`` →
  ``layerlens.instrument.adapters.frameworks.langgraph.lifecycle``

The adapter still accepts a positional ``stratix`` argument so the
ateam constructor pattern ``LayerLensLangGraphAdapter(stratix)``
continues to work.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from layerlens.instrument.adapters.frameworks.langgraph.state import (
    LangGraphStateAdapter,
)
from layerlens.instrument.adapters.frameworks.langgraph.lifecycle import (
    GraphExecution,
    LayerLensLangGraphAdapter,
    _TracedGraph,
)


class _MockStratix:
    """Mock STRATIX instance for testing."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append({"type": event_type, "payload": payload})

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


class _MockGraph:
    """Mock LangGraph compiled graph for testing."""

    def __init__(self, name: str = "test_graph") -> None:
        self.name = name
        self._invocations: list[dict[str, Any]] = []

    def invoke(self, state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        self._invocations.append({"state": state, "config": config})
        # Modify state to simulate processing
        return {**state, "processed": True}

    async def ainvoke(
        self, state: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self._invocations.append({"state": state, "config": config})
        return {**state, "processed": True}


class TestLayerLensLangGraphAdapter:
    """Tests for LayerLensLangGraphAdapter."""

    def test_adapter_initialization(self) -> None:
        """Test adapter initializes correctly."""
        stratix = _MockStratix()
        adapter = LayerLensLangGraphAdapter(stratix)

        assert adapter._stratix is stratix
        assert adapter._state_adapter is not None
        assert adapter._emit_environment_config is True

    def test_adapter_with_custom_state_adapter(self) -> None:
        """Test adapter with custom state adapter."""
        stratix = _MockStratix()
        state_adapter = LangGraphStateAdapter(exclude_keys=["secret"])
        adapter = LayerLensLangGraphAdapter(stratix, state_adapter=state_adapter)

        assert adapter._state_adapter is state_adapter

    def test_wrap_graph_returns_traced_graph(self) -> None:
        """Test wrap_graph returns a traced graph wrapper."""
        stratix = _MockStratix()
        adapter = LayerLensLangGraphAdapter(stratix)
        graph = _MockGraph()

        traced = adapter.wrap_graph(graph)

        assert isinstance(traced, _TracedGraph)

    def test_on_graph_start_creates_execution(self) -> None:
        """Test on_graph_start creates execution tracking."""
        stratix = _MockStratix()
        adapter = LayerLensLangGraphAdapter(stratix)

        execution = adapter.on_graph_start(
            graph_id="test",
            execution_id="exec-1",
            initial_state={"input": "hello"},
        )

        assert isinstance(execution, GraphExecution)
        assert execution.graph_id == "test"
        assert execution.execution_id == "exec-1"
        assert execution.start_time_ns > 0

    def test_on_graph_start_emits_events(self) -> None:
        """Test on_graph_start emits expected events."""
        stratix = _MockStratix()
        adapter = LayerLensLangGraphAdapter(stratix)

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

    def test_on_graph_end_completes_execution(self) -> None:
        """Test on_graph_end completes execution tracking."""
        stratix = _MockStratix()
        adapter = LayerLensLangGraphAdapter(stratix)

        execution = adapter.on_graph_start(
            graph_id="test",
            execution_id="exec-1",
            initial_state={"input": "hello"},
        )

        adapter.on_graph_end(execution, {"output": "world"})

        assert execution.end_time_ns is not None
        assert isinstance(execution.end_time_ns, int)

    def test_on_graph_end_emits_output_event(self) -> None:
        """Test on_graph_end emits agent.output event."""
        stratix = _MockStratix()
        adapter = LayerLensLangGraphAdapter(stratix)

        execution = adapter.on_graph_start(
            graph_id="test",
            execution_id="exec-1",
            initial_state={"input": "hello"},
        )

        adapter.on_graph_end(execution, {"output": "world"})

        output_events = stratix.get_events("agent.output")
        assert len(output_events) == 1
        assert output_events[0]["payload"]["graph_id"] == "test"

    def test_on_graph_end_emits_state_change_when_changed(self) -> None:
        """Test on_graph_end emits state change when state changed."""
        stratix = _MockStratix()
        adapter = LayerLensLangGraphAdapter(stratix)

        execution = adapter.on_graph_start(
            graph_id="test",
            execution_id="exec-1",
            initial_state={"count": 0},
        )

        adapter.on_graph_end(execution, {"count": 1})  # Different state

        state_change_events = stratix.get_events("agent.state.change")
        assert len(state_change_events) == 1

    def test_on_graph_end_no_state_change_when_same(self) -> None:
        """Test on_graph_end doesn't emit state change when state unchanged."""
        stratix = _MockStratix()
        adapter = LayerLensLangGraphAdapter(stratix)

        state: dict[str, Any] = {"count": 0}
        execution = adapter.on_graph_start(
            graph_id="test",
            execution_id="exec-1",
            initial_state=state,
        )

        adapter.on_graph_end(execution, state)  # Same state

        state_change_events = stratix.get_events("agent.state.change")
        assert len(state_change_events) == 0

    def test_on_graph_end_handles_error(self) -> None:
        """Test on_graph_end handles execution errors."""
        stratix = _MockStratix()
        adapter = LayerLensLangGraphAdapter(stratix)

        execution = adapter.on_graph_start(
            graph_id="test",
            execution_id="exec-1",
            initial_state={},
        )

        adapter.on_graph_end(execution, {}, error=ValueError("test error"))

        assert execution.error == "test error"

    def test_disable_environment_config(self) -> None:
        """Test disabling environment.config emission."""
        stratix = _MockStratix()
        adapter = LayerLensLangGraphAdapter(stratix, emit_environment_config=False)

        adapter.on_graph_start(
            graph_id="test",
            execution_id="exec-1",
            initial_state={},
        )

        env_events = stratix.get_events("environment.config")
        assert len(env_events) == 0

    def test_node_tracking(self) -> None:
        """Test node execution tracking."""
        stratix = _MockStratix()
        adapter = LayerLensLangGraphAdapter(stratix)

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

    def test_invoke_executes_graph(self) -> None:
        """Test invoke executes the underlying graph."""
        stratix = _MockStratix()
        adapter = LayerLensLangGraphAdapter(stratix)
        graph = _MockGraph()

        traced = adapter.wrap_graph(graph)
        result = traced.invoke({"input": "test"})

        assert result["processed"] is True
        assert len(graph._invocations) == 1

    def test_invoke_emits_lifecycle_events(self) -> None:
        """Test invoke emits start and end events."""
        stratix = _MockStratix()
        adapter = LayerLensLangGraphAdapter(stratix)
        graph = _MockGraph()

        traced = adapter.wrap_graph(graph)
        traced.invoke({"input": "test"})

        # Check events were emitted
        assert len(stratix.get_events("agent.input")) == 1
        assert len(stratix.get_events("agent.output")) == 1

    def test_invoke_handles_exception(self) -> None:
        """Test invoke handles graph exceptions."""
        stratix = _MockStratix()
        adapter = LayerLensLangGraphAdapter(stratix)

        class FailingGraph:
            name = "failing"

            def invoke(
                self, state: dict[str, Any], config: dict[str, Any] | None = None
            ) -> dict[str, Any]:
                raise ValueError("Graph failed")

        traced = adapter.wrap_graph(FailingGraph())

        with pytest.raises(ValueError, match="Graph failed"):
            traced.invoke({})

        # Should still emit output event with error
        output_events = stratix.get_events("agent.output")
        assert len(output_events) == 1
        assert output_events[0]["payload"]["error"] == "Graph failed"

    def test_attribute_proxying(self) -> None:
        """Test that attributes are proxied to underlying graph."""
        stratix = _MockStratix()
        adapter = LayerLensLangGraphAdapter(stratix)

        class CustomGraph:
            name = "custom"
            custom_attr = "value"

            def invoke(
                self, state: dict[str, Any], config: dict[str, Any] | None = None
            ) -> dict[str, Any]:
                return state

        traced = adapter.wrap_graph(CustomGraph())

        assert traced.name == "custom"
        assert traced.custom_attr == "value"

    def test_execution_counting(self) -> None:
        """Test that executions are counted."""
        stratix = _MockStratix()
        adapter = LayerLensLangGraphAdapter(stratix)
        graph = _MockGraph()

        traced = adapter.wrap_graph(graph)

        traced.invoke({})
        traced.invoke({})
        traced.invoke({})

        assert traced._execution_count == 3


class TestTracedGraphAsync:
    """Async tests for _TracedGraph.

    Uses ``asyncio.run`` (rather than ``@pytest.mark.asyncio``) because
    ``pytest-asyncio`` is not in the project's dev requirements; the
    existing ``test_langgraph.py`` smoke test follows the same pattern.
    """

    def test_ainvoke_executes_graph(self) -> None:
        """Test ainvoke executes the underlying graph."""
        stratix = _MockStratix()
        adapter = LayerLensLangGraphAdapter(stratix)
        graph = _MockGraph()

        traced = adapter.wrap_graph(graph)
        result = asyncio.run(traced.ainvoke({"input": "test"}))

        assert result["processed"] is True

    def test_ainvoke_emits_events(self) -> None:
        """Test ainvoke emits lifecycle events."""
        stratix = _MockStratix()
        adapter = LayerLensLangGraphAdapter(stratix)
        graph = _MockGraph()

        traced = adapter.wrap_graph(graph)
        asyncio.run(traced.ainvoke({"input": "test"}))

        assert len(stratix.get_events("agent.input")) == 1
        assert len(stratix.get_events("agent.output")) == 1
