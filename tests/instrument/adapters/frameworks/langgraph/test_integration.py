"""Integration tests for the LangGraph adapter using the REAL LangGraph SDK.

Ported from ``ateam/tests/adapters/langgraph/test_integration.py``.

These tests verify that ``LayerLensLangGraphAdapter`` correctly traces
events from actual LangGraph operations — not mocks. The SDK must be
installed (``pip install 'layerlens[langgraph]'``); tests are skipped
otherwise via ``pytest.importorskip``.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Annotated, TypedDict

import pytest

langgraph = pytest.importorskip("langgraph", reason="langgraph not installed")

from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402

from layerlens.instrument.adapters._base.adapter import (  # noqa: E402
    AdapterStatus,
    AdapterCapability,
)
from layerlens.instrument.adapters._base.capture import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.langgraph.nodes import (  # noqa: E402
    NodeTracer,
    create_traced_node,
)
from layerlens.instrument.adapters.frameworks.langgraph.state import (  # noqa: E402
    MessageListAdapter,
    LangGraphStateAdapter,
)
from layerlens.instrument.adapters.frameworks.langgraph.tools import (  # noqa: E402
    ToolTracer,
    LangGraphToolNode,
    trace_langgraph_tool,
)
from layerlens.instrument.adapters.frameworks.langgraph.handoff import (  # noqa: E402
    HandoffDetector,
)
from layerlens.instrument.adapters.frameworks.langgraph.lifecycle import (  # noqa: E402
    LayerLensLangGraphAdapter,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Real event collector — not a mock
# ---------------------------------------------------------------------------


class _EventCollector:
    """Accumulates events for assertions. Not a mock."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.traces_started: int = 0
        self.traces_ended: int = 0

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append({"type": event_type, "payload": payload})

    def start_trace(self, **kwargs: Any) -> str:
        self.traces_started += 1
        return f"trace-{self.traces_started}"

    def end_trace(self, **kwargs: Any) -> None:
        self.traces_ended += 1

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


# ---------------------------------------------------------------------------
# LangGraph state definitions using real TypedDict + annotations
# ---------------------------------------------------------------------------


class SimpleState(TypedDict):
    """Minimal graph state."""

    value: str
    steps: Annotated[list[str], operator.add]


class MessageState(TypedDict):
    """State with LangGraph message reducer."""

    messages: Annotated[list[Any], add_messages]


# ---------------------------------------------------------------------------
# Adapter construction with real LangGraph types
# ---------------------------------------------------------------------------


class TestAdapterWithRealSDK:
    """Verify adapter constructs and interacts with real LangGraph classes."""

    def test_adapter_framework_metadata(self) -> None:
        """Adapter should report langgraph as its framework."""
        collector = _EventCollector()
        adapter = LayerLensLangGraphAdapter(stratix=collector)
        assert adapter.FRAMEWORK == "langgraph"
        assert adapter.VERSION is not None

    def test_adapter_connect_disconnect(self) -> None:
        """connect() and disconnect() lifecycle should not raise."""
        collector = _EventCollector()
        adapter = LayerLensLangGraphAdapter(stratix=collector)
        adapter.connect()
        assert adapter._status == AdapterStatus.HEALTHY
        adapter.disconnect()
        assert adapter._status == AdapterStatus.DISCONNECTED

    def test_adapter_capabilities(self) -> None:
        """Adapter should declare expected capabilities."""
        collector = _EventCollector()
        adapter = LayerLensLangGraphAdapter(stratix=collector)
        info = adapter.get_adapter_info()
        assert AdapterCapability.TRACE_TOOLS in info.capabilities
        assert AdapterCapability.TRACE_STATE in info.capabilities
        assert AdapterCapability.TRACE_HANDOFFS in info.capabilities
        assert AdapterCapability.REPLAY in info.capabilities

    def test_capture_config_propagates(self) -> None:
        """CaptureConfig should be accessible on the adapter."""
        collector = _EventCollector()
        config = CaptureConfig(
            l3_model_metadata=True,
            l5a_tool_calls=False,
        )
        adapter = LayerLensLangGraphAdapter(
            stratix=collector,
            capture_config=config,
        )
        assert adapter._capture_config.l3_model_metadata is True
        assert adapter._capture_config.l5a_tool_calls is False

    def test_health_check_returns_framework_info(self) -> None:
        """health_check should include framework name."""
        collector = _EventCollector()
        adapter = LayerLensLangGraphAdapter(stratix=collector)
        adapter.connect()
        health = adapter.health_check()
        assert health.framework_name == "langgraph"
        assert health.status == AdapterStatus.HEALTHY


# ---------------------------------------------------------------------------
# StateGraph construction with real LangGraph types
# ---------------------------------------------------------------------------


class TestStateGraphTracing:
    """Verify adapter traces real LangGraph StateGraph execution."""

    def _build_simple_graph(self) -> StateGraph:
        """Build a real LangGraph StateGraph with two nodes."""

        def step_a(state: SimpleState) -> dict[str, Any]:
            return {"value": state["value"].upper(), "steps": ["step_a"]}

        def step_b(state: SimpleState) -> dict[str, Any]:
            return {"value": state["value"] + "!", "steps": ["step_b"]}

        graph = StateGraph(SimpleState)
        graph.add_node("step_a", step_a)
        graph.add_node("step_b", step_b)
        graph.add_edge(START, "step_a")
        graph.add_edge("step_a", "step_b")
        graph.add_edge("step_b", END)
        return graph

    def test_compile_real_state_graph(self) -> None:
        """A real StateGraph should compile without error."""
        graph = self._build_simple_graph()
        compiled = graph.compile()
        assert compiled is not None

    def test_invoke_real_state_graph(self) -> None:
        """Invoking a real compiled graph should produce correct output."""
        graph = self._build_simple_graph()
        compiled = graph.compile()
        result = compiled.invoke({"value": "hello", "steps": []})
        assert result["value"] == "HELLO!"
        assert "step_a" in result["steps"]
        assert "step_b" in result["steps"]

    def test_wrap_graph_traces_execution(self) -> None:
        """wrap_graph should trace start and end of graph execution."""
        collector = _EventCollector()
        adapter = LayerLensLangGraphAdapter(stratix=collector)
        adapter.connect()

        graph = self._build_simple_graph()
        compiled = graph.compile()
        traced = adapter.wrap_graph(compiled)

        result = traced.invoke({"value": "hello", "steps": []})

        # Graph should still produce correct output
        assert result["value"] == "HELLO!"
        assert "step_a" in result["steps"]
        assert "step_b" in result["steps"]

        # Adapter should have recorded executions
        assert len(adapter._executions) == 1
        execution = adapter._executions[0]
        assert execution.end_time_ns is not None
        assert execution.error is None

    def test_wrap_graph_emits_agent_input_output(self) -> None:
        """wrap_graph should emit agent.input and agent.output events."""
        collector = _EventCollector()
        adapter = LayerLensLangGraphAdapter(stratix=collector)
        adapter.connect()

        graph = self._build_simple_graph()
        compiled = graph.compile()
        traced = adapter.wrap_graph(compiled)
        traced.invoke({"value": "test", "steps": []})

        # Check trace events on the adapter (emit_dict_event stores them)
        event_types = [e.get("event_type") for e in adapter._trace_events]
        assert "agent.input" in event_types
        assert "agent.output" in event_types

    def test_wrap_graph_detects_state_change(self) -> None:
        """Graph execution that changes state should emit agent.state.change."""
        collector = _EventCollector()
        adapter = LayerLensLangGraphAdapter(stratix=collector)
        adapter.connect()

        graph = self._build_simple_graph()
        compiled = graph.compile()
        traced = adapter.wrap_graph(compiled)
        traced.invoke({"value": "hello", "steps": []})

        # State changed (hello -> HELLO!), so state.change should be emitted
        execution = adapter._executions[0]
        assert execution.initial_state_hash != execution.final_state_hash

        state_change_events = [
            e for e in adapter._trace_events if e.get("event_type") == "agent.state.change"
        ]
        assert len(state_change_events) >= 1


# ---------------------------------------------------------------------------
# Node execution events
# ---------------------------------------------------------------------------


class TestNodeExecutionEvents:
    """Verify node-level tracing with real LangGraph state types."""

    def test_node_tracer_tracks_execution(self) -> None:
        """NodeTracer should record node executions with state hashes."""
        collector = _EventCollector()
        tracer = NodeTracer(stratix_instance=collector)

        state_before: SimpleState = {"value": "hello", "steps": []}
        state_after: SimpleState = {"value": "HELLO", "steps": ["processed"]}

        with tracer.trace_node("process_node", state_before) as ctx:
            ctx.set_result(state_after)

        assert len(tracer._executions) == 1
        execution = tracer._executions[0]
        assert execution.node_name == "process_node"
        assert execution.start_time_ns > 0
        assert execution.end_time_ns is not None
        assert execution.end_time_ns >= execution.start_time_ns
        assert execution.error is None

    def test_node_tracer_detects_state_change(self) -> None:
        """NodeTracer should detect when state changes between before/after."""
        collector = _EventCollector()
        tracer = NodeTracer(stratix_instance=collector)

        state_before: SimpleState = {"value": "a", "steps": []}
        state_after: SimpleState = {"value": "b", "steps": ["changed"]}

        with tracer.trace_node("mutating_node", state_before) as ctx:
            ctx.set_result(state_after)

        execution = tracer._executions[0]
        # Hashes should differ since state was modified
        assert execution.state_hash_before != execution.state_hash_after

    def test_node_tracer_no_change_when_state_identical(self) -> None:
        """NodeTracer should report no change when state is unchanged."""
        collector = _EventCollector()
        tracer = NodeTracer(stratix_instance=collector)

        state: SimpleState = {"value": "same", "steps": []}

        with tracer.trace_node("noop_node", state) as ctx:
            ctx.set_result(state)

        execution = tracer._executions[0]
        assert execution.state_hash_before == execution.state_hash_after

    def test_node_tracer_captures_error(self) -> None:
        """NodeTracer should record errors that occur during node execution."""
        collector = _EventCollector()
        tracer = NodeTracer(stratix_instance=collector)

        state: SimpleState = {"value": "test", "steps": []}

        with pytest.raises(ValueError, match="deliberate"):
            with tracer.trace_node("failing_node", state):
                raise ValueError("deliberate failure")

        execution = tracer._executions[0]
        assert execution.error is not None
        assert "deliberate failure" in execution.error

    def test_node_tracer_emits_state_change_event(self) -> None:
        """NodeTracer should emit agent.state.change via collector when state changes."""
        collector = _EventCollector()
        tracer = NodeTracer(stratix_instance=collector)

        state_before: SimpleState = {"value": "x", "steps": []}
        state_after: SimpleState = {"value": "y", "steps": ["mutated"]}

        with tracer.trace_node("emit_node", state_before) as ctx:
            ctx.set_result(state_after)

        state_events = collector.get_events("agent.state.change")
        assert len(state_events) == 1
        assert state_events[0]["payload"]["node_name"] == "emit_node"

    def test_create_traced_node_wraps_function(self) -> None:
        """create_traced_node should wrap a function and trace it."""
        collector = _EventCollector()

        def process(state: dict[str, Any]) -> dict[str, Any]:
            return {**state, "value": state["value"].upper()}

        traced = create_traced_node(process, stratix_instance=collector)

        result = traced({"value": "hello", "steps": []})
        assert result["value"] == "HELLO"

    def test_node_decorator_preserves_function_name(self) -> None:
        """The decorate method should preserve the original function name."""
        tracer = NodeTracer()

        @tracer.decorate
        def my_custom_node(state: dict[str, Any]) -> dict[str, Any]:
            return state

        assert my_custom_node.__name__ == "my_custom_node"


# ---------------------------------------------------------------------------
# Conditional edge tracking
# ---------------------------------------------------------------------------


class TestConditionalEdgeTracking:
    """Verify adapter works with LangGraph conditional edges."""

    def test_conditional_edge_routing(self) -> None:
        """A graph with conditional edges should route correctly."""

        def classifier(state: SimpleState) -> dict[str, Any]:
            return {"steps": ["classified"]}

        def positive_handler(state: SimpleState) -> dict[str, Any]:
            return {"value": "positive: " + state["value"], "steps": ["positive"]}

        def negative_handler(state: SimpleState) -> dict[str, Any]:
            return {"value": "negative: " + state["value"], "steps": ["negative"]}

        def route_decision(state: SimpleState) -> str:
            if "good" in state["value"]:
                return "positive"
            return "negative"

        graph = StateGraph(SimpleState)
        graph.add_node("classify", classifier)
        graph.add_node("positive", positive_handler)
        graph.add_node("negative", negative_handler)
        graph.add_edge(START, "classify")
        graph.add_conditional_edges(
            "classify",
            route_decision,
            {"positive": "positive", "negative": "negative"},
        )
        graph.add_edge("positive", END)
        graph.add_edge("negative", END)

        compiled = graph.compile()

        # Test positive path
        result = compiled.invoke({"value": "good news", "steps": []})
        assert "positive" in result["value"]
        assert "positive" in result["steps"]

        # Test negative path
        result = compiled.invoke({"value": "bad news", "steps": []})
        assert "negative" in result["value"]
        assert "negative" in result["steps"]

    def test_conditional_edge_traced_via_adapter(self) -> None:
        """Conditional routing should be visible in adapter trace events."""
        collector = _EventCollector()
        adapter = LayerLensLangGraphAdapter(stratix=collector)
        adapter.connect()

        def route_decision(state: SimpleState) -> str:
            return "branch_a" if state["value"] == "a" else "branch_b"

        def branch_a(state: SimpleState) -> dict[str, Any]:
            return {"value": "went_a", "steps": ["branch_a"]}

        def branch_b(state: SimpleState) -> dict[str, Any]:
            return {"value": "went_b", "steps": ["branch_b"]}

        graph = StateGraph(SimpleState)
        graph.add_node("branch_a", branch_a)
        graph.add_node("branch_b", branch_b)
        graph.add_conditional_edges(
            START,
            route_decision,
            {"branch_a": "branch_a", "branch_b": "branch_b"},
        )
        graph.add_edge("branch_a", END)
        graph.add_edge("branch_b", END)

        compiled = graph.compile()
        traced = adapter.wrap_graph(compiled)

        result = traced.invoke({"value": "a", "steps": []})
        assert result["value"] == "went_a"

        # Adapter should have traced the execution
        assert len(adapter._executions) == 1
        execution = adapter._executions[0]
        assert execution.error is None


# ---------------------------------------------------------------------------
# State adapter with real LangGraph state types
# ---------------------------------------------------------------------------


class TestStateAdapterWithRealTypes:
    """Verify state adapter works with real LangGraph TypedDict states."""

    def test_snapshot_typed_dict_state(self) -> None:
        """State adapter should snapshot a real TypedDict state."""
        adapter = LangGraphStateAdapter()
        state: SimpleState = {"value": "hello", "steps": ["init"]}
        snapshot = adapter.snapshot(state)

        assert snapshot.state == {"value": "hello", "steps": ["init"]}
        assert snapshot.hash is not None
        assert snapshot.timestamp_ns > 0

    def test_diff_detects_modified_keys(self) -> None:
        """diff() should detect modified keys in TypedDict state."""
        adapter = LangGraphStateAdapter()
        before = adapter.snapshot({"value": "old", "steps": []})
        after = adapter.snapshot({"value": "new", "steps": ["changed"]})

        diff = adapter.diff(before, after)
        assert "value" in diff["modified"]
        assert diff["modified"]["value"]["before"] == "old"
        assert diff["modified"]["value"]["after"] == "new"

    def test_diff_detects_added_keys(self) -> None:
        """diff() should detect keys added between snapshots."""
        adapter = LangGraphStateAdapter()
        before = adapter.snapshot({"value": "x"})
        after = adapter.snapshot({"value": "x", "extra": "new_key"})

        diff = adapter.diff(before, after)
        assert "extra" in diff["added"]

    def test_include_keys_filter(self) -> None:
        """include_keys should limit which keys are tracked."""
        adapter = LangGraphStateAdapter(include_keys=["value"])
        state: SimpleState = {"value": "test", "steps": ["a", "b"]}
        snapshot = adapter.snapshot(state)

        # Only "value" should be in the snapshot
        assert "value" in snapshot.state
        assert "steps" not in snapshot.state

    def test_exclude_keys_filter(self) -> None:
        """exclude_keys should omit specified keys from tracking."""
        adapter = LangGraphStateAdapter(exclude_keys=["steps"])
        state: SimpleState = {"value": "test", "steps": ["a", "b"]}
        snapshot = adapter.snapshot(state)

        assert "value" in snapshot.state
        assert "steps" not in snapshot.state


# ---------------------------------------------------------------------------
# MessageListAdapter with add_messages reducer
# ---------------------------------------------------------------------------


class TestMessageListAdapterWithRealTypes:
    """Verify MessageListAdapter works with LangGraph message state."""

    def test_message_state_snapshot(self) -> None:
        """MessageListAdapter should snapshot message-based state."""
        adapter = MessageListAdapter(message_key="messages")
        state: MessageState = {"messages": [{"role": "user", "content": "hello"}]}
        snapshot = adapter.snapshot(state)

        assert "messages" in snapshot.state
        assert len(snapshot.state["messages"]) == 1

    def test_get_new_messages_after_append(self) -> None:
        """get_new_messages should return messages added between snapshots."""
        adapter = MessageListAdapter(message_key="messages")

        before_state: dict[str, Any] = {"messages": [{"role": "user", "content": "hi"}]}
        after_state: dict[str, Any] = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello!"},
            ],
        }

        before = adapter.snapshot(before_state)
        after = adapter.snapshot(after_state)

        new_msgs = adapter.get_new_messages(before, after)
        assert len(new_msgs) == 1
        assert new_msgs[0]["content"] == "hello!"


# ---------------------------------------------------------------------------
# Tool tracing with real LangGraph patterns
# ---------------------------------------------------------------------------


class TestToolTracingWithRealTypes:
    """Verify tool tracing works with LangGraph node patterns."""

    def test_trace_langgraph_tool_decorator(self) -> None:
        """@trace_langgraph_tool should trace a function and preserve output."""
        collector = _EventCollector()

        @trace_langgraph_tool(stratix_instance=collector)
        def search(query: str) -> str:
            return f"Results for: {query}"

        result = search("langgraph")
        assert result == "Results for: langgraph"

        tool_events = collector.get_events("tool.call")
        assert len(tool_events) == 1
        assert tool_events[0]["payload"]["tool_name"] == "search"

    def test_trace_langgraph_tool_without_args(self) -> None:
        """@trace_langgraph_tool without args should still work as a decorator."""

        @trace_langgraph_tool
        def simple_tool(x: int) -> int:
            return x * 2

        result = simple_tool(5)
        assert result == 10

    def test_tool_node_as_graph_node(self) -> None:
        """LangGraphToolNode should be callable as a LangGraph node."""
        collector = _EventCollector()

        def lookup(data: Any) -> str:
            return "found_it"

        node = LangGraphToolNode(
            tool_func=lookup,
            stratix_instance=collector,
            tool_name="lookup_tool",
        )

        result = node({"query": "test"})
        assert result["tool_output"] == "found_it"

        tool_events = collector.get_events("tool.call")
        assert len(tool_events) == 1
        assert tool_events[0]["payload"]["tool_name"] == "lookup_tool"

    def test_tool_tracer_captures_error(self) -> None:
        """ToolTracer should capture errors from traced tool functions."""
        collector = _EventCollector()
        tracer = ToolTracer(stratix_instance=collector)

        @tracer.trace
        def failing_tool() -> str:
            raise RuntimeError("tool broke")

        with pytest.raises(RuntimeError, match="tool broke"):
            failing_tool()

        tool_events = collector.get_events("tool.call")
        assert len(tool_events) == 1
        assert "tool broke" in tool_events[0]["payload"]["error"]


# ---------------------------------------------------------------------------
# Handoff detection in multi-node graphs
# ---------------------------------------------------------------------------


class TestHandoffDetectionWithRealGraphs:
    """Verify handoff detection with real LangGraph node transitions."""

    def test_handoff_detector_detects_agent_switch(self) -> None:
        """HandoffDetector should detect when current agent changes."""
        collector = _EventCollector()
        detector = HandoffDetector(stratix_instance=collector)
        detector.register_agents("researcher", "writer")
        detector.set_current_agent("researcher")

        handoff = detector.detect_handoff("writer")
        assert handoff is not None
        assert handoff.from_agent == "researcher"
        assert handoff.to_agent == "writer"

        # Should have emitted an event
        handoff_events = collector.get_events("agent.handoff")
        assert len(handoff_events) == 1

    def test_no_handoff_when_same_agent(self) -> None:
        """HandoffDetector should return None when agent doesn't change."""
        detector = HandoffDetector()
        detector.set_current_agent("researcher")

        handoff = detector.detect_handoff("researcher")
        assert handoff is None

    def test_handoff_detector_in_adapter(self) -> None:
        """Adapter should invoke handoff detector on node_start if attached."""
        collector = _EventCollector()
        detector = HandoffDetector(stratix_instance=collector)
        detector.register_agents("node_a", "node_b")
        detector.set_current_agent("node_a")

        adapter = LayerLensLangGraphAdapter(
            stratix=collector,
            handoff_detector=detector,
        )
        adapter.connect()

        execution = adapter.on_graph_start(
            graph_id="test",
            execution_id="exec-1",
            initial_state={"value": "x"},
        )

        # Transition to node_b should trigger handoff detection
        adapter.on_node_start(execution, "node_b", {"value": "x"})

        handoff_events = collector.get_events("agent.handoff")
        assert len(handoff_events) == 1
        assert handoff_events[0]["payload"]["from_agent"] == "node_a"
        assert handoff_events[0]["payload"]["to_agent"] == "node_b"


# ---------------------------------------------------------------------------
# Graph error handling
# ---------------------------------------------------------------------------


class TestGraphErrorHandling:
    """Verify adapter handles errors in real graph execution."""

    def test_wrap_graph_captures_error(self) -> None:
        """wrap_graph should capture and re-raise errors from the graph."""

        def failing_node(state: SimpleState) -> dict[str, Any]:
            raise RuntimeError("node exploded")

        graph = StateGraph(SimpleState)
        graph.add_node("fail", failing_node)
        graph.add_edge(START, "fail")
        graph.add_edge("fail", END)
        compiled = graph.compile()

        collector = _EventCollector()
        adapter = LayerLensLangGraphAdapter(stratix=collector)
        adapter.connect()
        traced = adapter.wrap_graph(compiled)

        with pytest.raises(RuntimeError, match="node exploded"):
            traced.invoke({"value": "test", "steps": []})

        # Execution should record the error
        assert len(adapter._executions) == 1
        assert adapter._executions[0].error is not None
        assert "node exploded" in adapter._executions[0].error


# ---------------------------------------------------------------------------
# Replay serialization
# ---------------------------------------------------------------------------


class TestReplaySerialization:
    """Verify adapter produces valid replay traces."""

    def test_serialize_for_replay(self) -> None:
        """serialize_for_replay should return a ReplayableTrace."""
        collector = _EventCollector()
        adapter = LayerLensLangGraphAdapter(stratix=collector)
        adapter.connect()

        trace = adapter.serialize_for_replay()
        assert trace.adapter_name == "LayerLensLangGraphAdapter"
        assert trace.framework == "langgraph"
        assert trace.trace_id is not None
        assert "capture_config" in trace.config

    def test_serialize_after_execution_captures_events(self) -> None:
        """Replay trace after graph execution should include emitted events."""
        collector = _EventCollector()
        adapter = LayerLensLangGraphAdapter(stratix=collector)
        adapter.connect()

        def passthrough(state: SimpleState) -> dict[str, Any]:
            return {"steps": ["done"]}

        graph = StateGraph(SimpleState)
        graph.add_node("pass", passthrough)
        graph.add_edge(START, "pass")
        graph.add_edge("pass", END)
        compiled = graph.compile()

        traced = adapter.wrap_graph(compiled)
        traced.invoke({"value": "test", "steps": []})

        replay = adapter.serialize_for_replay()
        # Should have at least agent.input and agent.output events
        assert len(replay.events) >= 2
