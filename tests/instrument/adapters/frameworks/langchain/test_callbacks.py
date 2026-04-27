"""Tests for LayerLens LangChain Callback Handler.

Ported as-is from ``ateam/tests/adapters/langchain/test_callbacks.py``.

Translation rules applied:
* ``stratix.sdk.python.adapters.langchain.callbacks`` →
  ``layerlens.instrument.adapters.frameworks.langchain.callbacks``
* ``STRATIXCallbackHandler`` → ``LayerLensCallbackHandler``
* ``stratix.sdk.python.adapters.sinks`` →
  ``layerlens.instrument.adapters._base.sinks``
* ``stratix.storage.traces.InMemoryTraceStore`` does not exist in
  ``stratix-python`` (it is a server-side construct in the framework
  repo). The single test that depended on it
  (``test_events_flow_to_trace_store_via_sink``) is marked ``xfail`` per
  the CLAUDE.md "never silently drop tests" rule. All other sink
  integration tests use the duck-typed ``RecordingSink`` and pass.
"""

from __future__ import annotations

from uuid import uuid4
from typing import Any

import pytest

from layerlens.instrument.adapters._base.sinks import EventSink
from layerlens.instrument.adapters.frameworks.langchain.callbacks import (
    LayerLensCallbackHandler,
)


class MockStratix:
    """Mock STRATIX instance for testing."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append({"type": event_type, "payload": payload})

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


class MockGeneration:
    """Mock generation."""

    def __init__(self, text: str) -> None:
        self.text = text


class MockLLMResult:
    """Mock LLM result."""

    def __init__(self, text: str = "Generated text") -> None:
        self.generations = [[MockGeneration(text)]]
        self.llm_output = {"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class TestSTRATIXCallbackHandler:
    """Tests for LayerLensCallbackHandler (legacy ``STRATIX`` name kept in class
    name for diff-friendliness with the ateam source)."""

    def test_initialization(self) -> None:
        """Test handler initializes correctly."""
        handler = LayerLensCallbackHandler()

        assert handler._emit_llm_events is True
        assert handler._emit_tool_events is True
        assert handler._emit_agent_events is True
        assert handler._events == []

    def test_initialization_with_stratix(self) -> None:
        """Test initialization with STRATIX instance."""
        stratix = MockStratix()
        handler = LayerLensCallbackHandler(stratix)

        assert handler._stratix is stratix

    def test_disable_llm_events(self) -> None:
        """Test disabling LLM events."""
        handler = LayerLensCallbackHandler(emit_llm_events=False)
        run_id = uuid4()

        handler.on_llm_start(
            serialized={"name": "test"},
            prompts=["Hello"],
            run_id=run_id,
        )

        assert len(handler._llm_calls) == 0

    def test_disable_tool_events(self) -> None:
        """Test disabling tool events."""
        handler = LayerLensCallbackHandler(emit_tool_events=False)
        run_id = uuid4()

        handler.on_tool_start(
            serialized={"name": "search"},
            input_str="query",
            run_id=run_id,
        )

        assert len(handler._tool_calls) == 0


class TestLLMCallbacks:
    """Tests for LLM callback methods."""

    def test_on_llm_start(self) -> None:
        """Test on_llm_start creates context."""
        stratix = MockStratix()
        handler = LayerLensCallbackHandler(stratix)
        run_id = uuid4()

        handler.on_llm_start(
            serialized={"name": "gpt-4", "kwargs": {"model_name": "gpt-4"}},
            prompts=["Hello, world!"],
            run_id=run_id,
        )

        assert str(run_id) in handler._llm_calls
        ctx = handler._llm_calls[str(run_id)]
        assert ctx.prompts == ["Hello, world!"]
        assert ctx.model == "gpt-4"

    def test_on_llm_end(self) -> None:
        """Test on_llm_end emits model.invoke event."""
        stratix = MockStratix()
        handler = LayerLensCallbackHandler(stratix)
        run_id = uuid4()

        # Start
        handler.on_llm_start(
            serialized={"name": "gpt-4"},
            prompts=["Test prompt"],
            run_id=run_id,
        )

        # End
        handler.on_llm_end(
            response=MockLLMResult("Test output"),
            run_id=run_id,
        )

        # Should emit model.invoke
        events = handler.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["prompts"] == ["Test prompt"]
        assert events[0]["payload"]["output"] == "Test output"

    def test_on_llm_end_captures_token_usage(self) -> None:
        """Test on_llm_end captures token usage."""
        stratix = MockStratix()
        handler = LayerLensCallbackHandler(stratix)
        run_id = uuid4()

        handler.on_llm_start(
            serialized={"name": "test"},
            prompts=["prompt"],
            run_id=run_id,
        )
        handler.on_llm_end(
            response=MockLLMResult(),
            run_id=run_id,
        )

        events = handler.get_events("model.invoke")
        assert events[0]["payload"]["token_usage"]["prompt_tokens"] == 10

    def test_on_llm_error(self) -> None:
        """Test on_llm_error emits event with error."""
        stratix = MockStratix()
        handler = LayerLensCallbackHandler(stratix)
        run_id = uuid4()

        handler.on_llm_start(
            serialized={"name": "test"},
            prompts=["prompt"],
            run_id=run_id,
        )
        handler.on_llm_error(
            error=ValueError("LLM error"),
            run_id=run_id,
        )

        events = handler.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["error"] == "LLM error"

    def test_extract_provider_openai(self) -> None:
        """Test OpenAI provider extraction."""
        handler = LayerLensCallbackHandler()

        provider = handler._extract_provider(
            {"name": "ChatOpenAI", "id": ["langchain", "chat_models", "openai"]}
        )

        assert provider == "openai"

    def test_extract_provider_anthropic(self) -> None:
        """Test Anthropic provider extraction."""
        handler = LayerLensCallbackHandler()

        provider = handler._extract_provider({"name": "ChatAnthropic"})

        assert provider == "anthropic"


class TestToolCallbacks:
    """Tests for tool callback methods."""

    def test_on_tool_start(self) -> None:
        """Test on_tool_start creates context."""
        handler = LayerLensCallbackHandler()
        run_id = uuid4()

        handler.on_tool_start(
            serialized={"name": "search_tool"},
            input_str="search query",
            run_id=run_id,
        )

        assert str(run_id) in handler._tool_calls
        ctx = handler._tool_calls[str(run_id)]
        assert ctx.tool_name == "search_tool"
        assert ctx.tool_input == "search query"

    def test_on_tool_start_with_inputs(self) -> None:
        """Test on_tool_start with structured inputs."""
        handler = LayerLensCallbackHandler()
        run_id = uuid4()

        handler.on_tool_start(
            serialized={"name": "calculator"},
            input_str="2+2",
            run_id=run_id,
            inputs={"expression": "2+2"},
        )

        ctx = handler._tool_calls[str(run_id)]
        assert ctx.tool_input == {"expression": "2+2"}

    def test_on_tool_end(self) -> None:
        """Test on_tool_end emits tool.call event."""
        stratix = MockStratix()
        handler = LayerLensCallbackHandler(stratix)
        run_id = uuid4()

        handler.on_tool_start(
            serialized={"name": "calculator"},
            input_str="2+2",
            run_id=run_id,
        )
        handler.on_tool_end(
            output="4",
            run_id=run_id,
        )

        events = handler.get_events("tool.call")
        assert len(events) == 1
        assert events[0]["payload"]["tool_name"] == "calculator"
        assert events[0]["payload"]["output"] == "4"

    def test_on_tool_error(self) -> None:
        """Test on_tool_error emits event with error."""
        stratix = MockStratix()
        handler = LayerLensCallbackHandler(stratix)
        run_id = uuid4()

        handler.on_tool_start(
            serialized={"name": "failing_tool"},
            input_str="input",
            run_id=run_id,
        )
        handler.on_tool_error(
            error=RuntimeError("Tool failed"),
            run_id=run_id,
        )

        events = handler.get_events("tool.call")
        assert events[0]["payload"]["error"] == "Tool failed"


class TestAgentCallbacks:
    """Tests for agent callback methods."""

    def test_on_agent_action(self) -> None:
        """Test on_agent_action emits tool.call event."""
        stratix = MockStratix()
        handler = LayerLensCallbackHandler(stratix)
        run_id = uuid4()

        class MockAction:
            tool = "search"
            tool_input = "query"

        handler.on_agent_action(
            action=MockAction(),
            run_id=run_id,
        )

        events = handler.get_events("tool.call")
        assert len(events) == 1
        assert events[0]["payload"]["tool_name"] == "search"
        assert events[0]["payload"]["tool_input"] == "query"

    def test_on_agent_finish(self) -> None:
        """Test on_agent_finish emits agent.output event."""
        stratix = MockStratix()
        handler = LayerLensCallbackHandler(stratix)
        run_id = uuid4()

        class MockFinish:
            return_values = {"output": "Final answer"}
            log = "Reasoning log"

        handler.on_agent_finish(
            finish=MockFinish(),
            run_id=run_id,
        )

        events = handler.get_events("agent.output")
        assert len(events) == 1
        assert events[0]["payload"]["output"] == {"output": "Final answer"}


class TestCallbackHandlerHelpers:
    """Tests for callback handler helper methods."""

    def test_get_events_all(self) -> None:
        """Test get_events returns all events."""
        handler = LayerLensCallbackHandler()
        run_id = uuid4()

        handler.on_llm_start(
            serialized={"name": "test"},
            prompts=["prompt"],
            run_id=run_id,
        )
        handler.on_llm_end(
            response=MockLLMResult(),
            run_id=run_id,
        )

        all_events = handler.get_events()
        assert len(all_events) == 1

    def test_get_events_filtered(self) -> None:
        """Test get_events with type filter."""
        handler = LayerLensCallbackHandler()
        run_id = uuid4()

        handler.on_llm_start(
            serialized={"name": "test"},
            prompts=["prompt"],
            run_id=run_id,
        )
        handler.on_llm_end(
            response=MockLLMResult(),
            run_id=run_id,
        )

        handler.on_tool_start(
            serialized={"name": "tool"},
            input_str="input",
            run_id=uuid4(),
        )
        handler.on_tool_end(output="output", run_id=uuid4())

        llm_events = handler.get_events("model.invoke")
        assert len(llm_events) == 1

    def test_clear_events(self) -> None:
        """Test clear_events clears all events."""
        handler = LayerLensCallbackHandler()
        run_id = uuid4()

        handler.on_llm_start(
            serialized={"name": "test"},
            prompts=["prompt"],
            run_id=run_id,
        )
        handler.on_llm_end(
            response=MockLLMResult(),
            run_id=run_id,
        )

        handler.clear_events()

        assert len(handler.get_events()) == 0

    def test_duration_tracked(self) -> None:
        """Test duration is tracked for events."""
        handler = LayerLensCallbackHandler()
        run_id = uuid4()

        handler.on_llm_start(
            serialized={"name": "test"},
            prompts=["prompt"],
            run_id=run_id,
        )
        handler.on_llm_end(
            response=MockLLMResult(),
            run_id=run_id,
        )

        events = handler.get_events("model.invoke")
        assert isinstance(events[0]["payload"]["duration_ns"], int)


class TestChainCallbacks:
    """Tests for chain/LangGraph node callback methods."""

    def test_on_chain_start_langgraph_node(self) -> None:
        """Test on_chain_start with langgraph_node emits agent.input."""
        stratix = MockStratix()
        handler = LayerLensCallbackHandler(stratix)
        run_id = uuid4()

        handler.on_chain_start(
            serialized={"name": "RunnableSequence"},
            inputs={"messages": [{"role": "user", "content": "Hello"}]},
            run_id=run_id,
            metadata={
                "langgraph_node": "researcher",
                "langgraph_step": 1,
                "langgraph_triggers": ["start:start_node"],
            },
        )

        # Context should be stored
        assert str(run_id) in handler._chain_calls
        ctx = handler._chain_calls[str(run_id)]
        assert ctx.node_name == "researcher"

        # run_id -> node mapping should exist
        assert handler._run_to_node[str(run_id)] == "researcher"

        # agent.input event should be emitted
        events = handler.get_events("agent.input")
        assert len(events) == 1
        assert events[0]["payload"]["node_name"] == "researcher"
        assert events[0]["payload"]["langgraph_step"] == 1

    def test_on_chain_end_langgraph_node(self) -> None:
        """Test on_chain_end for a LangGraph node emits agent.output with duration."""
        stratix = MockStratix()
        handler = LayerLensCallbackHandler(stratix)
        run_id = uuid4()

        handler.on_chain_start(
            serialized={"name": "RunnableSequence"},
            inputs={"query": "test"},
            run_id=run_id,
            metadata={"langgraph_node": "writer"},
        )
        handler.on_chain_end(
            outputs={"result": "done"},
            run_id=run_id,
        )

        events = handler.get_events("agent.output")
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload["node_name"] == "writer"
        assert isinstance(payload["duration_ns"], int)
        assert payload["duration_ns"] >= 0
        assert "done" in payload["output"]

        # Context should be cleaned up
        assert str(run_id) not in handler._chain_calls
        assert str(run_id) not in handler._run_to_node

    def test_on_chain_error_langgraph_node(self) -> None:
        """Test on_chain_error for a LangGraph node emits agent.output with error."""
        stratix = MockStratix()
        handler = LayerLensCallbackHandler(stratix)
        run_id = uuid4()

        handler.on_chain_start(
            serialized={"name": "RunnableSequence"},
            inputs={"query": "test"},
            run_id=run_id,
            metadata={"langgraph_node": "analyzer"},
        )
        handler.on_chain_error(
            error=RuntimeError("Node failed"),
            run_id=run_id,
        )

        events = handler.get_events("agent.output")
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload["node_name"] == "analyzer"
        assert payload["error"] == "Node failed"
        assert isinstance(payload["duration_ns"], int)

    def test_non_langgraph_chain_ignored(self) -> None:
        """Test that chains without langgraph_node metadata emit no events."""
        stratix = MockStratix()
        handler = LayerLensCallbackHandler(stratix)
        run_id = uuid4()

        handler.on_chain_start(
            serialized={"name": "RunnableSequence"},
            inputs={"query": "test"},
            run_id=run_id,
            metadata={},
        )
        handler.on_chain_end(
            outputs={"result": "done"},
            run_id=run_id,
        )

        assert len(handler.get_events()) == 0
        assert str(run_id) not in handler._chain_calls
        assert str(run_id) not in handler._run_to_node

    def test_llm_attributed_to_node(self) -> None:
        """Test that LLM calls within a node get node_name in their payload."""
        stratix = MockStratix()
        handler = LayerLensCallbackHandler(stratix)
        chain_run_id = uuid4()
        llm_run_id = uuid4()

        # Start chain (LangGraph node)
        handler.on_chain_start(
            serialized={"name": "RunnableSequence"},
            inputs={"query": "test"},
            run_id=chain_run_id,
            metadata={"langgraph_node": "warren_buffett_agent"},
        )

        # LLM call within the node
        handler.on_llm_start(
            serialized={"name": "gpt-4"},
            prompts=["Analyze this stock"],
            run_id=llm_run_id,
            parent_run_id=chain_run_id,
        )
        handler.on_llm_end(
            response=MockLLMResult("Buy recommendation"),
            run_id=llm_run_id,
            parent_run_id=chain_run_id,
        )

        events = handler.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["node_name"] == "warren_buffett_agent"

    def test_tool_attributed_to_node(self) -> None:
        """Test that tool calls within a node get node_name in their payload."""
        stratix = MockStratix()
        handler = LayerLensCallbackHandler(stratix)
        chain_run_id = uuid4()
        tool_run_id = uuid4()

        # Start chain (LangGraph node)
        handler.on_chain_start(
            serialized={"name": "RunnableSequence"},
            inputs={"query": "test"},
            run_id=chain_run_id,
            metadata={"langgraph_node": "researcher"},
        )

        # Tool call within the node
        handler.on_tool_start(
            serialized={"name": "web_search"},
            input_str="latest AI news",
            run_id=tool_run_id,
            parent_run_id=chain_run_id,
        )
        handler.on_tool_end(
            output="Search results...",
            run_id=tool_run_id,
            parent_run_id=chain_run_id,
        )

        events = handler.get_events("tool.call")
        assert len(events) == 1
        assert events[0]["payload"]["node_name"] == "researcher"

    def test_nested_chain_inherits_node(self) -> None:
        """Test that sub-chains inherit node name and LLM calls get attributed."""
        stratix = MockStratix()
        handler = LayerLensCallbackHandler(stratix)
        node_run_id = uuid4()
        sub_chain_run_id = uuid4()
        llm_run_id = uuid4()

        # Start LangGraph node chain
        handler.on_chain_start(
            serialized={"name": "RunnableSequence"},
            inputs={"query": "test"},
            run_id=node_run_id,
            metadata={"langgraph_node": "planner"},
        )

        # Sub-chain starts within the node (no langgraph_node in metadata)
        handler.on_chain_start(
            serialized={"name": "RunnableSequence"},
            inputs={"sub_input": "detail"},
            run_id=sub_chain_run_id,
            parent_run_id=node_run_id,
            metadata={},
        )

        # LLM call within the sub-chain
        handler.on_llm_start(
            serialized={"name": "gpt-4"},
            prompts=["Plan the approach"],
            run_id=llm_run_id,
            parent_run_id=sub_chain_run_id,
        )
        handler.on_llm_end(
            response=MockLLMResult("Step 1: ..."),
            run_id=llm_run_id,
            parent_run_id=sub_chain_run_id,
        )

        # LLM event should have the inherited node_name
        events = handler.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["node_name"] == "planner"


# ---------------------------------------------------------------------------
# Sink integration tests
# ---------------------------------------------------------------------------


class RecordingSink(EventSink):
    """Simple in-memory sink for testing."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any], int]] = []
        self.flushed = False
        self.closed = False

    def send(self, event_type: str, payload: dict[str, Any], timestamp_ns: int) -> None:
        self.events.append((event_type, payload, timestamp_ns))

    def flush(self) -> None:
        self.flushed = True

    def close(self) -> None:
        self.closed = True


class TestCallbackHandlerSinkIntegration:
    """Integration tests: callback handler -> event sink -> store."""

    @pytest.mark.xfail(
        reason=(
            "Depends on stratix.storage.traces.InMemoryTraceStore which is a "
            "server-side construct in the framework repo (ateam) and is not "
            "shipped in stratix-python. Sink protocol coverage is otherwise "
            "exercised via RecordingSink in this module. Re-enable once an "
            "equivalent in-memory trace store ships in layerlens.instrument "
            "(or a duck-typed ateam store is vendored)."
        ),
        strict=True,
        raises=ImportError,
    )
    def test_events_flow_to_trace_store_via_sink(self) -> None:
        """Events emitted by callback handler reach the TraceStore."""
        # The original ateam test imports stratix.storage.traces.InMemoryTraceStore
        # which has no equivalent in this SDK. The import is performed inside
        # the test body so the xfail is triggered by the ImportError as
        # declared above, instead of failing at module import time.
        from stratix.storage.traces import InMemoryTraceStore  # type: ignore[import-not-found]

        from layerlens.instrument.adapters._base.sinks import TraceStoreSink

        store = InMemoryTraceStore()
        sink = TraceStoreSink(store=store)
        stratix = MockStratix()
        handler = LayerLensCallbackHandler(stratix=stratix, event_sinks=[sink])

        run_id = uuid4()
        handler.on_llm_start(
            serialized={"name": "gpt-4"},
            prompts=["Hello"],
            run_id=run_id,
        )
        handler.on_llm_end(
            response=MockLLMResult("World"),
            run_id=run_id,
        )

        events = store.get_events_by_trace(sink.trace_id)
        assert len(events) == 1
        assert events[0].event_type == "model.invoke"

    def test_multiple_sinks_receive_same_events(self) -> None:
        """All configured sinks receive every event."""
        sink_a = RecordingSink()
        sink_b = RecordingSink()
        stratix = MockStratix()
        handler = LayerLensCallbackHandler(stratix=stratix, event_sinks=[sink_a, sink_b])

        run_id = uuid4()
        handler.on_tool_start(
            serialized={"name": "calc"},
            input_str="2+2",
            run_id=run_id,
        )
        handler.on_tool_end(output="4", run_id=run_id)

        assert len(sink_a.events) == 1
        assert len(sink_b.events) == 1
        assert sink_a.events[0][0] == "tool.call"
        assert sink_b.events[0][0] == "tool.call"

    def test_no_sinks_preserves_existing_behavior(self) -> None:
        """Handler works identically when no sinks are configured."""
        stratix = MockStratix()
        handler = LayerLensCallbackHandler(stratix=stratix)

        run_id = uuid4()
        handler.on_llm_start(
            serialized={"name": "gpt-4"},
            prompts=["Hello"],
            run_id=run_id,
        )
        handler.on_llm_end(
            response=MockLLMResult("World"),
            run_id=run_id,
        )

        assert len(handler.get_events("model.invoke")) == 1
        assert len(handler._event_sinks) == 0

    def test_disconnect_closes_sinks(self) -> None:
        """Calling disconnect() flushes and closes all sinks."""
        sink = RecordingSink()
        handler = LayerLensCallbackHandler(event_sinks=[sink])
        handler.connect()

        handler.disconnect()

        assert sink.flushed
        assert sink.closed
