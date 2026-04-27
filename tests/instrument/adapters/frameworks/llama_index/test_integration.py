"""Integration tests for LlamaIndex adapter using the REAL LlamaIndex SDK.

Ported from ``ateam/tests/adapters/llama_index/test_integration.py``.

These tests verify that ``LlamaIndexAdapter`` correctly instruments
actual LlamaIndex types — not mocks. The SDK must be installed::

    pip install 'layerlens[llama-index]'  # or pip install llama-index-core

Tests are skipped if ``llama-index-core`` is not installed via
``pytest.importorskip``.

Translation rules applied:
* ``stratix.sdk.python.adapters.base`` ->
  ``layerlens.instrument.adapters._base``
* ``stratix.sdk.python.adapters.capture`` ->
  ``layerlens.instrument.adapters._base``
* ``stratix.sdk.python.adapters.llama_index.lifecycle`` ->
  ``layerlens.instrument.adapters.frameworks.llama_index.lifecycle``
* ``StratixEventHandler`` (the inner handler class name returned by
  ``_create_event_handler().class_name()``) is preserved verbatim from
  the adapter source — it is the public LlamaIndex-side handler name
  rather than a brand symbol, and renaming would require changing the
  adapter itself, which is outside the scope of this test-restoration
  PR.
"""

from __future__ import annotations

from typing import Any

import pytest

llama_index_core = pytest.importorskip("llama_index.core", reason="llama-index-core not installed")

from llama_index.core.instrumentation import (  # noqa: E402
    get_dispatcher,  # pyright: ignore[reportPrivateImportUsage]
)
from llama_index.core.instrumentation.events import BaseEvent  # noqa: E402
from llama_index.core.instrumentation.event_handlers import BaseEventHandler  # noqa: E402

from layerlens.instrument.adapters._base import (  # noqa: E402
    AdapterStatus,
    CaptureConfig,
    AdapterCapability,
)
from layerlens.instrument.adapters.frameworks.llama_index.lifecycle import (  # noqa: E402
    LlamaIndexAdapter,
)

# ---------------------------------------------------------------------------
# Event collector — not a mock
# ---------------------------------------------------------------------------


class EventCollector:
    """Real event collector that accumulates events for assertions."""

    org_id: str = "test-org"

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
# Adapter construction with real LlamaIndex types
# ---------------------------------------------------------------------------


class TestAdapterWithRealSDK:
    """Verify adapter constructs and connects with real LlamaIndex classes."""

    def test_connect_detects_llama_index_version(self) -> None:
        """Adapter should detect the installed llama-index version on connect."""
        collector = EventCollector()
        adapter = LlamaIndexAdapter(stratix=collector)
        adapter.connect()

        assert adapter.is_connected
        assert adapter._status == AdapterStatus.HEALTHY
        health = adapter.health_check()
        assert health.framework_version is not None
        assert health.framework_version != "unknown"

    def test_adapter_info_capabilities(self) -> None:
        """Adapter info should report correct capabilities."""
        collector = EventCollector()
        adapter = LlamaIndexAdapter(stratix=collector)

        info = adapter.get_adapter_info()
        assert info.name == "LlamaIndexAdapter"
        assert info.framework == "llama_index"
        assert AdapterCapability.TRACE_TOOLS in info.capabilities
        assert AdapterCapability.TRACE_MODELS in info.capabilities
        assert AdapterCapability.TRACE_STATE in info.capabilities
        assert AdapterCapability.TRACE_HANDOFFS in info.capabilities

    def test_capture_config_propagates(self) -> None:
        """CaptureConfig correctly controls which events are captured."""
        collector = EventCollector()
        config = CaptureConfig(
            l3_model_metadata=True,
            l5a_tool_calls=False,
            l1_agent_io=True,
        )
        adapter = LlamaIndexAdapter(stratix=collector, capture_config=config)
        assert adapter._capture_config.l3_model_metadata is True
        assert adapter._capture_config.l5a_tool_calls is False


# ---------------------------------------------------------------------------
# Instrumentation Module integration
# ---------------------------------------------------------------------------


class TestInstrumentationModule:
    """Verify adapter integrates with the real LlamaIndex Instrumentation Module."""

    def test_create_event_handler_returns_base_event_handler(self) -> None:
        """_create_event_handler should return a valid BaseEventHandler subclass."""
        collector = EventCollector()
        adapter = LlamaIndexAdapter(stratix=collector)
        adapter.connect()

        handler = adapter._create_event_handler()
        assert handler is not None
        assert isinstance(handler, BaseEventHandler)
        assert handler.class_name() == "StratixEventHandler"

    def test_event_handler_handle_accepts_base_event(self) -> None:
        """The handler's handle() method should accept a BaseEvent without raising."""
        collector = EventCollector()
        adapter = LlamaIndexAdapter(stratix=collector)
        adapter.connect()

        handler = adapter._create_event_handler()

        # Create a minimal BaseEvent subclass for testing
        class TestEvent(BaseEvent):
            """Minimal event subclass for testing."""

            pass

        # handle() should not raise, even for unknown event types
        handler.handle(TestEvent())

    def test_instrument_workflow_registers_handler_on_dispatcher(self) -> None:
        """instrument_workflow should add a handler to the LlamaIndex dispatcher."""
        collector = EventCollector()
        adapter = LlamaIndexAdapter(stratix=collector)
        adapter.connect()

        # Get baseline handler count
        dispatcher = get_dispatcher()
        baseline_handlers = len(getattr(dispatcher, "event_handlers", []))

        adapter.instrument_workflow(workflow=None)

        handlers = getattr(dispatcher, "event_handlers", [])
        assert len(handlers) > baseline_handlers
        assert adapter._event_handler is not None

        # Verify the handler was actually registered
        assert adapter._event_handler in handlers

        # Cleanup: disconnect to unregister
        adapter.disconnect()

    def test_disconnect_unregisters_handler(self) -> None:
        """disconnect() should remove the event handler from the dispatcher."""
        collector = EventCollector()
        adapter = LlamaIndexAdapter(stratix=collector)
        adapter.connect()

        adapter.instrument_workflow(workflow=None)
        registered_handler = adapter._event_handler
        assert registered_handler is not None

        adapter.disconnect()
        assert adapter._event_handler is None

        dispatcher = get_dispatcher()
        handlers = getattr(dispatcher, "event_handlers", [])
        assert registered_handler not in handlers


# ---------------------------------------------------------------------------
# Lifecycle hooks and event emission
# ---------------------------------------------------------------------------


class TestLifecycleHooks:
    """Verify lifecycle hooks emit correct events."""

    def test_on_agent_start_emits_agent_input(self) -> None:
        """on_agent_start should emit an agent.input event."""
        collector = EventCollector()
        adapter = LlamaIndexAdapter(stratix=collector)
        adapter.connect()

        adapter.on_agent_start(agent_name="research_agent", input_data="Find papers on AI")

        events = collector.get_events("agent.input")
        assert len(events) == 1
        assert events[0]["payload"]["framework"] == "llama_index"
        assert events[0]["payload"]["agent_name"] == "research_agent"
        assert events[0]["payload"]["timestamp_ns"] > 0

    def test_on_agent_end_emits_agent_output_with_duration(self) -> None:
        """on_agent_end should emit agent.output with duration_ns."""
        collector = EventCollector()
        adapter = LlamaIndexAdapter(stratix=collector)
        adapter.connect()

        adapter.on_agent_start(agent_name="summarizer", input_data="Summarize this")
        adapter.on_agent_end(agent_name="summarizer", output="Summary done")

        events = collector.get_events("agent.output")
        assert len(events) == 1
        assert events[0]["payload"]["agent_name"] == "summarizer"
        assert events[0]["payload"]["duration_ns"] >= 0

    def test_on_agent_end_captures_error(self) -> None:
        """on_agent_end should capture errors."""
        collector = EventCollector()
        adapter = LlamaIndexAdapter(stratix=collector)
        adapter.connect()

        adapter.on_agent_end(
            agent_name="broken",
            output=None,
            error=ValueError("Index not found"),
        )

        events = collector.get_events("agent.output")
        assert len(events) == 1
        assert "Index not found" in events[0]["payload"]["error"]

    def test_on_tool_use_emits_tool_call(self) -> None:
        """on_tool_use should emit a tool.call event."""
        collector = EventCollector()
        adapter = LlamaIndexAdapter(stratix=collector)
        adapter.connect()

        adapter.on_tool_use(
            tool_name="query_engine",
            tool_input={"query": "What is RAG?"},
            tool_output={"answer": "Retrieval-Augmented Generation"},
        )

        events = collector.get_events("tool.call")
        assert len(events) == 1
        assert events[0]["payload"]["tool_name"] == "query_engine"
        assert events[0]["payload"]["tool_input"]["query"] == "What is RAG?"

    def test_on_llm_call_emits_model_invoke(self) -> None:
        """on_llm_call should emit a model.invoke event."""
        collector = EventCollector()
        adapter = LlamaIndexAdapter(stratix=collector)
        adapter.connect()

        adapter.on_llm_call(
            provider="openai",
            model="gpt-4o",
            tokens_prompt=150,
            tokens_completion=80,
            latency_ms=320.5,
        )

        events = collector.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["model"] == "gpt-4o"
        assert events[0]["payload"]["provider"] == "openai"
        assert events[0]["payload"]["tokens_prompt"] == 150
        assert events[0]["payload"]["tokens_completion"] == 80
        assert events[0]["payload"]["latency_ms"] == 320.5

    def test_on_handoff_emits_agent_handoff(self) -> None:
        """on_handoff should emit an agent.handoff event with context hash."""
        collector = EventCollector()
        adapter = LlamaIndexAdapter(stratix=collector)
        adapter.connect()

        adapter.on_handoff(
            from_agent="router",
            to_agent="specialist",
            context="User wants financial analysis",
        )

        events = collector.get_events("agent.handoff")
        assert len(events) == 1
        assert events[0]["payload"]["from_agent"] == "router"
        assert events[0]["payload"]["to_agent"] == "specialist"
        assert events[0]["payload"]["context_hash"] is not None
        assert len(events[0]["payload"]["context_hash"]) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# Event routing from LlamaIndex event types
# ---------------------------------------------------------------------------


class TestEventRouting:
    """Verify _handle_event routes LlamaIndex event types to correct Stratix events."""

    def test_unknown_event_type_does_not_raise(self) -> None:
        """Unknown event types should be silently ignored."""
        collector = EventCollector()
        adapter = LlamaIndexAdapter(stratix=collector)
        adapter.connect()

        class UnknownEvent(BaseEvent):
            pass

        # Should not raise
        adapter._handle_event(UnknownEvent())
        assert len(collector.events) == 0

    def test_not_connected_skips_events(self) -> None:
        """Events should be skipped when adapter is not connected."""
        collector = EventCollector()
        adapter = LlamaIndexAdapter(stratix=collector)
        # Do NOT call connect

        class SomeEvent(BaseEvent):
            pass

        adapter._handle_event(SomeEvent())
        assert len(collector.events) == 0


# ---------------------------------------------------------------------------
# CaptureConfig gating
# ---------------------------------------------------------------------------


class TestCaptureConfigGating:
    """Verify CaptureConfig correctly gates events."""

    def test_minimal_config_gates_l3_and_l5(self) -> None:
        """Minimal config should suppress model.invoke and tool.call."""
        collector = EventCollector()
        adapter = LlamaIndexAdapter(stratix=collector, capture_config=CaptureConfig.minimal())
        adapter.connect()

        adapter.on_llm_call(model="gpt-4o")
        adapter.on_tool_use(tool_name="search")

        assert len(collector.get_events("model.invoke")) == 0
        assert len(collector.get_events("tool.call")) == 0

    def test_minimal_config_allows_cross_cutting(self) -> None:
        """Cross-cutting events should always be emitted."""
        collector = EventCollector()
        adapter = LlamaIndexAdapter(stratix=collector, capture_config=CaptureConfig.minimal())
        adapter.connect()

        adapter.on_handoff(from_agent="a", to_agent="b")

        assert len(collector.get_events("agent.handoff")) == 1

    def test_full_config_emits_all(self) -> None:
        """Full config should emit all event types."""
        collector = EventCollector()
        adapter = LlamaIndexAdapter(stratix=collector, capture_config=CaptureConfig.full())
        adapter.connect()

        adapter.on_llm_call(model="gpt-4o", tokens_prompt=10, tokens_completion=5)
        adapter.on_tool_use(tool_name="search", tool_output="result")
        adapter.on_agent_start(agent_name="agent", input_data="q")
        adapter.on_agent_end(agent_name="agent", output="a")

        assert len(collector.get_events("model.invoke")) == 1
        assert len(collector.get_events("tool.call")) == 1
        assert len(collector.get_events("agent.input")) == 1
        assert len(collector.get_events("agent.output")) == 1


# ---------------------------------------------------------------------------
# Adapter lifecycle
# ---------------------------------------------------------------------------


class TestAdapterLifecycle:
    """Verify full adapter lifecycle with real SDK."""

    def test_connect_disconnect(self) -> None:
        """connect() and disconnect() should not raise."""
        collector = EventCollector()
        adapter = LlamaIndexAdapter(stratix=collector)
        adapter.connect()
        assert adapter._status == AdapterStatus.HEALTHY
        adapter.disconnect()
        assert adapter._status == AdapterStatus.DISCONNECTED

    def test_adapter_has_framework_metadata(self) -> None:
        """Adapter should expose its framework name and version."""
        collector = EventCollector()
        adapter = LlamaIndexAdapter(stratix=collector)
        assert adapter.FRAMEWORK == "llama_index"
        assert adapter.VERSION is not None

    def test_replay_trace_accumulates_events(self) -> None:
        """Events should accumulate for replay serialization."""
        collector = EventCollector()
        adapter = LlamaIndexAdapter(stratix=collector)
        adapter.connect()

        adapter.on_agent_start(agent_name="a", input_data="q")
        adapter.on_agent_end(agent_name="a", output="a")

        trace = adapter.serialize_for_replay()
        assert len(trace.events) == 2
        assert trace.adapter_name == "LlamaIndexAdapter"
        assert trace.framework == "llama_index"

    def test_agent_config_emitted_once_per_name(self) -> None:
        """Environment config should only be emitted once per agent name."""
        collector = EventCollector()
        adapter = LlamaIndexAdapter(stratix=collector)
        adapter.connect()

        class FakeEvent:
            agent_id: str = "my_agent"
            tools: Any = None

        adapter._emit_agent_config("my_agent", FakeEvent())
        adapter._emit_agent_config("my_agent", FakeEvent())

        config_events = collector.get_events("environment.config")
        assert len(config_events) == 1
        assert config_events[0]["payload"]["agent_name"] == "my_agent"
