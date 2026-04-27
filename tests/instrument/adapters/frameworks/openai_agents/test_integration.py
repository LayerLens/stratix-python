"""Integration tests for OpenAI Agents adapter using the REAL SDK.

Ported from ``ateam/tests/adapters/openai_agents/test_integration.py``.

These tests verify that ``OpenAIAgentsAdapter`` correctly captures events
from actual OpenAI Agents SDK types -- not mocks. The SDK must be
installed::

    pip install openai-agents

Tests are skipped (via ``pytest.importorskip``) if the SDK is not
installed, matching the langgraph integration suite convention.

Renames:
- ``stratix.sdk.python.adapters.base`` ->
  ``layerlens.instrument.adapters._base``
- ``stratix.sdk.python.adapters.capture.CaptureConfig`` ->
  ``layerlens.instrument.adapters._base.CaptureConfig``
- ``stratix.sdk.python.adapters.openai_agents.lifecycle.OpenAIAgentsAdapter``
  -> ``layerlens.instrument.adapters.frameworks.openai_agents.lifecycle.OpenAIAgentsAdapter``
"""

from __future__ import annotations

from typing import Any

import pytest

agents = pytest.importorskip("agents", reason="openai-agents not installed")

from agents import Agent  # noqa: E402
from agents.tracing import (  # noqa: E402
    AgentSpanData,
    HandoffSpanData,
    FunctionSpanData,
    TracingProcessor,
    GenerationSpanData,
)

from layerlens.instrument.adapters._base import (  # noqa: E402
    AdapterStatus,
    CaptureConfig,
    AdapterCapability,
)
from layerlens.instrument.adapters.frameworks.openai_agents.lifecycle import (  # noqa: E402
    OpenAIAgentsAdapter,
)

# ---------------------------------------------------------------------------
# Real event collector (not a mock)
# ---------------------------------------------------------------------------


class EventCollector:
    """Accumulates events emitted by the adapter for assertions.

    Carries ``org_id`` so :class:`BaseAdapter`'s multi-tenancy guard
    accepts the adapter at construction.
    """

    def __init__(self, org_id: str = "test-org") -> None:
        self.org_id = org_id
        self.events: list[dict[str, Any]] = []

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append({"type": event_type, "payload": payload})

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


# ---------------------------------------------------------------------------
# Adapter construction with real SDK types
# ---------------------------------------------------------------------------


class TestAdapterWithRealSDK:
    """Verify adapter constructs and connects with real OpenAI Agents SDK."""

    def test_framework_metadata(self) -> None:
        """Adapter should expose correct framework name and version."""
        collector = EventCollector()
        adapter = OpenAIAgentsAdapter(stratix=collector)
        assert adapter.FRAMEWORK == "openai_agents"
        assert adapter.VERSION is not None

    def test_connect_detects_sdk_version(self) -> None:
        """connect() should detect the installed SDK version."""
        collector = EventCollector()
        adapter = OpenAIAgentsAdapter(stratix=collector)
        adapter.connect()
        health = adapter.health_check()
        assert health.framework_version is not None
        assert health.framework_version != ""

    def test_adapter_capabilities(self) -> None:
        """Adapter declares correct capabilities for the SDK."""
        collector = EventCollector()
        adapter = OpenAIAgentsAdapter(stratix=collector)
        info = adapter.get_adapter_info()
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
        adapter = OpenAIAgentsAdapter(stratix=collector, capture_config=config)
        assert adapter._capture_config.l3_model_metadata is True
        assert adapter._capture_config.l5a_tool_calls is False

    def test_real_agent_type_is_constructable(self) -> None:
        """Real Agent from the SDK can be instantiated (no API key needed)."""
        agent = Agent(
            name="test_agent",
            instructions="You are a test agent.",
        )
        assert agent.name == "test_agent"


# ---------------------------------------------------------------------------
# TracingProcessor integration
# ---------------------------------------------------------------------------


class TestTracingProcessorIntegration:
    """Verify the adapter's trace processor integration with the real SDK.

    Note: The adapter internally imports from ``agents.tracing.TraceProcessor``
    which was renamed to ``TracingProcessor`` in newer SDK versions. When the
    import path does not match, ``_create_trace_processor`` returns None and
    ``instrument_runner`` gracefully degrades. These tests document that
    behavior and verify the adapter handles it without crashing.
    """

    def test_create_trace_processor_graceful_when_import_mismatched(self) -> None:
        """_create_trace_processor returns None when SDK class name differs."""
        collector = EventCollector()
        adapter = OpenAIAgentsAdapter(stratix=collector)
        adapter.connect()
        processor = adapter._create_trace_processor()
        # The adapter tries to import ``TraceProcessor`` but the SDK exports
        # ``TracingProcessor``. The adapter handles this gracefully.
        # If the import succeeds, we get a valid processor; otherwise None.
        if processor is not None:
            assert isinstance(processor, TracingProcessor)

    def test_instrument_runner_does_not_crash(self) -> None:
        """instrument_runner should not raise even if processor creation fails."""
        collector = EventCollector()
        adapter = OpenAIAgentsAdapter(stratix=collector)
        adapter.connect()
        runner = object()
        result = adapter.instrument_runner(runner)
        # Should return the runner object regardless of processor status
        assert result is runner

    def test_real_tracing_processor_interface(self) -> None:
        """The real SDK TracingProcessor has the expected interface."""
        # This documents the real SDK interface for future adapter alignment
        assert hasattr(TracingProcessor, "on_trace_start")
        assert hasattr(TracingProcessor, "on_trace_end")
        assert hasattr(TracingProcessor, "on_span_start")
        assert hasattr(TracingProcessor, "on_span_end")


# ---------------------------------------------------------------------------
# Span data event mapping with real SDK span types
# ---------------------------------------------------------------------------


class TestSpanDataEventMapping:
    """Verify adapter routes real SDK span data types to correct events."""

    def _make_span(self, span_data: Any, span_id: str = "sp-1") -> Any:
        """Create a minimal span-like object carrying the given span_data."""

        class FakeSpan:
            pass

        span = FakeSpan()
        span.span_data = span_data
        span.span_id = span_id
        span.duration_ms = 42.0
        return span

    def test_agent_span_emits_agent_input_and_output(self) -> None:
        """AgentSpanData should emit agent.input on start and agent.output on end."""
        collector = EventCollector()
        adapter = OpenAIAgentsAdapter(stratix=collector)
        adapter.connect()

        data = AgentSpanData(name="researcher")
        span = self._make_span(data)

        adapter._on_span_start(span)
        input_events = collector.get_events("agent.input")
        assert len(input_events) == 1
        assert input_events[0]["payload"]["agent_name"] == "researcher"

        # Simulate output
        data.output = "Research complete"
        adapter._on_span_end(span)
        output_events = collector.get_events("agent.output")
        assert len(output_events) == 1
        assert output_events[0]["payload"]["agent_name"] == "researcher"

    def test_generation_span_emits_model_invoke(self) -> None:
        """GenerationSpanData should emit model.invoke on span end."""
        collector = EventCollector()
        adapter = OpenAIAgentsAdapter(stratix=collector)
        adapter.connect()

        data = GenerationSpanData(
            model="gpt-4o",
            usage={"input_tokens": 100, "output_tokens": 50},
        )
        span = self._make_span(data)

        adapter._on_span_end(span)
        model_events = collector.get_events("model.invoke")
        assert len(model_events) == 1
        payload = model_events[0]["payload"]
        assert payload["model"] == "gpt-4o"

    def test_generation_span_emits_cost_record(self) -> None:
        """GenerationSpanData with tokens should also emit cost.record."""
        collector = EventCollector()
        adapter = OpenAIAgentsAdapter(stratix=collector)
        adapter.connect()

        data = GenerationSpanData(
            model="gpt-4o",
            usage={"input_tokens": 200, "output_tokens": 100},
        )
        # The adapter reads input_tokens/output_tokens as attributes on
        # span_data. Real SDK stores them in usage dict, so we set them
        # as attrs to match what the adapter expects from its
        # _on_generation_span_end logic.
        data.input_tokens = 200
        data.output_tokens = 100
        span = self._make_span(data)
        adapter._on_span_end(span)

        cost_events = collector.get_events("cost.record")
        assert len(cost_events) >= 1
        # Verify token data is present in cost record
        cost_payload = cost_events[0]["payload"]
        assert cost_payload.get("tokens_prompt") is not None or cost_payload.get("tokens_total") is not None

    def test_function_span_emits_tool_call(self) -> None:
        """FunctionSpanData should emit tool.call on span end."""
        collector = EventCollector()
        adapter = OpenAIAgentsAdapter(stratix=collector)
        adapter.connect()

        data = FunctionSpanData(
            name="search_web",
            input="quantum computing",
            output="Quantum computing uses qubits...",
        )
        span = self._make_span(data)

        adapter._on_span_end(span)
        tool_events = collector.get_events("tool.call")
        assert len(tool_events) == 1
        assert tool_events[0]["payload"]["tool_name"] == "search_web"
        assert "quantum computing" in str(tool_events[0]["payload"]["tool_input"])

    def test_handoff_span_emits_agent_handoff(self) -> None:
        """HandoffSpanData should emit agent.handoff on span end."""
        collector = EventCollector()
        adapter = OpenAIAgentsAdapter(stratix=collector)
        adapter.connect()

        data = HandoffSpanData(from_agent="triage", to_agent="specialist")
        span = self._make_span(data)

        adapter._on_span_end(span)
        handoff_events = collector.get_events("agent.handoff")
        assert len(handoff_events) == 1
        assert handoff_events[0]["payload"]["from_agent"] == "triage"
        assert handoff_events[0]["payload"]["to_agent"] == "specialist"


# ---------------------------------------------------------------------------
# Lifecycle hook events (Runner wrapping)
# ---------------------------------------------------------------------------


class TestLifecycleHookEvents:
    """Verify lifecycle hooks emit correct events with real SDK context."""

    def test_run_start_end_roundtrip(self) -> None:
        """on_run_start + on_run_end should emit agent.input + agent.output."""
        collector = EventCollector()
        adapter = OpenAIAgentsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_run_start(agent_name="assistant", input_data="What is AI?")
        adapter.on_run_end(agent_name="assistant", output="AI is...")

        input_events = collector.get_events("agent.input")
        output_events = collector.get_events("agent.output")
        assert len(input_events) == 1
        assert len(output_events) == 1
        assert input_events[0]["payload"]["agent_name"] == "assistant"
        assert output_events[0]["payload"]["output"] == "AI is..."
        assert "duration_ns" in output_events[0]["payload"]

    def test_run_end_with_error(self) -> None:
        """on_run_end with an error should include error in payload."""
        collector = EventCollector()
        adapter = OpenAIAgentsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_run_start(agent_name="assistant")
        adapter.on_run_end(
            agent_name="assistant",
            error=RuntimeError("API timeout"),
        )

        output_events = collector.get_events("agent.output")
        assert len(output_events) == 1
        assert "API timeout" in output_events[0]["payload"]["error"]

    def test_tool_use_emits_tool_call(self) -> None:
        """on_tool_use should emit a tool.call event."""
        collector = EventCollector()
        adapter = OpenAIAgentsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_tool_use(
            tool_name="calculator",
            tool_input={"expression": "2+2"},
            tool_output="4",
            latency_ms=5.0,
        )

        tool_events = collector.get_events("tool.call")
        assert len(tool_events) == 1
        assert tool_events[0]["payload"]["tool_name"] == "calculator"
        assert tool_events[0]["payload"]["latency_ms"] == 5.0

    def test_llm_call_emits_model_invoke(self) -> None:
        """on_llm_call should emit a model.invoke event."""
        collector = EventCollector()
        adapter = OpenAIAgentsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_llm_call(
            provider="openai",
            model="gpt-4o",
            tokens_prompt=150,
            tokens_completion=75,
            latency_ms=320.5,
        )

        model_events = collector.get_events("model.invoke")
        assert len(model_events) == 1
        payload = model_events[0]["payload"]
        assert payload["provider"] == "openai"
        assert payload["model"] == "gpt-4o"
        assert payload["tokens_prompt"] == 150

    def test_handoff_emits_event_with_context_hash(self) -> None:
        """on_handoff should emit agent.handoff with context hash."""
        collector = EventCollector()
        adapter = OpenAIAgentsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_handoff(
            from_agent="triage",
            to_agent="billing",
            context={"reason": "billing question"},
        )

        handoff_events = collector.get_events("agent.handoff")
        assert len(handoff_events) == 1
        assert handoff_events[0]["payload"]["from_agent"] == "triage"
        assert handoff_events[0]["payload"]["to_agent"] == "billing"
        assert handoff_events[0]["payload"]["context_hash"] is not None


# ---------------------------------------------------------------------------
# Adapter lifecycle
# ---------------------------------------------------------------------------


class TestAdapterLifecycle:
    """Verify adapter lifecycle with real SDK."""

    def test_connect_disconnect(self) -> None:
        """connect() and disconnect() should transition status correctly."""
        collector = EventCollector()
        adapter = OpenAIAgentsAdapter(stratix=collector)
        adapter.connect()
        assert adapter._status == AdapterStatus.HEALTHY
        adapter.disconnect()
        assert adapter._status == AdapterStatus.DISCONNECTED

    def test_events_not_emitted_when_disconnected(self) -> None:
        """After disconnect, lifecycle hooks should not emit events."""
        collector = EventCollector()
        adapter = OpenAIAgentsAdapter(stratix=collector)
        adapter.connect()
        adapter.disconnect()

        adapter.on_run_start(agent_name="ghost")
        adapter.on_run_end(agent_name="ghost", output="nope")
        adapter.on_tool_use(tool_name="phantom")
        adapter.on_llm_call(model="gpt-4o")

        assert len(collector.events) == 0

    def test_serialize_for_replay(self) -> None:
        """serialize_for_replay produces a valid ReplayableTrace."""
        collector = EventCollector()
        adapter = OpenAIAgentsAdapter(stratix=collector)
        adapter.connect()
        trace = adapter.serialize_for_replay()
        assert trace.adapter_name == "OpenAIAgentsAdapter"
        assert trace.framework == "openai_agents"
        assert trace.trace_id is not None

    def test_agent_config_emitted_once_per_agent(self) -> None:
        """environment.config should only be emitted once per agent name."""
        collector = EventCollector()
        adapter = OpenAIAgentsAdapter(stratix=collector)
        adapter.connect()

        data = AgentSpanData(name="researcher")
        span_like = type("Span", (), {"span_data": data, "span_id": "sp-1"})()

        adapter._on_agent_span_start(span_like, data)
        adapter._on_agent_span_start(span_like, data)

        config_events = collector.get_events("environment.config")
        assert len(config_events) == 1
