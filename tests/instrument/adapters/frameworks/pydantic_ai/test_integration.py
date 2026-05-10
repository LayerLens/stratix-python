"""Integration tests for PydanticAI adapter using the REAL SDK.

These tests verify that PydanticAIAdapter correctly captures events
from actual PydanticAI types -- not mocks. The SDK must be installed:
    pip install pydantic-ai

Tests are skipped if pydantic-ai is not installed.

Ported as-is from ``ateam/tests/adapters/pydantic_ai/test_integration.py``.

Translation rules applied:
* ``stratix.sdk.python.adapters.pydantic_ai.lifecycle`` →
  ``layerlens.instrument.adapters.frameworks.pydantic_ai.lifecycle``
* ``stratix.sdk.python.adapters.base`` →
  ``layerlens.instrument.adapters._base``
* ``stratix.sdk.python.adapters.capture.CaptureConfig`` →
  ``layerlens.instrument.adapters._base.CaptureConfig``
* ``stratix.sdk.python.adapters.replay_models.ReplayableTrace`` →
  ``layerlens.instrument.adapters._base.ReplayableTrace``
* ``stratix.sdk.python.adapters.registry._ADAPTER_MODULES`` →
  ``layerlens.instrument.adapters._base.registry._ADAPTER_MODULES``
* The wrapper marker attribute renamed by the source from
  ``_stratix_original`` to ``_layerlens_original``.

Multi-tenancy: per the transitional "stratix attribute" pattern (see
migration doc §2.3 step 2 — keystone PR #118 still DRAFT), the
``MockStratix`` / ``EventCollector`` test stub gets an ``org_id``
attribute. The post-merge sweep PR will rebase to canonical kwarg once
#118 lands.
"""

from __future__ import annotations

from typing import Any

import pytest

pydantic_ai = pytest.importorskip("pydantic_ai", reason="pydantic-ai not installed")

from pydantic_ai import Agent  # noqa: E402

from layerlens.instrument.adapters._base import AdapterCapability, AdapterStatus  # noqa: E402
from layerlens.instrument.adapters._base import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.pydantic_ai.lifecycle import PydanticAIAdapter  # noqa: E402

# ---------------------------------------------------------------------------
# Real event collector (not a mock)
# ---------------------------------------------------------------------------


class EventCollector:
    """Accumulates events emitted by the adapter for assertions."""

    def __init__(self) -> None:
        self.org_id: str = "test-org"
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
    """Verify adapter constructs and connects with real PydanticAI classes."""

    def test_framework_metadata(self) -> None:
        """Adapter should expose correct framework name and version."""
        collector = EventCollector()
        adapter = PydanticAIAdapter(stratix=collector)
        assert adapter.FRAMEWORK == "pydantic_ai"
        assert adapter.VERSION is not None

    def test_connect_detects_sdk_version(self) -> None:
        """connect() should detect the installed PydanticAI version."""
        collector = EventCollector()
        adapter = PydanticAIAdapter(stratix=collector)
        adapter.connect()
        health = adapter.health_check()
        assert health.framework_version is not None
        assert health.framework_version != ""

    def test_adapter_capabilities(self) -> None:
        """Adapter declares correct capabilities for the SDK."""
        collector = EventCollector()
        adapter = PydanticAIAdapter(stratix=collector)
        info = adapter.get_adapter_info()
        assert AdapterCapability.TRACE_TOOLS in info.capabilities
        assert AdapterCapability.TRACE_MODELS in info.capabilities
        assert AdapterCapability.TRACE_STATE in info.capabilities

    def test_capture_config_propagates(self) -> None:
        """CaptureConfig correctly controls which events are captured."""
        collector = EventCollector()
        config = CaptureConfig(
            l3_model_metadata=True,
            l5a_tool_calls=False,
            l1_agent_io=True,
        )
        adapter = PydanticAIAdapter(stratix=collector, capture_config=config)
        assert adapter._capture_config.l3_model_metadata is True
        assert adapter._capture_config.l5a_tool_calls is False

    def test_real_agent_type_is_constructable(self) -> None:
        """Real Agent from PydanticAI can be instantiated (no API key needed)."""
        agent = Agent(
            "test",
            system_prompt="You are a helpful assistant.",
        )
        assert agent is not None


# ---------------------------------------------------------------------------
# Agent wrapping integration
# ---------------------------------------------------------------------------


class TestAgentWrapping:
    """Verify the adapter correctly wraps real PydanticAI Agent methods."""

    def test_instrument_agent_wraps_run_methods(self) -> None:
        """instrument_agent should wrap run and run_sync on a real Agent."""
        collector = EventCollector()
        adapter = PydanticAIAdapter(stratix=collector)
        adapter.connect()

        agent = Agent("test", system_prompt="test agent")
        adapter.instrument_agent(agent)

        # run should be wrapped (has _layerlens_original)
        assert hasattr(agent.run, "_layerlens_original")
        # run_sync should be wrapped if it exists
        if hasattr(agent, "run_sync"):
            assert hasattr(agent.run_sync, "_layerlens_original")

    def test_instrument_agent_emits_config(self) -> None:
        """instrument_agent should emit environment.config for the agent."""
        collector = EventCollector()
        adapter = PydanticAIAdapter(stratix=collector)
        adapter.connect()

        agent = Agent("test", system_prompt="You help with tests.")
        adapter.instrument_agent(agent)

        config_events = collector.get_events("environment.config")
        assert len(config_events) == 1
        assert config_events[0]["payload"]["framework"] == "pydantic_ai"

    def test_instrument_agent_idempotent(self) -> None:
        """Wrapping the same agent twice should not double-wrap."""
        collector = EventCollector()
        adapter = PydanticAIAdapter(stratix=collector)
        adapter.connect()

        agent = Agent("test", system_prompt="test")
        adapter.instrument_agent(agent)
        first_run = agent.run
        adapter.instrument_agent(agent)
        # Should be the same wrapped function, not double-wrapped
        assert agent.run is first_run

    def test_disconnect_unwraps_agent(self) -> None:
        """disconnect() should restore original methods on wrapped agents."""
        collector = EventCollector()
        adapter = PydanticAIAdapter(stratix=collector)
        adapter.connect()

        agent = Agent("test", system_prompt="test")
        adapter.instrument_agent(agent)
        # After wrapping, run should have the _layerlens_original marker
        assert hasattr(agent.run, "_layerlens_original")
        adapter.disconnect()
        # After disconnect, the _layerlens_original marker should be gone
        # (original method restored)
        assert not hasattr(agent.run, "_layerlens_original")

    def test_config_emitted_once_per_agent_name(self) -> None:
        """environment.config should only be emitted once per unique agent name."""
        collector = EventCollector()
        adapter = PydanticAIAdapter(stratix=collector)
        adapter.connect()

        agent1 = Agent("test", system_prompt="agent one")
        agent2 = Agent("test", system_prompt="agent two")
        adapter.instrument_agent(agent1)
        adapter.instrument_agent(agent2)

        config_events = collector.get_events("environment.config")
        # Same name "Agent" (class name) -- only one config should be emitted
        # Note: PydanticAI Agent does not always expose .name;
        # the adapter falls back to the class name
        assert len(config_events) >= 1


# ---------------------------------------------------------------------------
# Lifecycle hook events
# ---------------------------------------------------------------------------


class TestLifecycleHookEvents:
    """Verify lifecycle hooks emit correct events with real SDK context."""

    def test_run_start_end_roundtrip(self) -> None:
        """on_run_start + on_run_end should emit agent.input + agent.output + state.change."""
        collector = EventCollector()
        adapter = PydanticAIAdapter(stratix=collector)
        adapter.connect()

        adapter.on_run_start(agent_name="assistant", input_data="What is AI?")
        adapter.on_run_end(agent_name="assistant", output="AI is...")

        input_events = collector.get_events("agent.input")
        output_events = collector.get_events("agent.output")
        state_events = collector.get_events("agent.state.change")

        assert len(input_events) == 1
        assert len(output_events) == 1
        assert len(state_events) == 1

        assert input_events[0]["payload"]["agent_name"] == "assistant"
        assert output_events[0]["payload"]["output"] == "AI is..."
        assert "duration_ns" in output_events[0]["payload"]
        assert state_events[0]["payload"]["event_subtype"] == "run_complete"

    def test_run_end_with_error_emits_failed_state(self) -> None:
        """on_run_end with error should emit run_failed state change."""
        collector = EventCollector()
        adapter = PydanticAIAdapter(stratix=collector)
        adapter.connect()

        adapter.on_run_start(agent_name="assistant")
        adapter.on_run_end(
            agent_name="assistant",
            error=RuntimeError("Model unavailable"),
        )

        output_events = collector.get_events("agent.output")
        state_events = collector.get_events("agent.state.change")

        assert len(output_events) == 1
        assert "Model unavailable" in output_events[0]["payload"]["error"]
        assert state_events[0]["payload"]["event_subtype"] == "run_failed"

    def test_tool_use_emits_tool_call(self) -> None:
        """on_tool_use should emit a tool.call event."""
        collector = EventCollector()
        adapter = PydanticAIAdapter(stratix=collector)
        adapter.connect()

        adapter.on_tool_use(
            tool_name="get_weather",
            tool_input={"city": "Paris"},
            tool_output={"temp": 22, "unit": "C"},
            latency_ms=12.3,
        )

        tool_events = collector.get_events("tool.call")
        assert len(tool_events) == 1
        assert tool_events[0]["payload"]["tool_name"] == "get_weather"
        assert tool_events[0]["payload"]["latency_ms"] == 12.3

    def test_llm_call_emits_model_invoke(self) -> None:
        """on_llm_call should emit a model.invoke event."""
        collector = EventCollector()
        adapter = PydanticAIAdapter(stratix=collector)
        adapter.connect()

        adapter.on_llm_call(
            provider="anthropic",
            model="claude-opus-4-6",
            tokens_prompt=200,
            tokens_completion=100,
            latency_ms=450.0,
        )

        model_events = collector.get_events("model.invoke")
        assert len(model_events) == 1
        payload = model_events[0]["payload"]
        assert payload["provider"] == "anthropic"
        assert payload["model"] == "claude-opus-4-6"
        assert payload["tokens_prompt"] == 200

    def test_handoff_emits_event_with_context_hash(self) -> None:
        """on_handoff should emit agent.handoff with context hash."""
        collector = EventCollector()
        adapter = PydanticAIAdapter(stratix=collector)
        adapter.connect()

        adapter.on_handoff(
            from_agent="router",
            to_agent="specialist",
            context={"topic": "billing"},
        )

        handoff_events = collector.get_events("agent.handoff")
        assert len(handoff_events) == 1
        assert handoff_events[0]["payload"]["from_agent"] == "router"
        assert handoff_events[0]["payload"]["to_agent"] == "specialist"
        assert handoff_events[0]["payload"]["context_hash"] is not None


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------


class TestProviderDetection:
    """Verify _detect_provider with real model name strings."""

    def test_openai_models(self) -> None:
        adapter = PydanticAIAdapter()
        assert adapter._detect_provider("gpt-4o") == "openai"
        assert adapter._detect_provider("o1-preview") == "openai"
        assert adapter._detect_provider("o3-mini") == "openai"

    def test_anthropic_models(self) -> None:
        adapter = PydanticAIAdapter()
        assert adapter._detect_provider("claude-opus-4-6") == "anthropic"
        assert adapter._detect_provider("claude-3-haiku") == "anthropic"

    def test_google_models(self) -> None:
        adapter = PydanticAIAdapter()
        assert adapter._detect_provider("gemini-2.0-flash") == "google"

    def test_unknown_model(self) -> None:
        adapter = PydanticAIAdapter()
        assert adapter._detect_provider("custom-model-v1") is None

    def test_none_model(self) -> None:
        adapter = PydanticAIAdapter()
        assert adapter._detect_provider(None) is None


# ---------------------------------------------------------------------------
# Adapter lifecycle
# ---------------------------------------------------------------------------


class TestAdapterLifecycle:
    """Verify adapter lifecycle with real SDK."""

    def test_connect_disconnect(self) -> None:
        """connect() and disconnect() should transition status correctly."""
        collector = EventCollector()
        adapter = PydanticAIAdapter(stratix=collector)
        adapter.connect()
        assert adapter._status == AdapterStatus.HEALTHY
        adapter.disconnect()
        assert adapter._status == AdapterStatus.DISCONNECTED

    def test_events_not_emitted_when_disconnected(self) -> None:
        """After disconnect, lifecycle hooks should not emit events."""
        collector = EventCollector()
        adapter = PydanticAIAdapter(stratix=collector)
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
        adapter = PydanticAIAdapter(stratix=collector)
        adapter.connect()
        trace = adapter.serialize_for_replay()
        assert trace.adapter_name == "PydanticAIAdapter"
        assert trace.framework == "pydantic_ai"
        assert trace.trace_id is not None


# ---------------------------------------------------------------------------
# Capture config gating
# ---------------------------------------------------------------------------


class TestCaptureConfigGating:
    """Verify capture config gates events correctly with real SDK."""

    def test_minimal_config_blocks_model_and_tool_events(self) -> None:
        """CaptureConfig.minimal() should block L3 and L5a events."""
        collector = EventCollector()
        config = CaptureConfig.minimal()
        adapter = PydanticAIAdapter(stratix=collector, capture_config=config)
        adapter.connect()

        adapter.on_llm_call(model="gpt-4o")
        adapter.on_tool_use(tool_name="test")

        assert len(collector.get_events("model.invoke")) == 0
        assert len(collector.get_events("tool.call")) == 0

    def test_cross_cutting_always_emitted_under_minimal(self) -> None:
        """Cross-cutting events should always be emitted even under minimal config."""
        collector = EventCollector()
        config = CaptureConfig.minimal()
        adapter = PydanticAIAdapter(stratix=collector, capture_config=config)
        adapter.connect()

        adapter.emit_dict_event(
            "agent.state.change",
            {"framework": "pydantic_ai", "event_subtype": "test"},
        )

        assert len(collector.get_events("agent.state.change")) == 1

    def test_content_capture_controls_messages(self) -> None:
        """When capture_content=False, messages should not be included in model.invoke."""
        collector = EventCollector()
        config = CaptureConfig(capture_content=False)
        adapter = PydanticAIAdapter(stratix=collector, capture_config=config)
        adapter.connect()

        adapter.on_llm_call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "secret"}],
        )

        model_events = collector.get_events("model.invoke")
        assert len(model_events) == 1
        assert "messages" not in model_events[0]["payload"]
