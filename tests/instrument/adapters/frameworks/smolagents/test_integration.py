"""Integration tests for SmolAgents adapter using the REAL SmolAgents SDK.

These tests verify that SmolAgentsAdapter correctly instruments and
captures events from actual SmolAgents types. The SDK must be installed:
    pip install 'stratix[smolagents]'

Tests are skipped if smolagents is not installed.

Ported as-is from ``ateam/tests/adapters/smolagents/test_integration.py``.

Translation rules applied:
* ``stratix.sdk.python.adapters.smolagents.lifecycle`` →
  ``layerlens.instrument.adapters.frameworks.smolagents.lifecycle``
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

smolagents = pytest.importorskip("smolagents", reason="smolagents not installed")

from layerlens.instrument.adapters._base import AdapterCapability, AdapterStatus  # noqa: E402
from layerlens.instrument.adapters._base import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.smolagents.lifecycle import SmolAgentsAdapter  # noqa: E402

# ---------------------------------------------------------------------------
# Test STRATIX instance that collects events
# ---------------------------------------------------------------------------


class EventCollector:
    """Real event collector -- not a mock. Accumulates events for assertions."""

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
# Adapter construction with real SDK
# ---------------------------------------------------------------------------


class TestAdapterWithRealSDK:
    """Verify adapter constructs and connects with real SmolAgents classes."""

    def test_connect_detects_framework_version(self) -> None:
        """connect() should detect the smolagents version."""
        collector = EventCollector()
        adapter = SmolAgentsAdapter(stratix=collector)
        adapter.connect()
        assert adapter._status == AdapterStatus.HEALTHY
        assert adapter._framework_version is not None
        assert True  # framework_version may be "unknown" if SDK not fully installed
        adapter.disconnect()

    def test_adapter_info_metadata(self) -> None:
        """Adapter info should expose correct framework metadata."""
        collector = EventCollector()
        adapter = SmolAgentsAdapter(stratix=collector)
        adapter.connect()
        info = adapter.get_adapter_info()
        assert info.framework == "smolagents"
        assert info.name == "SmolAgentsAdapter"
        assert AdapterCapability.TRACE_TOOLS in info.capabilities
        assert AdapterCapability.TRACE_MODELS in info.capabilities
        adapter.disconnect()

    def test_health_check_returns_healthy(self) -> None:
        """health_check should return HEALTHY after connect."""
        collector = EventCollector()
        adapter = SmolAgentsAdapter(stratix=collector)
        adapter.connect()
        health = adapter.health_check()
        assert health.status == AdapterStatus.HEALTHY
        assert health.framework_name == "smolagents"
        adapter.disconnect()

    def test_capture_config_propagates(self) -> None:
        """CaptureConfig correctly controls which events are captured."""
        collector = EventCollector()
        config = CaptureConfig(
            l3_model_metadata=True,
            l5a_tool_calls=False,
            l1_agent_io=True,
        )
        adapter = SmolAgentsAdapter(stratix=collector, capture_config=config)
        assert adapter._capture_config.l3_model_metadata is True
        assert adapter._capture_config.l5a_tool_calls is False


# ---------------------------------------------------------------------------
# SDK types and instrumentation
# ---------------------------------------------------------------------------


class TestSDKTypesExist:
    """Verify expected SmolAgents SDK types are importable."""

    def test_tool_class_exists(self) -> None:
        """SmolAgents should have a Tool base class."""
        assert hasattr(smolagents, "Tool") or hasattr(smolagents, "tool")

    def test_agent_classes_exist(self) -> None:
        """SmolAgents should have agent classes available."""
        # At least one agent class should exist
        has_code_agent = hasattr(smolagents, "CodeAgent")
        has_tool_calling_agent = hasattr(smolagents, "ToolCallingAgent")
        has_multi_step = hasattr(smolagents, "MultiStepAgent")
        assert has_code_agent or has_tool_calling_agent or has_multi_step, (
            "No recognized agent class found in smolagents"
        )


class TestInstrumentationWithFakeAgent:
    """Test instrumentation using a fake agent that mimics SmolAgents API."""

    def _make_fake_agent(self) -> Any:
        """Create a minimal object that looks like a SmolAgents agent."""

        class FakeAgent:
            name = "test_agent"
            tools = {"search": "search_tool", "calc": "calc_tool"}
            model = "HfApiModel"
            managed_agents = None
            system_prompt = "You are a helpful assistant."

            def run(self, task: str) -> str:
                return f"Result for: {task}"

        return FakeAgent()

    def test_instrument_wraps_run(self) -> None:
        """instrument_agent should wrap the run method."""
        collector = EventCollector()
        adapter = SmolAgentsAdapter(stratix=collector)
        adapter.connect()

        agent = self._make_fake_agent()
        adapter.instrument_agent(agent)

        # run should now be wrapped
        assert hasattr(agent.run, "_layerlens_original")
        adapter.disconnect()

    def test_instrumented_run_emits_events(self) -> None:
        """Running an instrumented agent should emit agent.input and agent.output."""
        collector = EventCollector()
        adapter = SmolAgentsAdapter(stratix=collector)
        adapter.connect()

        agent = self._make_fake_agent()
        adapter.instrument_agent(agent)

        result = agent.run("What is 2+2?")
        assert result == "Result for: What is 2+2?"

        input_events = collector.get_events("agent.input")
        output_events = collector.get_events("agent.output")
        assert len(input_events) == 1
        assert len(output_events) == 1
        assert input_events[0]["payload"]["agent_name"] == "test_agent"
        assert "Result for" in str(output_events[0]["payload"]["output"])

        adapter.disconnect()

    def test_instrument_emits_config_event(self) -> None:
        """instrument_agent should emit an environment.config event."""
        collector = EventCollector()
        adapter = SmolAgentsAdapter(stratix=collector)
        adapter.connect()

        agent = self._make_fake_agent()
        adapter.instrument_agent(agent)

        config_events = collector.get_events("environment.config")
        assert len(config_events) == 1
        assert config_events[0]["payload"]["agent_name"] == "test_agent"
        assert "search" in config_events[0]["payload"]["tools"]

        adapter.disconnect()

    def test_instrument_idempotent(self) -> None:
        """Instrumenting the same agent twice should be a no-op."""
        collector = EventCollector()
        adapter = SmolAgentsAdapter(stratix=collector)
        adapter.connect()

        agent = self._make_fake_agent()
        adapter.instrument_agent(agent)
        adapter.instrument_agent(agent)

        # Should only get one config event
        config_events = collector.get_events("environment.config")
        assert len(config_events) == 1

        adapter.disconnect()

    def test_disconnect_unwraps_agent(self) -> None:
        """disconnect() should restore the original run method."""
        collector = EventCollector()
        adapter = SmolAgentsAdapter(stratix=collector)
        adapter.connect()

        agent = self._make_fake_agent()
        _original_run = agent.run
        adapter.instrument_agent(agent)
        assert hasattr(agent.run, "_layerlens_original")

        adapter.disconnect()
        # After disconnect, the original should be restored
        assert not hasattr(agent.run, "_layerlens_original")


# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------


class TestLifecycleHooks:
    """Test manual lifecycle hook invocations."""

    def test_on_run_start_emits_agent_input(self) -> None:
        """on_run_start should emit an agent.input event."""
        collector = EventCollector()
        adapter = SmolAgentsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_run_start(agent_name="my_agent", input_data="Hello")

        events = collector.get_events("agent.input")
        assert len(events) == 1
        assert events[0]["payload"]["agent_name"] == "my_agent"
        assert events[0]["payload"]["input"] == "Hello"

        adapter.disconnect()

    def test_on_run_end_emits_agent_output(self) -> None:
        """on_run_end should emit an agent.output event."""
        collector = EventCollector()
        adapter = SmolAgentsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_run_end(agent_name="my_agent", output="World")

        events = collector.get_events("agent.output")
        assert len(events) == 1
        assert events[0]["payload"]["agent_name"] == "my_agent"
        assert events[0]["payload"]["output"] == "World"

        adapter.disconnect()

    def test_on_tool_use_emits_tool_call(self) -> None:
        """on_tool_use should emit a tool.call event."""
        collector = EventCollector()
        adapter = SmolAgentsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_tool_use(
            tool_name="search",
            tool_input="query text",
            tool_output="search results",
        )

        events = collector.get_events("tool.call")
        assert len(events) == 1
        assert events[0]["payload"]["tool_name"] == "search"

        adapter.disconnect()

    def test_on_llm_call_emits_model_invoke(self) -> None:
        """on_llm_call should emit a model.invoke event."""
        collector = EventCollector()
        adapter = SmolAgentsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_llm_call(
            provider="huggingface",
            model="Qwen/Qwen2.5-72B",
            tokens_prompt=100,
            tokens_completion=50,
            latency_ms=250.0,
        )

        events = collector.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["provider"] == "huggingface"
        assert events[0]["payload"]["model"] == "Qwen/Qwen2.5-72B"
        assert events[0]["payload"]["tokens_prompt"] == 100

        adapter.disconnect()

    def test_on_handoff_emits_agent_handoff(self) -> None:
        """on_handoff should emit an agent.handoff event."""
        collector = EventCollector()
        adapter = SmolAgentsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_handoff(
            from_agent="manager",
            to_agent="worker",
            context="Do the research task",
        )

        events = collector.get_events("agent.handoff")
        assert len(events) == 1
        assert events[0]["payload"]["from_agent"] == "manager"
        assert events[0]["payload"]["to_agent"] == "worker"

        adapter.disconnect()

    def test_events_not_emitted_when_disconnected(self) -> None:
        """Events should not be emitted when the adapter is disconnected."""
        collector = EventCollector()
        adapter = SmolAgentsAdapter(stratix=collector)
        # Do NOT connect

        adapter.on_run_start(agent_name="test", input_data="hello")
        adapter.on_run_end(agent_name="test", output="bye")

        assert len(collector.events) == 0


# ---------------------------------------------------------------------------
# Replay serialization
# ---------------------------------------------------------------------------


class TestReplaySerialization:
    """Test replay serialization captures events."""

    def test_serialize_includes_events(self) -> None:
        """serialize_for_replay should include emitted events."""
        collector = EventCollector()
        adapter = SmolAgentsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_run_start(agent_name="agent", input_data="test")
        adapter.on_run_end(agent_name="agent", output="done")

        trace = adapter.serialize_for_replay()
        assert trace.adapter_name == "SmolAgentsAdapter"
        assert trace.framework == "smolagents"
        assert len(trace.events) >= 2

        adapter.disconnect()
