"""Integration tests for AWS Strands adapter using the REAL Strands SDK.

These tests verify that StrandsAdapter correctly instruments and captures
events from actual Strands agents. The SDK must be installed:
    pip install 'stratix[strands]'

Tests are skipped if strands-agents is not installed.

Ported as-is from ``ateam/tests/adapters/strands/test_integration.py``.

Translation rules applied:
* ``stratix.sdk.python.adapters.strands.lifecycle`` →
  ``layerlens.instrument.adapters.frameworks.strands.lifecycle``
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

strands = pytest.importorskip("strands", reason="strands-agents not installed")

from layerlens.instrument.adapters._base import AdapterCapability, AdapterStatus  # noqa: E402
from layerlens.instrument.adapters._base import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.strands.lifecycle import StrandsAdapter  # noqa: E402

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
    """Verify adapter constructs and connects with real Strands SDK."""

    def test_connect_detects_framework_version(self) -> None:
        """connect() should detect the strands version."""
        collector = EventCollector()
        adapter = StrandsAdapter(stratix=collector)
        adapter.connect()
        assert adapter._status == AdapterStatus.HEALTHY
        # Version may or may not be set depending on SDK
        adapter.disconnect()

    def test_adapter_info_metadata(self) -> None:
        """Adapter info should expose correct framework metadata."""
        collector = EventCollector()
        adapter = StrandsAdapter(stratix=collector)
        adapter.connect()
        info = adapter.get_adapter_info()
        assert info.framework == "strands"
        assert info.name == "StrandsAdapter"
        assert AdapterCapability.TRACE_TOOLS in info.capabilities
        assert AdapterCapability.TRACE_MODELS in info.capabilities
        assert AdapterCapability.TRACE_STATE in info.capabilities
        adapter.disconnect()

    def test_health_check_returns_healthy(self) -> None:
        """health_check should return HEALTHY after connect."""
        collector = EventCollector()
        adapter = StrandsAdapter(stratix=collector)
        adapter.connect()
        health = adapter.health_check()
        assert health.status == AdapterStatus.HEALTHY
        assert health.framework_name == "strands"
        adapter.disconnect()

    def test_capture_config_propagates(self) -> None:
        """CaptureConfig correctly controls which events are captured."""
        collector = EventCollector()
        config = CaptureConfig(
            l3_model_metadata=True,
            l5a_tool_calls=False,
        )
        adapter = StrandsAdapter(stratix=collector, capture_config=config)
        assert adapter._capture_config.l3_model_metadata is True
        assert adapter._capture_config.l5a_tool_calls is False


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------


class TestProviderDetection:
    """Verify _detect_provider works with real model IDs."""

    def setup_method(self) -> None:
        self.adapter = StrandsAdapter()

    def test_detect_bedrock_claude(self) -> None:
        assert self.adapter._detect_provider("anthropic.claude-v2") == "bedrock"

    def test_detect_bedrock_titan(self) -> None:
        assert self.adapter._detect_provider("amazon.titan-embed") == "bedrock"

    def test_detect_bedrock_llama(self) -> None:
        assert self.adapter._detect_provider("meta.llama3-70b") == "bedrock"

    def test_detect_openai(self) -> None:
        assert self.adapter._detect_provider("gpt-4o") == "openai"

    def test_detect_google(self) -> None:
        assert self.adapter._detect_provider("gemini-1.5-pro") == "google"

    def test_detect_default_bedrock(self) -> None:
        """Strands defaults to Bedrock for unrecognized models."""
        assert self.adapter._detect_provider("custom-model-v1") == "bedrock"

    def test_detect_none(self) -> None:
        assert self.adapter._detect_provider(None) is None


# ---------------------------------------------------------------------------
# Instrumentation with fake agent
# ---------------------------------------------------------------------------


class TestInstrumentationWithFakeAgent:
    """Test instrumentation using a fake agent that mimics Strands Agent API."""

    def _make_fake_agent(self) -> Any:
        """Create a minimal object that looks like a Strands Agent."""

        class FakeAgent:
            name = "strands_test_agent"
            model = "anthropic.claude-3-sonnet-20240229-v1:0"
            tools = {"web_search": "WebSearchTool"}
            system_prompt = "You are a helpful assistant."

            def __call__(self, prompt: str) -> str:
                return f"Response to: {prompt}"

            def invoke(self, message: str) -> str:
                return f"Invoked: {message}"

        return FakeAgent()

    def test_instrument_wraps_call(self) -> None:
        """instrument_agent should wrap __call__ and invoke."""
        collector = EventCollector()
        adapter = StrandsAdapter(stratix=collector)
        adapter.connect()

        agent = self._make_fake_agent()
        adapter.instrument_agent(agent)

        assert hasattr(agent.__call__, "_layerlens_original")
        assert hasattr(agent.invoke, "_layerlens_original")
        adapter.disconnect()

    def test_instrumented_call_emits_events(self) -> None:
        """Calling an instrumented agent should emit agent.input and agent.output."""
        collector = EventCollector()
        adapter = StrandsAdapter(stratix=collector)
        adapter.connect()

        agent = self._make_fake_agent()
        adapter.instrument_agent(agent)

        result = agent("What is AI?")
        assert result == "Response to: What is AI?"

        input_events = collector.get_events("agent.input")
        output_events = collector.get_events("agent.output")
        assert len(input_events) == 1
        assert len(output_events) >= 1
        assert input_events[0]["payload"]["agent_name"] == "strands_test_agent"

        adapter.disconnect()

    def test_instrument_emits_config_event(self) -> None:
        """instrument_agent should emit an environment.config event."""
        collector = EventCollector()
        adapter = StrandsAdapter(stratix=collector)
        adapter.connect()

        agent = self._make_fake_agent()
        adapter.instrument_agent(agent)

        config_events = collector.get_events("environment.config")
        assert len(config_events) == 1
        assert config_events[0]["payload"]["agent_name"] == "strands_test_agent"
        assert "web_search" in config_events[0]["payload"]["tools"]

        adapter.disconnect()

    def test_instrument_idempotent(self) -> None:
        """Instrumenting the same agent twice should be a no-op."""
        collector = EventCollector()
        adapter = StrandsAdapter(stratix=collector)
        adapter.connect()

        agent = self._make_fake_agent()
        adapter.instrument_agent(agent)
        adapter.instrument_agent(agent)

        config_events = collector.get_events("environment.config")
        assert len(config_events) == 1

        adapter.disconnect()

    def test_disconnect_unwraps_agent(self) -> None:
        """disconnect() should restore original methods."""
        collector = EventCollector()
        adapter = StrandsAdapter(stratix=collector)
        adapter.connect()

        agent = self._make_fake_agent()
        adapter.instrument_agent(agent)
        assert hasattr(agent.__call__, "_layerlens_original")

        adapter.disconnect()
        assert not hasattr(agent.__call__, "_layerlens_original")


# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------


class TestLifecycleHooks:
    """Test manual lifecycle hook invocations."""

    def test_on_run_start_emits_agent_input(self) -> None:
        collector = EventCollector()
        adapter = StrandsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_run_start(agent_name="my_agent", input_data="Hello")

        events = collector.get_events("agent.input")
        assert len(events) == 1
        assert events[0]["payload"]["agent_name"] == "my_agent"
        adapter.disconnect()

    def test_on_run_end_emits_agent_output_and_state_change(self) -> None:
        """on_run_end should emit agent.output AND agent.state.change."""
        collector = EventCollector()
        adapter = StrandsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_run_end(agent_name="my_agent", output="Done")

        output_events = collector.get_events("agent.output")
        state_events = collector.get_events("agent.state.change")
        assert len(output_events) == 1
        assert len(state_events) == 1
        assert state_events[0]["payload"]["event_subtype"] == "run_complete"
        adapter.disconnect()

    def test_on_run_end_with_error_emits_failed_state(self) -> None:
        """on_run_end with error should emit run_failed state."""
        collector = EventCollector()
        adapter = StrandsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_run_end(
            agent_name="my_agent",
            output=None,
            error=ValueError("Something went wrong"),
        )

        state_events = collector.get_events("agent.state.change")
        assert len(state_events) == 1
        assert state_events[0]["payload"]["event_subtype"] == "run_failed"
        adapter.disconnect()

    def test_on_tool_use_emits_tool_call(self) -> None:
        collector = EventCollector()
        adapter = StrandsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_tool_use(
            tool_name="web_search",
            tool_input="quantum computing",
            tool_output="Quantum uses qubits...",
        )

        events = collector.get_events("tool.call")
        assert len(events) == 1
        assert events[0]["payload"]["tool_name"] == "web_search"
        adapter.disconnect()

    def test_on_llm_call_emits_model_invoke(self) -> None:
        collector = EventCollector()
        adapter = StrandsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_llm_call(
            provider="bedrock",
            model="anthropic.claude-3-sonnet",
            tokens_prompt=200,
            tokens_completion=100,
            latency_ms=500.0,
        )

        events = collector.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["provider"] == "bedrock"
        adapter.disconnect()

    def test_events_not_emitted_when_disconnected(self) -> None:
        collector = EventCollector()
        adapter = StrandsAdapter(stratix=collector)

        adapter.on_run_start(agent_name="test", input_data="hello")
        assert len(collector.events) == 0


# ---------------------------------------------------------------------------
# Replay serialization
# ---------------------------------------------------------------------------


class TestReplaySerialization:
    """Test replay serialization."""

    def test_serialize_includes_events(self) -> None:
        collector = EventCollector()
        adapter = StrandsAdapter(stratix=collector)
        adapter.connect()

        adapter.on_run_start(agent_name="agent", input_data="test")
        adapter.on_run_end(agent_name="agent", output="done")

        trace = adapter.serialize_for_replay()
        assert trace.adapter_name == "StrandsAdapter"
        assert trace.framework == "strands"
        assert len(trace.events) >= 2
        adapter.disconnect()
