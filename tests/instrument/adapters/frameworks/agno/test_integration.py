"""Integration tests for the Agno adapter using the real Agno SDK.

Ported from ``ateam/tests/adapters/agno/test_integration.py``.

These tests verify that ``AgnoAdapter`` correctly instruments actual
Agno ``Agent`` classes — not mocks. The Agno SDK must be installed::

    pip install 'layerlens[agno]'

Tests are skipped via ``pytest.importorskip`` if ``agno`` is not
available in the environment.

Renames from the ateam original:
- ``stratix.sdk.python.adapters.agno.lifecycle`` →
  ``layerlens.instrument.adapters.frameworks.agno.lifecycle``
- ``stratix.sdk.python.adapters.base`` /
  ``stratix.sdk.python.adapters.capture`` →
  ``layerlens.instrument.adapters._base``
- The wrapper marker attribute renamed from ``_stratix_original`` to
  ``_layerlens_original`` (matches the Agno adapter source).
"""

from __future__ import annotations

from typing import Any

import pytest

agno_mod = pytest.importorskip("agno", reason="agno not installed")

from agno.agent import Agent  # noqa: E402
from agno.models.openai import OpenAIChat  # noqa: E402

from layerlens.instrument.adapters._base import (  # noqa: E402
    AdapterStatus,
    CaptureConfig,
    AdapterCapability,
)
from layerlens.instrument.adapters.frameworks.agno.lifecycle import AgnoAdapter  # noqa: E402

# ---------------------------------------------------------------------------
# Event collector — not a mock
# ---------------------------------------------------------------------------


class EventCollector:
    """Real event collector that accumulates events for assertions."""

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
# Adapter construction with real Agno types
# ---------------------------------------------------------------------------


class TestAdapterWithRealSDK:
    """Verify adapter constructs and connects with real Agno classes."""

    def test_connect_detects_agno_version(self) -> None:
        """Adapter should detect the installed Agno version on connect."""
        collector = EventCollector()
        adapter = AgnoAdapter(stratix=collector)
        adapter.connect()

        assert adapter.is_connected
        assert adapter._status == AdapterStatus.HEALTHY
        health = adapter.health_check()
        # framework_version should be populated when agno is installed
        assert health.framework_version is not None

    def test_adapter_info_capabilities(self) -> None:
        """Adapter info should report correct capabilities."""
        collector = EventCollector()
        adapter = AgnoAdapter(stratix=collector)

        info = adapter.get_adapter_info()
        assert info.name == "AgnoAdapter"
        assert info.framework == "agno"
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
        adapter = AgnoAdapter(stratix=collector, capture_config=config)
        assert adapter._capture_config.l3_model_metadata is True
        assert adapter._capture_config.l5a_tool_calls is False


# ---------------------------------------------------------------------------
# Agent instrumentation with real Agno Agent
# ---------------------------------------------------------------------------


class TestAgentInstrumentation:
    """Verify adapter instruments real Agno Agent instances."""

    def _make_agent(self, name: str = "test_agent") -> Agent:
        """Create a real Agno Agent.

        Note: We construct the Agent but do NOT call ``run()`` since
        that requires a valid API key. We test instrumentation wrapping
        only.
        """
        return Agent(
            name=name,
            description="A test agent for integration testing",
            instructions=["You are a helpful test agent."],
            markdown=False,
        )

    def test_instrument_agent_wraps_run(self) -> None:
        """instrument_agent should wrap run() on a real Agno Agent."""
        collector = EventCollector()
        adapter = AgnoAdapter(stratix=collector)
        adapter.connect()

        agent = self._make_agent("researcher")
        adapter.instrument_agent(agent)

        assert hasattr(agent.run, "_layerlens_original")

    def test_instrument_agent_emits_environment_config(self) -> None:
        """Wrapping an agent should emit environment.config."""
        collector = EventCollector()
        adapter = AgnoAdapter(stratix=collector)
        adapter.connect()

        agent = self._make_agent("analyst")
        adapter.instrument_agent(agent)

        config_events = collector.get_events("environment.config")
        assert len(config_events) == 1
        payload = config_events[0]["payload"]
        assert payload["framework"] == "agno"
        assert payload["agent_name"] == "analyst"

    def test_instrument_agent_captures_description(self) -> None:
        """Environment config should include agent description."""
        collector = EventCollector()
        adapter = AgnoAdapter(stratix=collector)
        adapter.connect()

        agent = self._make_agent("describer")
        adapter.instrument_agent(agent)

        config_events = collector.get_events("environment.config")
        payload = config_events[0]["payload"]
        assert "description" in payload
        assert "test agent" in payload["description"].lower()

    def test_instrument_agent_captures_instructions_with_full_config(self) -> None:
        """With capture_content=True, environment.config should include instructions."""
        collector = EventCollector()
        config = CaptureConfig(capture_content=True)
        adapter = AgnoAdapter(stratix=collector, capture_config=config)
        adapter.connect()

        agent = self._make_agent("instructor")
        adapter.instrument_agent(agent)

        config_events = collector.get_events("environment.config")
        payload = config_events[0]["payload"]
        assert "instructions" in payload

    def test_instrument_agent_idempotent(self) -> None:
        """Instrumenting the same agent twice should not double-wrap."""
        collector = EventCollector()
        adapter = AgnoAdapter(stratix=collector)
        adapter.connect()

        agent = self._make_agent()
        adapter.instrument_agent(agent)
        first_run = agent.run
        adapter.instrument_agent(agent)

        assert agent.run is first_run
        assert len(adapter._originals) == 1

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Pre-existing assertion flaw inherited from ateam: ``agent.run`` on "
            "a real Agno ``Agent`` is a bound method, so ``agent.run is original_run`` "
            "is False even when unwrap restored the original underlying function "
            "(bound methods are constructed fresh on each attribute access). The "
            "assertion only passes against ``MagicMock`` agents (see "
            "``test_disconnect_unwraps`` in test_lifecycle.py). Verifying real "
            "unwrap on Agno's Pydantic ``Agent`` requires comparing "
            "``agent.run.__func__`` and accounting for ``setattr`` semantics on "
            "Pydantic v2 models — out of scope for the test-restoration port."
        ),
    )
    def test_disconnect_unwraps_real_agent(self) -> None:
        """disconnect() should restore original methods on real agents."""
        collector = EventCollector()
        adapter = AgnoAdapter(stratix=collector)
        adapter.connect()

        agent = self._make_agent()
        original_run = agent.run
        adapter.instrument_agent(agent)
        assert hasattr(agent.run, "_layerlens_original")

        adapter.disconnect()
        assert agent.run is original_run


# ---------------------------------------------------------------------------
# Lifecycle hooks and event emission
# ---------------------------------------------------------------------------


class TestLifecycleHooks:
    """Verify lifecycle hooks emit correct events."""

    def test_on_run_start_emits_agent_input(self) -> None:
        """on_run_start should emit an agent.input event."""
        collector = EventCollector()
        adapter = AgnoAdapter(stratix=collector)
        adapter.connect()

        adapter.on_run_start(agent_name="research_agent", input_data="Find papers on AI")

        events = collector.get_events("agent.input")
        assert len(events) == 1
        assert events[0]["payload"]["framework"] == "agno"
        assert events[0]["payload"]["agent_name"] == "research_agent"
        assert events[0]["payload"]["timestamp_ns"] > 0

    def test_on_run_end_emits_agent_output_with_duration(self) -> None:
        """on_run_end should emit agent.output with duration_ns."""
        collector = EventCollector()
        adapter = AgnoAdapter(stratix=collector)
        adapter.connect()

        adapter.on_run_start(agent_name="summarizer", input_data="Summarize")
        adapter.on_run_end(agent_name="summarizer", output="Summary done")

        events = collector.get_events("agent.output")
        assert len(events) == 1
        assert events[0]["payload"]["agent_name"] == "summarizer"
        assert events[0]["payload"]["duration_ns"] >= 0

    def test_on_run_end_emits_state_change(self) -> None:
        """on_run_end should also emit agent.state.change."""
        collector = EventCollector()
        adapter = AgnoAdapter(stratix=collector)
        adapter.connect()

        adapter.on_run_end(agent_name="test", output="done")

        events = collector.get_events("agent.state.change")
        assert len(events) == 1
        assert events[0]["payload"]["event_subtype"] == "run_complete"

    def test_on_run_end_error_emits_run_failed(self) -> None:
        """on_run_end with error should emit state change as run_failed."""
        collector = EventCollector()
        adapter = AgnoAdapter(stratix=collector)
        adapter.connect()

        adapter.on_run_end(agent_name="broken", output=None, error=RuntimeError("API timeout"))

        output_events = collector.get_events("agent.output")
        assert len(output_events) == 1
        assert "API timeout" in output_events[0]["payload"]["error"]

        state_events = collector.get_events("agent.state.change")
        assert len(state_events) == 1
        assert state_events[0]["payload"]["event_subtype"] == "run_failed"

    def test_on_tool_use_emits_tool_call(self) -> None:
        """on_tool_use should emit a tool.call event."""
        collector = EventCollector()
        adapter = AgnoAdapter(stratix=collector)
        adapter.connect()

        adapter.on_tool_use(
            tool_name="web_search",
            tool_input={"query": "quantum computing"},
            tool_output={"results": ["paper1", "paper2"]},
            latency_ms=250.0,
        )

        events = collector.get_events("tool.call")
        assert len(events) == 1
        assert events[0]["payload"]["tool_name"] == "web_search"
        assert events[0]["payload"]["latency_ms"] == 250.0

    def test_on_tool_use_with_error(self) -> None:
        """on_tool_use should capture tool errors."""
        collector = EventCollector()
        adapter = AgnoAdapter(stratix=collector)
        adapter.connect()

        adapter.on_tool_use(
            tool_name="broken_tool",
            error=ConnectionError("Service unavailable"),
        )

        events = collector.get_events("tool.call")
        assert len(events) == 1
        assert events[0]["payload"]["error"] == "Service unavailable"

    def test_on_llm_call_emits_model_invoke(self) -> None:
        """on_llm_call should emit a model.invoke event."""
        collector = EventCollector()
        adapter = AgnoAdapter(stratix=collector)
        adapter.connect()

        adapter.on_llm_call(
            provider="openai",
            model="gpt-4o",
            tokens_prompt=200,
            tokens_completion=100,
            latency_ms=480.0,
        )

        events = collector.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["model"] == "gpt-4o"
        assert events[0]["payload"]["provider"] == "openai"
        assert events[0]["payload"]["tokens_prompt"] == 200
        assert events[0]["payload"]["tokens_completion"] == 100

    def test_on_handoff_emits_agent_handoff(self) -> None:
        """on_handoff should emit an agent.handoff event with context hash."""
        collector = EventCollector()
        adapter = AgnoAdapter(stratix=collector)
        adapter.connect()

        adapter.on_handoff(
            from_agent="leader",
            to_agent="researcher",
            context="Investigate market trends",
        )

        events = collector.get_events("agent.handoff")
        assert len(events) == 1
        assert events[0]["payload"]["from_agent"] == "leader"
        assert events[0]["payload"]["to_agent"] == "researcher"
        assert events[0]["payload"]["reason"] == "agno_team_delegation"
        assert events[0]["payload"]["context_hash"] is not None
        assert len(events[0]["payload"]["context_hash"]) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------


class TestProviderDetection:
    """Verify provider detection from model identifiers."""

    def test_detect_openai(self) -> None:
        adapter = AgnoAdapter()
        assert adapter._detect_provider("gpt-4o") == "openai"
        assert adapter._detect_provider("o1-preview") == "openai"

    def test_detect_anthropic(self) -> None:
        adapter = AgnoAdapter()
        assert adapter._detect_provider("claude-opus-4-20250514") == "anthropic"

    def test_detect_google(self) -> None:
        adapter = AgnoAdapter()
        assert adapter._detect_provider("gemini-3.1-pro") == "google"

    def test_detect_meta(self) -> None:
        adapter = AgnoAdapter()
        assert adapter._detect_provider("llama-3.1-70b") == "meta"

    def test_detect_unknown(self) -> None:
        adapter = AgnoAdapter()
        assert adapter._detect_provider("some-custom-model") is None
        assert adapter._detect_provider(None) is None


# ---------------------------------------------------------------------------
# CaptureConfig gating
# ---------------------------------------------------------------------------


class TestCaptureConfigGating:
    """Verify CaptureConfig correctly gates events."""

    def test_minimal_config_gates_l3_and_l5(self) -> None:
        """Minimal config should suppress model.invoke and tool.call."""
        collector = EventCollector()
        adapter = AgnoAdapter(stratix=collector, capture_config=CaptureConfig.minimal())
        adapter.connect()

        adapter.on_llm_call(model="gpt-4o")
        adapter.on_tool_use(tool_name="search")

        assert len(collector.get_events("model.invoke")) == 0
        assert len(collector.get_events("tool.call")) == 0

    def test_minimal_config_allows_cross_cutting(self) -> None:
        """Cross-cutting events should always be emitted."""
        collector = EventCollector()
        adapter = AgnoAdapter(stratix=collector, capture_config=CaptureConfig.minimal())
        adapter.connect()

        adapter.on_handoff(from_agent="a", to_agent="b")

        assert len(collector.get_events("agent.handoff")) == 1

    def test_full_config_emits_all(self) -> None:
        """Full config should emit all event types."""
        collector = EventCollector()
        adapter = AgnoAdapter(stratix=collector, capture_config=CaptureConfig.full())
        adapter.connect()

        adapter.on_llm_call(model="gpt-4o", tokens_prompt=10, tokens_completion=5)
        adapter.on_tool_use(tool_name="search", tool_output="result")
        adapter.on_run_start(agent_name="agent", input_data="q")
        adapter.on_run_end(agent_name="agent", output="a")

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
        adapter = AgnoAdapter(stratix=collector)
        adapter.connect()
        assert adapter._status == AdapterStatus.HEALTHY
        adapter.disconnect()
        assert adapter._status == AdapterStatus.DISCONNECTED

    def test_adapter_has_framework_metadata(self) -> None:
        """Adapter should expose its framework name and version."""
        collector = EventCollector()
        adapter = AgnoAdapter(stratix=collector)
        assert adapter.FRAMEWORK == "agno"
        assert adapter.VERSION is not None

    def test_replay_trace_accumulates_events(self) -> None:
        """Events should accumulate for replay serialization."""
        collector = EventCollector()
        adapter = AgnoAdapter(stratix=collector)
        adapter.connect()

        adapter.on_run_start(agent_name="a", input_data="q")
        adapter.on_run_end(agent_name="a", output="a")

        trace = adapter.serialize_for_replay()
        # on_run_end emits agent.output + agent.state.change = 3 total
        assert len(trace.events) >= 3
        assert trace.adapter_name == "AgnoAdapter"
        assert trace.framework == "agno"

    def test_agent_config_emitted_once_per_name(self) -> None:
        """Environment config should only be emitted once per agent name."""
        collector = EventCollector()
        adapter = AgnoAdapter(stratix=collector)
        adapter.connect()

        agent = Agent(name="unique_agent", markdown=False)
        adapter.instrument_agent(agent)
        # Manually try to emit again
        adapter._emit_agent_config("unique_agent", agent)

        config_events = collector.get_events("environment.config")
        assert len(config_events) == 1


# ---------------------------------------------------------------------------
# Real Agno model type verification
# ---------------------------------------------------------------------------


class TestAgnoModelTypes:
    """Verify adapter works with real Agno model types."""

    def test_openai_chat_model_has_expected_attributes(self) -> None:
        """OpenAIChat model should have the attributes the adapter inspects."""
        model = OpenAIChat(id="gpt-4o")
        # The adapter reads model.id to extract the model name
        assert model.id == "gpt-4o"

    def test_agent_with_model_captures_model_in_config(self) -> None:
        """Agent constructed with a model should have its model in config."""
        collector = EventCollector()
        adapter = AgnoAdapter(stratix=collector)
        adapter.connect()

        model = OpenAIChat(id="gpt-4o")
        agent = Agent(name="model_agent", model=model, markdown=False)
        adapter.instrument_agent(agent)

        config_events = collector.get_events("environment.config")
        assert len(config_events) == 1
        payload = config_events[0]["payload"]
        # The adapter calls str(model) on the model attribute
        assert "model" in payload
