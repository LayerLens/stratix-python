"""Integration tests for Google ADK adapter using real SDK types.

Ported as-is from ``ateam/tests/adapters/google_adk/test_integration.py``.

Translation rules applied:
* ``stratix.sdk.python.adapters.base`` →
  ``layerlens.instrument.adapters._base.adapter``
* ``stratix.sdk.python.adapters.capture`` →
  ``layerlens.instrument.adapters._base.capture``
* ``stratix.sdk.python.adapters.google_adk.lifecycle`` →
  ``layerlens.instrument.adapters.frameworks.google_adk.lifecycle``

These tests verify that GoogleADKAdapter correctly captures events
from actual Google ADK constructs -- not mocks. The adapter uses
google-adk (google.adk) or google-genai (google.genai) as its SDK.

Since google-adk is not yet widely available as a public PyPI package,
these tests focus on adapter behavior that can be validated without
requiring authentication or network access: construction, configuration,
event mapping, callback wiring, and lifecycle hooks.

When google-adk IS installed, tests additionally verify SDK version
detection and callback type compatibility.

Tests skip cleanly when neither google-adk nor google-genai is installed.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from layerlens.instrument.adapters._base.adapter import (
    AdapterStatus,
    AdapterCapability,
)
from layerlens.instrument.adapters._base.capture import CaptureConfig
from layerlens.instrument.adapters.frameworks.google_adk.lifecycle import (
    GoogleADKAdapter,
)

# ---------------------------------------------------------------------------
# Try to import google-adk; tests that need it will skipif unavailable
# ---------------------------------------------------------------------------

_has_google_adk = False
_has_google_genai = False

try:
    import google.adk as _adk  # type: ignore[import-untyped]

    _has_google_adk = bool(_adk)
except ImportError:
    pass

try:
    import google.genai as _genai  # type: ignore[import-untyped]  # noqa: F401

    _has_google_genai = True
except ImportError:
    pass

needs_google_sdk = pytest.mark.skipif(
    not (_has_google_adk or _has_google_genai),
    reason="Neither google-adk nor google-genai installed",
)


# ---------------------------------------------------------------------------
# EventCollector -- real collector, not a mock
# ---------------------------------------------------------------------------


class EventCollector:
    """Real event collector -- accumulates events for assertions."""

    # Multi-tenant test stand-in: every recording client carries an
    # org_id so adapters constructed with this collector pass the
    # BaseAdapter fail-fast check.
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
# Fake ADK objects for callback testing (no auth required)
# ---------------------------------------------------------------------------


class FakeAgent:
    """Minimal agent stub matching Google ADK Agent callback surface."""

    def __init__(self, name: str = "test_agent", model: str = "gemini-2.0-flash") -> None:
        self.name = name
        self.model = model
        self.description = "A test agent"
        self.instruction = "You are a helpful assistant."
        self.tools: list[Any] = []
        self.sub_agents: list[Any] = []
        # Callback slots that GoogleADKAdapter.instrument_agent() sets
        self.before_agent_callback: Any = None
        self.after_agent_callback: Any = None
        self.before_model_callback: Any = None
        self.after_model_callback: Any = None
        self.before_tool_callback: Any = None
        self.after_tool_callback: Any = None


class FakeSession:
    """Minimal session stub."""

    def __init__(self, session_id: str = "session-001") -> None:
        self.id = session_id


class FakeCallbackContext:
    """Minimal callback context matching Google ADK callback signature."""

    def __init__(
        self,
        agent: FakeAgent | None = None,
        session: FakeSession | None = None,
        user_content: str | None = None,
        agent_output: str | None = None,
    ) -> None:
        self.agent = agent or FakeAgent()
        self.session = session or FakeSession()
        self.user_content = user_content
        self.agent_output = agent_output
        self.model = getattr(self.agent, "model", None)


class FakeUsageMetadata:
    """Minimal usage metadata matching google.genai response."""

    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.prompt_token_count = prompt_tokens
        self.candidates_token_count = completion_tokens


class FakeLLMResponse:
    """Minimal LLM response matching google.genai GenerateContentResponse."""

    def __init__(
        self,
        model: str | None = None,
        usage: FakeUsageMetadata | None = None,
    ) -> None:
        self.model = model
        self.usage_metadata = usage


# ---------------------------------------------------------------------------
# Adapter construction
# ---------------------------------------------------------------------------


class TestAdapterConstruction:
    """Verify adapter constructs correctly with various configurations."""

    def test_adapter_framework_metadata(self) -> None:
        """Adapter should expose correct framework name and version."""
        adapter: GoogleADKAdapter = GoogleADKAdapter(org_id="test-org")
        assert adapter.FRAMEWORK == "google_adk"
        assert adapter.VERSION == "0.1.0"

    def test_adapter_capabilities(self) -> None:
        """Google ADK adapter must declare all four capabilities."""
        collector = EventCollector()
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector)
        adapter.connect()
        info = adapter.get_adapter_info()
        assert AdapterCapability.TRACE_HANDOFFS in info.capabilities
        assert AdapterCapability.TRACE_TOOLS in info.capabilities
        assert AdapterCapability.TRACE_MODELS in info.capabilities
        assert AdapterCapability.TRACE_STATE in info.capabilities
        assert info.name == "GoogleADKAdapter"

    def test_capture_config_propagates(self) -> None:
        """CaptureConfig correctly controls which events are captured."""
        collector = EventCollector()
        config = CaptureConfig(
            l3_model_metadata=True,
            l5a_tool_calls=False,
            l1_agent_io=True,
        )
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector, capture_config=config)
        assert adapter._capture_config.l3_model_metadata is True
        assert adapter._capture_config.l5a_tool_calls is False

    @needs_google_sdk
    def test_connect_detects_sdk_version(self) -> None:
        """connect() should discover the installed google-adk or genai version."""
        collector = EventCollector()
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector)
        adapter.connect()
        health = adapter.health_check()
        assert health.framework_version is not None
        assert health.framework_version != ""

    def test_connect_without_sdk_still_healthy(self) -> None:
        """connect() should succeed even without SDK (framework_version may be None)."""
        collector = EventCollector()
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector)
        adapter.connect()
        assert adapter._status == AdapterStatus.HEALTHY


# ---------------------------------------------------------------------------
# Agent instrumentation via callback wiring
# ---------------------------------------------------------------------------


class TestAgentInstrumentation:
    """Verify instrument_agent wires callbacks onto the agent object."""

    def test_instrument_agent_sets_all_callbacks(self) -> None:
        """instrument_agent should set all 6 callback slots on the agent."""
        collector = EventCollector()
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector)
        adapter.connect()

        agent = FakeAgent(name="weather_agent")
        adapter.instrument_agent(agent)

        assert agent.before_agent_callback is not None
        assert agent.after_agent_callback is not None
        assert agent.before_model_callback is not None
        assert agent.after_model_callback is not None
        assert agent.before_tool_callback is not None
        assert agent.after_tool_callback is not None

    def test_before_agent_callback_emits_agent_input(self) -> None:
        """Calling the wired before_agent_callback should emit agent.input."""
        collector = EventCollector()
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector)
        adapter.connect()

        agent = FakeAgent(name="planner_agent")
        adapter.instrument_agent(agent)

        ctx = FakeCallbackContext(agent=agent, user_content="Plan a trip to Paris")
        agent.before_agent_callback(ctx)

        events = collector.get_events("agent.input")
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload["framework"] == "google_adk"
        assert payload["agent_name"] == "planner_agent"
        assert payload["input"] == "Plan a trip to Paris"

    def test_after_agent_callback_emits_agent_output(self) -> None:
        """Calling the wired after_agent_callback should emit agent.output."""
        collector = EventCollector()
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector)
        adapter.connect()

        agent = FakeAgent(name="planner_agent")
        adapter.instrument_agent(agent)

        ctx_start = FakeCallbackContext(agent=agent, user_content="Plan trip")
        agent.before_agent_callback(ctx_start)

        ctx_end = FakeCallbackContext(agent=agent, agent_output="Here is your itinerary...")
        agent.after_agent_callback(ctx_end)

        events = collector.get_events("agent.output")
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload["agent_name"] == "planner_agent"
        assert payload["output"] == "Here is your itinerary..."
        assert payload["duration_ns"] >= 0

    def test_before_after_model_callback_emits_model_invoke(self) -> None:
        """Model callback pair should emit model.invoke with usage info."""
        collector = EventCollector()
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector)
        adapter.connect()

        agent = FakeAgent(name="chat_agent", model="gemini-2.0-flash")
        adapter.instrument_agent(agent)

        ctx = FakeCallbackContext(agent=agent)
        llm_request = {"messages": [{"role": "user", "content": "hello"}]}

        agent.before_model_callback(ctx, llm_request)

        llm_response = FakeLLMResponse(
            model="gemini-2.0-flash",
            usage=FakeUsageMetadata(prompt_tokens=50, completion_tokens=30),
        )
        agent.after_model_callback(ctx, llm_response)

        model_events = collector.get_events("model.invoke")
        assert len(model_events) == 1
        payload = model_events[0]["payload"]
        assert payload["framework"] == "google_adk"
        assert payload["model"] == "gemini-2.0-flash"
        assert payload["provider"] == "google"
        assert payload["tokens_prompt"] == 50
        assert payload["tokens_completion"] == 30
        assert "latency_ms" in payload

    def test_model_callback_emits_cost_record(self) -> None:
        """Model callback should also emit cost.record when usage data present."""
        collector = EventCollector()
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector)
        adapter.connect()

        agent = FakeAgent(name="chat_agent")
        adapter.instrument_agent(agent)

        ctx = FakeCallbackContext(agent=agent)
        agent.before_model_callback(ctx, {})
        llm_response = FakeLLMResponse(
            model="gemini-2.0-flash",
            usage=FakeUsageMetadata(prompt_tokens=100, completion_tokens=50),
        )
        agent.after_model_callback(ctx, llm_response)

        cost_events = collector.get_events("cost.record")
        assert len(cost_events) == 1
        assert cost_events[0]["payload"]["tokens_total"] == 150

    def test_tool_callbacks_emit_tool_call(self) -> None:
        """Tool callback pair should emit tool.call with input/output."""
        collector = EventCollector()
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector)
        adapter.connect()

        agent = FakeAgent(name="search_agent")
        adapter.instrument_agent(agent)

        ctx = FakeCallbackContext(agent=agent)
        tool_input = {"query": "weather in Seattle"}

        agent.before_tool_callback(ctx, "get_weather", tool_input)
        agent.after_tool_callback(
            ctx, "get_weather", tool_input, {"temperature": "62F", "condition": "cloudy"}
        )

        events = collector.get_events("tool.call")
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload["tool_name"] == "get_weather"
        assert payload["tool_input"]["query"] == "weather in Seattle"
        assert payload["tool_output"]["temperature"] == "62F"
        assert "latency_ms" in payload

    def test_agent_config_emitted_once(self) -> None:
        """environment.config should be emitted only once per agent."""
        collector = EventCollector()
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector)
        adapter.connect()

        agent = FakeAgent(name="dedupe_agent")
        adapter.instrument_agent(agent)

        ctx = FakeCallbackContext(agent=agent)
        agent.before_agent_callback(ctx)
        agent.before_agent_callback(ctx)
        agent.before_agent_callback(ctx)

        config_events = collector.get_events("environment.config")
        assert len(config_events) == 1
        assert config_events[0]["payload"]["agent_name"] == "dedupe_agent"


# ---------------------------------------------------------------------------
# Lifecycle hooks (manual API)
# ---------------------------------------------------------------------------


class TestLifecycleHooks:
    """Verify manual lifecycle hook methods emit correct events."""

    def test_agent_start_end_roundtrip(self) -> None:
        """on_agent_start + on_agent_end should emit input/output with duration."""
        collector = EventCollector()
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector)
        adapter.connect()

        adapter.on_agent_start(agent_name="summarizer", input_data="Summarize this doc")
        adapter.on_agent_end(agent_name="summarizer", output="Here is the summary...")

        input_events = collector.get_events("agent.input")
        assert len(input_events) == 1
        assert input_events[0]["payload"]["input"] == "Summarize this doc"

        output_events = collector.get_events("agent.output")
        assert len(output_events) == 1
        assert output_events[0]["payload"]["output"] == "Here is the summary..."
        assert output_events[0]["payload"]["duration_ns"] >= 0

    def test_agent_end_with_error(self) -> None:
        """on_agent_end with error should include error string."""
        collector = EventCollector()
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector)
        adapter.connect()

        adapter.on_agent_start(agent_name="failing", input_data="hello")
        adapter.on_agent_end(agent_name="failing", error=RuntimeError("Model quota exceeded"))

        events = collector.get_events("agent.output")
        assert len(events) == 1
        assert "Model quota exceeded" in events[0]["payload"]["error"]

    def test_on_tool_use(self) -> None:
        """on_tool_use should emit tool.call."""
        collector = EventCollector()
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector)
        adapter.connect()

        adapter.on_tool_use(
            tool_name="google_search",
            tool_input={"query": "AI news"},
            tool_output={"results": ["result1", "result2"]},
            latency_ms=340.0,
        )

        events = collector.get_events("tool.call")
        assert len(events) == 1
        assert events[0]["payload"]["latency_ms"] == 340.0

    def test_on_llm_call(self) -> None:
        """on_llm_call should emit model.invoke."""
        collector = EventCollector()
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector)
        adapter.connect()

        adapter.on_llm_call(
            provider="google",
            model="gemini-2.0-flash",
            tokens_prompt=200,
            tokens_completion=100,
        )

        events = collector.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["model"] == "gemini-2.0-flash"

    def test_disconnected_adapter_emits_nothing(self) -> None:
        """Lifecycle hooks should no-op when adapter is disconnected."""
        collector = EventCollector()
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector)
        # NOT calling connect()

        adapter.on_agent_start(agent_name="test", input_data="hello")
        adapter.on_agent_end(agent_name="test", output="world")
        adapter.on_tool_use(tool_name="t", tool_input={})
        adapter.on_llm_call(model="m")
        adapter.on_handoff(from_agent="a", to_agent="b")

        assert len(collector.events) == 0


# ---------------------------------------------------------------------------
# Handoff events
# ---------------------------------------------------------------------------


class TestHandoffEvents:
    """Verify agent.handoff events for transfer_to_agent patterns."""

    def test_handoff_event(self) -> None:
        """on_handoff should emit agent.handoff with context hash."""
        collector = EventCollector()
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector)
        adapter.connect()

        adapter.on_handoff(
            from_agent="orchestrator",
            to_agent="specialist",
            context="Handle billing inquiry #123",
        )

        events = collector.get_events("agent.handoff")
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload["from_agent"] == "orchestrator"
        assert payload["to_agent"] == "specialist"
        assert payload["reason"] == "transfer_to_agent"
        expected_hash = hashlib.sha256(b"Handle billing inquiry #123").hexdigest()
        assert payload["context_hash"] == expected_hash
        assert payload["context_preview"] == "Handle billing inquiry #123"

    def test_handoff_without_context(self) -> None:
        """Handoff without context should have None hashes."""
        collector = EventCollector()
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector)
        adapter.connect()

        adapter.on_handoff(from_agent="a", to_agent="b")

        events = collector.get_events("agent.handoff")
        assert len(events) == 1
        assert events[0]["payload"]["context_hash"] is None

    def test_handoff_always_emitted_with_minimal_config(self) -> None:
        """agent.handoff is cross-cutting and should emit even with minimal config."""
        collector = EventCollector()
        config = CaptureConfig.minimal()
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector, capture_config=config)
        adapter.connect()

        adapter.on_handoff(from_agent="x", to_agent="y")

        events = collector.get_events("agent.handoff")
        assert len(events) == 1


# ---------------------------------------------------------------------------
# CaptureConfig gating
# ---------------------------------------------------------------------------


class TestCaptureConfigGating:
    """Verify that CaptureConfig correctly gates event emission."""

    def test_minimal_config_blocks_l3_l5(self) -> None:
        """Minimal config should block model.invoke and tool.call."""
        collector = EventCollector()
        config = CaptureConfig.minimal()
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector, capture_config=config)
        adapter.connect()

        adapter.on_llm_call(model="gemini-2.0-flash")
        adapter.on_tool_use(tool_name="search")

        assert len(collector.get_events("model.invoke")) == 0
        assert len(collector.get_events("tool.call")) == 0

    def test_minimal_config_allows_l1(self) -> None:
        """Minimal config should still allow agent.input/output."""
        collector = EventCollector()
        config = CaptureConfig.minimal()
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector, capture_config=config)
        adapter.connect()

        adapter.on_agent_start(agent_name="test", input_data="hello")
        adapter.on_agent_end(agent_name="test", output="world")

        assert len(collector.get_events("agent.input")) == 1
        assert len(collector.get_events("agent.output")) == 1

    def test_content_capture_disabled(self) -> None:
        """capture_content=False should exclude messages from model.invoke."""
        collector = EventCollector()
        config = CaptureConfig(capture_content=False)
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector, capture_config=config)
        adapter.connect()

        adapter.on_llm_call(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "secret"}],
        )

        events = collector.get_events("model.invoke")
        assert len(events) == 1
        assert "messages" not in events[0]["payload"]


# ---------------------------------------------------------------------------
# Adapter lifecycle
# ---------------------------------------------------------------------------


class TestAdapterLifecycle:
    """Verify adapter lifecycle management."""

    def test_connect_disconnect(self) -> None:
        """connect() and disconnect() should not raise."""
        collector = EventCollector()
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector)
        adapter.connect()
        assert adapter._status == AdapterStatus.HEALTHY
        adapter.disconnect()
        assert adapter._status == AdapterStatus.DISCONNECTED

    def test_disconnect_clears_state(self) -> None:
        """disconnect() should clear all internal tracking state."""
        collector = EventCollector()
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector)
        adapter.connect()

        adapter.on_agent_start(agent_name="test", input_data="hello")
        adapter.disconnect()

        assert len(adapter._seen_agents) == 0
        assert len(adapter._model_call_starts) == 0
        assert len(adapter._tool_call_starts) == 0
        assert len(adapter._agent_starts) == 0

    def test_serialization_for_replay(self) -> None:
        """serialize_for_replay should produce a valid ReplayableTrace."""
        collector = EventCollector()
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=collector)
        adapter.connect()

        adapter.on_agent_start(agent_name="test", input_data="hello")
        adapter.on_agent_end(agent_name="test", output="world")

        trace = adapter.serialize_for_replay()
        assert trace.adapter_name == "GoogleADKAdapter"
        assert trace.framework == "google_adk"
        assert len(trace.events) >= 2

    def test_null_stratix_pattern(self) -> None:
        """Adapter should work (no-op) without a stratix instance."""
        adapter: GoogleADKAdapter = GoogleADKAdapter(org_id="test-org")
        adapter.connect()
        # Should not raise
        adapter.on_agent_start(agent_name="test", input_data="hello")
        adapter.on_agent_end(agent_name="test", output="world")
        adapter.on_tool_use(tool_name="test", tool_input={})
        adapter.on_llm_call(model="test")
        adapter.on_handoff(from_agent="a", to_agent="b")
