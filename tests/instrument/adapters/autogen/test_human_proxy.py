"""Tests for AutoGen Human-in-the-Loop Tracing."""

import pytest

from layerlens.instrument.adapters.autogen.lifecycle import AutoGenAdapter
from layerlens.instrument.adapters.autogen.human_proxy import HumanProxyTracer


class MockStratix:
    """Mock STRATIX instance for testing."""

    def __init__(self):
        self.events = []

    def emit(self, event_type: str, payload: dict):
        self.events.append({"type": event_type, "payload": payload})

    def get_events(self, event_type: str = None):
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


class MockUserProxyAgent:
    """Mock AutoGen UserProxyAgent."""

    def __init__(self, name="user_proxy", response="yes"):
        self.name = name
        self._response = response

    def get_human_input(self, prompt="", **kwargs):
        return self._response


class TestHumanProxyTracer:
    """Tests for HumanProxyTracer."""

    def test_tracer_initialization(self):
        """Test tracer initializes correctly."""
        adapter = AutoGenAdapter()
        tracer = HumanProxyTracer(adapter)

        assert tracer.interaction_count == 0

    def test_wrap_agent(self):
        """Test wrap_agent wraps get_human_input."""
        adapter = AutoGenAdapter()
        tracer = HumanProxyTracer(adapter)
        agent = MockUserProxyAgent()

        result = tracer.wrap_agent(agent)

        assert result is agent
        assert hasattr(agent, "_stratix_human_tracer")
        assert hasattr(agent.get_human_input, "_stratix_original")

    def test_wrapped_get_human_input_calls_original(self):
        """Test wrapped method returns original response."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()
        tracer = HumanProxyTracer(adapter)

        agent = MockUserProxyAgent(response="yes, proceed")
        tracer.wrap_agent(agent)

        result = agent.get_human_input("Do you approve?")
        assert result == "yes, proceed"

    def test_wrapped_emits_events(self):
        """Test wrapped method emits request and response events."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()
        tracer = HumanProxyTracer(adapter)

        agent = MockUserProxyAgent(response="yes")
        tracer.wrap_agent(agent)

        agent.get_human_input("Approve this?")

        events = stratix.get_events("agent.input")
        assert len(events) == 2  # request + response

        # Request event
        assert events[0]["payload"]["input_type"] == "human_input_request"
        assert events[0]["payload"]["role"] == "HUMAN"

        # Response event
        assert events[1]["payload"]["input_type"] == "approval"
        assert events[1]["payload"]["response_preview"] == "yes"

    def test_interaction_count_increments(self):
        """Test interaction counter increments."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()
        tracer = HumanProxyTracer(adapter)

        agent = MockUserProxyAgent()
        tracer.wrap_agent(agent)

        agent.get_human_input("q1")
        agent.get_human_input("q2")

        assert tracer.interaction_count == 2

    def test_classify_approval(self):
        """Test approval classification."""
        adapter = AutoGenAdapter()
        tracer = HumanProxyTracer(adapter)

        for word in ("y", "yes", "approve", "ok", "okay", "sure", "proceed", "continue"):
            assert tracer._classify_input(word) == "approval"

    def test_classify_rejection(self):
        """Test rejection classification."""
        adapter = AutoGenAdapter()
        tracer = HumanProxyTracer(adapter)

        for word in ("n", "no", "reject", "deny", "stop", "cancel", "abort"):
            assert tracer._classify_input(word) == "rejection"

    def test_classify_custom_input(self):
        """Test custom input classification."""
        adapter = AutoGenAdapter()
        tracer = HumanProxyTracer(adapter)

        assert tracer._classify_input("Use a different approach") == "custom_input"

    def test_classify_empty_input(self):
        """Test empty input classification."""
        adapter = AutoGenAdapter()
        tracer = HumanProxyTracer(adapter)

        assert tracer._classify_input("") == "empty"

    def test_response_latency_captured(self):
        """Test response latency is captured."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()
        tracer = HumanProxyTracer(adapter)

        agent = MockUserProxyAgent(response="yes")
        tracer.wrap_agent(agent)

        agent.get_human_input("prompt")

        events = stratix.get_events("agent.input")
        # Response event should have latency
        response_event = events[1]
        assert "response_latency_ms" in response_event["payload"]
        assert response_event["payload"]["response_latency_ms"] >= 0
