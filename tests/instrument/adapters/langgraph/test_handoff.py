"""Tests for STRATIX LangGraph Handoff Detection."""

import pytest

from layerlens.instrument.adapters.langgraph.handoff import (
    HandoffDetector,
    AgentHandoff,
    detect_handoff,
    SupervisorHandoffTracker,
    create_handoff_aware_router,
)


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


class TestHandoffDetector:
    """Tests for HandoffDetector."""

    def test_initialization(self):
        """Test detector initializes correctly."""
        detector = HandoffDetector()

        assert detector._current_agent is None
        assert detector._handoffs == []

    def test_register_agent(self):
        """Test registering agents."""
        detector = HandoffDetector()

        detector.register_agent("agent1")
        detector.register_agent("agent2")

        assert "agent1" in detector._registered_agents
        assert "agent2" in detector._registered_agents

    def test_register_multiple_agents(self):
        """Test registering multiple agents at once."""
        detector = HandoffDetector()

        detector.register_agents("agent1", "agent2", "agent3")

        assert len(detector._registered_agents) == 3

    def test_is_handoff_different_agents(self):
        """Test is_handoff returns True for different agents."""
        detector = HandoffDetector()

        assert detector.is_handoff("agent1", "agent2") is True

    def test_is_handoff_same_agent(self):
        """Test is_handoff returns False for same agent."""
        detector = HandoffDetector()

        assert detector.is_handoff("agent1", "agent1") is False

    def test_detect_handoff_creates_handoff(self):
        """Test detect_handoff creates handoff record."""
        detector = HandoffDetector()
        detector.set_current_agent("agent1")

        handoff = detector.detect_handoff("agent2")

        assert handoff is not None
        assert handoff.from_agent == "agent1"
        assert handoff.to_agent == "agent2"

    def test_detect_handoff_returns_none_same_agent(self):
        """Test detect_handoff returns None for same agent."""
        detector = HandoffDetector()
        detector.set_current_agent("agent1")

        handoff = detector.detect_handoff("agent1")

        assert handoff is None

    def test_detect_handoff_updates_current_agent(self):
        """Test detect_handoff updates current agent."""
        detector = HandoffDetector()
        detector.set_current_agent("agent1")

        detector.detect_handoff("agent2")

        assert detector._current_agent == "agent2"

    def test_detect_handoff_emits_event(self):
        """Test detect_handoff emits agent.handoff event."""
        stratix = MockStratix()
        detector = HandoffDetector(stratix)
        detector.set_current_agent("agent1")

        detector.detect_handoff("agent2")

        handoff_events = stratix.get_events("agent.handoff")
        assert len(handoff_events) == 1
        assert handoff_events[0]["payload"]["from_agent"] == "agent1"
        assert handoff_events[0]["payload"]["to_agent"] == "agent2"

    def test_emit_handoff_explicit(self):
        """Test explicit emit_handoff."""
        stratix = MockStratix()
        detector = HandoffDetector(stratix)

        handoff = detector.emit_handoff("from", "to", reason="Task delegation")

        assert handoff.from_agent == "from"
        assert handoff.to_agent == "to"
        assert handoff.reason == "Task delegation"

    def test_emit_handoff_with_state_context(self):
        """Test emit_handoff extracts context from state."""
        stratix = MockStratix()
        detector = HandoffDetector(stratix)

        state = {"task": "Search for documents", "query": "STRATIX"}
        detector.emit_handoff("researcher", "writer", state=state)

        handoff_events = stratix.get_events("agent.handoff")
        context = handoff_events[0]["payload"]["context"]
        assert "task" in context
        assert "query" in context

    def test_context_extraction_truncates_long_values(self):
        """Test context extraction truncates long strings."""
        detector = HandoffDetector()

        state = {"task": "x" * 600}  # Long string
        context = detector._extract_context(state)

        assert len(context["task"]) <= 503  # 500 + "..."

    def test_context_extraction_summarizes_long_lists(self):
        """Test context extraction summarizes long lists."""
        detector = HandoffDetector()

        state = {"messages": list(range(20))}
        context = detector._extract_context(state)

        assert "[20 items]" in context["messages"]


class TestDetectHandoffFunction:
    """Tests for detect_handoff utility function."""

    def test_returns_handoff_for_different_agents(self):
        """Test returns handoff for different agents."""
        handoff = detect_handoff("agent1", "agent2")

        assert handoff is not None
        assert handoff.from_agent == "agent1"
        assert handoff.to_agent == "agent2"

    def test_returns_none_for_same_agent(self):
        """Test returns None for same agent."""
        handoff = detect_handoff("agent1", "agent1")

        assert handoff is None

    def test_emits_event_with_stratix(self):
        """Test emits event when STRATIX provided."""
        stratix = MockStratix()

        detect_handoff("agent1", "agent2", stratix_instance=stratix)

        handoff_events = stratix.get_events("agent.handoff")
        assert len(handoff_events) == 1


class TestSupervisorHandoffTracker:
    """Tests for SupervisorHandoffTracker."""

    def test_initialization(self):
        """Test tracker initializes correctly."""
        tracker = SupervisorHandoffTracker()

        assert tracker._supervisor_name == "supervisor"

    def test_custom_supervisor_name(self):
        """Test custom supervisor name."""
        tracker = SupervisorHandoffTracker(supervisor_name="orchestrator")

        assert tracker._supervisor_name == "orchestrator"

    def test_register_worker(self):
        """Test registering workers."""
        tracker = SupervisorHandoffTracker()

        tracker.register_worker("researcher")
        tracker.register_worker("writer")

        assert "researcher" in tracker._detector._registered_agents
        assert "writer" in tracker._detector._registered_agents

    def test_route_to_creates_handoff(self):
        """Test route_to creates handoff from supervisor."""
        stratix = MockStratix()
        tracker = SupervisorHandoffTracker(stratix)

        handoff = tracker.route_to("researcher", reason="Starting research")

        assert handoff.from_agent == "supervisor"
        assert handoff.to_agent == "researcher"
        assert handoff.reason == "Starting research"

    def test_route_to_default_reason(self):
        """Test route_to uses default reason when not provided."""
        stratix = MockStratix()
        tracker = SupervisorHandoffTracker(stratix)

        handoff = tracker.route_to("researcher")

        assert "Supervisor routed" in handoff.reason

    def test_route_to_from_last_worker(self):
        """Test route_to tracks handoff from last worker."""
        stratix = MockStratix()
        tracker = SupervisorHandoffTracker(stratix)

        tracker.route_to("researcher")
        handoff = tracker.route_to("writer")

        assert handoff.from_agent == "researcher"
        assert handoff.to_agent == "writer"

    def test_return_to_supervisor(self):
        """Test return_to_supervisor creates handoff."""
        stratix = MockStratix()
        tracker = SupervisorHandoffTracker(stratix)

        tracker.route_to("researcher")
        handoff = tracker.return_to_supervisor()

        assert handoff is not None
        assert handoff.from_agent == "researcher"
        assert handoff.to_agent == "supervisor"

    def test_return_to_supervisor_none_when_at_supervisor(self):
        """Test return_to_supervisor returns None when already at supervisor."""
        tracker = SupervisorHandoffTracker()

        handoff = tracker.return_to_supervisor()

        assert handoff is None


class TestCreateHandoffAwareRouter:
    """Tests for create_handoff_aware_router."""

    def test_creates_router_function(self):
        """Test creates router function."""

        def simple_router(state):
            return state.get("next_agent", "default")

        router = create_handoff_aware_router(simple_router)

        assert callable(router)

    def test_router_returns_next_agent(self):
        """Test router returns correct next agent."""

        def simple_router(state):
            return state.get("next")

        router = create_handoff_aware_router(simple_router)
        result = router({"next": "agent2"})

        assert result["next"] == "agent2"

    def test_router_emits_handoff_events(self):
        """Test router emits handoff events."""
        stratix = MockStratix()

        def alternating_router(state):
            return "agent_b" if state.get("current") == "agent_a" else "agent_a"

        router = create_handoff_aware_router(alternating_router, stratix)

        # First call
        router({"current": None})
        # Second call (different agent)
        router({"current": "agent_a"})

        handoff_events = stratix.get_events("agent.handoff")
        # First call sets agent_a, second detects handoff to agent_b
        assert len(handoff_events) == 1
        assert handoff_events[0]["payload"]["to_agent"] == "agent_b"


class TestAgentHandoff:
    """Tests for AgentHandoff dataclass."""

    def test_handoff_creation(self):
        """Test handoff creation."""
        handoff = AgentHandoff(
            from_agent="agent1",
            to_agent="agent2",
            timestamp_ns=12345,
            reason="Task delegation",
        )

        assert handoff.from_agent == "agent1"
        assert handoff.to_agent == "agent2"
        assert handoff.timestamp_ns == 12345
        assert handoff.reason == "Task delegation"

    def test_handoff_optional_fields(self):
        """Test handoff with optional fields as None."""
        handoff = AgentHandoff(
            from_agent="a",
            to_agent="b",
            timestamp_ns=0,
        )

        assert handoff.context is None
        assert handoff.reason is None
