"""Tests for CrewAI Delegation Tracking."""

import pytest

from layerlens.instrument.adapters.crewai.lifecycle import CrewAIAdapter
from layerlens.instrument.adapters.crewai.delegation import CrewDelegationTracker


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


class TestCrewDelegationTracker:
    """Tests for CrewDelegationTracker."""

    def test_tracker_initialization(self):
        """Test tracker initializes correctly."""
        adapter = CrewAIAdapter()
        tracker = CrewDelegationTracker(adapter)

        assert tracker.delegation_count == 0

    def test_track_delegation_emits_handoff(self):
        """Test track_delegation emits agent.handoff event."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()
        tracker = CrewDelegationTracker(adapter)

        tracker.track_delegation("manager", "researcher", "research AI")

        events = stratix.get_events("agent.handoff")
        assert len(events) == 1
        assert events[0]["payload"]["from_agent"] == "manager"
        assert events[0]["payload"]["to_agent"] == "researcher"
        assert events[0]["payload"]["reason"] == "delegation"

    def test_track_delegation_increments_counter(self):
        """Test delegation counter increments."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()
        tracker = CrewDelegationTracker(adapter)

        tracker.track_delegation("a", "b")
        tracker.track_delegation("b", "c")

        assert tracker.delegation_count == 2

    def test_track_delegation_with_context(self):
        """Test delegation with context includes preview and hash."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()
        tracker = CrewDelegationTracker(adapter)

        tracker.track_delegation("manager", "writer", "write a detailed report")

        events = stratix.get_events("agent.handoff")
        assert events[0]["payload"]["context_preview"] == "write a detailed report"
        assert events[0]["payload"]["context_hash"]  # non-empty

    def test_track_delegation_none_context(self):
        """Test delegation with None context."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()
        tracker = CrewDelegationTracker(adapter)

        tracker.track_delegation("a", "b", context=None)

        events = stratix.get_events("agent.handoff")
        assert events[0]["payload"]["context_preview"] is None

    def test_context_hash_deterministic(self):
        """Test context hash is deterministic for same input."""
        adapter = CrewAIAdapter()
        tracker = CrewDelegationTracker(adapter)

        hash1 = tracker._hash_context("same context")
        hash2 = tracker._hash_context("same context")

        assert hash1 == hash2

    def test_context_hash_differs_for_different_input(self):
        """Test context hash differs for different input."""
        adapter = CrewAIAdapter()
        tracker = CrewDelegationTracker(adapter)

        hash1 = tracker._hash_context("context a")
        hash2 = tracker._hash_context("context b")

        assert hash1 != hash2

    def test_long_context_truncated(self):
        """Test long context is truncated in preview."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()
        tracker = CrewDelegationTracker(adapter)

        long_context = "x" * 1000
        tracker.track_delegation("a", "b", context=long_context)

        events = stratix.get_events("agent.handoff")
        preview = events[0]["payload"]["context_preview"]
        assert len(preview) == 500

    def test_delegation_seq_in_payload(self):
        """Test delegation sequence number is in payload."""
        stratix = MockStratix()
        adapter = CrewAIAdapter(stratix=stratix)
        adapter.connect()
        tracker = CrewDelegationTracker(adapter)

        tracker.track_delegation("a", "b")
        tracker.track_delegation("b", "c")

        events = stratix.get_events("agent.handoff")
        assert events[0]["payload"]["delegation_seq"] == 1
        assert events[1]["payload"]["delegation_seq"] == 2
