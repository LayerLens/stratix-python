"""Tests for STRATIX causality (vector clock) types."""

import pytest

from layerlens.instrument.schema.causality import SparseVectorClock, VectorClockManager


class TestSparseVectorClock:
    """Tests for SparseVectorClock."""

    def test_empty_clock(self):
        """Test creating an empty vector clock."""
        vc = SparseVectorClock.empty()
        assert len(vc) == 0
        assert vc.get("any") == 0

    def test_for_agent(self):
        """Test creating a clock for an agent."""
        vc = SparseVectorClock.for_agent("agent_1")
        assert vc.get("agent:agent_1") == 1

    def test_for_tool(self):
        """Test creating a clock for a tool."""
        vc = SparseVectorClock.for_tool("lookup")
        assert vc.get("tool:lookup") == 1

    def test_increment_immutable(self):
        """Test that increment returns a new clock."""
        vc1 = SparseVectorClock.empty()
        vc2 = vc1.increment("agent:a")
        assert vc1.get("agent:a") == 0  # Original unchanged
        assert vc2.get("agent:a") == 1  # New has increment

    def test_multiple_increments(self):
        """Test multiple increments."""
        vc = SparseVectorClock.empty()
        vc = vc.increment("agent:a")
        vc = vc.increment("agent:a")
        vc = vc.increment("agent:a")
        assert vc.get("agent:a") == 3

    def test_merge_takes_max(self):
        """Test that merge takes the maximum of each participant."""
        vc1 = SparseVectorClock(entries={"agent:a": 3, "agent:b": 1})
        vc2 = SparseVectorClock(entries={"agent:a": 2, "agent:c": 5})
        merged = vc1.merge(vc2)
        assert merged.get("agent:a") == 3  # max(3, 2)
        assert merged.get("agent:b") == 1  # only in vc1
        assert merged.get("agent:c") == 5  # only in vc2

    def test_merge_and_increment(self):
        """Test merge followed by increment."""
        vc1 = SparseVectorClock(entries={"agent:a": 2})
        vc2 = SparseVectorClock(entries={"agent:b": 3})
        result = vc1.merge_and_increment(vc2, "agent:a")
        assert result.get("agent:a") == 3  # merged max(2,0) then +1
        assert result.get("agent:b") == 3  # from vc2

    def test_happens_before_basic(self):
        """Test basic happens-before relationship."""
        vc1 = SparseVectorClock(entries={"agent:a": 1})
        vc2 = SparseVectorClock(entries={"agent:a": 2})
        assert vc1.happens_before(vc2) is True
        assert vc2.happens_before(vc1) is False

    def test_happens_before_multiple_participants(self):
        """Test happens-before with multiple participants."""
        vc1 = SparseVectorClock(entries={"agent:a": 1, "agent:b": 2})
        vc2 = SparseVectorClock(entries={"agent:a": 2, "agent:b": 3})
        assert vc1.happens_before(vc2) is True
        assert vc2.happens_before(vc1) is False

    def test_happens_before_new_participant(self):
        """Test happens-before with new participant."""
        vc1 = SparseVectorClock(entries={"agent:a": 1})
        vc2 = SparseVectorClock(entries={"agent:a": 1, "agent:b": 1})
        assert vc1.happens_before(vc2) is True

    def test_concurrent_detection(self):
        """Test detection of concurrent events."""
        vc1 = SparseVectorClock(entries={"agent:a": 2, "agent:b": 1})
        vc2 = SparseVectorClock(entries={"agent:a": 1, "agent:b": 2})
        assert vc1.is_concurrent_with(vc2) is True
        assert vc2.is_concurrent_with(vc1) is True

    def test_equality(self):
        """Test clock equality."""
        vc1 = SparseVectorClock(entries={"agent:a": 1, "agent:b": 2})
        vc2 = SparseVectorClock(entries={"agent:a": 1, "agent:b": 2})
        vc3 = SparseVectorClock(entries={"agent:a": 1, "agent:b": 3})
        assert vc1 == vc2
        assert vc1 != vc3

    def test_comparison_operators(self):
        """Test comparison operators."""
        vc1 = SparseVectorClock(entries={"agent:a": 1})
        vc2 = SparseVectorClock(entries={"agent:a": 2})
        assert vc1 < vc2
        assert vc1 <= vc2
        assert vc2 > vc1
        assert vc2 >= vc1

    def test_participants(self):
        """Test getting participant set."""
        vc = SparseVectorClock(entries={"agent:a": 1, "tool:b": 2})
        assert vc.participants() == {"agent:a", "tool:b"}

    def test_iteration(self):
        """Test iterating over clock."""
        vc = SparseVectorClock(entries={"agent:a": 1, "tool:b": 2})
        items = dict(vc)
        assert items == {"agent:a": 1, "tool:b": 2}

    def test_to_dict(self):
        """Test conversion to dictionary."""
        vc = SparseVectorClock(entries={"agent:a": 1, "tool:b": 2})
        assert vc.to_dict() == {"agent:a": 1, "tool:b": 2}


class TestVectorClockManager:
    """Tests for VectorClockManager."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = VectorClockManager("agent:test")
        assert manager.local_participant_id == "agent:test"
        assert len(manager.current_clock) == 0

    def test_emit_increments_local(self):
        """Test that emit increments local participant."""
        manager = VectorClockManager("agent:test")
        clock = manager.emit()
        assert clock.get("agent:test") == 1
        clock = manager.emit()
        assert clock.get("agent:test") == 2

    def test_receive_merges_and_increments(self):
        """Test that receive merges and increments."""
        manager = VectorClockManager("agent:a")
        manager.emit()  # agent:a = 1

        remote = SparseVectorClock(entries={"agent:b": 5})
        clock = manager.receive(remote)
        assert clock.get("agent:a") == 2  # incremented
        assert clock.get("agent:b") == 5  # merged

    def test_synchronize_merges_without_increment(self):
        """Test that synchronize merges without incrementing."""
        manager = VectorClockManager("agent:a")
        manager.emit()  # agent:a = 1

        other = SparseVectorClock(entries={"agent:b": 5})
        clock = manager.synchronize(other)
        assert clock.get("agent:a") == 1  # NOT incremented
        assert clock.get("agent:b") == 5  # merged

    def test_causal_queries(self):
        """Test causal relationship queries."""
        manager = VectorClockManager("agent:a")
        manager.emit()
        manager.emit()  # agent:a = 2

        earlier = SparseVectorClock(entries={"agent:a": 1})
        later = SparseVectorClock(entries={"agent:a": 3})
        concurrent = SparseVectorClock(entries={"agent:b": 1})

        assert manager.is_after(earlier) is True
        assert manager.is_before(later) is True
        assert manager.is_concurrent(concurrent) is True
