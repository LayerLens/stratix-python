"""Tests for STRATIX identity types."""

import uuid
from datetime import datetime, timezone

import pytest

from layerlens.instrument.schema.identity import (
    EvaluationId,
    TrialId,
    TraceId,
    SpanId,
    AgentId,
    SequenceId,
    SequenceIdAllocator,
    VectorClock,
    Timestamps,
    IdentityEnvelope,
)


class TestEvaluationId:
    """Tests for EvaluationId."""

    def test_generate_creates_valid_uuid(self):
        """Test that generate creates a valid UUID."""
        eval_id = EvaluationId.generate()
        # Should be parseable as UUID
        uuid.UUID(eval_id)

    def test_validate_accepts_valid_uuid(self):
        """Test that validate accepts valid UUIDs."""
        valid_uuid = str(uuid.uuid4())
        result = EvaluationId.validate(valid_uuid)
        assert result == valid_uuid

    def test_validate_rejects_invalid_uuid(self):
        """Test that validate rejects invalid UUIDs."""
        with pytest.raises(ValueError):
            EvaluationId.validate("not-a-uuid")


class TestSequenceId:
    """Tests for SequenceId."""

    def test_validate_accepts_non_negative(self):
        """Test that validate accepts non-negative integers."""
        assert SequenceId.validate(0) == 0
        assert SequenceId.validate(42) == 42

    def test_validate_rejects_negative(self):
        """Test that validate rejects negative integers."""
        with pytest.raises(ValueError):
            SequenceId.validate(-1)


class TestSequenceIdAllocator:
    """Tests for SequenceIdAllocator."""

    def test_starts_at_zero_by_default(self):
        """Test that allocator starts at zero."""
        allocator = SequenceIdAllocator()
        assert allocator.current == 0

    def test_next_increments(self):
        """Test that next() increments the sequence."""
        allocator = SequenceIdAllocator()
        assert allocator.next() == 1
        assert allocator.next() == 2
        assert allocator.next() == 3

    def test_monotonic_validation(self):
        """Test monotonicity validation."""
        allocator = SequenceIdAllocator()
        allocator.next()  # Now at 1
        assert allocator.validate_monotonic(SequenceId(2)) is True
        assert allocator.validate_monotonic(SequenceId(1)) is False
        assert allocator.validate_monotonic(SequenceId(0)) is False

    def test_custom_start(self):
        """Test starting from a custom value."""
        allocator = SequenceIdAllocator(start=100)
        assert allocator.current == 100
        assert allocator.next() == 101


class TestVectorClock:
    """Tests for VectorClock."""

    def test_empty_clock(self):
        """Test creating an empty vector clock."""
        vc = VectorClock.empty()
        assert vc.clock == {}
        assert vc.get("any") == 0

    def test_increment(self):
        """Test incrementing a participant."""
        vc = VectorClock.empty()
        vc2 = vc.increment("agent:a")
        assert vc2.get("agent:a") == 1
        # Original unchanged
        assert vc.get("agent:a") == 0

    def test_merge_takes_max(self):
        """Test that merge takes the maximum of each participant."""
        vc1 = VectorClock(clock={"agent:a": 3, "agent:b": 1})
        vc2 = VectorClock(clock={"agent:a": 2, "agent:c": 5})
        merged = vc1.merge(vc2)
        assert merged.get("agent:a") == 3  # max(3, 2)
        assert merged.get("agent:b") == 1  # only in vc1
        assert merged.get("agent:c") == 5  # only in vc2

    def test_happens_before(self):
        """Test happens-before relationship."""
        vc1 = VectorClock(clock={"agent:a": 1})
        vc2 = VectorClock(clock={"agent:a": 2})
        assert vc1.happens_before(vc2) is True
        assert vc2.happens_before(vc1) is False

    def test_concurrent(self):
        """Test concurrent detection."""
        vc1 = VectorClock(clock={"agent:a": 2, "agent:b": 1})
        vc2 = VectorClock(clock={"agent:a": 1, "agent:b": 2})
        assert vc1.concurrent_with(vc2) is True
        assert vc2.concurrent_with(vc1) is True


class TestTimestamps:
    """Tests for Timestamps."""

    def test_now_creates_valid_timestamps(self):
        """Test that now() creates valid timestamps."""
        ts = Timestamps.now()
        assert ts.wall_clock is not None
        assert ts.monotonic_ns >= 0
        assert ts.vector_clock is not None

    def test_wall_clock_parsing(self):
        """Test that wall_clock parses ISO format."""
        ts = Timestamps(
            wall_clock="2024-01-01T00:00:00Z",
            monotonic_ns=0,
        )
        assert ts.wall_clock.year == 2024


class TestIdentityEnvelope:
    """Tests for IdentityEnvelope."""

    def test_create_generates_ids(self):
        """Test that create generates missing IDs."""
        envelope = IdentityEnvelope.create(
            event_type="agent.input",
            agent_id="test_agent",
        )
        assert envelope.evaluation_id is not None
        assert envelope.trial_id is not None
        assert envelope.trace_id is not None
        assert envelope.span_id is not None
        assert envelope.agent_id == "test_agent"
        assert envelope.event_type == "agent.input"

    def test_validates_uuid_fields(self):
        """Test that UUID fields are validated."""
        with pytest.raises(ValueError):
            IdentityEnvelope(
                evaluation_id="not-a-uuid",
                trial_id=str(uuid.uuid4()),
                trace_id=str(uuid.uuid4()),
                span_id=str(uuid.uuid4()),
                sequence_id=0,
                agent_id="test",
                event_type="test",
                timestamps=Timestamps.now(),
            )

    def test_sequence_id_monotonicity(self):
        """Test that sequence_id must be non-negative."""
        with pytest.raises(ValueError):
            IdentityEnvelope(
                evaluation_id=str(uuid.uuid4()),
                trial_id=str(uuid.uuid4()),
                trace_id=str(uuid.uuid4()),
                span_id=str(uuid.uuid4()),
                sequence_id=-1,  # Invalid
                agent_id="test",
                event_type="test",
                timestamps=Timestamps.now(),
            )
