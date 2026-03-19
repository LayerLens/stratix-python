"""Tests for STRATIX Python SDK Context Propagation."""

import pytest

from layerlens.instrument import STRATIX, STRATIXContext, get_current_context, context_scope
from layerlens.instrument._context import set_current_context, reset_context


class TestContextBasics:
    """Tests for basic context functionality."""

    def test_context_creation(self):
        """Test creating a context."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = STRATIXContext(stratix=stratix)

        assert ctx.evaluation_id is not None
        assert ctx.trial_id is not None
        assert ctx.trace_id is not None
        assert ctx.sequence_id == 0

    def test_context_with_explicit_ids(self):
        """Test creating a context with explicit IDs."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = STRATIXContext(
            stratix=stratix,
            evaluation_id="11111111-1111-1111-1111-111111111111",
            trial_id="22222222-2222-2222-2222-222222222222",
            trace_id="33333333-3333-3333-3333-333333333333",
        )

        assert ctx.evaluation_id == "11111111-1111-1111-1111-111111111111"
        assert ctx.trial_id == "22222222-2222-2222-2222-222222222222"
        assert ctx.trace_id == "33333333-3333-3333-3333-333333333333"


class TestContextPropagation:
    """Tests for context propagation."""

    def test_get_current_context_none(self):
        """Test getting context when none is set."""
        # Ensure no context is set
        token = set_current_context(None)
        try:
            ctx = get_current_context()
            assert ctx is None
        finally:
            reset_context(token)

    def test_set_and_get_context(self):
        """Test setting and getting current context."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = STRATIXContext(stratix=stratix)
        token = set_current_context(ctx)

        try:
            current = get_current_context()
            assert current is ctx
        finally:
            reset_context(token)

    def test_context_scope_manager(self):
        """Test using context_scope as context manager."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = STRATIXContext(stratix=stratix)

        # Save current context
        before = get_current_context()

        with context_scope(ctx) as scoped_ctx:
            assert scoped_ctx is ctx
            assert get_current_context() is ctx

        # Context should be restored after exiting
        assert get_current_context() is before


class TestSequenceIds:
    """Tests for sequence ID allocation."""

    def test_sequence_id_starts_at_zero(self):
        """Test that sequence ID starts at 0."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = STRATIXContext(stratix=stratix)
        assert ctx.sequence_id == 0

    def test_next_sequence_id_increments(self):
        """Test that next_sequence_id increments."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = STRATIXContext(stratix=stratix)

        seq1 = ctx.next_sequence_id()
        seq2 = ctx.next_sequence_id()
        seq3 = ctx.next_sequence_id()

        assert seq1 == 1
        assert seq2 == 2
        assert seq3 == 3


class TestVectorClock:
    """Tests for vector clock management."""

    def test_vector_clock_starts_empty(self):
        """Test that vector clock starts empty."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = STRATIXContext(stratix=stratix)
        vc = ctx.vector_clock

        assert len(vc.clock) == 0

    def test_increment_vector_clock(self):
        """Test incrementing vector clock."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = STRATIXContext(stratix=stratix)

        vc = ctx.increment_vector_clock()

        # Should have incremented for agent
        assert len(vc.clock) == 1
        assert vc.clock[f"agent:{stratix.agent_id}"] == 1

    def test_multiple_increments(self):
        """Test multiple vector clock increments."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = STRATIXContext(stratix=stratix)

        ctx.increment_vector_clock()
        ctx.increment_vector_clock()
        vc = ctx.increment_vector_clock()

        assert vc.clock[f"agent:{stratix.agent_id}"] == 3

    def test_merge_vector_clock(self):
        """Test merging vector clocks."""
        from layerlens.instrument.schema.identity import VectorClock

        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = STRATIXContext(stratix=stratix)

        # Local increment
        ctx.increment_vector_clock()

        # Remote clock
        remote = VectorClock(clock={"agent:other_agent": 5})

        # Merge
        merged = ctx.merge_vector_clock(remote)

        # Should have both entries, with local incremented
        assert merged.clock[f"agent:{stratix.agent_id}"] == 2
        assert merged.clock["agent:other_agent"] == 5


class TestSpanStack:
    """Tests for span stack management."""

    def test_start_span(self):
        """Test starting a span."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = STRATIXContext(stratix=stratix)

        span_id = ctx.start_span()

        assert span_id is not None
        assert ctx.current_span_id == span_id

    def test_nested_spans(self):
        """Test nested span management."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = STRATIXContext(stratix=stratix)

        span1 = ctx.start_span()
        span2 = ctx.start_span()

        assert ctx.current_span_id == span2
        assert ctx.parent_span_id == span1

        # End inner span
        ended = ctx.end_span()
        assert ended == span2
        assert ctx.current_span_id == span1
        assert ctx.parent_span_id is None

    def test_end_span_restores_parent(self):
        """Test that ending a span restores the parent."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = STRATIXContext(stratix=stratix)

        span1 = ctx.start_span()
        span2 = ctx.start_span()
        span3 = ctx.start_span()

        # End all spans
        ctx.end_span()
        assert ctx.current_span_id == span2

        ctx.end_span()
        assert ctx.current_span_id == span1

        ctx.end_span()
        assert ctx.current_span_id is None


class TestContextSerialization:
    """Tests for context serialization."""

    def test_to_dict(self):
        """Test serializing context to dict."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = STRATIXContext(
            stratix=stratix,
            evaluation_id="11111111-1111-1111-1111-111111111111",
            trial_id="22222222-2222-2222-2222-222222222222",
            trace_id="33333333-3333-3333-3333-333333333333",
        )

        ctx.start_span()
        ctx.increment_vector_clock()

        data = ctx.to_dict()

        assert data["evaluation_id"] == "11111111-1111-1111-1111-111111111111"
        assert data["trial_id"] == "22222222-2222-2222-2222-222222222222"
        assert data["trace_id"] == "33333333-3333-3333-3333-333333333333"
        assert data["span_id"] is not None
        assert "vector_clock" in data

    def test_from_dict(self):
        """Test restoring context from dict."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        data = {
            "evaluation_id": "11111111-1111-1111-1111-111111111111",
            "trial_id": "22222222-2222-2222-2222-222222222222",
            "trace_id": "33333333-3333-3333-3333-333333333333",
            "span_id": "44444444-4444-4444-4444-444444444444",
            "parent_span_id": "55555555-5555-5555-5555-555555555555",
            "vector_clock": {"agent:test_agent": 5},
        }

        ctx = STRATIXContext.from_dict(data, stratix)

        assert ctx.evaluation_id == "11111111-1111-1111-1111-111111111111"
        assert ctx.trial_id == "22222222-2222-2222-2222-222222222222"
        assert ctx.trace_id == "33333333-3333-3333-3333-333333333333"
        assert ctx.current_span_id == "44444444-4444-4444-4444-444444444444"
        assert ctx.parent_span_id == "55555555-5555-5555-5555-555555555555"


class TestChildContext:
    """Tests for child context creation."""

    def test_create_child_context(self):
        """Test creating a child context."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        parent = STRATIXContext(stratix=stratix)
        parent.start_span()
        parent.increment_vector_clock()

        child = parent.create_child_context()

        # Should share IDs
        assert child.evaluation_id == parent.evaluation_id
        assert child.trial_id == parent.trial_id
        assert child.trace_id == parent.trace_id

        # Should have parent's span in stack
        assert child.parent_span_id == parent.current_span_id
