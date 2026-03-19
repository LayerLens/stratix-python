"""Tests for STRATIX base event types."""

import pytest

from layerlens.instrument.schema.event import STRATIXEvent, STRATIXEventBuilder
from layerlens.instrument.schema.events import AgentInputEvent, AgentOutputEvent, MessageRole
from layerlens.instrument.schema.privacy import PrivacyLevel


class TestSTRATIXEvent:
    """Tests for STRATIXEvent."""

    def test_create_complete_event(self):
        """Test creating a complete STRATIX event."""
        payload = AgentInputEvent.create(
            message="Hello!",
            role=MessageRole.HUMAN,
        )
        event = STRATIXEvent.create(
            payload=payload,
            agent_id="test_agent",
        )

        # Check all envelopes are present
        assert event.identity is not None
        assert event.privacy is not None
        assert event.attestation is not None
        assert event.payload is not None

        # Check identity
        assert event.identity.agent_id == "test_agent"
        assert event.identity.event_type == "agent.input"

        # Check privacy
        assert event.privacy.level == PrivacyLevel.CLEARTEXT

        # Check attestation
        assert event.attestation.hash.startswith("sha256:")

    def test_create_with_privacy_level(self):
        """Test creating event with different privacy levels."""
        payload = AgentInputEvent.create(message="Secret info")

        # Hashed
        event = STRATIXEvent.create(
            payload=payload,
            agent_id="test_agent",
            privacy_level=PrivacyLevel.HASHED,
        )
        assert event.privacy.level == PrivacyLevel.HASHED

        # Not provided
        event = STRATIXEvent.create(
            payload=payload,
            agent_id="test_agent",
            privacy_level=PrivacyLevel.NOT_PROVIDED,
        )
        assert event.privacy.level == PrivacyLevel.NOT_PROVIDED

    def test_event_consistency_validation(self):
        """Test that event type consistency is validated."""
        payload = AgentInputEvent.create(message="Hello")
        event = STRATIXEvent.create(
            payload=payload,
            agent_id="test_agent",
        )
        # identity.event_type should match payload.event_type
        assert event.identity.event_type == payload.event_type

    def test_is_attestable(self):
        """Test attestability check."""
        payload = AgentInputEvent.create(message="Hello")
        event = STRATIXEvent.create(
            payload=payload,
            agent_id="test_agent",
        )
        assert event.is_attestable() is True

    def test_get_event_type(self):
        """Test getting event type."""
        payload = AgentInputEvent.create(message="Hello")
        event = STRATIXEvent.create(
            payload=payload,
            agent_id="test_agent",
        )
        assert event.get_event_type() == "agent.input"

    def test_get_layer(self):
        """Test getting layer from event."""
        payload = AgentInputEvent.create(message="Hello")
        event = STRATIXEvent.create(
            payload=payload,
            agent_id="test_agent",
        )
        assert event.get_layer() == "L1"


class TestSTRATIXEventBuilder:
    """Tests for STRATIXEventBuilder."""

    def test_builder_initialization(self):
        """Test builder initialization."""
        builder = STRATIXEventBuilder(agent_id="test_agent")
        assert builder.sequence_id == 0
        assert builder.last_hash is None
        assert builder.event_count == 0

    def test_builder_increments_sequence(self):
        """Test that builder increments sequence IDs."""
        builder = STRATIXEventBuilder(agent_id="test_agent")

        event1 = builder.build(AgentInputEvent.create(message="First"))
        assert event1.identity.sequence_id == 1

        event2 = builder.build(AgentOutputEvent.create(message="Second"))
        assert event2.identity.sequence_id == 2

    def test_builder_maintains_hash_chain(self):
        """Test that builder maintains continuous hash chain."""
        builder = STRATIXEventBuilder(agent_id="test_agent")

        event1 = builder.build(AgentInputEvent.create(message="First"))
        assert event1.attestation.previous_hash is None

        event2 = builder.build(AgentOutputEvent.create(message="Second"))
        assert event2.attestation.previous_hash == event1.attestation.hash

        event3 = builder.build(AgentInputEvent.create(message="Third"))
        assert event3.attestation.previous_hash == event2.attestation.hash

    def test_builder_verify_chain(self):
        """Test chain verification."""
        builder = STRATIXEventBuilder(agent_id="test_agent")
        builder.build(AgentInputEvent.create(message="First"))
        builder.build(AgentOutputEvent.create(message="Second"))
        builder.build(AgentInputEvent.create(message="Third"))
        assert builder.verify_chain() is True

    def test_builder_get_events(self):
        """Test getting all events."""
        builder = STRATIXEventBuilder(agent_id="test_agent")
        builder.build(AgentInputEvent.create(message="First"))
        builder.build(AgentOutputEvent.create(message="Second"))
        events = builder.get_events()
        assert len(events) == 2

    def test_builder_preserves_ids(self):
        """Test that builder preserves evaluation/trial/trace IDs."""
        eval_id = "11111111-1111-1111-1111-111111111111"
        trial_id = "22222222-2222-2222-2222-222222222222"
        trace_id = "33333333-3333-3333-3333-333333333333"
        builder = STRATIXEventBuilder(
            agent_id="test_agent",
            evaluation_id=eval_id,
            trial_id=trial_id,
            trace_id=trace_id,
        )

        event1 = builder.build(AgentInputEvent.create(message="First"))
        event2 = builder.build(AgentOutputEvent.create(message="Second"))

        # All events should have same evaluation/trial/trace
        assert event1.identity.evaluation_id == eval_id
        assert event2.identity.evaluation_id == eval_id
        assert event1.identity.trial_id == trial_id
        assert event2.identity.trial_id == trial_id
