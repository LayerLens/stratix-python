"""
STRATIX Base Event Model

Defines the base STRATIXEvent that composes:
- Identity envelope (required)
- Privacy envelope (required for payload-bearing events)
- Attestation envelope (required)
- Payload (event-specific data)

From Step 1 specification, all events MUST include:
1. Core identity model (evaluation_id, trial_id, trace_id, span_id, etc.)
2. Privacy envelope (for payload-bearing events)
3. Hashing & attestation envelope
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar, Union

from pydantic import BaseModel, Field, model_validator

from layerlens.instrument.schema.attestation import AttestationEnvelope, HashScope
from layerlens.instrument.schema.identity import IdentityEnvelope, Timestamps, VectorClock
from layerlens.instrument.schema.privacy import PrivacyEnvelope, PrivacyLevel
from layerlens.instrument.schema.events.l1_io import AgentInputEvent, AgentOutputEvent
from layerlens.instrument.schema.events.l2_code import AgentCodeEvent
from layerlens.instrument.schema.events.l3_model import ModelInvokeEvent
from layerlens.instrument.schema.events.l4_environment import EnvironmentConfigEvent, EnvironmentMetricsEvent
from layerlens.instrument.schema.events.l5_tools import ToolCallEvent, ToolLogicEvent, ToolEnvironmentEvent
from layerlens.instrument.schema.events.cross_cutting import (
    AgentStateChangeEvent,
    CostRecordEvent,
    PolicyViolationEvent,
    AgentHandoffEvent,
)
from layerlens.instrument.schema.events.replay import (
    TraceCheckpointEvent,
    TraceReplayStartEvent,
    TraceReplayEndEvent,
)
from layerlens.instrument.schema.events.feedback import (
    ExplicitFeedbackEvent,
    ImplicitFeedbackEvent,
    AnnotationFeedbackEvent,
)


# Type variable for event payloads
PayloadT = TypeVar("PayloadT", bound=BaseModel)

# Union of all event payload types (13 existing + 6 new = 19 total)
EventPayload = Union[
    # L1: Agent I/O
    AgentInputEvent,
    AgentOutputEvent,
    # L2: Agent Code
    AgentCodeEvent,
    # L3: Model
    ModelInvokeEvent,
    # L4: Environment
    EnvironmentConfigEvent,
    EnvironmentMetricsEvent,
    # L5: Tools
    ToolCallEvent,
    ToolLogicEvent,
    ToolEnvironmentEvent,
    # Cross-cutting
    AgentStateChangeEvent,
    CostRecordEvent,
    PolicyViolationEvent,
    AgentHandoffEvent,
    # Replay
    TraceCheckpointEvent,
    TraceReplayStartEvent,
    TraceReplayEndEvent,
    # Feedback
    ExplicitFeedbackEvent,
    ImplicitFeedbackEvent,
    AnnotationFeedbackEvent,
]


class STRATIXEvent(BaseModel, Generic[PayloadT]):
    """
    Base STRATIX Event with all required envelopes.

    NORMATIVE:
    - Events without identity envelope are INVALID
    - Events without privacy envelope are INVALID (for payload-bearing events)
    - Events without attestation envelope are INVALID

    Structure:
    {
        "identity": { ... },     // Required: evaluation_id, trial_id, etc.
        "privacy": { ... },      // Required: level, payload_hash, etc.
        "attestation": { ... },  // Required: hash, hash_scope, etc.
        "payload": { ... }       // Event-specific data
    }
    """

    identity: IdentityEnvelope = Field(
        description="Identity envelope with evaluation/trial/trace IDs"
    )
    privacy: PrivacyEnvelope = Field(
        description="Privacy envelope with level and payload hash"
    )
    attestation: AttestationEnvelope = Field(
        description="Attestation envelope with hash chain"
    )
    payload: PayloadT = Field(
        description="Event-specific payload data"
    )

    @model_validator(mode="after")
    def validate_event_consistency(self) -> STRATIXEvent[PayloadT]:
        """Validate that the event is internally consistent."""
        # Ensure event_type in identity matches payload
        payload_event_type = getattr(self.payload, "event_type", None)
        if payload_event_type and self.identity.event_type != payload_event_type:
            raise ValueError(
                f"Identity event_type ({self.identity.event_type}) "
                f"doesn't match payload event_type ({payload_event_type})"
            )
        return self

    @classmethod
    def create(
        cls,
        payload: PayloadT,
        agent_id: str,
        evaluation_id: str | None = None,
        trial_id: str | None = None,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        parent_agent_id: str | None = None,
        sequence_id: int = 0,
        vector_clock: VectorClock | None = None,
        privacy_level: PrivacyLevel = PrivacyLevel.CLEARTEXT,
        privacy_reason: str | None = None,
        previous_hash: str | None = None,
        signing_key_id: str | None = None,
    ) -> STRATIXEvent[PayloadT]:
        """
        Create a complete STRATIX event with all envelopes.

        Args:
            payload: The event-specific payload
            agent_id: The agent identifier
            evaluation_id: Evaluation ID (generated if not provided)
            trial_id: Trial ID (generated if not provided)
            trace_id: Trace ID (generated if not provided)
            parent_span_id: Parent span ID
            parent_agent_id: Parent agent ID
            sequence_id: Sequence number for this event
            vector_clock: Vector clock for causal ordering
            privacy_level: Privacy level for the payload
            privacy_reason: Reason for privacy level selection
            previous_hash: Previous hash in the chain
            signing_key_id: Signing key identifier

        Returns:
            Complete STRATIXEvent instance
        """
        # Get event type from payload
        event_type = getattr(payload, "event_type", "unknown")

        # Create identity envelope
        identity = IdentityEnvelope.create(
            event_type=event_type,
            agent_id=agent_id,
            evaluation_id=evaluation_id,
            trial_id=trial_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            parent_agent_id=parent_agent_id,
            sequence_id=sequence_id,
            vector_clock=vector_clock,
        )

        # Create privacy envelope based on level
        payload_dict = payload.model_dump()
        if privacy_level == PrivacyLevel.CLEARTEXT:
            privacy = PrivacyEnvelope.for_cleartext(payload_dict, privacy_reason)
        elif privacy_level == PrivacyLevel.HASHED:
            privacy = PrivacyEnvelope.for_hashed(payload_dict, privacy_reason)
        elif privacy_level == PrivacyLevel.NOT_PROVIDED:
            privacy = PrivacyEnvelope.for_not_provided(privacy_reason)
        else:
            # Default to cleartext
            privacy = PrivacyEnvelope.for_cleartext(payload_dict, privacy_reason)

        # Create attestation envelope
        # Hash the full event data (identity + privacy + payload)
        event_data = {
            "identity": identity.model_dump(),
            "privacy": privacy.model_dump(),
            "payload": payload_dict,
        }
        attestation = AttestationEnvelope.create_event_hash(
            event_data=event_data,
            previous_hash=previous_hash,
            signing_key_id=signing_key_id,
        )

        return cls(
            identity=identity,
            privacy=privacy,
            attestation=attestation,
            payload=payload,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return self.model_dump()

    def get_event_type(self) -> str:
        """Get the event type."""
        return self.identity.event_type

    def get_layer(self) -> str | None:
        """Get the layer from the payload if available."""
        return getattr(self.payload, "layer", None)

    def is_attestable(self) -> bool:
        """Check if this event is properly attestable."""
        return (
            self.attestation.hash is not None
            and self.privacy.payload_hash is not None
        )


class STRATIXEventBuilder:
    """
    Builder for creating STRATIX events with proper chain management.

    Handles:
    - Sequence ID allocation
    - Vector clock management
    - Hash chain continuity
    """

    def __init__(
        self,
        agent_id: str,
        evaluation_id: str | None = None,
        trial_id: str | None = None,
        trace_id: str | None = None,
        signing_key_id: str | None = None,
    ):
        """
        Initialize the event builder.

        Args:
            agent_id: The agent identifier
            evaluation_id: Evaluation ID (generated if not provided)
            trial_id: Trial ID (generated if not provided)
            trace_id: Trace ID (generated if not provided)
            signing_key_id: Signing key identifier
        """
        import uuid
        self._agent_id = agent_id
        self._evaluation_id = evaluation_id or str(uuid.uuid4())
        self._trial_id = trial_id or str(uuid.uuid4())
        self._trace_id = trace_id or str(uuid.uuid4())
        self._signing_key_id = signing_key_id
        self._sequence_id = 0
        self._vector_clock = VectorClock.empty()
        self._last_hash: str | None = None
        self._events: list[STRATIXEvent] = []

    @property
    def sequence_id(self) -> int:
        """Get current sequence ID."""
        return self._sequence_id

    @property
    def last_hash(self) -> str | None:
        """Get the last hash in the chain."""
        return self._last_hash

    @property
    def event_count(self) -> int:
        """Get the number of events built."""
        return len(self._events)

    def _next_sequence_id(self) -> int:
        """Allocate the next sequence ID."""
        self._sequence_id += 1
        return self._sequence_id

    def _increment_vector_clock(self) -> VectorClock:
        """Increment the vector clock for this agent."""
        participant_id = f"agent:{self._agent_id}"
        self._vector_clock = self._vector_clock.increment(participant_id)
        return self._vector_clock

    def build(
        self,
        payload: PayloadT,
        privacy_level: PrivacyLevel = PrivacyLevel.CLEARTEXT,
        privacy_reason: str | None = None,
        parent_span_id: str | None = None,
        parent_agent_id: str | None = None,
    ) -> STRATIXEvent[PayloadT]:
        """
        Build a new event with proper sequencing and hash chaining.

        Args:
            payload: The event-specific payload
            privacy_level: Privacy level for the payload
            privacy_reason: Reason for privacy level selection
            parent_span_id: Parent span ID
            parent_agent_id: Parent agent ID

        Returns:
            Complete STRATIXEvent instance
        """
        # Allocate sequence ID and update vector clock
        seq_id = self._next_sequence_id()
        vc = self._increment_vector_clock()

        # Create the event
        event = STRATIXEvent.create(
            payload=payload,
            agent_id=self._agent_id,
            evaluation_id=self._evaluation_id,
            trial_id=self._trial_id,
            trace_id=self._trace_id,
            parent_span_id=parent_span_id,
            parent_agent_id=parent_agent_id,
            sequence_id=seq_id,
            vector_clock=vc,
            privacy_level=privacy_level,
            privacy_reason=privacy_reason,
            previous_hash=self._last_hash,
            signing_key_id=self._signing_key_id,
        )

        # Update hash chain
        self._last_hash = event.attestation.hash
        self._events.append(event)

        return event

    def get_events(self) -> list[STRATIXEvent]:
        """Get all events built so far."""
        return list(self._events)

    def verify_chain(self) -> bool:
        """Verify the hash chain integrity."""
        if len(self._events) == 0:
            return True

        # First event should have no previous hash
        if self._events[0].attestation.previous_hash is not None:
            return False

        # Each subsequent event should link to the previous
        for i in range(1, len(self._events)):
            expected = self._events[i - 1].attestation.hash
            actual = self._events[i].attestation.previous_hash
            if actual != expected:
                return False

        return True
