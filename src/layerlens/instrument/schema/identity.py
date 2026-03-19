"""
STRATIX Core Identity Model

Defines the identity envelope required by all STRATIX events as specified
in Step 1: Canonical Event & Trace Schema.

All events MUST include the identity envelope with:
- evaluation_id, trial_id, trace_id, span_id
- sequence_id (strictly monotonic per agent)
- agent_id, parent_agent_id
- event_type
- timestamps (wall_clock, monotonic_ns, vector_clock)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, ClassVar

from pydantic import BaseModel, Field, field_validator, model_validator


# Type aliases with validation
class EvaluationId(str):
    """Unique identifier for an evaluation run."""

    @classmethod
    def generate(cls) -> EvaluationId:
        """Generate a new evaluation ID."""
        return cls(str(uuid.uuid4()))

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: str) -> EvaluationId:
        if not isinstance(v, str):
            raise ValueError("EvaluationId must be a string")
        try:
            uuid.UUID(v)
        except ValueError as e:
            raise ValueError(f"EvaluationId must be a valid UUID: {e}") from e
        return cls(v)


class TrialId(str):
    """Unique identifier for a trial within an evaluation."""

    @classmethod
    def generate(cls) -> TrialId:
        """Generate a new trial ID."""
        return cls(str(uuid.uuid4()))

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: str) -> TrialId:
        if not isinstance(v, str):
            raise ValueError("TrialId must be a string")
        try:
            uuid.UUID(v)
        except ValueError as e:
            raise ValueError(f"TrialId must be a valid UUID: {e}") from e
        return cls(v)


class TraceId(str):
    """Unique identifier for a trace (compatible with OpenTelemetry)."""

    @classmethod
    def generate(cls) -> TraceId:
        """Generate a new trace ID."""
        return cls(str(uuid.uuid4()))

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: str) -> TraceId:
        if not isinstance(v, str):
            raise ValueError("TraceId must be a string")
        try:
            uuid.UUID(v)
        except ValueError as e:
            raise ValueError(f"TraceId must be a valid UUID: {e}") from e
        return cls(v)


class SpanId(str):
    """Unique identifier for a span within a trace."""

    @classmethod
    def generate(cls) -> SpanId:
        """Generate a new span ID."""
        return cls(str(uuid.uuid4()))

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: str) -> SpanId:
        if not isinstance(v, str):
            raise ValueError("SpanId must be a string")
        try:
            uuid.UUID(v)
        except ValueError as e:
            raise ValueError(f"SpanId must be a valid UUID: {e}") from e
        return cls(v)


class AgentId(str):
    """Identifier for an agent (not required to be UUID, can be semantic)."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: str) -> AgentId:
        if not isinstance(v, str):
            raise ValueError("AgentId must be a string")
        if len(v) == 0:
            raise ValueError("AgentId cannot be empty")
        if len(v) > 256:
            raise ValueError("AgentId cannot exceed 256 characters")
        return cls(v)


class SequenceId(int):
    """
    Monotonically increasing sequence number per agent.

    NORMATIVE: sequence_id is strictly monotonic per agent.
    Each event emitted by an agent must have a sequence_id greater
    than all previous events from that agent.
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: int) -> SequenceId:
        if not isinstance(v, int):
            raise ValueError("SequenceId must be an integer")
        if v < 0:
            raise ValueError("SequenceId must be non-negative")
        return cls(v)


class SequenceIdAllocator:
    """
    Thread-safe allocator for monotonically increasing sequence IDs.

    Each agent should have its own allocator to ensure monotonicity.
    """

    def __init__(self, start: int = 0):
        self._current = start
        self._lock_available = True
        try:
            import threading
            self._lock: threading.Lock | None = threading.Lock()
        except ImportError:
            self._lock = None

    def next(self) -> SequenceId:
        """Allocate the next sequence ID."""
        if self._lock is not None:
            with self._lock:
                self._current += 1
                return SequenceId(self._current)
        else:
            self._current += 1
            return SequenceId(self._current)

    @property
    def current(self) -> SequenceId:
        """Get the current sequence ID without incrementing."""
        return SequenceId(self._current)

    def validate_monotonic(self, seq_id: SequenceId) -> bool:
        """Validate that a sequence ID is greater than the current."""
        return seq_id > self._current


class VectorClock(BaseModel):
    """
    Sparse vector clock for causal ordering across distributed agents.

    NORMATIVE: Vector clocks are sparse and only include active participants.
    Keys are participant IDs in the format:
    - agent:{agent_id}
    - tool:{name}
    - grader:{id}
    """

    clock: dict[str, int] = Field(default_factory=dict)

    @classmethod
    def empty(cls) -> VectorClock:
        """Create an empty vector clock."""
        return cls(clock={})

    def increment(self, participant_id: str) -> VectorClock:
        """
        Increment the clock for a participant and return a new clock.

        Args:
            participant_id: The participant ID (e.g., "agent:support_agent")

        Returns:
            A new VectorClock with the incremented value
        """
        new_clock = self.clock.copy()
        new_clock[participant_id] = new_clock.get(participant_id, 0) + 1
        return VectorClock(clock=new_clock)

    def merge(self, other: VectorClock) -> VectorClock:
        """
        Merge two vector clocks by taking the max of each participant.

        NORMATIVE: On receiving remote context (handoff/tool response):
        - Merge vector clocks by max(participant)
        - Increment local participant for the receive event

        Args:
            other: The other vector clock to merge with

        Returns:
            A new VectorClock with merged values
        """
        merged = {}
        all_keys = set(self.clock.keys()) | set(other.clock.keys())
        for key in all_keys:
            merged[key] = max(
                self.clock.get(key, 0),
                other.clock.get(key, 0)
            )
        return VectorClock(clock=merged)

    def happens_before(self, other: VectorClock) -> bool:
        """
        Check if this clock happens-before another.

        Returns True if all entries in self are <= corresponding entries
        in other, and at least one is strictly less.
        """
        at_least_one_less = False
        for key in self.clock:
            self_val = self.clock.get(key, 0)
            other_val = other.clock.get(key, 0)
            if self_val > other_val:
                return False
            if self_val < other_val:
                at_least_one_less = True
        # Check for new entries in other
        for key in other.clock:
            if key not in self.clock and other.clock[key] > 0:
                at_least_one_less = True
        return at_least_one_less

    def concurrent_with(self, other: VectorClock) -> bool:
        """Check if this clock is concurrent with another (neither happens-before)."""
        return not self.happens_before(other) and not other.happens_before(self)

    def get(self, participant_id: str) -> int:
        """Get the clock value for a participant."""
        return self.clock.get(participant_id, 0)

    def __getitem__(self, key: str) -> int:
        return self.clock.get(key, 0)

    def model_dump(self, **kwargs) -> dict:
        """Serialize to dictionary."""
        return self.clock


class Timestamps(BaseModel):
    """
    Timestamp envelope for events.

    Contains:
    - wall_clock: RFC3339 formatted wall clock time
    - monotonic_ns: Monotonic nanoseconds (for ordering within same process)
    - vector_clock: Sparse vector clock for distributed causal ordering
    """

    wall_clock: datetime = Field(
        description="RFC3339 formatted wall clock timestamp"
    )
    monotonic_ns: int = Field(
        ge=0,
        description="Monotonic nanoseconds for local ordering"
    )
    vector_clock: VectorClock = Field(
        default_factory=VectorClock.empty,
        description="Sparse vector clock for causal ordering"
    )

    @classmethod
    def now(cls, vector_clock: VectorClock | None = None) -> Timestamps:
        """Create a Timestamps with current time."""
        import time
        return cls(
            wall_clock=datetime.now(timezone.utc),
            monotonic_ns=time.monotonic_ns(),
            vector_clock=vector_clock or VectorClock.empty()
        )

    @field_validator("wall_clock", mode="before")
    @classmethod
    def parse_wall_clock(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

    def model_dump(self, **kwargs) -> dict:
        """Serialize to dictionary with proper datetime handling."""
        data = super().model_dump(**kwargs)
        # Convert datetime to ISO format string
        if isinstance(data.get("wall_clock"), datetime):
            data["wall_clock"] = data["wall_clock"].isoformat()
        return data


class IdentityEnvelope(BaseModel):
    """
    Core identity envelope required by ALL STRATIX events.

    NORMATIVE: Events without this envelope are invalid.

    From Step 1 specification:
    - evaluation_id: UUID for the evaluation run
    - trial_id: UUID for this specific trial
    - trace_id: UUID for the trace (OTel compatible)
    - span_id: UUID for this span
    - parent_span_id: UUID of parent span (null for root)
    - sequence_id: Strictly monotonic per agent
    - agent_id: String identifier for the agent
    - parent_agent_id: String identifier for parent agent (null for root)
    - event_type: Type of event
    - timestamps: Wall clock, monotonic, and vector clock
    """

    evaluation_id: str = Field(
        description="Unique identifier for the evaluation run"
    )
    trial_id: str = Field(
        description="Unique identifier for the trial"
    )
    trace_id: str = Field(
        description="Unique identifier for the trace (OTel compatible)"
    )
    span_id: str = Field(
        description="Unique identifier for this span"
    )
    parent_span_id: str | None = Field(
        default=None,
        description="Identifier of the parent span (null for root)"
    )
    sequence_id: int = Field(
        ge=0,
        description="Strictly monotonic sequence number per agent"
    )
    agent_id: str = Field(
        min_length=1,
        max_length=256,
        description="Identifier for the agent"
    )
    parent_agent_id: str | None = Field(
        default=None,
        description="Identifier for the parent agent (null for root)"
    )
    event_type: str = Field(
        min_length=1,
        description="Type of event (e.g., agent.input, model.invoke)"
    )
    timestamps: Timestamps = Field(
        description="Timestamp envelope with wall clock, monotonic, and vector clock"
    )

    @field_validator("evaluation_id", "trial_id", "trace_id", "span_id")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        """Validate that ID fields are valid UUIDs."""
        try:
            uuid.UUID(v)
        except ValueError as e:
            raise ValueError(f"Must be a valid UUID: {e}") from e
        return v

    @field_validator("parent_span_id")
    @classmethod
    def validate_optional_uuid(cls, v: str | None) -> str | None:
        """Validate optional UUID field."""
        if v is not None:
            try:
                uuid.UUID(v)
            except ValueError as e:
                raise ValueError(f"Must be a valid UUID: {e}") from e
        return v

    @classmethod
    def create(
        cls,
        event_type: str,
        agent_id: str,
        evaluation_id: str | None = None,
        trial_id: str | None = None,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        parent_agent_id: str | None = None,
        sequence_id: int = 0,
        vector_clock: VectorClock | None = None,
    ) -> IdentityEnvelope:
        """
        Create a new identity envelope with auto-generated IDs.

        Args:
            event_type: The type of event
            agent_id: The agent identifier
            evaluation_id: Optional evaluation ID (generated if not provided)
            trial_id: Optional trial ID (generated if not provided)
            trace_id: Optional trace ID (generated if not provided)
            parent_span_id: Optional parent span ID
            parent_agent_id: Optional parent agent ID
            sequence_id: The sequence number for this event
            vector_clock: Optional vector clock

        Returns:
            A new IdentityEnvelope instance
        """
        return cls(
            evaluation_id=evaluation_id or str(uuid.uuid4()),
            trial_id=trial_id or str(uuid.uuid4()),
            trace_id=trace_id or str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            parent_span_id=parent_span_id,
            sequence_id=sequence_id,
            agent_id=agent_id,
            parent_agent_id=parent_agent_id,
            event_type=event_type,
            timestamps=Timestamps.now(vector_clock),
        )
