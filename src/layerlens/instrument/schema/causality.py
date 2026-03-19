"""
STRATIX Causality Model - Sparse Vector Clocks

Implements sparse vector clocks for causal ordering across distributed
agents, tools, and graders as specified in Step 1.

NORMATIVE:
- Vector clocks are sparse and only include active participants
- On receiving remote context (handoff/tool response):
  - Merge vector clocks by max(participant)
  - Increment local participant for the receive event

Participant ID Format:
- agent:{agent_id}
- tool:{name}
- grader:{id}
"""

from __future__ import annotations

from typing import Iterator

from pydantic import BaseModel, Field


class SparseVectorClock(BaseModel):
    """
    Sparse vector clock implementation for distributed causal ordering.

    Vector clocks track logical time across multiple participants without
    requiring global coordination. Each participant maintains its own counter,
    and clocks are merged on communication boundaries.

    Properties:
    - Sparse: Only stores non-zero entries
    - Immutable operations: merge/increment return new instances
    - Causality detection: Can determine happens-before relationships
    """

    entries: dict[str, int] = Field(
        default_factory=dict,
        description="Mapping of participant ID to logical time"
    )

    class Config:
        frozen = True  # Make immutable

    @classmethod
    def empty(cls) -> SparseVectorClock:
        """Create an empty vector clock."""
        return cls(entries={})

    @classmethod
    def for_agent(cls, agent_id: str, initial_time: int = 1) -> SparseVectorClock:
        """Create a vector clock initialized for a single agent."""
        return cls(entries={f"agent:{agent_id}": initial_time})

    @classmethod
    def for_tool(cls, tool_name: str, initial_time: int = 1) -> SparseVectorClock:
        """Create a vector clock initialized for a tool."""
        return cls(entries={f"tool:{tool_name}": initial_time})

    @classmethod
    def for_grader(cls, grader_id: str, initial_time: int = 1) -> SparseVectorClock:
        """Create a vector clock initialized for a grader."""
        return cls(entries={f"grader:{grader_id}": initial_time})

    @staticmethod
    def make_agent_id(agent_id: str) -> str:
        """Create a participant ID for an agent."""
        return f"agent:{agent_id}"

    @staticmethod
    def make_tool_id(tool_name: str) -> str:
        """Create a participant ID for a tool."""
        return f"tool:{tool_name}"

    @staticmethod
    def make_grader_id(grader_id: str) -> str:
        """Create a participant ID for a grader."""
        return f"grader:{grader_id}"

    def get(self, participant_id: str) -> int:
        """
        Get the logical time for a participant.

        Args:
            participant_id: The participant ID

        Returns:
            The logical time (0 if not present)
        """
        return self.entries.get(participant_id, 0)

    def __getitem__(self, participant_id: str) -> int:
        """Get the logical time for a participant."""
        return self.get(participant_id)

    def increment(self, participant_id: str) -> SparseVectorClock:
        """
        Increment the clock for a participant.

        This is a send/local event operation.

        Args:
            participant_id: The participant ID to increment

        Returns:
            A new SparseVectorClock with the incremented value
        """
        new_entries = dict(self.entries)
        new_entries[participant_id] = new_entries.get(participant_id, 0) + 1
        return SparseVectorClock(entries=new_entries)

    def increment_agent(self, agent_id: str) -> SparseVectorClock:
        """Convenience method to increment an agent's clock."""
        return self.increment(self.make_agent_id(agent_id))

    def increment_tool(self, tool_name: str) -> SparseVectorClock:
        """Convenience method to increment a tool's clock."""
        return self.increment(self.make_tool_id(tool_name))

    def increment_grader(self, grader_id: str) -> SparseVectorClock:
        """Convenience method to increment a grader's clock."""
        return self.increment(self.make_grader_id(grader_id))

    def merge(self, other: SparseVectorClock) -> SparseVectorClock:
        """
        Merge two vector clocks by taking the maximum of each participant.

        NORMATIVE: On receiving remote context (handoff/tool response):
        - Merge vector clocks by max(participant)

        This is a receive operation. After merging, you typically also
        increment the local participant.

        Args:
            other: The other vector clock to merge with

        Returns:
            A new SparseVectorClock with merged values
        """
        merged = dict(self.entries)
        for participant_id, time in other.entries.items():
            merged[participant_id] = max(merged.get(participant_id, 0), time)
        return SparseVectorClock(entries=merged)

    def merge_and_increment(
        self,
        other: SparseVectorClock,
        local_participant_id: str
    ) -> SparseVectorClock:
        """
        Merge with another clock and increment local participant.

        NORMATIVE: On receiving remote context:
        - Merge vector clocks by max(participant)
        - Increment local participant for the receive event

        Args:
            other: The other vector clock to merge with
            local_participant_id: The local participant to increment

        Returns:
            A new SparseVectorClock with merged and incremented values
        """
        return self.merge(other).increment(local_participant_id)

    def happens_before(self, other: SparseVectorClock) -> bool:
        """
        Check if this clock happens-before another.

        Clock A happens-before Clock B if:
        1. For all participants, A[p] <= B[p]
        2. There exists at least one participant where A[p] < B[p]

        Args:
            other: The other vector clock to compare with

        Returns:
            True if this clock happens-before other
        """
        at_least_one_less = False

        # Check all participants in self
        for participant_id, self_time in self.entries.items():
            other_time = other.get(participant_id)
            if self_time > other_time:
                return False
            if self_time < other_time:
                at_least_one_less = True

        # Check for participants only in other (new participants)
        for participant_id, other_time in other.entries.items():
            if participant_id not in self.entries and other_time > 0:
                at_least_one_less = True

        return at_least_one_less

    def happens_after(self, other: SparseVectorClock) -> bool:
        """
        Check if this clock happens-after another.

        Equivalent to other.happens_before(self).
        """
        return other.happens_before(self)

    def is_concurrent_with(self, other: SparseVectorClock) -> bool:
        """
        Check if this clock is concurrent with another.

        Two clocks are concurrent if neither happens-before the other.
        This indicates potential race conditions or independent events.

        Args:
            other: The other vector clock to compare with

        Returns:
            True if the clocks are concurrent
        """
        return not self.happens_before(other) and not other.happens_before(self)

    def is_equal(self, other: SparseVectorClock) -> bool:
        """
        Check if two clocks are logically equal.

        Two clocks are equal if they have the same time for all participants.
        """
        all_participants = set(self.entries.keys()) | set(other.entries.keys())
        for participant_id in all_participants:
            if self.get(participant_id) != other.get(participant_id):
                return False
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SparseVectorClock):
            return NotImplemented
        return self.is_equal(other)

    def __lt__(self, other: SparseVectorClock) -> bool:
        """Less than means happens-before."""
        return self.happens_before(other)

    def __le__(self, other: SparseVectorClock) -> bool:
        """Less than or equal means happens-before or equal."""
        return self.happens_before(other) or self.is_equal(other)

    def __gt__(self, other: SparseVectorClock) -> bool:
        """Greater than means happens-after."""
        return self.happens_after(other)

    def __ge__(self, other: SparseVectorClock) -> bool:
        """Greater than or equal means happens-after or equal."""
        return self.happens_after(other) or self.is_equal(other)

    def participants(self) -> set[str]:
        """Get the set of all participants in this clock."""
        return set(self.entries.keys())

    def __iter__(self) -> Iterator[tuple[str, int]]:
        """Iterate over (participant_id, time) pairs."""
        return iter(self.entries.items())

    def __len__(self) -> int:
        """Get the number of participants in this clock."""
        return len(self.entries)

    def to_dict(self) -> dict[str, int]:
        """Convert to a plain dictionary."""
        return dict(self.entries)

    def model_dump(self, **kwargs) -> dict[str, int]:
        """Serialize to dictionary (returns just the entries for JSON compat)."""
        return self.to_dict()


class VectorClockManager:
    """
    Manager for tracking vector clocks across multiple participants.

    Provides a convenient interface for:
    - Registering participants
    - Emitting events (increment local)
    - Receiving events (merge and increment)
    - Querying causal relationships
    """

    def __init__(self, local_participant_id: str):
        """
        Initialize the vector clock manager.

        Args:
            local_participant_id: The participant ID for this node
        """
        self._local_id = local_participant_id
        self._clock = SparseVectorClock.empty()

    @property
    def local_participant_id(self) -> str:
        """Get the local participant ID."""
        return self._local_id

    @property
    def current_clock(self) -> SparseVectorClock:
        """Get the current vector clock."""
        return self._clock

    def emit(self) -> SparseVectorClock:
        """
        Record a local event (send/emit).

        Increments the local participant's logical time.

        Returns:
            The updated vector clock
        """
        self._clock = self._clock.increment(self._local_id)
        return self._clock

    def receive(self, remote_clock: SparseVectorClock) -> SparseVectorClock:
        """
        Record a receive event from a remote participant.

        Merges the remote clock and increments local time.

        Args:
            remote_clock: The vector clock from the remote event

        Returns:
            The updated vector clock
        """
        self._clock = self._clock.merge_and_increment(remote_clock, self._local_id)
        return self._clock

    def synchronize(self, other_clock: SparseVectorClock) -> SparseVectorClock:
        """
        Synchronize with another clock without incrementing.

        Use this for state synchronization without counting as a new event.

        Args:
            other_clock: The clock to synchronize with

        Returns:
            The updated vector clock
        """
        self._clock = self._clock.merge(other_clock)
        return self._clock

    def get_time(self, participant_id: str | None = None) -> int:
        """
        Get the logical time for a participant.

        Args:
            participant_id: The participant ID (defaults to local)

        Returns:
            The logical time
        """
        pid = participant_id or self._local_id
        return self._clock.get(pid)

    def is_after(self, other_clock: SparseVectorClock) -> bool:
        """Check if current clock happens-after another."""
        return self._clock.happens_after(other_clock)

    def is_before(self, other_clock: SparseVectorClock) -> bool:
        """Check if current clock happens-before another."""
        return self._clock.happens_before(other_clock)

    def is_concurrent(self, other_clock: SparseVectorClock) -> bool:
        """Check if current clock is concurrent with another."""
        return self._clock.is_concurrent_with(other_clock)
