"""
STRATIX LangGraph State Adapter

Adapts LangGraph graph state for STRATIX state tracking.

Note: This adapter is designed specifically for LangGraph state management
and doesn't extend the base StateAdapter which is designed for STRATIX core integration.
"""

from __future__ import annotations

import json
import hashlib
from typing import Any, TypeVar
from dataclasses import dataclass

StateT = TypeVar("StateT")


@dataclass
class StateSnapshot:
    """Snapshot of graph state at a point in time."""

    state: dict[str, Any]
    hash: str
    timestamp_ns: int


class LangGraphStateAdapter:
    """
    State adapter for LangGraph graph state.

    Captures state snapshots at node boundaries and detects mutations.

    Usage:
        adapter = LangGraphStateAdapter()

        # Before node
        before_snapshot = adapter.snapshot(state)

        # After node
        after_snapshot = adapter.snapshot(state)

        # Check for changes
        if adapter.has_changed(before_snapshot, after_snapshot):
            changes = adapter.diff(before_snapshot, after_snapshot)
    """

    def __init__(
        self, include_keys: list[str] | None = None, exclude_keys: list[str] | None = None
    ) -> None:
        """
        Initialize the state adapter.

        Args:
            include_keys: Only track these keys (if specified)
            exclude_keys: Exclude these keys from tracking
        """
        self._include_keys = set(include_keys) if include_keys else None
        self._exclude_keys = set(exclude_keys) if exclude_keys else set()

    def snapshot(self, state: Any) -> StateSnapshot:
        """
        Create a snapshot of the current state.

        Args:
            state: LangGraph state (typically a dict or TypedDict)

        Returns:
            StateSnapshot with hash for comparison
        """
        import time

        # Convert state to dictionary
        state_dict = self._to_dict(state)

        # Filter keys if configured
        filtered_state = self._filter_state(state_dict)

        # Compute hash
        state_hash = self._compute_hash(filtered_state)

        return StateSnapshot(
            state=filtered_state,
            hash=state_hash,
            timestamp_ns=time.time_ns(),
        )

    def has_changed(self, before: StateSnapshot, after: StateSnapshot) -> bool:
        """
        Check if state has changed between snapshots.

        Args:
            before: Snapshot before operation
            after: Snapshot after operation

        Returns:
            True if state changed
        """
        return before.hash != after.hash

    def diff(self, before: StateSnapshot, after: StateSnapshot) -> dict[str, Any]:
        """
        Compute the difference between two snapshots.

        Args:
            before: Snapshot before operation
            after: Snapshot after operation

        Returns:
            Dictionary describing changes:
            {
                "added": {"key": value},
                "removed": {"key": value},
                "modified": {"key": {"before": old, "after": new}}
            }
        """
        added = {}
        removed = {}
        modified = {}

        before_keys = set(before.state.keys())
        after_keys = set(after.state.keys())

        # Added keys
        for key in after_keys - before_keys:
            added[key] = after.state[key]

        # Removed keys
        for key in before_keys - after_keys:
            removed[key] = before.state[key]

        # Modified keys
        for key in before_keys & after_keys:
            if before.state[key] != after.state[key]:
                modified[key] = {
                    "before": before.state[key],
                    "after": after.state[key],
                }

        return {
            "added": added,
            "removed": removed,
            "modified": modified,
        }

    def get_hash(self, state: Any) -> str:
        """
        Compute hash of state without creating full snapshot.

        The returned value is prefixed with ``sha256:`` so it can be
        passed directly to :class:`AgentStateChangeEvent.create` (whose
        ``before_hash`` / ``after_hash`` Pydantic validators require the
        algorithm prefix per spec ``02-event-schema-spec.md``). Callers
        that need the raw 64-char hex digest (for example, for byte-wise
        comparison with an externally produced digest) should use
        :meth:`get_hash_unprefixed`.

        Args:
            state: LangGraph state

        Returns:
            Hash string in the canonical ``sha256:<64-hex>`` form.
        """
        state_dict = self._to_dict(state)
        filtered = self._filter_state(state_dict)
        return self._compute_hash(filtered)

    def get_hash_unprefixed(self, state: Any) -> str:
        """
        Compute hash of state and return the bare 64-char hex digest.

        Provided for callers that need the unprefixed SHA-256 hex value
        (for byte-wise comparison or interoperability with consumers
        that supply their own algorithm prefix). Internal LangGraph
        adapter code paths use :meth:`get_hash` so the result is
        directly accepted by :class:`AgentStateChangeEvent.create`.

        Args:
            state: LangGraph state

        Returns:
            Bare 64-character hex SHA-256 digest (no ``sha256:`` prefix).
        """
        state_dict = self._to_dict(state)
        filtered = self._filter_state(state_dict)
        digest = self._compute_hash(filtered)
        # ``_compute_hash`` always returns the prefixed form; strip it.
        return digest.removeprefix("sha256:")

    def _to_dict(self, state: Any) -> dict[str, Any]:
        """Convert state to dictionary."""
        if isinstance(state, dict):
            return dict(state)
        elif hasattr(state, "__dict__"):
            return dict(state.__dict__)
        elif hasattr(state, "_asdict"):  # NamedTuple
            return state._asdict()  # type: ignore[no-any-return]
        else:
            # Try to treat as dict-like
            try:
                return dict(state)
            except (TypeError, ValueError):
                return {"__value__": state}

    def _filter_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Apply include/exclude filters."""
        if self._include_keys:
            state = {k: v for k, v in state.items() if k in self._include_keys}

        if self._exclude_keys:
            state = {k: v for k, v in state.items() if k not in self._exclude_keys}

        return state

    def _compute_hash(self, state: dict[str, Any]) -> str:
        """Compute SHA-256 hash of state.

        Returns the canonical ``sha256:<64-hex>`` form so the value is
        directly accepted by :class:`AgentStateChangeEvent` Pydantic
        validators (spec ``02-event-schema-spec.md`` requires the
        algorithm prefix). Snapshot equality comparisons in
        :meth:`has_changed` are unaffected — both sides are prefixed
        identically.
        """
        # Canonical JSON serialization
        try:
            serialized = json.dumps(state, sort_keys=True, default=str)
        except TypeError:
            # Fallback for non-serializable objects
            serialized = str(state)

        return f"sha256:{hashlib.sha256(serialized.encode()).hexdigest()}"


class MessageListAdapter(LangGraphStateAdapter):
    """
    Specialized adapter for LangGraph message-based state.

    LangGraph commonly uses a messages list in state.
    This adapter optimizes tracking for message append patterns.
    """

    def __init__(self, message_key: str = "messages") -> None:
        """
        Initialize the message list adapter.

        Args:
            message_key: Key in state that contains messages list
        """
        super().__init__()  # Initialize parent with defaults
        self._message_key = message_key
        self._last_message_count = 0

    def snapshot(self, state: Any) -> StateSnapshot:
        """Create snapshot with message count optimization."""
        snapshot = LangGraphStateAdapter.snapshot(self, state)

        # Track message count for efficient change detection
        state_dict = self._to_dict(state)
        if self._message_key in state_dict:
            messages = state_dict[self._message_key]
            if isinstance(messages, list):
                self._last_message_count = len(messages)

        return snapshot

    def get_new_messages(self, before: StateSnapshot, after: StateSnapshot) -> list[Any]:
        """
        Get messages added between snapshots.

        Args:
            before: Snapshot before
            after: Snapshot after

        Returns:
            List of new messages
        """
        before_messages = before.state.get(self._message_key, [])
        after_messages = after.state.get(self._message_key, [])

        if not isinstance(before_messages, list) or not isinstance(after_messages, list):
            return []

        # Assume messages are appended, not inserted
        if len(after_messages) > len(before_messages):
            return after_messages[len(before_messages) :]

        return []
