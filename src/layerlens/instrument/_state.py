"""
STRATIX State Adapter

From Step 4 specification:
- StateAdapter: Enables state snapshot capture for agent.state.change events
- Adapters should capture before/after hashes for state changes
- Framework-specific adapters extend this base class
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from layerlens.instrument.schema.events.cross_cutting import AgentStateChangeEvent, StateType
from layerlens.instrument._context import get_current_context

if TYPE_CHECKING:
    from layerlens.instrument._core import STRATIX


class StateAdapter(ABC):
    """
    Base class for framework-specific state adapters.

    State adapters enable automatic capture of agent state changes
    for emit as agent.state.change events.

    Subclasses must implement:
    - snapshot(): Capture current state
    - get_state_keys(): Return list of tracked state keys
    """

    def __init__(self, stratix: "STRATIX"):
        """
        Initialize the state adapter.

        Args:
            stratix: The STRATIX instance
        """
        self._stratix = stratix
        self._last_snapshot: dict[str, Any] | None = None
        self._last_hash: str | None = None

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """
        Capture the current state.

        Returns:
            Dictionary representing the current state
        """
        pass

    @abstractmethod
    def get_state_keys(self) -> list[str]:
        """
        Get the list of state keys being tracked.

        Returns:
            List of state key names
        """
        pass

    def compute_hash(self, state: dict[str, Any]) -> str:
        """
        Compute a deterministic hash of the state.

        Args:
            state: The state dictionary

        Returns:
            SHA-256 hash of the state
        """
        # Canonical JSON serialization
        canonical = json.dumps(state, sort_keys=True, default=str)
        return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()

    def capture_change(self, trigger: str = "unknown") -> AgentStateChangeEvent | None:
        """
        Capture a state change if the state has changed.

        Args:
            trigger: What triggered this state change (e.g., "tool_call", "model_invoke")

        Returns:
            AgentStateChangeEvent if state changed, None otherwise
        """
        current = self.snapshot()
        current_hash = self.compute_hash(current)

        # Check if state actually changed
        if self._last_hash is not None and current_hash == self._last_hash:
            return None

        # Compute delta
        delta = self._compute_delta(self._last_snapshot, current)

        # Create event - use internal state type for adapter-tracked state
        before_hash = self._last_hash or "sha256:" + "0" * 64
        event = AgentStateChangeEvent.create(
            state_type=StateType.INTERNAL,
            before_hash=before_hash,
            after_hash=current_hash,
        )

        # Update last snapshot
        self._last_snapshot = current
        self._last_hash = current_hash

        return event

    def emit_change(self, trigger: str = "unknown") -> None:
        """
        Capture and emit a state change event if the state has changed.

        Args:
            trigger: What triggered this state change
        """
        ctx = get_current_context()
        if ctx is None:
            return

        event = self.capture_change(trigger)
        if event is not None:
            self._stratix._emit_event(ctx, event)

    def _compute_delta(
        self, before: dict[str, Any] | None, after: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Compute the delta between two states.

        Args:
            before: Previous state (None for initial)
            after: Current state

        Returns:
            Dictionary of changed keys with their new values
        """
        if before is None:
            return after

        delta = {}
        all_keys = set(before.keys()) | set(after.keys())

        for key in all_keys:
            before_val = before.get(key)
            after_val = after.get(key)
            if before_val != after_val:
                delta[key] = {"before": before_val, "after": after_val}

        return delta

    def initialize(self) -> None:
        """
        Initialize the adapter by taking an initial snapshot.

        Should be called when the trial starts.
        """
        self._last_snapshot = self.snapshot()
        self._last_hash = self.compute_hash(self._last_snapshot)


class DictStateAdapter(StateAdapter):
    """
    Simple state adapter for dictionary-based state.

    Useful for testing or simple agents that maintain state in a dict.
    """

    def __init__(self, stratix: "STRATIX", state_dict: dict[str, Any]):
        """
        Initialize with a reference to the state dictionary.

        Args:
            stratix: The STRATIX instance
            state_dict: The dictionary to track
        """
        super().__init__(stratix)
        self._state_dict = state_dict

    def snapshot(self) -> dict[str, Any]:
        """Capture current state from the tracked dictionary."""
        # Deep copy to avoid mutation issues
        import copy
        return copy.deepcopy(self._state_dict)

    def get_state_keys(self) -> list[str]:
        """Get keys from the tracked dictionary."""
        return list(self._state_dict.keys())
