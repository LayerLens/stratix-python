"""
A2A Task Lifecycle State Machine

Tracks A2A task state transitions:
  submitted → working → completed | failed | cancelled
                     → input_required → working → ...
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskState(str, Enum):
    """A2A task states."""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Valid state transitions
_VALID_TRANSITIONS: dict[TaskState, set[TaskState]] = {
    TaskState.SUBMITTED: {TaskState.WORKING, TaskState.FAILED, TaskState.CANCELLED},
    TaskState.WORKING: {
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.CANCELLED,
        TaskState.INPUT_REQUIRED,
    },
    TaskState.INPUT_REQUIRED: {TaskState.WORKING, TaskState.CANCELLED, TaskState.FAILED},
    TaskState.COMPLETED: set(),
    TaskState.FAILED: set(),
    TaskState.CANCELLED: set(),
}

# Terminal states
TERMINAL_STATES = frozenset({TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED})


class TaskStateMachine:
    """
    Tracks the lifecycle of a single A2A task.

    Validates state transitions and records transition history.
    """

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.state = TaskState.SUBMITTED
        self.history: list[tuple[TaskState, TaskState]] = []

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def transition(self, new_state: TaskState | str) -> bool:
        """
        Attempt a state transition.

        Args:
            new_state: Target state.

        Returns:
            True if transition was valid and applied, False otherwise.
        """
        if isinstance(new_state, str):
            try:
                new_state = TaskState(new_state)
            except ValueError:
                logger.warning(
                    "Task %s: unknown state '%s'", self.task_id, new_state,
                )
                return False

        if new_state not in _VALID_TRANSITIONS.get(self.state, set()):
            logger.warning(
                "Task %s: invalid transition %s → %s",
                self.task_id,
                self.state.value,
                new_state.value,
            )
            return False

        old_state = self.state
        self.state = new_state
        self.history.append((old_state, new_state))
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "state": self.state.value,
            "history": [(a.value, b.value) for a, b in self.history],
        }
