"""A2A task state machine.

Validates transitions on ``submitted → working → completed | failed |
cancelled`` (with an ``input_required → working`` loop) so the A2A adapter
can drop or flag out-of-order status updates instead of emitting them blindly.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


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

TERMINAL_STATES = frozenset({TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED})


class TaskStateMachine:
    """Tracks and validates a single A2A task's state transitions."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.state: TaskState = TaskState.SUBMITTED
        self.history: list[tuple[TaskState, TaskState]] = []

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def transition(self, new_state: TaskState | str) -> bool:
        if isinstance(new_state, str):
            try:
                new_state = TaskState(new_state)
            except ValueError:
                log.warning("Task %s: unknown state %r", self.task_id, new_state)
                return False
        if new_state not in _VALID_TRANSITIONS.get(self.state, set()):
            log.warning(
                "Task %s: invalid transition %s → %s",
                self.task_id,
                self.state.value,
                new_state.value,
            )
            return False
        self.history.append((self.state, new_state))
        self.state = new_state
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "state": self.state.value,
            "history": [(a.value, b.value) for a, b in self.history],
        }
