"""A2A task state machine — the REAL a2a-sdk 1.1.0 ``TaskState`` vocabulary.

Values + spellings are pinned to ``a2a.compat.v0_3.types.TaskState`` (spec §4.1.3):
``submitted, working, input-required, completed, canceled, failed, rejected,
auth-required, unknown``. The two security-relevant terminals the old FSM was
missing — ``rejected`` (the peer DECLINED the task) and the ``auth-required``
handshake state — are modeled here so a rejected/failed task is never silently
mapped to ``completed`` (D5). Note the spec spellings: hyphen in
``input-required``/``auth-required`` and SINGLE-L ``canceled``.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"
    REJECTED = "rejected"
    AUTH_REQUIRED = "auth-required"
    UNKNOWN = "unknown"


# Spec §4.1.3 terminal states. ``rejected`` is terminal (the agent declined) —
# it is NOT a benign completion; ``_record_transition`` must reach it, not drop
# it as "unknown" and leave the FSM hanging at a non-terminal state.
TERMINAL_STATES = frozenset({TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED, TaskState.REJECTED})

# Any non-terminal state may reach any terminal one (a server can fail/reject/
# cancel from anywhere), plus the working <-> input-required/auth-required loops.
_NON_TERMINAL = (
    TaskState.SUBMITTED,
    TaskState.WORKING,
    TaskState.INPUT_REQUIRED,
    TaskState.AUTH_REQUIRED,
    TaskState.UNKNOWN,
)

_VALID_TRANSITIONS: dict[TaskState, set[TaskState]] = {
    TaskState.SUBMITTED: {
        TaskState.WORKING,
        TaskState.INPUT_REQUIRED,
        TaskState.AUTH_REQUIRED,
        *TERMINAL_STATES,
    },
    TaskState.WORKING: {
        TaskState.INPUT_REQUIRED,
        TaskState.AUTH_REQUIRED,
        *TERMINAL_STATES,
    },
    TaskState.INPUT_REQUIRED: {
        TaskState.WORKING,
        TaskState.AUTH_REQUIRED,
        *TERMINAL_STATES,
    },
    TaskState.AUTH_REQUIRED: {
        TaskState.WORKING,
        TaskState.INPUT_REQUIRED,
        *TERMINAL_STATES,
    },
    TaskState.UNKNOWN: {
        TaskState.WORKING,
        TaskState.INPUT_REQUIRED,
        TaskState.AUTH_REQUIRED,
        *TERMINAL_STATES,
    },
    TaskState.COMPLETED: set(),
    TaskState.FAILED: set(),
    TaskState.CANCELED: set(),
    TaskState.REJECTED: set(),
}


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
