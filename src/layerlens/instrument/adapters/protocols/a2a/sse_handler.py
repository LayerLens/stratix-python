"""A2A SSE (Server-Sent Events) stream tap — late-attribution safe (A8 / D4).

The ``message/stream`` path delivers a sequence of ``TaskStatusUpdateEvent`` /
``TaskArtifactUpdateEvent`` frames over an (often async / cross-thread) stream.
The trace context (the collector) lives in a ContextVar resolved at EMIT time —
but a streamed completion can arrive long after the ``trace_context()`` block
that opened the stream has exited, on a different thread/task, where the
ContextVar reads its default (``None``) and the event is DROPPED or
mis-attributed to a later trace.

Fix (mirrors ``bedrock_agents.py`` ``_CompletionProxy``): SNAPSHOT the collector
at stream-OPEN (construction time, while the opener's context is still ambient)
and RE-ESTABLISH it for the duration of each emitted SSE event, so every frame
lands on the stream's own trace no matter who drains it. Each frame also drives
the task FSM and reads the event's REAL ``final`` / ``task_id`` / ``status.state``
so the lifecycle reaches the correct terminal state (instead of being recorded as
opaque ``protocol.stream.event`` blobs).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

from ...._events import A2A_TASK_UPDATED
from ...._context import _current_collector
from .task_lifecycle import TaskState, TaskStateMachine

log = logging.getLogger(__name__)


class A2ASSEHandler:
    """Tap an A2A ``message/stream`` SSE event stream for instrumentation.

    Construct this WHILE the opening trace context is ambient — it snapshots the
    current collector then, and re-establishes it around every emitted event.
    """

    def __init__(self, task_id: str, adapter: Any) -> None:
        self._task_id = str(task_id)
        self._adapter = adapter
        self._sequence = 0
        # SNAPSHOT at stream-open: whichever collector is ambient NOW is the
        # stream's trace. A late frame re-establishes THIS, not the (possibly
        # None / possibly a later) ambient one (A8 / D4).
        self._collector = _current_collector.get()
        self._fsm = TaskStateMachine(self._task_id)

    def process_event(self, event_data: Any) -> Any:
        """Observe one SSE frame, attributing it to the stream's trace."""
        collector = self._collector
        if collector is None:
            # Stream opened with no ambient collector — nothing to attribute to.
            self._sequence += 1
            return event_data

        token = _current_collector.set(collector)
        try:
            self._dispatch(event_data)
        except Exception:  # never break the customer stream
            log.warning("layerlens: error observing a2a SSE event", exc_info=True)
        finally:
            _current_collector.reset(token)
        self._sequence += 1
        return event_data

    def _dispatch(self, event_data: Any) -> None:
        state = _event_state(event_data)
        is_final = _event_final(event_data)
        task_id = _event_task_id(event_data) or self._task_id
        payload: dict[str, Any] = {
            "task_id": task_id,
            "sequence_in_stream": self._sequence,
            "payload_hash": _payload_hash(event_data),
        }
        if is_final is not None:
            payload["final"] = is_final
        if state is not None:
            payload["status"] = state
            self._record_transition(state)
        # Emit the lifecycle update on the stream's trace (NOT a generic opaque
        # stream blob): a streamed completion is now attributed to the task span
        # and the FSM reaches the right terminal state.
        self._adapter.emit(A2A_TASK_UPDATED, payload)

    def _record_transition(self, state: str) -> None:
        # Advance from WORKING (the live state) so the terminal transition is
        # valid; the FSM drops invalid/unknown states rather than mislabeling.
        if self._fsm.state is TaskState.SUBMITTED:
            self._fsm.transition(TaskState.WORKING)
        self._fsm.transition(state)

    def process_stream(self, events: Any) -> Any:
        for event in events:
            self.process_event(event)
        return events

    @property
    def events_processed(self) -> int:
        return self._sequence


def _event_state(event: Any) -> Optional[str]:
    """The TaskState string from a TaskStatusUpdateEvent (real obj or dict)."""
    status = event.get("status") if isinstance(event, dict) else getattr(event, "status", None)
    if status is None:
        return None
    state = status.get("state") if isinstance(status, dict) else getattr(status, "state", None)
    if state is None:
        return None
    return getattr(state, "value", str(state))


def _event_final(event: Any) -> Optional[bool]:
    final = event.get("final") if isinstance(event, dict) else getattr(event, "final", None)
    return bool(final) if final is not None else None


def _event_task_id(event: Any) -> Optional[str]:
    # camelCase on the wire (``taskId``) or snake_case on a pydantic object.
    if isinstance(event, dict):
        tid = event.get("taskId") or event.get("task_id")
    else:
        tid = getattr(event, "task_id", None)
    return str(tid) if tid else None


def _payload_hash(event: Any) -> str:
    return "sha256:" + hashlib.sha256(str(event).encode()).hexdigest()
