"""A2A SSE (Server-Sent Events) stream tap.

Given a callable that emits events into the current collector, wraps an A2A
task-update SSE stream and produces a ``protocol.stream.event`` for each
event while passing the original payload through unchanged. Sequence numbers
and payload hashes are included so UIs can reconstruct event ordering.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable

log = logging.getLogger(__name__)


class A2ASSEHandler:
    """Tap an A2A SSE event stream for instrumentation."""

    def __init__(self, task_id: str, emit_fn: Callable[[str, dict[str, Any]], None]) -> None:
        self._task_id = task_id
        self._emit_fn = emit_fn
        self._sequence = 0

    def process_event(self, event_data: dict[str, Any]) -> dict[str, Any]:
        payload_str = str(event_data)
        payload_hash = "sha256:" + hashlib.sha256(payload_str.encode()).hexdigest()
        summary = payload_str if len(payload_str) <= 200 else payload_str[:200]
        self._emit_fn(
            "protocol.stream.event",
            {
                "protocol": "a2a",
                "task_id": self._task_id,
                "sequence_in_stream": self._sequence,
                "payload_hash": payload_hash,
                "payload_summary": summary,
            },
        )
        self._sequence += 1
        return event_data

    def process_stream(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for event in events:
            self.process_event(event)
        return events

    @property
    def events_processed(self) -> int:
        return self._sequence
