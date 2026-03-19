"""
A2A SSE Stream Handler

Captures and forwards A2A SSE (Server-Sent Events) stream events,
translating them to Stratix protocol.stream.event events.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class A2ASSEHandler:
    """
    Handles A2A SSE streams for task update subscriptions.

    Wraps an SSE event generator, emitting protocol.stream.event for each
    event received while forwarding all events unchanged to the consumer.
    """

    def __init__(
        self,
        task_id: str,
        emit_fn: Callable[..., None],
    ) -> None:
        self._task_id = task_id
        self._emit_fn = emit_fn
        self._sequence = 0

    def process_event(self, event_data: dict[str, Any]) -> dict[str, Any]:
        """
        Process a single SSE event.

        Emits a protocol.stream.event and returns the event unchanged.

        Args:
            event_data: The SSE event payload (parsed JSON).

        Returns:
            The original event_data, unmodified.
        """
        from layerlens.instrument.schema.events.protocol import ProtocolStreamEvent

        payload_str = str(event_data)
        payload_hash = f"sha256:{hashlib.sha256(payload_str.encode()).hexdigest()}"

        stream_event = ProtocolStreamEvent.create(
            protocol="a2a",
            sequence_in_stream=self._sequence,
            payload_hash=payload_hash,
            payload_summary=payload_str[:200] if len(payload_str) > 200 else payload_str,
        )
        self._emit_fn(stream_event)
        self._sequence += 1

        return event_data

    def process_stream(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Process an entire SSE event stream.

        Args:
            events: Ordered list of SSE event payloads.

        Returns:
            The original events list, unmodified.
        """
        for event in events:
            self.process_event(event)
        return events

    @property
    def events_processed(self) -> int:
        return self._sequence
