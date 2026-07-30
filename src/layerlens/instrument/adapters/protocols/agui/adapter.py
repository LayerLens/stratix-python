"""AG-UI (Agent-User Interface) protocol adapter.

Instruments the CopilotKit-style agent↔frontend SSE stream. Designed to sit
as middleware around an SSE response stream without modifying the agent or
frontend. Observes events, reconstructs the textual message buffer, tracks
tool-call fragments, and emits ``agui.state`` / ``agui.message`` /
``agui.tool_call``.
"""

from __future__ import annotations

import json
import uuid
import logging
from typing import Any, Dict, Callable, Iterator, AsyncIterator

from ...._events import AGUI_STATE, AGENT_ERROR, AGUI_MESSAGE, AGUI_TOOL_CALL, PROTOCOL_STREAM_EVENT
from .event_mapper import map_agui_to_stratix, build_run_error_payload
from .state_handler import StateDeltaHandler
from .._base_protocol import BaseProtocolAdapter

log = logging.getLogger(__name__)


class AGUIProtocolAdapter(BaseProtocolAdapter):
    PROTOCOL = "agui"
    PROTOCOL_VERSION = "0.1.0"

    def __init__(self, *, capture_config: Any = None) -> None:
        super().__init__(capture_config=capture_config)
        self._state_handler = StateDeltaHandler()

    def connect(self, target: Any = None, **kwargs: Any) -> Any:  # noqa: ARG002
        self._client = target
        # Common attach points exposed by CopilotKit-compatible runtimes.
        for attr in ("dispatch_event", "emit_event", "publish"):
            if hasattr(target, attr):
                orig = getattr(target, attr)
                self._originals[attr] = orig
                setattr(target, attr, self._wrap_event_dispatch(orig))
        return target

    # --- middleware wrappers ---

    def wrap_stream(self, stream: Iterator[Any]) -> Iterator[Any]:
        """Wrap a sync SSE iterator; emit telemetry as events pass through."""
        state = _StreamState()
        for event in stream:
            self._observe(event, state)
            yield event
        self._flush(state)

    async def wrap_async_stream(self, stream: AsyncIterator[Any]) -> AsyncIterator[Any]:
        """Wrap an async SSE iterator without interrupting the pass-through."""
        state = _StreamState()
        async for event in stream:
            self._observe(event, state)
            yield event
        self._flush(state)

    def _wrap_event_dispatch(self, original: Callable[..., Any]) -> Callable[..., Any]:
        adapter = self
        state = _StreamState()

        def wrapped(event: Any, *args: Any, **kwargs: Any) -> Any:
            adapter._observe(event, state)
            return original(event, *args, **kwargs)

        return wrapped

    # --- event inspection ---

    def _observe(self, event: Any, state: "_StreamState") -> None:
        etype = _event_type(event)
        if etype == "TEXT_MESSAGE_CONTENT":
            # Key the buffer by the protocol's messageId (exactly as tool-call
            # fragments below are keyed by toolCallId). connect() shares ONE
            # _StreamState across every dispatched event, so a single shared
            # scalar accumulator would cross-contaminate concurrent/interleaved
            # dispatch sessions (run A's deltas bleed into run B's message and
            # run B's own message flushes empty). Per-messageId buffering isolates
            # them; a stream that omits messageId falls back to a single key
            # (sequential single-message streams stay correct).
            mid = _event_field(event, "messageId")
            state.text_buffers[mid] = (state.text_buffers.get(mid) or "") + (_event_field(event, "delta") or "")
            return
        if etype == "TEXT_MESSAGE_END":
            mid = _event_field(event, "messageId")
            self.emit(AGUI_MESSAGE, {"text": state.text_buffers.pop(mid, "")})
            return
        if etype == "TOOL_CALL_START":
            tc_id = _event_field(event, "toolCallId") or uuid.uuid4().hex[:16]
            state.tool_calls[tc_id] = {
                "tool_name": _event_field(event, "toolCallName"),
                "arguments": "",
            }
            return
        if etype == "TOOL_CALL_ARGS":
            tc_id = _event_field(event, "toolCallId")
            if tc_id and tc_id in state.tool_calls:
                state.tool_calls[tc_id]["arguments"] += _event_field(event, "delta") or ""
            return
        if etype == "TOOL_CALL_END":
            tc_id = _event_field(event, "toolCallId")
            if tc_id and tc_id in state.tool_calls:
                call = state.tool_calls.pop(tc_id)
                try:
                    call["arguments"] = json.loads(call["arguments"])
                except (ValueError, TypeError):
                    pass
                self.emit(AGUI_TOOL_CALL, call)
            return
        if etype == "STATE_SNAPSHOT":
            snapshot = _event_field(event, "state") or {}
            before_hash, after_hash = self._state_handler.apply_snapshot(snapshot if isinstance(snapshot, dict) else {})
            self.emit(
                AGUI_STATE,
                {
                    "state_event": etype,
                    "state": snapshot,
                    "before_hash": before_hash,
                    "after_hash": after_hash,
                },
            )
            return
        if etype == "RUN_ERROR":
            # A run failure surfaces as agent.error (not lifecycle stream
            # telemetry) so the trace's derived status is error, not completed.
            self.emit(
                AGENT_ERROR,
                build_run_error_payload(_event_field(event, "message"), _event_field(event, "code")),
            )
            return
        if etype == "STATE_DELTA":
            operations = _event_field(event, "delta") or []
            ops = operations if isinstance(operations, list) else []
            before_hash, after_hash = self._state_handler.apply_delta(ops)
            self.emit(
                AGUI_STATE,
                {
                    "state_event": etype,
                    "operations": ops,
                    "before_hash": before_hash,
                    "after_hash": after_hash,
                },
            )
            return
        # Fallback: use the event-type → stratix-event map so lifecycle + step
        # events still produce telemetry instead of being silently dropped.
        if etype:
            mapping = map_agui_to_stratix(etype)
            self.emit(
                (
                    PROTOCOL_STREAM_EVENT
                    if mapping["stratix_event"] == "protocol.stream.event"
                    else mapping["stratix_event"]
                ),
                {
                    "agui_event": etype,
                    "category": mapping["category"],
                    "payload": event if isinstance(event, dict) else None,
                },
            )

    def _flush(self, state: "_StreamState") -> None:
        for text in state.text_buffers.values():
            if text:
                self.emit(AGUI_MESSAGE, {"text": text, "reason": "stream_closed"})
        for call in state.tool_calls.values():
            self.emit(AGUI_TOOL_CALL, {**call, "reason": "stream_closed"})


class _StreamState:
    __slots__ = ("text_buffers", "tool_calls")

    def __init__(self) -> None:
        # Buffers keyed by AG-UI messageId (None when a stream omits it); the
        # dict is what isolates concurrent dispatch sessions sharing one state.
        self.text_buffers: Dict[Any, str] = {}
        self.tool_calls: Dict[str, Dict[str, Any]] = {}


def _event_type(event: Any) -> str | None:
    if isinstance(event, dict):
        return event.get("type") or event.get("event")
    return getattr(event, "type", None) or getattr(event, "event", None)


def _event_field(event: Any, name: str) -> Any:
    if isinstance(event, dict):
        return event.get(name)
    return getattr(event, name, None)


def instrument_agui(target: Any) -> AGUIProtocolAdapter:
    from ..._registry import get, register

    existing = get("agui")
    if existing is not None:
        existing.disconnect()
    adapter = AGUIProtocolAdapter()
    adapter.connect(target)
    register("agui", adapter)
    return adapter


def uninstrument_agui() -> None:
    from ..._registry import unregister

    unregister("agui")
