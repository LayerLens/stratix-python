"""Concurrency / dispatch-session isolation floor for the AG-UI adapter
(``agui.conc`` cell, partial -> solid).

``wrap_stream`` / ``wrap_async_stream`` create a fresh ``_StreamState`` per call,
so a single stream is already isolated. But ``connect()`` attaches
``_wrap_event_dispatch``, whose ``wrapped`` closure captures ONE ``_StreamState``
at connect() time and shares it across EVERY dispatched event. A CopilotKit-style
runtime dispatches events for many concurrent agent runs through that one attach
point, so two interleaved dispatch sessions share ``state.text_buffer`` and their
``TEXT_MESSAGE_CONTENT`` deltas cross-contaminate: the AG-UI protocol identifies
every text event by ``messageId`` (exactly as tool-call events carry
``toolCallId``, which the adapter already keys on), yet the shared scalar buffer
ignores it — so run A's text bleeds into run B's ``agui.message`` and run B's own
message is emitted empty.

This is the same class of latent race W3 fixed for ms_agent_framework / langgraph
(shared per-adapter handoff state under concurrency). The fix keys text buffering
by ``messageId`` — mirroring the existing per-``toolCallId`` tool-call keying — so
the shared dispatch state is naturally isolated per protocol message.

Everything here is offline: the only mock is the trace collector's upload client;
the adapter, its dispatch wrapping, and the AG-UI events are real.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

from layerlens.instrument._context import _current_collector
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.protocols.agui.adapter import AGUIProtocolAdapter

_FULL = CaptureConfig(capture_content=True)


def _messages(events: List[Dict[str, Any]]) -> List[str]:
    return [e["payload"]["text"] for e in events if e["event_type"] == "agui.message"]


def _tool_calls(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [e["payload"] for e in events if e["event_type"] == "agui.tool_call"]


def _connected_dispatch(config: CaptureConfig) -> tuple[Any, TraceCollector]:
    """Return a runtime whose ``dispatch_event`` is the REAL adapter-wrapped
    attach point, plus the ambient collector its emits land in."""
    collector = TraceCollector(object(), config)
    runtime = SimpleNamespace(dispatched=[])

    def dispatch_event(event: Any, *args: Any, **kwargs: Any) -> Any:
        runtime.dispatched.append(event)
        return event

    runtime.dispatch_event = dispatch_event
    adapter = AGUIProtocolAdapter(capture_config=config)
    adapter.connect(runtime)  # wraps runtime.dispatch_event via _wrap_event_dispatch
    return runtime, collector


# Two independent AG-UI agent runs, each a real protocol message (unique
# ``messageId``) + a real tool call (unique ``toolCallId``), dispatched
# INTERLEAVED through the one connected attach point.
_SESSION_A_TEXT = "Alpha assistant answer from run A"
_SESSION_B_TEXT = "Beta assistant answer from run B"


def _interleaved_events() -> List[Dict[str, Any]]:
    return [
        {"type": "TEXT_MESSAGE_START", "messageId": "msg-a"},
        {"type": "TEXT_MESSAGE_START", "messageId": "msg-b"},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "msg-a", "delta": "Alpha assistant "},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "msg-b", "delta": "Beta assistant "},
        {"type": "TOOL_CALL_START", "toolCallId": "tc-a", "toolCallName": "search_a"},
        {"type": "TOOL_CALL_START", "toolCallId": "tc-b", "toolCallName": "search_b"},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "msg-a", "delta": "answer from run A"},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "msg-b", "delta": "answer from run B"},
        {"type": "TOOL_CALL_ARGS", "toolCallId": "tc-a", "delta": '{"q": "alpha"}'},
        {"type": "TOOL_CALL_ARGS", "toolCallId": "tc-b", "delta": '{"q": "beta"}'},
        {"type": "TEXT_MESSAGE_END", "messageId": "msg-a"},
        {"type": "TOOL_CALL_END", "toolCallId": "tc-a"},
        {"type": "TEXT_MESSAGE_END", "messageId": "msg-b"},
        {"type": "TOOL_CALL_END", "toolCallId": "tc-b"},
    ]


class TestDispatchSessionIsolation:
    def test_interleaved_dispatch_messages_do_not_cross_contaminate(self) -> None:
        runtime, collector = _connected_dispatch(_FULL)
        token = _current_collector.set(collector)
        try:
            for ev in _interleaved_events():
                runtime.dispatch_event(ev)
        finally:
            _current_collector.reset(token)

        msgs = _messages(collector.events)

        # Each run's message is emitted with ONLY its own text (bite today: run A's
        # message carries both runs' deltas concatenated and run B's is empty).
        assert _SESSION_A_TEXT in msgs, f"run A's message lost/contaminated; got {msgs!r}"
        assert _SESSION_B_TEXT in msgs, f"run B's message lost/contaminated; got {msgs!r}"

        # No emitted message may mix the two runs' text (the shared-buffer race).
        for m in msgs:
            assert not ("Alpha" in m and "Beta" in m), f"cross-contaminated agui.message: {m!r}"

        # And no empty message leaked (the drained-buffer symptom for the 2nd END).
        assert "" not in msgs, f"an empty agui.message leaked (drained shared buffer): {msgs!r}"

    def test_interleaved_dispatch_tool_calls_stay_isolated(self) -> None:
        runtime, collector = _connected_dispatch(_FULL)
        token = _current_collector.set(collector)
        try:
            for ev in _interleaved_events():
                runtime.dispatch_event(ev)
        finally:
            _current_collector.reset(token)

        calls = {c["tool_name"]: c["arguments"] for c in _tool_calls(collector.events)}
        # tool_calls is keyed by toolCallId, so the two runs' calls stay isolated;
        # this guards that the messageId fix does not regress that isolation.
        assert calls.get("search_a") == {"q": "alpha"}, f"run A tool-call args wrong: {calls!r}"
        assert calls.get("search_b") == {"q": "beta"}, f"run B tool-call args wrong: {calls!r}"

    def test_passthrough_is_preserved_under_interleave(self) -> None:
        # The wrapper must forward every event to the original dispatch_event
        # unchanged (observing must not swallow or reorder the runtime's stream).
        runtime, collector = _connected_dispatch(_FULL)
        token = _current_collector.set(collector)
        try:
            evs = _interleaved_events()
            for ev in evs:
                runtime.dispatch_event(ev)
        finally:
            _current_collector.reset(token)
        assert runtime.dispatched == evs, "dispatch wrapper dropped/reordered pass-through events"
