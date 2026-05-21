"""Sample: AG-UI middleware wrapping a synthetic SSE stream."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.protocols.agui import AGUIProtocolAdapter

SAMPLE_STREAM = [
    {"type": "TEXT_MESSAGE_CONTENT", "delta": "Hello "},
    {"type": "TEXT_MESSAGE_CONTENT", "delta": "world"},
    {"type": "TEXT_MESSAGE_END"},
    {"type": "TOOL_CALL_START", "toolCallId": "tc1", "toolCallName": "lookup"},
    {"type": "TOOL_CALL_ARGS", "toolCallId": "tc1", "delta": '{"q": "gravity'},
    {"type": "TOOL_CALL_ARGS", "toolCallId": "tc1", "delta": '"}'},
    {"type": "TOOL_CALL_END", "toolCallId": "tc1"},
    {"type": "STATE_SNAPSHOT", "state": {"turn": 1}},
]


def main() -> None:
    adapter = AGUIProtocolAdapter()
    with capture_events("agui"):
        for _ in adapter.wrap_stream(iter(SAMPLE_STREAM)):
            pass


if __name__ == "__main__":
    main()
