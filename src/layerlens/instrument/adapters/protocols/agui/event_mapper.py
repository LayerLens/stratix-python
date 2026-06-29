"""Map AG-UI event types to layerlens event names.

AG-UI defines 16 event types across five categories (lifecycle, text,
tool, state, special). The adapter delegates to ``map_agui_to_stratix``
so new AG-UI event types only need a single line here to start flowing
through instrumentation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class AGUIEventType(str, Enum):
    """All known AG-UI event types."""

    # Lifecycle
    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"
    # Text messages
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
    # Tool calls
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_END = "TOOL_CALL_END"
    TOOL_CALL_RESULT = "TOOL_CALL_RESULT"
    # State
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"
    MESSAGES_SNAPSHOT = "MESSAGES_SNAPSHOT"
    # Special
    STEP_STARTED = "STEP_STARTED"
    STEP_FINISHED = "STEP_FINISHED"
    RAW = "RAW"


_AGUI_EVENT_MAP: dict[str, dict[str, str]] = {
    # Run lifecycle + state events map to SUPPRESSIBLE L6b stream types, not the
    # ALWAYS-ENABLED agent.state.change — so l6b_protocol_streams=False /
    # minimal() actually suppress them, and their raw event (carried under
    # data/payload) is stripped under capture_content=False (LAY-3578).
    "RUN_STARTED": {"stratix_event": "protocol.stream.event", "category": "lifecycle"},
    "RUN_FINISHED": {"stratix_event": "protocol.stream.event", "category": "lifecycle"},
    "RUN_ERROR": {"stratix_event": "protocol.stream.event", "category": "lifecycle"},
    "TEXT_MESSAGE_START": {
        "stratix_event": "protocol.stream.event",
        "category": "text",
    },
    "TEXT_MESSAGE_CONTENT": {
        "stratix_event": "protocol.stream.event",
        "category": "text",
    },
    "TEXT_MESSAGE_END": {"stratix_event": "protocol.stream.event", "category": "text"},
    "TOOL_CALL_START": {"stratix_event": "tool.call", "category": "tool"},
    "TOOL_CALL_ARGS": {"stratix_event": "protocol.stream.event", "category": "tool"},
    "TOOL_CALL_END": {"stratix_event": "protocol.stream.event", "category": "tool"},
    "TOOL_CALL_RESULT": {"stratix_event": "tool.call", "category": "tool"},
    # agui.state matches the adapter's explicit STATE_* branches (reconciles the
    # wrap_stream vs middleware dual-path) and is L6b-suppressible with content
    # keys covering state/operations/payload/data.
    "STATE_SNAPSHOT": {"stratix_event": "agui.state", "category": "state"},
    "STATE_DELTA": {"stratix_event": "agui.state", "category": "state"},
    "MESSAGES_SNAPSHOT": {"stratix_event": "agui.state", "category": "state"},
    "STEP_STARTED": {"stratix_event": "protocol.stream.event", "category": "special"},
    "STEP_FINISHED": {"stratix_event": "protocol.stream.event", "category": "special"},
    "RAW": {"stratix_event": "protocol.stream.event", "category": "special"},
}


def map_agui_to_stratix(agui_event_type: str) -> dict[str, Any]:
    """Return the ``{stratix_event, category}`` mapping for an AG-UI type."""
    return _AGUI_EVENT_MAP.get(
        agui_event_type,
        {"stratix_event": "protocol.stream.event", "category": "unknown"},
    )


def get_all_agui_event_types() -> list[str]:
    return list(_AGUI_EVENT_MAP.keys())
