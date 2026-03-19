"""
AG-UI Event Type Mapper

Maps AG-UI event types to Stratix event types according to the
five AG-UI event categories: Lifecycle, Text Messages, Tool Calls,
State Management, and Special.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class AGUIEventType(str, Enum):
    """AG-UI event types."""
    # Lifecycle
    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"
    # Text Messages
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
    # Tool Calls
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_END = "TOOL_CALL_END"
    TOOL_CALL_RESULT = "TOOL_CALL_RESULT"
    # State Management
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"
    MESSAGES_SNAPSHOT = "MESSAGES_SNAPSHOT"
    # Special
    STEP_STARTED = "STEP_STARTED"
    STEP_FINISHED = "STEP_FINISHED"
    RAW = "RAW"


# AG-UI event type → Stratix mapping
_AGUI_EVENT_MAP: dict[str, dict[str, Any]] = {
    # Lifecycle → agent.state.change
    "RUN_STARTED": {"stratix_event": "agent.state.change", "category": "lifecycle"},
    "RUN_FINISHED": {"stratix_event": "agent.state.change", "category": "lifecycle"},
    "RUN_ERROR": {"stratix_event": "agent.state.change", "category": "lifecycle"},
    # Text Messages → protocol.stream.event (L6b gated)
    "TEXT_MESSAGE_START": {"stratix_event": "protocol.stream.event", "category": "text"},
    "TEXT_MESSAGE_CONTENT": {"stratix_event": "protocol.stream.event", "category": "text"},
    "TEXT_MESSAGE_END": {"stratix_event": "protocol.stream.event", "category": "text"},
    # Tool Calls → tool.call (L5a) + protocol.stream.event for streaming args
    "TOOL_CALL_START": {"stratix_event": "tool.call", "category": "tool"},
    "TOOL_CALL_ARGS": {"stratix_event": "protocol.stream.event", "category": "tool"},
    "TOOL_CALL_END": {"stratix_event": "protocol.stream.event", "category": "tool"},
    "TOOL_CALL_RESULT": {"stratix_event": "tool.call", "category": "tool"},
    # State Management → agent.state.change + protocol.stream.event
    "STATE_SNAPSHOT": {"stratix_event": "agent.state.change", "category": "state"},
    "STATE_DELTA": {"stratix_event": "agent.state.change", "category": "state"},
    "MESSAGES_SNAPSHOT": {"stratix_event": "agent.state.change", "category": "state"},
    # Special → protocol.stream.event
    "STEP_STARTED": {"stratix_event": "protocol.stream.event", "category": "special"},
    "STEP_FINISHED": {"stratix_event": "protocol.stream.event", "category": "special"},
    "RAW": {"stratix_event": "protocol.stream.event", "category": "special"},
}


def map_agui_to_stratix(agui_event_type: str) -> dict[str, Any]:
    """
    Map an AG-UI event type to its Stratix mapping.

    Args:
        agui_event_type: AG-UI event type string.

    Returns:
        Mapping dict with stratix_event and category keys.
        Returns a default mapping for unknown event types.
    """
    return _AGUI_EVENT_MAP.get(
        agui_event_type,
        {"stratix_event": "protocol.stream.event", "category": "unknown"},
    )


def get_all_agui_event_types() -> list[str]:
    """Return all known AG-UI event type strings."""
    return list(_AGUI_EVENT_MAP.keys())
