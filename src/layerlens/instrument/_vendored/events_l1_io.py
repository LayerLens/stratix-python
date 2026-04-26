"""Vendored snapshot of ``stratix.core.events.l1_io``.

Source: ``A:/github/layerlens/ateam/stratix/core/events/l1_io.py``
Source SHA: 7359c0e38d74e02aa1b27c34daef7a958abbd002

Compatibility shims applied for Python 3.9 + Pydantic 2:
- ``enum.StrEnum`` (added in Python 3.11) replaced with
  ``(str, Enum)`` mixin.
- PEP-604 union syntax (``X | None``) on Pydantic field annotations
  rewritten as ``Optional[X]``.

Updates require re-vendoring — see ``__init__.py`` for the workflow.
"""

# STRATIX Layer 1 Events - Agent Inputs & Outputs
#
# {
#     "event_type": "agent.input | agent.output",
#     "layer": "L1",
#     "content": {
#         "role": "human | system | agent",
#         "message": "string"
#     }
# }

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import Field, BaseModel


class MessageRole(str, Enum):
    """Role of the message sender."""

    HUMAN = "human"
    SYSTEM = "system"
    AGENT = "agent"


class MessageContent(BaseModel):
    """Content structure for L1 events."""

    role: MessageRole = Field(description="Role of the message sender")
    message: str = Field(description="The message content")
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Optional metadata about the message"
    )


class AgentInputEvent(BaseModel):
    """Layer 1 Event: Agent Input.

    Represents an inbound message to the agent (from human or system).

    NORMATIVE: Must be emitted for every inbound human/system message.
    """

    event_type: str = Field(default="agent.input", description="Event type identifier")
    layer: str = Field(default="L1", description="Layer identifier")
    content: MessageContent = Field(description="Message content")

    @classmethod
    def create(
        cls,
        message: str,
        role: MessageRole = MessageRole.HUMAN,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AgentInputEvent:
        """Create an agent input event."""
        return cls(
            content=MessageContent(
                role=role,
                message=message,
                metadata=metadata,
            )
        )


class AgentOutputEvent(BaseModel):
    """Layer 1 Event: Agent Output.

    Represents an outbound message from the agent.

    NORMATIVE: Must be emitted for every outbound agent message.
    """

    event_type: str = Field(default="agent.output", description="Event type identifier")
    layer: str = Field(default="L1", description="Layer identifier")
    content: MessageContent = Field(description="Message content")

    @classmethod
    def create(
        cls,
        message: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AgentOutputEvent:
        """Create an agent output event."""
        return cls(
            content=MessageContent(
                role=MessageRole.AGENT,
                message=message,
                metadata=metadata,
            )
        )


__all__ = [
    "MessageRole",
    "MessageContent",
    "AgentInputEvent",
    "AgentOutputEvent",
]
