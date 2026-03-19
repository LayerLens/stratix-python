"""
STRATIX Layer 1 Events - Agent Inputs & Outputs

From Step 1 specification:
{
    "event_type": "agent.input | agent.output",
    "layer": "L1",
    "content": {
        "role": "human | system | agent",
        "message": "string"
    }
}
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role of the message sender."""
    HUMAN = "human"
    SYSTEM = "system"
    AGENT = "agent"


class MessageContent(BaseModel):
    """Content structure for L1 events."""
    role: MessageRole = Field(
        description="Role of the message sender"
    )
    message: str = Field(
        description="The message content"
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata about the message"
    )


class AgentInputEvent(BaseModel):
    """
    Layer 1 Event: Agent Input

    Represents an inbound message to the agent (from human or system).

    NORMATIVE: Must be emitted for every inbound human/system message.
    """
    event_type: str = Field(
        default="agent.input",
        description="Event type identifier"
    )
    layer: str = Field(
        default="L1",
        description="Layer identifier"
    )
    content: MessageContent = Field(
        description="Message content"
    )

    @classmethod
    def create(
        cls,
        message: str,
        role: MessageRole = MessageRole.HUMAN,
        metadata: dict[str, Any] | None = None,
    ) -> AgentInputEvent:
        """
        Create an agent input event.

        Args:
            message: The input message
            role: Role of the sender (default: human)
            metadata: Optional metadata

        Returns:
            AgentInputEvent instance
        """
        return cls(
            content=MessageContent(
                role=role,
                message=message,
                metadata=metadata,
            )
        )


class AgentOutputEvent(BaseModel):
    """
    Layer 1 Event: Agent Output

    Represents an outbound message from the agent.

    NORMATIVE: Must be emitted for every outbound agent message.
    """
    event_type: str = Field(
        default="agent.output",
        description="Event type identifier"
    )
    layer: str = Field(
        default="L1",
        description="Layer identifier"
    )
    content: MessageContent = Field(
        description="Message content"
    )

    @classmethod
    def create(
        cls,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentOutputEvent:
        """
        Create an agent output event.

        Args:
            message: The output message
            metadata: Optional metadata

        Returns:
            AgentOutputEvent instance
        """
        return cls(
            content=MessageContent(
                role=MessageRole.AGENT,
                message=message,
                metadata=metadata,
            )
        )
