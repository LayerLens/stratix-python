"""
STRATIX Layer 5 Events - Tool/Action Execution

From Step 1 specification:

Layer 5a - Tool/Action Execution:
{
    "event_type": "tool.call",
    "layer": "L5a",
    "tool": {
        "name": "string",
        "version": "string",
        "integration": "library | service | agent"
    },
    "input": { },
    "output": { }
}

Layer 5b - Tool Business Logic:
{
    "event_type": "tool.logic",
    "layer": "L5b",
    "logic": {
        "description": "string",
        "rules": ["rule1", "rule2"]
    }
}

Layer 5c - Tool Environment:
{
    "event_type": "tool.environment",
    "layer": "L5c",
    "environment": {
        "api": "uri",
        "permissions": ["scope1"]
    }
}
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class IntegrationType(str, Enum):
    """Type of tool integration."""
    LIBRARY = "library"
    SCRIPT = "script"
    SERVICE = "service"
    AGENT = "agent"


class ToolInfo(BaseModel):
    """Tool information for L5a events."""
    name: str = Field(
        description="Tool name"
    )
    version: str = Field(
        description="Tool version (or 'unavailable')"
    )
    integration: IntegrationType = Field(
        description="Type of integration"
    )


class ToolCallEvent(BaseModel):
    """
    Layer 5a Event: Tool Call

    Represents a tool/action invocation.

    NORMATIVE:
    - Must be emitted for every tool/action invocation
    - tool.call must include integration type
    - tool version required (or explicitly 'unavailable')
    """
    event_type: str = Field(
        default="tool.call",
        description="Event type identifier"
    )
    layer: str = Field(
        default="L5a",
        description="Layer identifier"
    )
    tool: ToolInfo = Field(
        description="Tool information"
    )
    input: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool input parameters"
    )
    output: dict[str, Any] | None = Field(
        default=None,
        description="Tool output (null if error/pending)"
    )
    error: str | None = Field(
        default=None,
        description="Error message if tool failed"
    )
    latency_ms: float | None = Field(
        default=None,
        ge=0,
        description="Execution latency in milliseconds"
    )

    @classmethod
    def create(
        cls,
        name: str,
        version: str = "unavailable",
        integration: IntegrationType = IntegrationType.LIBRARY,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        error: str | None = None,
        latency_ms: float | None = None,
    ) -> ToolCallEvent:
        """
        Create a tool call event.

        Args:
            name: Tool name
            version: Tool version
            integration: Integration type
            input_data: Tool input parameters
            output_data: Tool output
            error: Error message if failed
            latency_ms: Execution latency

        Returns:
            ToolCallEvent instance
        """
        return cls(
            tool=ToolInfo(
                name=name,
                version=version,
                integration=integration,
            ),
            input=input_data or {},
            output=output_data,
            error=error,
            latency_ms=latency_ms,
        )


class ToolLogicInfo(BaseModel):
    """Tool business logic information for L5b events."""
    description: str = Field(
        description="Description of the business logic"
    )
    rules: list[str] = Field(
        default_factory=list,
        description="Business rules applied"
    )


class ToolLogicEvent(BaseModel):
    """
    Layer 5b Event: Tool Business Logic

    Represents the business logic applied during tool execution.
    """
    event_type: str = Field(
        default="tool.logic",
        description="Event type identifier"
    )
    layer: str = Field(
        default="L5b",
        description="Layer identifier"
    )
    logic: ToolLogicInfo = Field(
        description="Business logic information"
    )

    @classmethod
    def create(
        cls,
        description: str,
        rules: list[str] | None = None,
    ) -> ToolLogicEvent:
        """
        Create a tool logic event.

        Args:
            description: Description of the business logic
            rules: List of rules applied

        Returns:
            ToolLogicEvent instance
        """
        return cls(
            logic=ToolLogicInfo(
                description=description,
                rules=rules or [],
            )
        )


class ToolEnvironmentInfo(BaseModel):
    """Tool environment information for L5c events."""
    api: str | None = Field(
        default=None,
        description="API endpoint URI"
    )
    permissions: list[str] = Field(
        default_factory=list,
        description="Required permissions/scopes"
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional environment configuration"
    )


class ToolEnvironmentEvent(BaseModel):
    """
    Layer 5c Event: Tool Environment

    Represents the execution environment for a tool.
    """
    event_type: str = Field(
        default="tool.environment",
        description="Event type identifier"
    )
    layer: str = Field(
        default="L5c",
        description="Layer identifier"
    )
    environment: ToolEnvironmentInfo = Field(
        description="Tool environment information"
    )

    @classmethod
    def create(
        cls,
        api: str | None = None,
        permissions: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> ToolEnvironmentEvent:
        """
        Create a tool environment event.

        Args:
            api: API endpoint URI
            permissions: Required permissions
            config: Additional configuration

        Returns:
            ToolEnvironmentEvent instance
        """
        return cls(
            environment=ToolEnvironmentInfo(
                api=api,
                permissions=permissions or [],
                config=config or {},
            )
        )
