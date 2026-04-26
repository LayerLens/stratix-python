"""Vendored snapshot of ``stratix.core.events.l5_tools``.

Source: ``A:/github/layerlens/ateam/stratix/core/events/l5_tools.py``
Source SHA: 7359c0e38d74e02aa1b27c34daef7a958abbd002

Compatibility shims applied for Python 3.9 + Pydantic 2:
- ``enum.StrEnum`` (added in Python 3.11) replaced with
  ``(str, Enum)`` mixin.
- PEP-604 union syntax (``X | None``) on Pydantic field annotations
  rewritten as ``Optional[X]``.

Updates require re-vendoring — see ``__init__.py`` for the workflow.
"""

# STRATIX Layer 5 Events - Tool/Action Execution
#
# Layer 5a - Tool/Action Execution:
# {
#     "event_type": "tool.call",
#     "layer": "L5a",
#     "tool": {
#         "name": "string",
#         "version": "string",
#         "integration": "library | service | agent"
#     },
#     "input": { },
#     "output": { }
# }
#
# Layer 5b - Tool Business Logic:
# {
#     "event_type": "tool.logic",
#     "layer": "L5b",
#     "logic": {
#         "description": "string",
#         "rules": ["rule1", "rule2"]
#     }
# }
#
# Layer 5c - Tool Environment:
# {
#     "event_type": "tool.environment",
#     "layer": "L5c",
#     "environment": {
#         "api": "uri",
#         "permissions": ["scope1"]
#     }
# }

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import Field, BaseModel


class IntegrationType(str, Enum):
    """Type of tool integration."""

    LIBRARY = "library"
    SCRIPT = "script"
    SERVICE = "service"
    AGENT = "agent"


class ToolInfo(BaseModel):
    """Tool information for L5a events."""

    name: str = Field(description="Tool name")
    version: str = Field(description="Tool version (or 'unavailable')")
    integration: IntegrationType = Field(description="Type of integration")


class ToolCallEvent(BaseModel):
    """Layer 5a Event: Tool Call.

    Represents a tool/action invocation.

    NORMATIVE:
    - Must be emitted for every tool/action invocation
    - tool.call must include integration type
    - tool version required (or explicitly 'unavailable')
    """

    event_type: str = Field(default="tool.call", description="Event type identifier")
    layer: str = Field(default="L5a", description="Layer identifier")
    tool: ToolInfo = Field(description="Tool information")
    input: dict[str, Any] = Field(default_factory=dict, description="Tool input parameters")
    output: Optional[dict[str, Any]] = Field(
        default=None, description="Tool output (null if error/pending)"
    )
    error: Optional[str] = Field(default=None, description="Error message if tool failed")
    latency_ms: Optional[float] = Field(
        default=None, ge=0, description="Execution latency in milliseconds"
    )

    @classmethod
    def create(
        cls,
        name: str,
        version: str = "unavailable",
        integration: IntegrationType = IntegrationType.LIBRARY,
        input_data: Optional[dict[str, Any]] = None,
        output_data: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> ToolCallEvent:
        """Create a tool call event."""
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

    description: str = Field(description="Description of the business logic")
    rules: list[str] = Field(default_factory=list, description="Business rules applied")


class ToolLogicEvent(BaseModel):
    """Layer 5b Event: Tool Business Logic.

    Represents the business logic applied during tool execution.
    """

    event_type: str = Field(default="tool.logic", description="Event type identifier")
    layer: str = Field(default="L5b", description="Layer identifier")
    logic: ToolLogicInfo = Field(description="Business logic information")

    @classmethod
    def create(
        cls,
        description: str,
        rules: Optional[list[str]] = None,
    ) -> ToolLogicEvent:
        """Create a tool logic event."""
        return cls(
            logic=ToolLogicInfo(
                description=description,
                rules=rules or [],
            )
        )


class ToolEnvironmentInfo(BaseModel):
    """Tool environment information for L5c events."""

    api: Optional[str] = Field(default=None, description="API endpoint URI")
    permissions: list[str] = Field(default_factory=list, description="Required permissions/scopes")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Additional environment configuration"
    )


class ToolEnvironmentEvent(BaseModel):
    """Layer 5c Event: Tool Environment.

    Represents the execution environment for a tool.
    """

    event_type: str = Field(default="tool.environment", description="Event type identifier")
    layer: str = Field(default="L5c", description="Layer identifier")
    environment: ToolEnvironmentInfo = Field(description="Tool environment information")

    @classmethod
    def create(
        cls,
        api: Optional[str] = None,
        permissions: Optional[list[str]] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> ToolEnvironmentEvent:
        """Create a tool environment event."""
        return cls(
            environment=ToolEnvironmentInfo(
                api=api,
                permissions=permissions or [],
                config=config or {},
            )
        )


__all__ = [
    "IntegrationType",
    "ToolInfo",
    "ToolCallEvent",
    "ToolLogicInfo",
    "ToolLogicEvent",
    "ToolEnvironmentInfo",
    "ToolEnvironmentEvent",
]
