"""Aggregated re-exports of vendored ``stratix.core.events`` types.

Source: ``A:/github/layerlens/ateam/stratix/core/events/__init__.py``
Source SHA: 7359c0e38d74e02aa1b27c34daef7a958abbd002

Mirrors the surface that the langgraph and langchain framework adapters
import from ``stratix.core.events`` directly. Only the names that those
adapters actually reference at runtime are re-exported here — anything
else lives in the per-module vendored files.

Updates require re-vendoring — see ``__init__.py`` for the workflow.
"""

from __future__ import annotations

from layerlens.instrument._vendored.events_l1_io import (
    MessageRole,
    AgentInputEvent,
    AgentOutputEvent,
)
from layerlens.instrument._vendored.events_l2_code import (
    CodeInfo,
    AgentCodeEvent,
)
from layerlens.instrument._vendored.events_l3_model import ModelInvokeEvent
from layerlens.instrument._vendored.events_l5_tools import (
    ToolCallEvent,
    ToolLogicEvent,
    IntegrationType,
    ToolEnvironmentEvent,
)
from layerlens.instrument._vendored.events_protocol import (
    SkillInfo,
    AgentCardInfo,
    AgentCardEvent,
    AsyncTaskEvent,
    TaskCompletedEvent,
    TaskSubmittedEvent,
    ProtocolStreamEvent,
    McpAppInvocationEvent,
    ElicitationRequestEvent,
    ElicitationResponseEvent,
    StructuredToolOutputEvent,
)
from layerlens.instrument._vendored.events_cross_cutting import (
    StateType,
    ViolationType,
    CostRecordEvent,
    AgentHandoffEvent,
    PolicyViolationEvent,
    AgentStateChangeEvent,
)
from layerlens.instrument._vendored.events_l4_environment import (
    EnvironmentType,
    EnvironmentConfigEvent,
    EnvironmentMetricsEvent,
)

__all__ = [
    # L1
    "AgentInputEvent",
    "AgentOutputEvent",
    "MessageRole",
    # L2
    "AgentCodeEvent",
    "CodeInfo",
    # L3
    "ModelInvokeEvent",
    # L4
    "EnvironmentConfigEvent",
    "EnvironmentMetricsEvent",
    "EnvironmentType",
    # L5
    "ToolCallEvent",
    "ToolLogicEvent",
    "ToolEnvironmentEvent",
    "IntegrationType",
    # Cross-cutting
    "AgentStateChangeEvent",
    "CostRecordEvent",
    "PolicyViolationEvent",
    "AgentHandoffEvent",
    "StateType",
    "ViolationType",
    # Protocol
    "AgentCardEvent",
    "AgentCardInfo",
    "SkillInfo",
    "TaskSubmittedEvent",
    "TaskCompletedEvent",
    "ProtocolStreamEvent",
    "ElicitationRequestEvent",
    "ElicitationResponseEvent",
    "StructuredToolOutputEvent",
    "McpAppInvocationEvent",
    "AsyncTaskEvent",
]
