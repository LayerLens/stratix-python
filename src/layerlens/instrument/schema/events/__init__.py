"""
STRATIX Event Types

This module contains all layer-specific event types as defined in
the Canonical Event & Trace Schema.

Layer Structure:
- L1: Agent Inputs & Outputs
- L2: Agent Logic Code
- L3: Model Metadata
- L4a: Environment Configuration
- L4b: Environment Metrics
- L5a: Tool/Action Execution
- L5b: Tool Business Logic
- L5c: Tool Environment
- L6a: Protocol Discovery
- L6b: Protocol Streams
- L6c: Protocol Lifecycle

Cross-Cutting Events:
- agent.state.change: State mutations
- cost.record: Cost/usage tracking
- policy.violation: Policy failures
- agent.handoff: Multi-agent delegation
- protocol.task.submitted: A2A task lifecycle start
- protocol.task.completed: A2A task lifecycle end
- protocol.async_task: MCP/A2A async task lifecycle

Replay Events:
- trace.checkpoint: Resumable execution checkpoints
- trace.replay.start: Replay session start with parameter overrides
- trace.replay.end: Replay session end with diff summary

Feedback Events:
- feedback.explicit: Human ratings, thumbs, comments
- feedback.implicit: Behavioral signals (retry, abandonment, etc.)
- feedback.annotation: Expert annotation queue results

Evaluation Events:
- evaluation.result: Evaluation dimension and final scores

Protocol Events (Schema v1.2.0):
- protocol.agent_card: A2A Agent Card discovery (L6a)
- protocol.stream.event: AG-UI/A2A streaming event (L6b)
- protocol.task.submitted: A2A task submitted (cross-cutting)
- protocol.task.completed: A2A task completed (cross-cutting)
- protocol.async_task: MCP/A2A async task (cross-cutting)
- protocol.elicitation.request: MCP elicitation request (L5a)
- protocol.elicitation.response: MCP elicitation response (L5a)
- protocol.tool.structured_output: MCP structured output (L5a)
- protocol.mcp_app.invocation: MCP App invocation (L5a)
"""

from layerlens.instrument.schema.events.l1_io import (
    AgentInputEvent,
    AgentOutputEvent,
    MessageRole,
)
from layerlens.instrument.schema.events.l2_code import AgentCodeEvent
from layerlens.instrument.schema.events.l3_model import ModelInvokeEvent
from layerlens.instrument.schema.events.l4_environment import (
    EnvironmentConfigEvent,
    EnvironmentMetricsEvent,
    EnvironmentType,
)
from layerlens.instrument.schema.events.l5_tools import (
    ToolCallEvent,
    ToolLogicEvent,
    ToolEnvironmentEvent,
    IntegrationType,
)
from layerlens.instrument.schema.events.cross_cutting import (
    AgentStateChangeEvent,
    CostRecordEvent,
    PolicyViolationEvent,
    AgentHandoffEvent,
    StateType,
    ViolationType,
)
from layerlens.instrument.schema.events.replay import (
    TraceCheckpointEvent,
    TraceReplayStartEvent,
    TraceReplayEndEvent,
)
from layerlens.instrument.schema.events.feedback import (
    ExplicitFeedbackEvent,
    ImplicitFeedbackEvent,
    AnnotationFeedbackEvent,
)
from layerlens.instrument.schema.events.evaluation import (
    EvaluationResultEvent,
    EvaluationInfo,
)
from layerlens.instrument.schema.events.protocol import (
    AgentCardEvent,
    AgentCardInfo,
    SkillInfo,
    TaskSubmittedEvent,
    TaskCompletedEvent,
    ProtocolStreamEvent,
    ElicitationRequestEvent,
    ElicitationResponseEvent,
    StructuredToolOutputEvent,
    McpAppInvocationEvent,
    AsyncTaskEvent,
)

__all__ = [
    # L1
    "AgentInputEvent",
    "AgentOutputEvent",
    "MessageRole",
    # L2
    "AgentCodeEvent",
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
    # Replay
    "TraceCheckpointEvent",
    "TraceReplayStartEvent",
    "TraceReplayEndEvent",
    # Feedback
    "ExplicitFeedbackEvent",
    "ImplicitFeedbackEvent",
    "AnnotationFeedbackEvent",
    # Evaluation
    "EvaluationResultEvent",
    "EvaluationInfo",
    # Protocol (Schema v1.2.0)
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
