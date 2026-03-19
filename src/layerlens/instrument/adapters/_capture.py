"""
STRATIX Capture Configuration

Defines the CaptureConfig model that controls which telemetry layers
are active for a given adapter instance.

Layer Mapping:
- L1: Agent I/O (agent.input, agent.output)
- L2: Agent Code (agent.code)
- L3: Model Metadata (model.invoke)
- L4a: Environment Configuration (environment.config)
- L4b: Environment Metrics (environment.metrics)
- L5a: Tool/Action Execution (tool.call)
- L5b: Tool Business Logic (tool.logic)
- L5c: Tool Environment (tool.environment)

Cross-cutting events (agent.state.change, cost.record, policy.violation,
agent.handoff) are always enabled and cannot be disabled.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field


# Layers that cannot be disabled
_CROSS_CUTTING_LAYERS = frozenset({
    "cross_cutting_state",
    "cross_cutting_cost",
    "cross_cutting_policy",
    "cross_cutting_handoff",
})

# Event types that are always emitted regardless of config
ALWAYS_ENABLED_EVENT_TYPES = frozenset({
    "agent.state.change",
    "cost.record",
    "policy.violation",
    "agent.handoff",
    "evaluation.result",
    "protocol.task.submitted",
    "protocol.task.completed",
    "protocol.async_task",
})


class CaptureConfig(BaseModel):
    """
    Controls which telemetry layers are active.

    Each boolean flag corresponds to an STRATIX capture layer. When a flag
    is False, the adapter's emit_event() silently drops events for that
    layer instead of forwarding them to the STRATIX pipeline.

    Cross-cutting events (state changes, cost records, policy violations,
    handoffs) are always enabled and cannot be gated.
    """

    l1_agent_io: bool = Field(
        default=True,
        description="L1: Agent input/output messages",
    )
    l2_agent_code: bool = Field(
        default=False,
        description="L2: Agent code artifacts and hashes",
    )
    l3_model_metadata: bool = Field(
        default=True,
        description="L3: Model invocation metadata",
    )
    l4a_environment_config: bool = Field(
        default=True,
        description="L4a: Environment configuration snapshots",
    )
    l4b_environment_metrics: bool = Field(
        default=False,
        description="L4b: Environment runtime metrics",
    )
    l5a_tool_calls: bool = Field(
        default=True,
        description="L5a: Tool/action call input/output",
    )
    l5b_tool_logic: bool = Field(
        default=False,
        description="L5b: Tool business logic details",
    )
    l5c_tool_environment: bool = Field(
        default=False,
        description="L5c: Tool environment details",
    )
    l6a_protocol_discovery: bool = Field(
        default=True,
        description="L6a: Protocol discovery events (A2A Agent Cards).",
    )
    l6b_protocol_streams: bool = Field(
        default=True,
        description="L6b: Protocol stream events (AG-UI chunks, A2A SSE). "
                    "Set to False to capture only stream start/end events.",
    )
    l6c_protocol_lifecycle: bool = Field(
        default=True,
        description="L6c: Protocol lifecycle events (A2A tasks, async tasks).",
    )
    capture_content: bool = Field(
        default=True,
        description="Capture LLM message content on model.invoke events",
    )

    @property
    def otel_capture_content(self) -> bool:
        """Check if OTel content capture is enabled via env var.

        Content appears in OTel spans only when BOTH this CaptureConfig.capture_content
        AND the OTEL_GENAI_CAPTURE_MESSAGE_CONTENT env var are true.
        """
        env_val = os.environ.get("OTEL_GENAI_CAPTURE_MESSAGE_CONTENT", "").lower()
        return self.capture_content and env_val == "true"

    def is_layer_enabled(self, layer: str) -> bool:
        """
        Check whether a given layer is enabled.

        Cross-cutting events always return True.

        Args:
            layer: Layer identifier. Accepted formats:
                - Attribute names: "l1_agent_io", "l3_model_metadata", etc.
                - Short labels: "L1", "L3", "L5a", etc.
                - Event types: "agent.input", "model.invoke", etc.

        Returns:
            True if the layer is enabled or is a cross-cutting event.
        """
        # Cross-cutting events are always on
        if layer in _CROSS_CUTTING_LAYERS or layer in ALWAYS_ENABLED_EVENT_TYPES:
            return True

        # Direct attribute lookup
        if hasattr(self, layer):
            return bool(getattr(self, layer))

        # Short-label mapping
        label_map = {
            "L1": "l1_agent_io",
            "L2": "l2_agent_code",
            "L3": "l3_model_metadata",
            "L4a": "l4a_environment_config",
            "L4b": "l4b_environment_metrics",
            "L5a": "l5a_tool_calls",
            "L5b": "l5b_tool_logic",
            "L5c": "l5c_tool_environment",
            "L6a": "l6a_protocol_discovery",
            "L6b": "l6b_protocol_streams",
            "L6c": "l6c_protocol_lifecycle",
        }
        if layer in label_map:
            return bool(getattr(self, label_map[layer]))

        # Event-type mapping
        event_type_map = {
            "agent.input": "l1_agent_io",
            "agent.output": "l1_agent_io",
            "agent.lifecycle": "l1_agent_io",
            "agent.identity": "l1_agent_io",
            "agent.interaction": "l1_agent_io",
            "agent.code": "l2_agent_code",
            "model.invoke": "l3_model_metadata",
            "environment.config": "l4a_environment_config",
            "environment.metrics": "l4b_environment_metrics",
            "tool.call": "l5a_tool_calls",
            "tool.logic": "l5b_tool_logic",
            "tool.environment": "l5c_tool_environment",
            "protocol.agent_card": "l6a_protocol_discovery",
            "protocol.stream.event": "l6b_protocol_streams",
            "protocol.elicitation.request": "l5a_tool_calls",
            "protocol.elicitation.response": "l5a_tool_calls",
            "protocol.tool.structured_output": "l5a_tool_calls",
            "protocol.mcp_app.invocation": "l5a_tool_calls",
        }
        if layer in event_type_map:
            return bool(getattr(self, event_type_map[layer]))

        # Unknown layers default to disabled (safe-by-default)
        return False

    # --- Presets ---

    @classmethod
    def minimal(cls) -> CaptureConfig:
        """L1 only — lightweight production telemetry."""
        return cls(
            l1_agent_io=True,
            l2_agent_code=False,
            l3_model_metadata=False,
            l4a_environment_config=False,
            l4b_environment_metrics=False,
            l5a_tool_calls=False,
            l5b_tool_logic=False,
            l5c_tool_environment=False,
            l6a_protocol_discovery=True,
            l6b_protocol_streams=False,
            l6c_protocol_lifecycle=True,
            capture_content=False,
        )

    @classmethod
    def standard(cls) -> CaptureConfig:
        """L1 + L3 + L4a + L5a + L6 — recommended for most deployments."""
        return cls(
            l1_agent_io=True,
            l2_agent_code=False,
            l3_model_metadata=True,
            l4a_environment_config=True,
            l4b_environment_metrics=False,
            l5a_tool_calls=True,
            l5b_tool_logic=False,
            l5c_tool_environment=False,
            l6a_protocol_discovery=True,
            l6b_protocol_streams=True,
            l6c_protocol_lifecycle=True,
        )

    @classmethod
    def full(cls) -> CaptureConfig:
        """All layers enabled — development/debugging."""
        return cls(
            l1_agent_io=True,
            l2_agent_code=True,
            l3_model_metadata=True,
            l4a_environment_config=True,
            l4b_environment_metrics=True,
            l5a_tool_calls=True,
            l5b_tool_logic=True,
            l5c_tool_environment=True,
            l6a_protocol_discovery=True,
            l6b_protocol_streams=True,
            l6c_protocol_lifecycle=True,
        )
