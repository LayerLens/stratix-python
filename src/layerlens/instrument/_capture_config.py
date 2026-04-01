from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

# Maps event type strings to CaptureConfig field names
_EVENT_TYPE_MAP: Dict[str, str] = {
    # L1: Agent I/O
    "agent.input": "l1_agent_io",
    "agent.output": "l1_agent_io",
    "agent.lifecycle": "l1_agent_io",
    "agent.identity": "l1_agent_io",
    "agent.interaction": "l1_agent_io",
    # L2: Agent code
    "agent.code": "l2_agent_code",
    # L3: Model metadata
    "model.invoke": "l3_model_metadata",
    "embedding.create": "l3_model_metadata",
    # L4a: Environment config
    "environment.config": "l4a_environment_config",
    # L4b: Environment metrics
    "environment.metrics": "l4b_environment_metrics",
    # L5a: Tool calls
    "tool.call": "l5a_tool_calls",
    "tool.result": "l5a_tool_calls",
    "retrieval.query": "l5a_tool_calls",
    "protocol.elicitation.request": "l5a_tool_calls",
    "protocol.elicitation.response": "l5a_tool_calls",
    "protocol.tool.structured_output": "l5a_tool_calls",
    "protocol.mcp_app.invocation": "l5a_tool_calls",
    # L5b: Tool logic
    "tool.logic": "l5b_tool_logic",
    # L5c: Tool environment
    "tool.environment": "l5c_tool_environment",
    # L6a: Protocol discovery
    "protocol.agent_card": "l6a_protocol_discovery",
    # L6b: Protocol streams
    "protocol.stream.event": "l6b_protocol_streams",
    # L6c: Protocol lifecycle
    "protocol.lifecycle": "l6c_protocol_lifecycle",
}

# Events that are always emitted regardless of config
_ALWAYS_ENABLED = frozenset(
    {
        "agent.error",
        "agent.state.change",
        "cost.record",
        "policy.violation",
        "agent.handoff",
        "evaluation.result",
        "protocol.task.submitted",
        "protocol.task.completed",
        "protocol.async_task",
    }
)


@dataclass(frozen=True)
class CaptureConfig:
    """Controls which telemetry layers are captured.

    Each boolean flag corresponds to an L1-L6 capture layer.
    Use presets for common configurations: minimal(), standard(), full().
    """

    # L1: Agent I/O
    l1_agent_io: bool = True
    # L2: Agent code artifacts
    l2_agent_code: bool = False
    # L3: Model invocation metadata
    l3_model_metadata: bool = True
    # L4a: Environment configuration
    l4a_environment_config: bool = True
    # L4b: Environment metrics
    l4b_environment_metrics: bool = False
    # L5a: Tool/function calls
    l5a_tool_calls: bool = True
    # L5b: Tool internal logic
    l5b_tool_logic: bool = False
    # L5c: Tool environment
    l5c_tool_environment: bool = False
    # L6a: Protocol discovery (A2A Agent Cards)
    l6a_protocol_discovery: bool = True
    # L6b: Protocol streams (SSE, AG-UI)
    l6b_protocol_streams: bool = True
    # L6c: Protocol lifecycle (task events)
    l6c_protocol_lifecycle: bool = True
    # Gates LLM message content (prompts/completions) independently of L-layers
    capture_content: bool = True

    def redact_payload(
        self, event_type: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return a copy of payload with fields removed per config."""
        if not self.capture_content and event_type == "model.invoke":
            payload = {
                k: v
                for k, v in payload.items()
                if k not in ("messages", "output_message")
            }
        return payload

    def is_layer_enabled(self, event_type: str) -> bool:
        """Check if an event type is enabled by this config.

        Always-enabled events (cost.record, agent.error, etc.) return True.
        Mapped event types check their corresponding L-layer flag.
        Unknown event types return True (fail-open).
        """
        if event_type in _ALWAYS_ENABLED:
            return True
        field_name = _EVENT_TYPE_MAP.get(event_type)
        if field_name is None:
            return True  # fail-open for unknown event types
        return getattr(self, field_name)

    @classmethod
    def minimal(cls) -> CaptureConfig:
        """Lightweight production telemetry: agent I/O + protocol discovery/lifecycle."""
        return cls(
            l1_agent_io=True,
            l3_model_metadata=False,
            l4a_environment_config=False,
            l5a_tool_calls=False,
            l6a_protocol_discovery=True,
            l6b_protocol_streams=False,
            l6c_protocol_lifecycle=True,
            capture_content=True,
        )

    @classmethod
    def standard(cls) -> CaptureConfig:
        """Balanced telemetry: agent I/O, model metadata, tools, protocols. Same as default."""
        return cls()

    @classmethod
    def full(cls) -> CaptureConfig:
        """Full capture: all layers enabled. Development/debugging."""
        return cls(
            l2_agent_code=True,
            l4b_environment_metrics=True,
            l5b_tool_logic=True,
            l5c_tool_environment=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "l1_agent_io": self.l1_agent_io,
            "l2_agent_code": self.l2_agent_code,
            "l3_model_metadata": self.l3_model_metadata,
            "l4a_environment_config": self.l4a_environment_config,
            "l4b_environment_metrics": self.l4b_environment_metrics,
            "l5a_tool_calls": self.l5a_tool_calls,
            "l5b_tool_logic": self.l5b_tool_logic,
            "l5c_tool_environment": self.l5c_tool_environment,
            "l6a_protocol_discovery": self.l6a_protocol_discovery,
            "l6b_protocol_streams": self.l6b_protocol_streams,
            "l6c_protocol_lifecycle": self.l6c_protocol_lifecycle,
            "capture_content": self.capture_content,
        }
