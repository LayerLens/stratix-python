from __future__ import annotations

from typing import Any, Dict, FrozenSet
from dataclasses import dataclass

# Maps event type strings to CaptureConfig field names. EVERY content-bearing
# event type MUST appear here (or in _ALWAYS_ENABLED) so that disabling its
# L-layer actually suppresses it — an unmapped type fails OPEN (LAY-3578 / L2).
# The keys-must-match guard (tests/instrument/test_content_keys_guard.py) holds
# this true: every _CONTENT_KEYS entry is mapped here or always-enabled.
_EVENT_TYPE_MAP: Dict[str, str] = {
    # L1: Agent I/O
    "agent.input": "l1_agent_io",
    "agent.output": "l1_agent_io",
    "agent.lifecycle": "l1_agent_io",
    "agent.identity": "l1_agent_io",
    "agent.interaction": "l1_agent_io",
    "agent.step": "l1_agent_io",
    "agent.node.enter": "l1_agent_io",
    "agent.node.exit": "l1_agent_io",
    "conversation.started": "l1_agent_io",
    "conversation.ended": "l1_agent_io",
    "conversation.message": "l1_agent_io",
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
    # MCP tool/elicitation/structured-output/async surfaces ARE L5a tool calls;
    # the adapter emits these concrete strings (not the protocol.* aliases above),
    # so they must be mapped here or minimal()/l5a-off suppresses nothing (L2).
    "mcp.tool.call": "l5a_tool_calls",
    "mcp.tools.listed": "l5a_tool_calls",
    "mcp.elicitation": "l5a_tool_calls",
    "mcp.structured_output": "l5a_tool_calls",
    "mcp.async_task": "l5a_tool_calls",
    # L5b: Tool logic
    "tool.logic": "l5b_tool_logic",
    # L5c: Tool environment
    "tool.environment": "l5c_tool_environment",
    # L6a: Protocol discovery
    "protocol.agent_card": "l6a_protocol_discovery",
    "a2a.agent.discovered": "l6a_protocol_discovery",
    "a2a.agent.card": "l6a_protocol_discovery",
    "a2a.agent.card.served": "l6a_protocol_discovery",
    # L6b: Protocol streams (SSE, AG-UI)
    "protocol.stream.event": "l6b_protocol_streams",
    "agui.message": "l6b_protocol_streams",
    "agui.tool_call": "l6b_protocol_streams",
    "agui.state": "l6b_protocol_streams",
    # L6c: Protocol lifecycle (task / commerce / payment flow events). Mapped to
    # lifecycle (kept by minimal()) so the payment/commerce audit trail survives
    # a lightweight config but is suppressed when l6c is explicitly disabled.
    "protocol.lifecycle": "l6c_protocol_lifecycle",
    "a2a.task.created": "l6c_protocol_lifecycle",
    "a2a.task.updated": "l6c_protocol_lifecycle",
    "a2a.task.completed": "l6c_protocol_lifecycle",
    "a2a.delegation": "l6c_protocol_lifecycle",
    "payment.intent_mandate": "l6c_protocol_lifecycle",
    "payment.mandate_signed": "l6c_protocol_lifecycle",
    "payment.receipt_issued": "l6c_protocol_lifecycle",
    "commerce.supplier_discovered": "l6c_protocol_lifecycle",
    "commerce.catalog.browsed": "l6c_protocol_lifecycle",
    "commerce.checkout.started": "l6c_protocol_lifecycle",
    "commerce.checkout_completed": "l6c_protocol_lifecycle",
    "commerce.refund_issued": "l6c_protocol_lifecycle",
    "commerce.ui.surface_created": "l6c_protocol_lifecycle",
    "commerce.ui.user_action": "l6c_protocol_lifecycle",
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

# Request-parameter keys that can carry prompt/response content. Adapters must
# keep content out of ``capture_params`` (see providers/anthropic.py), but
# redaction has to hold even if one does not (LAY-3567 B1). ``tools`` /
# ``tool_choice`` / ``response_format`` carry the caller's tool/function
# JSON-Schema (names + natural-language descriptions + arg schemas), which is
# content, not a safe metric — strip them under capture_content=False (#17).
_CONTENT_PARAM_KEYS = frozenset(
    {
        "messages",
        "prompt",
        "contents",
        "input",
        "system",
        "output_message",
        "tools",
        "tool_choice",
        "functions",
        "function_call",
        "response_format",
    }
)

# Per-event-type CONTENT keys (LAY-3578 / LAY-3567). This is the SINGLE source of
# truth shared by adapter emit-time gating AND the collector-side backstop in
# ``redact_payload`` — so ``capture_content=False`` holds even when an adapter
# forgot to gate (the systemic class found 2026-06-24: str(exc) errors + ungated
# tool args / handoff context across protocol, framework, and provider adapters).
#
# Policy (team-reviewed): message/prompt/completion text, tool
# arguments/results, retrieval queries, state snapshots, raw stream payloads,
# free-text error strings (``error``/``error_message`` carry str(exc), which
# echoes the failing arguments), guardrail/handoff context, elicitation titles,
# delegation targets/skills, and financial details (amount, merchant,
# cumulative spend, supplier name, blocked reason) are CONTENT. Categories
# (error_type, error_code, reason_code), ids, counts, statuses, hashes,
# latencies, topology (from_agent/to_agent), and currencies stay METADATA so
# redaction never blinds observability.
_CONTENT_KEYS: Dict[str, FrozenSet[str]] = {
    # --- core agent / model / tool surfaces (backstop; adapters also gate) ---
    "agent.input": frozenset({"input", "messages", "content", "prompt", "system", "value"}),
    # ``error`` covers adapters that fold str(exc) onto agent.output/step (instead
    # of agent.error): smolagents, haystack, agno (census 2026-06-24).
    "agent.output": frozenset({"output", "output_message", "content", "messages", "value", "error"}),
    "agent.error": frozenset({"error", "error_message"}),
    "agent.step": frozenset({"input", "output", "messages", "content", "reason", "error", "code_action"}),
    # NB: agent.handoff `reason` is intentionally NOT stripped — it is a CATEGORY
    # constant (e.g. bedrock "supervisor_delegation"), kept for observability.
    # Free-text handoff content rides `context` (stripped). _handoff.py never
    # passes a free-text reason today (latent only); if it ever does, route it
    # through `context`, not `reason`.
    "agent.handoff": frozenset({"context", "input", "output", "messages"}),
    # Node lifecycle (langgraph): content + on_chain_error str(exc) ride these
    # — they were content-FREE-classified, so the backstop was a no-op for them.
    "agent.node.enter": frozenset({"input", "messages", "content"}),
    "agent.node.exit": frozenset({"input", "output", "messages", "content", "error"}),
    # Agent code / prompts / execution output (bedrock_agents, semantic_kernel,
    # langfuse). L2-gated, but make the backstop strip it too (defense in depth).
    "agent.code": frozenset({"code", "code_action", "output", "execution_error", "rendered_prompt", "input"}),
    # ``payload``/``data`` cover the AG-UI middleware + fallback raw-event
    # passthrough, which rides agent.state.change / tool.call (a raw protocol
    # event blob is content whichever type it lands on).
    "agent.state.change": frozenset(
        {"state", "status_message", "input", "output", "messages", "value", "payload", "data"}
    ),
    "agent.interaction": frozenset({"content", "input", "output", "messages"}),
    "conversation.message": frozenset({"content", "message"}),
    # ``error``: strands/langchain attach str(exc) onto model.invoke/tool.result.
    "model.invoke": frozenset({"messages", "output_message", "error"}),
    "embedding.create": frozenset({"input", "messages", "texts", "contents"}),
    "tool.call": frozenset({"arguments", "input", "args", "result", "payload", "data"}),
    "tool.result": frozenset({"result", "output", "content", "error"}),
    "retrieval.query": frozenset({"query", "input"}),
    "evaluation.result": frozenset({"comment", "explanation"}),
    # --- protocol surfaces ---
    "agui.message": frozenset({"text"}),
    "agui.tool_call": frozenset({"arguments", "result"}),
    "agui.state": frozenset({"state", "operations", "payload", "data"}),
    "protocol.stream.event": frozenset({"payload", "data"}),
    "mcp.tool.call": frozenset({"arguments", "result", "error"}),
    "mcp.async_task": frozenset({"error"}),
    "mcp.elicitation": frozenset({"title"}),
    "mcp.structured_output": frozenset({"validation_errors"}),
    "mcp.tools.listed": frozenset({"tool_names"}),
    "a2a.task.created": frozenset({"request", "skill"}),
    "a2a.task.updated": frozenset({"error", "error_message"}),
    "a2a.delegation": frozenset({"target_agent", "skill", "from_agent", "target_url"}),
    "a2a.agent.discovered": frozenset({"name", "skills"}),
    "payment.intent_mandate": frozenset({"amount", "merchant"}),
    "payment.mandate_signed": frozenset({"amount", "cumulative_spend", "reason"}),
    "payment.receipt_issued": frozenset({"amount", "merchant"}),
    "commerce.supplier_discovered": frozenset({"name"}),
    "commerce.catalog.browsed": frozenset({"query"}),
    "commerce.checkout_completed": frozenset({"amount"}),
    "commerce.refund_issued": frozenset({"amount", "reason"}),
}

# Backwards-compatible alias: the protocol subset (the original LAY-3578 map).
PROTOCOL_CONTENT_KEYS: Dict[str, FrozenSet[str]] = {
    k: v
    for k, v in _CONTENT_KEYS.items()
    if k.split(".")[0] in {"agui", "mcp", "a2a", "payment", "commerce", "protocol"}
}


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

    def redact_payload(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of *payload* with content fields removed per config.

        When ``capture_content`` is False this is the COLLECTOR-SIDE BACKSTOP:
        it strips every content field named in :data:`_CONTENT_KEYS` for the
        event type, regardless of whether the emitting adapter remembered to
        gate at emit time. Category/metadata (error_type, reason_code,
        tool_name, ids, counts, statuses, hashes, latencies, topology) is
        preserved so redaction does not blind observability. ``model.invoke``
        additionally has content stripped out of its ``parameters`` sub-dict.
        """
        if self.capture_content:
            return payload
        content_keys = _CONTENT_KEYS.get(event_type)
        if content_keys:
            payload = {k: v for k, v in payload.items() if k not in content_keys}
        if event_type == "model.invoke":
            parameters = payload.get("parameters")
            if isinstance(parameters, dict):
                payload = {
                    **payload,
                    "parameters": {k: v for k, v in parameters.items() if k not in _CONTENT_PARAM_KEYS},
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
        return bool(getattr(self, field_name))

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
