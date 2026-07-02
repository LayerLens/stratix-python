"""Canonical event names emitted by layerlens instrumentation.

Kept in a single module so adapters don't scatter string literals.
"""

from __future__ import annotations

from typing import Final

# Trace structure
# Synthesized structural root marker (LAY-364x / trace-root). Emitted by the
# collector at flush when a trace's leaf events all hang off a single ambient
# span the SDK never emitted an event for (provider-only / trace_context /
# framework _begin_run usage), so the trace always has a REAL captured root and
# the frontend never has to synthesize one. It is NOT an agent lifecycle event —
# it is a content-free structural marker, so it has its own type rather than
# polluting the agent.lifecycle stream.
TRACE_ROOT: Final[str] = "trace.root"

# LLM provider events
MODEL_INVOKE: Final[str] = "model.invoke"
COST_RECORD: Final[str] = "cost.record"
TOOL_CALL: Final[str] = "tool.call"
AGENT_ERROR: Final[str] = "agent.error"

# Framework events
AGENT_HANDOFF: Final[str] = "agent.handoff"

# Canonical, producer-DECLARED agent identity. Synthesized once per trace at
# flush from the honest name a producer already declared (a @stratix.trace name,
# a crew/agent name, a langgraph node), so the server + FE surface the Agent
# column from ONE place instead of chasing per-adapter keys — and NEVER from a
# model name, an API method name, a span_name, or a class default. Structural
# metadata (like the trace.root marker): attestation-covered, content-free.
AGENT_IDENTITY: Final[str] = "agent.identity"

# MCP protocol events
MCP_TOOL_CALL: Final[str] = "mcp.tool.call"
MCP_ELICITATION: Final[str] = "mcp.elicitation"
MCP_STRUCTURED_OUTPUT: Final[str] = "mcp.structured_output"
MCP_ASYNC_TASK: Final[str] = "mcp.async_task"
# Server-initiated nested LLM round-trip (sampling/createMessage). The server
# asks the CLIENT's LLM to generate — real, billable tokens that flow through
# the MCP transport, not through any provider adapter. Emitted alongside a
# cost.record so the agentic money-burning path is not invisible (D3 / LAY-3626).
MCP_SAMPLING: Final[str] = "mcp.sampling"

# A2A protocol events
A2A_AGENT_DISCOVERED: Final[str] = "a2a.agent.discovered"
A2A_AGENT_CARD_SERVED: Final[str] = "a2a.agent.card.served"
A2A_TASK_CREATED: Final[str] = "a2a.task.created"
A2A_TASK_UPDATED: Final[str] = "a2a.task.updated"
A2A_DELEGATION: Final[str] = "a2a.delegation"

# AG-UI protocol events
AGUI_STATE: Final[str] = "agui.state"
AGUI_MESSAGE: Final[str] = "agui.message"
AGUI_TOOL_CALL: Final[str] = "agui.tool_call"

# Generic protocol stream event (SSE / partial updates)
PROTOCOL_STREAM_EVENT: Final[str] = "protocol.stream.event"

# Commerce / payments protocol events
COMMERCE_UI_SURFACE_CREATED: Final[str] = "commerce.ui.surface_created"
COMMERCE_UI_USER_ACTION: Final[str] = "commerce.ui.user_action"
COMMERCE_SUPPLIER_DISCOVERED: Final[str] = "commerce.supplier_discovered"
COMMERCE_CHECKOUT_COMPLETED: Final[str] = "commerce.checkout_completed"
COMMERCE_REFUND_ISSUED: Final[str] = "commerce.refund_issued"
PAYMENT_INTENT_MANDATE: Final[str] = "payment.intent_mandate"
PAYMENT_CART_MANDATE: Final[str] = "payment.cart_mandate"
PAYMENT_MANDATE_SIGNED: Final[str] = "payment.mandate_signed"
PAYMENT_RECEIPT_ISSUED: Final[str] = "payment.receipt_issued"
POLICY_VIOLATION: Final[str] = "policy.violation"
