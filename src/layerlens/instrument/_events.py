"""Canonical event names emitted by layerlens instrumentation.

Kept in a single module so adapters don't scatter string literals.
"""

from __future__ import annotations

from typing import Final

# LLM provider events
MODEL_INVOKE: Final[str] = "model.invoke"
COST_RECORD: Final[str] = "cost.record"
TOOL_CALL: Final[str] = "tool.call"
AGENT_ERROR: Final[str] = "agent.error"

# Framework events
AGENT_HANDOFF: Final[str] = "agent.handoff"

# MCP protocol events
MCP_TOOL_CALL: Final[str] = "mcp.tool.call"
MCP_ELICITATION: Final[str] = "mcp.elicitation"
MCP_STRUCTURED_OUTPUT: Final[str] = "mcp.structured_output"
MCP_ASYNC_TASK: Final[str] = "mcp.async_task"

# A2A protocol events
A2A_AGENT_DISCOVERED: Final[str] = "a2a.agent.discovered"
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
PAYMENT_MANDATE_SIGNED: Final[str] = "payment.mandate_signed"
PAYMENT_RECEIPT_ISSUED: Final[str] = "payment.receipt_issued"
