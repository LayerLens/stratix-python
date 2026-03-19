"""
LayerLens Instrumentation SDK

Provides instrumentation for Python-based agent frameworks.

Usage:
    from layerlens.instrument import STRATIX

    stratix = STRATIX(
        policy_ref="stratix-policy-cs-v1@1.0.0",
        agent_id="support_agent",
        framework="langgraph",
        exporter="otel",
        endpoint="otel-collector:4317"
    )

    @stratix.trace_tool(name="lookup_order", version="1.0.0")
    def lookup_order(order_id: str) -> dict:
        ...
"""

from layerlens.instrument._core import STRATIX
from layerlens.instrument._context import STRATIXContext, get_current_context, context_scope
from layerlens.instrument._decorators import trace_tool, trace_model
from layerlens.instrument._state import StateAdapter, DictStateAdapter
from layerlens.instrument._emit import (
    emit,
    emit_input,
    emit_output,
    emit_tool_call,
    emit_model_invoke,
    emit_handoff,
)
from layerlens.instrument._cost import CostTracker, record_cost, record_token_cost
from layerlens.instrument._enforcement import (
    PolicyEnforcer,
    PolicyViolationError,
    check_tool_allowed,
    check_model_allowed,
    check_max_tokens,
)
from layerlens.instrument.exporters import Exporter, OTelExporter

__all__ = [
    # Core
    "STRATIX",
    "STRATIXContext",
    "get_current_context",
    "context_scope",
    # Decorators
    "trace_tool",
    "trace_model",
    # State
    "StateAdapter",
    "DictStateAdapter",
    # Emit API
    "emit",
    "emit_input",
    "emit_output",
    "emit_tool_call",
    "emit_model_invoke",
    "emit_handoff",
    # Cost tracking
    "CostTracker",
    "record_cost",
    "record_token_cost",
    # Enforcement
    "PolicyEnforcer",
    "PolicyViolationError",
    "check_tool_allowed",
    "check_model_allowed",
    "check_max_tokens",
    # Exporters
    "Exporter",
    "OTelExporter",
]
