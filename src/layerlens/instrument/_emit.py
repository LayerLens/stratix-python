"""
STRATIX Explicit Emit API

From Step 4 specification:
- Explicit emit API for cases not covered by decorators
- stratix.emit() for arbitrary events
- stratix.emit_tool_call() for manual tool call events
- stratix.emit_handoff() for agent handoffs

This module provides standalone emit functions that work with the current context.
"""

from __future__ import annotations

from typing import Any

from layerlens.instrument.schema.events import (
    AgentHandoffEvent,
    AgentInputEvent,
    AgentOutputEvent,
    ToolCallEvent,
    ModelInvokeEvent,
)
from layerlens.instrument.schema.events.l1_io import MessageRole
from layerlens.instrument.schema.events.l3_model import ModelInfo
from layerlens.instrument.schema.events.l5_tools import IntegrationType
from layerlens.instrument.schema.privacy import PrivacyLevel
from layerlens.instrument._context import get_current_context


def emit(payload: Any, privacy_level: PrivacyLevel | None = None) -> None:
    """
    Emit an arbitrary event.

    Args:
        payload: Event payload (must be a valid STRATIX event type)
        privacy_level: Optional privacy level override
    """
    ctx = get_current_context()
    if ctx is None:
        raise RuntimeError("No active STRATIX context. Call start_trial() first.")

    ctx.stratix._emit_event(ctx, payload, privacy_level)


def emit_input(message: str, role: str = "human") -> None:
    """
    Emit an agent input event.

    Args:
        message: The input message content
        role: Message role (human, system, assistant)
    """
    role_enum = MessageRole(role)
    payload = AgentInputEvent.create(message=message, role=role_enum)
    emit(payload)


def emit_output(message: str) -> None:
    """
    Emit an agent output event.

    Args:
        message: The output message content
    """
    payload = AgentOutputEvent.create(message=message)
    emit(payload)


def emit_tool_call(
    name: str,
    input_data: dict[str, Any] | None = None,
    output_data: dict[str, Any] | None = None,
    error: str | None = None,
    latency_ms: float | None = None,
    version: str = "unavailable",
    integration: str = "library",
) -> None:
    """
    Emit a tool call event.

    This is the explicit emit API for tool calls when decorators
    cannot be used (e.g., dynamic tool invocation).

    Args:
        name: Tool name
        input_data: Tool input parameters
        output_data: Tool output
        error: Error message if the tool failed
        latency_ms: Execution time in milliseconds
        version: Tool version
        integration: Integration type (library, service, agent, script)
    """
    integration_type = IntegrationType(integration)
    payload = ToolCallEvent.create(
        name=name,
        version=version,
        integration=integration_type,
        input_data=input_data,
        output_data=output_data,
        error=error,
        latency_ms=latency_ms,
    )
    emit(payload)


def emit_model_invoke(
    provider: str,
    name: str,
    version: str = "unavailable",
    parameters: dict[str, Any] | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    latency_ms: float | None = None,
) -> None:
    """
    Emit a model invocation event.

    This is the explicit emit API for model calls when wrappers
    cannot be used.

    Args:
        provider: Model provider (openai, anthropic, etc.)
        name: Model name
        version: Model version
        parameters: Model parameters (temperature, max_tokens, etc.)
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        total_tokens: Total tokens
        latency_ms: Invocation time in milliseconds
    """
    payload = ModelInvokeEvent.create(
        provider=provider,
        name=name,
        version=version,
        parameters=parameters,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
    )
    emit(payload)


def emit_handoff(
    source_agent: str,
    target_agent: str,
    context_passed: dict[str, Any] | None = None,
    privacy_level: str = "cleartext",
) -> None:
    """
    Emit an agent handoff event.

    From Step 4 specification:
    - Handoffs must preserve trace continuity
    - Context should be propagated to the target agent

    Args:
        source_agent: The agent handing off
        target_agent: The agent receiving the handoff
        context_passed: Context data passed to the target agent
        privacy_level: Privacy level of the handoff context
    """
    import hashlib
    import json

    ctx = get_current_context()
    if ctx is None:
        raise RuntimeError("No active STRATIX context. Call start_trial() first.")

    # Build context to be hashed
    vector_clock = ctx.vector_clock.model_dump()
    context_data = {
        "trace_id": ctx.trace_id,
        "span_id": ctx.current_span_id,
        "vector_clock": vector_clock,
        **(context_passed or {}),
    }

    # Compute context hash
    canonical = json.dumps(context_data, sort_keys=True, default=str)
    context_hash = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()

    payload = AgentHandoffEvent.create(
        from_agent=source_agent,
        to_agent=target_agent,
        handoff_context_hash=context_hash,
        context_privacy_level=privacy_level,
    )
    emit(payload)
