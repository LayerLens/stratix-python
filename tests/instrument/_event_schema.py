"""Event-schema contract lock (LAY-3583 / T9).

This module IS the in-repo schema document. It codifies the CURRENT payload
vocabulary so that any NEW drift fails loudly; ratcheting the documented
exceptions down is the future §3.6 convergence work (this file is its
worklist), NOT this phase — nothing here renames fields.

Canonical vocabulary
====================

* **Event types** — every payload uploaded by a unit suite must use an event
  type registered in :data:`KNOWN_EVENT_TYPES`. Adding a new event type is a
  deliberate act: add it here in the same PR.
* **Timing** — ``latency_ms`` (number) is the canonical duration field.
  ``duration_ns`` survives only where :data:`DURATION_NS_EXCEPTIONS` records
  today's drift (smolagents everywhere; crewai/strands/google_adk root
  events). New adapters must not use it.
* **Tokens** — provider-family events carry a ``usage`` dict whose keys come
  from :data:`USAGE_KEYS` (``prompt_tokens``/``completion_tokens``/...).
  Framework-family events use flat ``tokens_prompt``/``tokens_completion``.
  Mixing vocabularies in one payload is drift and fails.
* **cost.record** — must carry token counts; ``cost_usd`` remains optional
  because 15 of 18 framework emitters don't compute it (documented §3.6 gap
  and convergence-work item — locking it required would be a rename-scale
  change).

The lock is enforced automatically: the ``capture_trace`` /
``capture_framework_trace`` fixtures validate every uploaded event, so every
adapter unit suite participates without per-test wiring.
"""

from __future__ import annotations

from typing import Any, Dict, List
from numbers import Number

# ---------------------------------------------------------------------------
# Event-type registry
# ---------------------------------------------------------------------------

KNOWN_EVENT_TYPES = frozenset(
    {
        # agent family
        "agent.input",
        "agent.output",
        "agent.error",
        "agent.step",
        "agent.code",
        "agent.handoff",
        "agent.identity",
        "agent.interaction",
        "agent.lifecycle",
        "agent.state.change",
        "agent.node.enter",
        "agent.node.exit",
        # model / cost / environment
        "model.invoke",
        "cost.record",
        "embedding.create",
        "environment.config",
        "environment.metrics",
        # tools / retrieval
        "tool.call",
        "tool.result",
        "tool.logic",
        "tool.environment",
        "retrieval.query",
        # conversation (autogen group chat)
        "conversation.started",
        "conversation.ended",
        "conversation.message",
        # policy / evaluation
        "policy.violation",
        "evaluation.result",
        # protocol family
        "protocol.agent_card",
        "protocol.stream.event",
        "protocol.lifecycle",
        "protocol.task.submitted",
        "protocol.task.completed",
        "protocol.async_task",
        "protocol.elicitation.request",
        "protocol.elicitation.response",
        "protocol.tool.structured_output",
        "protocol.mcp_app.invocation",
        "mcp.tool.call",
        "mcp.tools.listed",
        "mcp.async_task",
        "mcp.elicitation",
        "mcp.structured_output",
        "a2a.task.created",
        "a2a.task.updated",
        "a2a.task.completed",
        "a2a.agent.card",
        "a2a.agent.card.served",
        "a2a.agent.discovered",
        "a2a.delegation",
        "agui.message",
        "agui.tool_call",
        "agui.state",
        "commerce.supplier_discovered",
        "commerce.catalog.browsed",
        "commerce.checkout.started",
        "commerce.checkout_completed",
        "commerce.refund_issued",
        "commerce.ui.surface_created",
        "commerce.ui.user_action",
        "payment.intent_mandate",
        "payment.mandate_signed",
        "payment.receipt_issued",
    }
)

#: ``usage`` dict keys a model.invoke / embedding.create payload may carry.
USAGE_KEYS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "thinking_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    }
)

#: Flat token fields used by the framework-family emitters.
FRAMEWORK_TOKEN_KEYS = frozenset({"tokens_prompt", "tokens_completion", "tokens_total"})

#: (framework marker, event_type) pairs allowed to emit ``duration_ns``.
#: ``"*"`` matches any event type from that adapter. THIS IS THE DRIFT TABLE
#: (stability report §3.6) — shrink it, never grow it.
DURATION_NS_EXCEPTIONS = frozenset(
    {
        ("smolagents", "*"),
        ("crewai", "*"),
        ("strands", "*"),
        ("google_adk", "*"),
    }
)


class EventSchemaViolation(AssertionError):
    pass


def _adapter_marker(payload: Dict[str, Any]) -> str:
    for key in ("framework", "provider", "protocol"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return "?"


def validate_event(event: Dict[str, Any]) -> List[str]:
    """Return schema violations for one uploaded event (empty = compliant)."""
    problems: List[str] = []
    event_type = event.get("event_type")
    payload = event.get("payload") or {}
    marker = _adapter_marker(payload)
    tag = f"[{marker}/{event_type}]"

    if event_type not in KNOWN_EVENT_TYPES:
        problems.append(
            f"{tag} unknown event type — register it in tests/instrument/_event_schema.py "
            "in the same PR that introduces it"
        )
        return problems

    latency = payload.get("latency_ms")
    if latency is not None and not isinstance(latency, Number):
        problems.append(f"{tag} latency_ms must be a number, got {type(latency).__name__}")

    if "duration_ns" in payload:
        allowed = (marker, "*") in DURATION_NS_EXCEPTIONS or (marker, event_type) in DURATION_NS_EXCEPTIONS
        if not allowed:
            problems.append(
                f"{tag} duration_ns is non-canonical (use latency_ms); only the documented "
                "drift table in DURATION_NS_EXCEPTIONS may carry it"
            )

    usage = payload.get("usage")
    if usage is not None:
        if not isinstance(usage, dict):
            problems.append(f"{tag} usage must be a dict")
        else:
            unknown = set(usage) - USAGE_KEYS
            if unknown:
                problems.append(f"{tag} unknown usage keys {sorted(unknown)} — extend USAGE_KEYS deliberately")
            for key, value in usage.items():
                if value is not None and not isinstance(value, int):
                    problems.append(f"{tag} usage.{key} must be int/None, got {type(value).__name__}")

    flat_tokens = FRAMEWORK_TOKEN_KEYS & set(payload)
    provider_style = {"prompt_tokens", "completion_tokens", "total_tokens"} & set(payload)
    if flat_tokens and provider_style:
        problems.append(
            f"{tag} mixes framework token vocabulary {sorted(flat_tokens)} with provider "
            f"vocabulary {sorted(provider_style)} in one payload"
        )
    for key in flat_tokens | provider_style:
        value = payload.get(key)
        if value is not None and not isinstance(value, int):
            problems.append(f"{tag} {key} must be int/None, got {type(value).__name__}")

    if event_type == "cost.record":
        has_tokens = bool(flat_tokens or provider_style)
        if not has_tokens:
            problems.append(f"{tag} cost.record without any token counts")
        cost = payload.get("cost_usd")
        if cost is not None and not isinstance(cost, Number):
            problems.append(f"{tag} cost_usd must be a number, got {type(cost).__name__}")

    return problems


def validate_events(events: List[Dict[str, Any]]) -> None:
    """Assert every uploaded event matches the locked schema."""
    problems: List[str] = []
    for event in events:
        problems.extend(validate_event(event))
    if problems:
        raise EventSchemaViolation(
            "event-schema contract violations (LAY-3583 — the schema lock lives in "
            "tests/instrument/_event_schema.py):\n  " + "\n  ".join(problems)
        )
