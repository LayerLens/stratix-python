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
* **cost.record** — must carry token counts; ``cost_usd`` is now computed
  centrally for priced models (BaseFrameworkAdapter._emit + each _fire), so the
  old "15 of 18 emitters don't compute it" no longer holds. It stays *optional*
  in the lock because ``calculate_cost`` legitimately returns ``None`` for
  unpriced models (ollama/local/custom). FOLLOW-UP (LAY-3621): ratchet to
  conditionally-required — when the model resolves in PRICING and tokens>0 —
  after a green full-adapter-matrix run.

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
        # trace structure
        # Synthesized structural root marker emitted by the collector at flush
        # (LAY-364x / trace-root) so every trace has a real captured root and the
        # FE never synthesizes one. Content-free; its own type (not agent.*).
        "trace.root",
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
        "mcp.tool.call",
        "mcp.tools.listed",
        "mcp.async_task",
        "mcp.elicitation",
        "mcp.sampling",
        "mcp.structured_output",
        "mcp.server.connected",
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
        "payment.cart_mandate",
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
        # INVARIANT (LAY-3626 / A11, fail-closed cost): a model that resolves to a
        # rate MUST carry cost_usd. The central price-on-emit chokepoint
        # (TraceCollector.emit) fills it, so a None here means a priced model's
        # cost was DROPPED — indistinguishable from a genuinely-unpriced
        # local/custom model unless we enforce it. None stays legal only for
        # unpriced models (ollama/local/custom).
        if cost is None and has_tokens:
            from layerlens.instrument.adapters.providers.pricing import is_priced

            if is_priced(payload.get("model"), payload.get("provider")):
                problems.append(
                    f"{tag} priced model {payload.get('model')!r} has no cost_usd — a priced "
                    "cost.record must carry cost_usd (the central price-on-emit chokepoint fills "
                    "it; None means the price was dropped). Unpriced local/custom models may omit it."
                )

    # INVARIANT (LAY-3620, redact-without-going-blind): agent.error must carry a
    # surviving CATEGORY. The capture_content=False backstop strips the free-text
    # error/error_message, so without error_type/error_code/status a failure
    # becomes indistinguishable from a benign event. Runs over every uploaded
    # event in every adapter suite — this is the population-complete net that
    # would have caught the error_type-blindness gap at authoring time.
    if event_type == "agent.error" and not any(payload.get(k) for k in ("error_type", "error_code", "status")):
        problems.append(
            f"{tag} agent.error has no surviving category — set error_type "
            "(or error_code/status). The redaction backstop strips the free-text "
            "error under capture_content=False, so the failure would otherwise vanish."
        )

    # INVARIANT (D1/D8, consent-faithful elicitation): an mcp.elicitation in the
    # RESPONSE phase MUST carry an ``action`` in {accept, decline, cancel}. The
    # action is the entire point of the real ElicitResult — a decline/cancel must
    # be distinguishable from an accept downstream (the old code hardcoded
    # "submit" and emitted no action at all, so a refused consent looked identical
    # to a granted one). Fail CLOSED: a response with no/invalid action is a
    # consent-record bug, not a benign omission.
    if event_type == "mcp.elicitation" and payload.get("phase") == "response":
        action = payload.get("action")
        if action not in {"accept", "decline", "cancel"}:
            problems.append(
                f"{tag} mcp.elicitation (phase=response) has action={action!r} — a response MUST "
                "carry action ∈ {accept, decline, cancel} read from the real ElicitResult so a "
                "decline/cancel is distinguishable from an accept (consent-faithful, D1)."
            )
        # A non-accept (refusal) must NOT carry a content-derived hash of a payload
        # the user did not submit — declined/cancelled telemetry hashes nothing.
        if action in {"decline", "cancel"} and payload.get("content_hash") is not None:
            problems.append(
                f"{tag} mcp.elicitation action={action!r} carries a content_hash — a refused "
                "elicitation must not hash a submitted payload (it has none)."
            )

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
