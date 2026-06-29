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
    # MCP sampling is a nested LLM call (the server samples the client's model),
    # so it is L3 model metadata — l3-off / minimal() suppresses it like any
    # other model.invoke. Its paired cost.record is _ALWAYS_ENABLED (cost).
    "mcp.sampling": "l3_model_metadata",
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
    "payment.cart_mandate": "l6c_protocol_lifecycle",
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
#
# Commerce / payment PII + PCI field names (A15 / UCP-Q2). Shared across the
# commerce.* content-key sets so a checkout/refund hook that carries the buyer's
# address or card cannot leak it under capture_content=False. Names cover both
# real specs the census pinned: Google UCP v2026-04-08 (``billing_address``,
# ``shipping_address``, ``payment``/``payment_instrument``/``instrument``,
# ``credential``, ``card``) and ACP/OpenAI+Stripe 2026-04-17
# (``PaymentMethodCard.number``/``cvc``/``exp_month``/``exp_year``, ``Buyer``
# email/phone, ``Address`` lines). ``card``/``payment``/``instrument`` strip the
# whole nested card/instrument object (the recursive stripper removes the key
# wherever it appears); the individual leaf names (``pan``/``number``/``cvc``…)
# catch a card whose fields were flattened to the top level.
_COMMERCE_PII_KEYS: FrozenSet[str] = frozenset(
    {
        # addresses (whole objects + flattened buyer-identity leaves)
        "billing_address",
        "shipping_address",
        "address",
        "email",
        "phone",
        "phone_number",
        "first_name",
        "last_name",
        "full_name",
        # card / payment instrument (whole objects)
        "card",
        "payment",
        "payment_instrument",
        "instrument",
        "credential",
        "payment_method",
        # flattened card leaves (PCI: PAN + CVC + expiry)
        "pan",
        "number",
        "card_number",
        "cvc",
        "cvv",
        "exp_month",
        "exp_year",
    }
)

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
    # MCP elicitation (D1/D2). The user-facing prompt rides the REAL field name
    # ``message`` ("Enter card number to confirm $499" — content, not a metadata
    # title); ``title`` is kept for back-compat. ``content_hash`` is a hash
    # DERIVED from the submitted form data (only present on accept) — a
    # content-derived value, so it is stripped under no-content. The surviving
    # fields are the consent CATEGORY (``action`` ∈ accept/decline/cancel), the
    # ``mode`` (form/url), ids, and latency — so a refusal is auditable WITHOUT
    # leaking what was (or would have been) submitted.
    "mcp.elicitation": frozenset({"title", "message", "content_hash", "response_hash"}),
    # MCP sampling (D3): the sampled completion text (``output``/``content``) and
    # the request prompt (``messages``/``system_prompt``) are content; the
    # surviving fields are model id, token counts, stop_reason, latency.
    "mcp.sampling": frozenset({"messages", "system_prompt", "content", "output", "prompt"}),
    "mcp.structured_output": frozenset({"validation_errors"}),
    "mcp.tools.listed": frozenset({"tool_names"}),
    "a2a.task.created": frozenset({"request", "skill", "skill_description"}),
    "a2a.task.updated": frozenset({"error", "error_message"}),
    # A2A delegation provenance (A15 / D3, user-approved 2026-06-25). The
    # DELEGATION TOPOLOGY survives capture_content=False so cross-agent
    # provenance is auditable under privacy-on (mirrors agent.handoff keeping
    # from_agent/to_agent): the redaction-SURVIVING fields are the delegator
    # id (``from_agent``), the delegatee ids (``to_agent``/``target_agent``),
    # the ``task_id``, and the keyed-HMAC ``delegation_fp`` of (target+skill)
    # for server-anchored verification. STRIPPED as content: the free-text
    # ``skill_description`` (what the skill DOES), the ``target_url``, and any
    # free-text ``context``/``skill`` blob.
    "a2a.delegation": frozenset({"skill", "skill_description", "target_url", "context"}),
    # Agent-card discovery/serving (D2). Free-text ``name``/``skills``/
    # ``description`` are content; the SURVIVING provenance is the signature
    # PRESENCE (``signature_present``/``signature_count``), the keyed-HMAC
    # ``signature_fp`` (never the raw JWS), ``agent_id``, ``protocolVersion``,
    # and ``authScheme``.
    "a2a.agent.discovered": frozenset({"name", "skills", "description"}),
    "a2a.agent.card.served": frozenset({"name", "skills", "description"}),
    # AP2 v0.2 (LAY-3625). Financial details are CONTENT: the binding cart
    # ``amount`` (the total value), the ``merchant``/``merchant_name`` the user
    # buys from, the intent ``merchants`` whitelist (a list of merchant names),
    # the user's free-text ``description`` (natural_language_description), the
    # running ``cumulative_spend``, and the free-text ``reason``/``detail`` that
    # interpolate amount/merchant. The redaction-SURVIVING fields are deliberately
    # NOT here: ids (cart_id/mandate ids/receipt_id), ``status``, ``reason_code``,
    # ``currency`` (a code, not a sum), ``cart_expiry``/``intent_expiry`` (an
    # instant, not money), and the merchant-signature PRESENCE + keyed-HMAC
    # ``merchant_signature_fp`` (provenance, not the raw JWT) — so a customer can
    # still audit WHO/WHEN/WHY under capture_content=False.
    "payment.intent_mandate": frozenset({"amount", "merchant", "merchants", "description"}),
    "payment.cart_mandate": frozenset({"amount", "merchant", "merchant_name", "description"}),
    "payment.mandate_signed": frozenset({"amount", "cumulative_spend", "reason", "merchant", "merchant_name"}),
    "payment.receipt_issued": frozenset({"amount", "merchant", "merchant_name"}),
    # Payment-guard block: the free-text ``reason``/``detail`` interpolate the
    # over-cap amount / off-whitelist merchant; ``reason_code`` (the category)
    # stays so a customer sees WHY a charge was refused under no-content.
    # (bedrock_agents' policy.violation carries action/stage/policies/ids only —
    # none of these keys — so this is a no-op for it.)
    "policy.violation": frozenset({"reason", "detail", "amount", "merchant", "merchant_name"}),
    "commerce.supplier_discovered": frozenset({"name"}),
    "commerce.catalog.browsed": frozenset({"query"}),
    # Commerce checkout/start PII (A15 / UCP-Q2 fail-open, user-approved
    # 2026-06-25). A checkout can carry the buyer's billing/shipping address,
    # the card / PAN / CVC / expiry, the tokenized payment instrument, and the
    # buyer's email/phone/name — all CONTENT that must be stripped under
    # capture_content=False (ACP rfc.delegate_payment §277 "logs MUST NOT
    # contain full PAN or CVC"; UCP "Never log raw credentials"). The previous
    # set listed only ``amount``, so any of these fields on a real checkout hook
    # LEAKED under no-content. The SURVIVING metadata is the financial-flow
    # skeleton: ids (session_id/supplier_id/order_id), ``currency`` (a code, not
    # a sum), counts, statuses, and latencies — so a customer can still audit
    # WHO/WHEN/HOW-MUCH without leaking the card or the address. (The amount-as-
    # a-value is content; the currency code is metadata.) The card/PAN/CVC are
    # ALSO scrubbed at the collector chokepoint regardless of capture_content
    # (_secret_scrub.SECRET_PATTERNS PAN/CVC) — defense in depth.
    "commerce.checkout.started": frozenset(_COMMERCE_PII_KEYS),
    "commerce.checkout_completed": frozenset({"amount"} | _COMMERCE_PII_KEYS),
    "commerce.refund_issued": frozenset({"amount", "reason"} | _COMMERCE_PII_KEYS),
}

# Backwards-compatible alias: the protocol subset (the original LAY-3578 map).
PROTOCOL_CONTENT_KEYS: Dict[str, FrozenSet[str]] = {
    k: v
    for k, v in _CONTENT_KEYS.items()
    if k.split(".")[0] in {"agui", "mcp", "a2a", "payment", "commerce", "protocol"}
}


def known_event_types() -> FrozenSet[str]:
    """The RUNTIME set of registered event-type strings.

    The single source of truth for "is this a real layerlens event type" inside
    ``src/`` — the union of every type the capture config knows how to gate
    (``_EVENT_TYPE_MAP``), every always-emitted type (``_ALWAYS_ENABLED``), and
    every type with a declared content-key set (``_CONTENT_KEYS``). The test-side
    schema lock (``tests/instrument/_event_schema.py::KNOWN_EVENT_TYPES``) is the
    mirror of this used in CI; this function is what runtime code (e.g. the
    replay fail-closed gate) consults, since ``src/`` cannot import from
    ``tests/``.

    Used by :func:`layerlens.replay.snapshot.replay_events` /
    :func:`load_snapshot` to REJECT a recorded event whose ``event_type`` is not
    registered (fail closed) — a garbage/forged recorded trace must not inject an
    arbitrary type into a fresh collector that then uploads it.
    """
    return frozenset(_EVENT_TYPE_MAP) | _ALWAYS_ENABLED | frozenset(_CONTENT_KEYS)


def _strip_content_keys(value: Any, content_keys: FrozenSet[str]) -> Any:
    """Recursively drop any key in *content_keys* from dicts anywhere in *value*.

    Returns a redacted copy; non-dict/list scalars pass through unchanged.
    Lists/tuples are walked element-wise so content nested in a list survives
    neither at the top level nor inside a wrapper object.
    """
    if isinstance(value, dict):
        return {k: _strip_content_keys(v, content_keys) for k, v in value.items() if k not in content_keys}
    if isinstance(value, list):
        return [_strip_content_keys(v, content_keys) for v in value]
    if isinstance(value, tuple):
        return tuple(_strip_content_keys(v, content_keys) for v in value)
    return value


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
    # Privacy-by-default (A10 / LAY-3628, product-approved 2026-06-25): the
    # out-of-the-box default REDACTS content. The default config (and standard()/
    # minimal()) emit structure + metadata only; raw prompt/response/tool content
    # is opt-in via full() or an explicit capture_content=True. This makes a
    # forgotten _CONTENT_KEYS entry fail CLOSED (the redactor runs by default)
    # rather than leaking a whole field. Secrets are scrubbed independently of
    # this flag at the collector chokepoint (_secret_scrub.scrub_payload).
    capture_content: bool = False

    def redact_payload(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of *payload* with content fields removed per config.

        When ``capture_content`` is False this is the COLLECTOR-SIDE BACKSTOP:
        it strips every content field named in :data:`_CONTENT_KEYS` for the
        event type, regardless of whether the emitting adapter remembered to
        gate at emit time. Category/metadata (error_type, reason_code,
        tool_name, ids, counts, statuses, hashes, latencies, topology) is
        preserved so redaction does not blind observability. ``model.invoke``
        additionally has content stripped out of its ``parameters`` sub-dict.

        Stripping is RECURSIVE (LAY-3572 / R1): a content key is removed wherever
        it appears in the payload tree — top-level, inside a ``metadata``/``extra``
        wrapper, or nested in a list element — because adapters routinely nest
        content (``model.invoke.parameters``, langgraph graph state, AG-UI raw
        passthrough). A non-recursive (top-level-only) strip left nested content
        leaking under ``capture_content=False``.
        """
        if self.capture_content:
            return payload
        content_keys = _CONTENT_KEYS.get(event_type)
        if content_keys:
            payload = _strip_content_keys(payload, content_keys)
        if event_type == "model.invoke":
            parameters = payload.get("parameters")
            if isinstance(parameters, dict):
                payload = {**payload, "parameters": _strip_content_keys(parameters, _CONTENT_PARAM_KEYS)}
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
            capture_content=False,
        )

    @classmethod
    def standard(cls) -> CaptureConfig:
        """Balanced telemetry: agent I/O, model metadata, tools, protocols. Same
        as the default — privacy-by-default, so content is redacted. Opt into raw
        content with ``full()`` or ``capture_content=True``."""
        return cls()

    @classmethod
    def full(cls) -> CaptureConfig:
        """Full capture: all layers enabled INCLUDING raw content. Development/
        debugging or an explicit content-capture opt-in (privacy review required)."""
        return cls(
            l2_agent_code=True,
            l4b_environment_metrics=True,
            l5b_tool_logic=True,
            l5c_tool_environment=True,
            capture_content=True,
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
