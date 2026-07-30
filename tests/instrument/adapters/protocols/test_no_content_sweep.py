"""L7 — cross-cutting "no SENTINEL under capture_content=False" sweep.

This is the single behavioral guard that catches the whole leak CLASS generically
(and would have caught L1/L3/L4/#4/#5/#12 at once): drive EVERY protocol adapter
through its real ``connect()``/emit path with a workload whose every content
field carries a unique SENTINEL, run it under ``capture_content=False``, and
assert the SENTINEL appears NOWHERE in any serialized event — recursing the whole
payload, not just known keys. It does not depend on remembering to add a
per-field key, so it also catches FUTURE leaks of the same class.

Tagged ``privacy_evidence`` so it doubles as SOC2/GDPR evidence ("no payment /
PII / delegation content leaves the SDK under capture_content=False").

Run both:
* adapter-side  (adapter constructed with capture_content=False), and
* collector-side (default adapter, no-content collector backstop).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, List, Tuple, Callable

import pytest

from layerlens.instrument._context import _current_collector
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig

# Run as a required CI gate via `-m invariant` (see .github/workflows/invariants.yaml).
pytestmark = pytest.mark.invariant

# One canary that looks like every kind of sensitive value at once.
SENTINEL = "SENTINEL-omni-4111111111111111-ACME-49.99-refund"
NO_CONTENT = CaptureConfig(capture_content=False)


def _build_ap2(cfg: CaptureConfig) -> Tuple[Any, Callable[[], None]]:
    # Real ap2 pydantic fixtures (LAY-3625) — skips in the base py3.9 venv where
    # the pinned ap2 SDK is absent; the other adapters in this parametrized sweep
    # still run there.
    pytest.importorskip("ap2")
    from ap2.models.mandate import CartMandate, CartContents, IntentMandate
    from ap2.models.payment_request import (
        PaymentItem,
        PaymentRequest,
        PaymentMethodData,
        PaymentDetailsInit,
        PaymentCurrencyAmount,
    )

    from layerlens.instrument.adapters.protocols.ap2 import AP2Guardrails, AP2ProtocolAdapter

    adapter = AP2ProtocolAdapter(guardrails=AP2Guardrails(merchant_whitelist=["ALLOWED"]), capture_config=cfg)
    adapter.connect(target=SimpleNamespace())  # no SDK to wrap; drive record_* directly

    # The SENTINEL rides every cleartext content surface: intent text + merchant
    # whitelist names, the cart merchant_name + binding total label, and the
    # blocked-guardrail free-text reason.
    item = PaymentItem(label=SENTINEL, amount=PaymentCurrencyAmount(currency="USD", value=49.99))
    details = PaymentDetailsInit(id="pd1", display_items=[item], total=item)
    pr = PaymentRequest(method_data=[PaymentMethodData(supported_methods="card")], details=details)
    cart = CartMandate(
        contents=CartContents(
            id="cart1",
            user_cart_confirmation_required=True,
            payment_request=pr,
            cart_expiry="2999-01-01T00:00:00Z",
            merchant_name=SENTINEL,  # off-whitelist -> blocked, reason embeds SENTINEL
        ),
        merchant_authorization="eyJhbGc.SENTINEL-merchant-sig.xyz",
    )
    intent = IntentMandate(
        natural_language_description=SENTINEL,
        intent_expiry="2999-01-01T00:00:00Z",
        merchants=[SENTINEL],
    )

    def go() -> None:
        adapter.record_intent_mandate(intent, mandate_id="m1")
        # Off-whitelist merchant -> blocked; the policy.violation detail + the
        # cart_mandate event must not leak the SENTINEL merchant/amount.
        try:
            adapter.record_cart_mandate(cart, intent_mandate_id="m1")
        except PermissionError:
            pass

    return adapter, go


def _build_ucp(cfg: CaptureConfig) -> Tuple[Any, Callable[[], None]]:
    from layerlens.instrument.adapters.protocols.ucp import UCPProtocolAdapter

    adapter = UCPProtocolAdapter(capture_config=cfg)
    target = SimpleNamespace(
        discover_suppliers=lambda **kw: [{"id": "s1", "name": SENTINEL}],
        browse_catalog=lambda **kw: [],
        start_checkout=lambda **kw: {"session_id": "c1"},
        complete_checkout=lambda *a, **kw: {"ok": True},
        issue_refund=lambda **kw: {"ok": True},
    )
    adapter.connect(target=target)

    def go() -> None:
        target.discover_suppliers(query=SENTINEL)
        target.browse_catalog(supplier_id="s1", query=SENTINEL)
        target.start_checkout(supplier_id="s1", session_id="c1")
        target.complete_checkout("c1", supplier_id="s1", amount=49.99)
        target.issue_refund(session_id="c1", amount=49.99, reason=SENTINEL)

    return adapter, go


def _build_a2a(cfg: CaptureConfig) -> Tuple[Any, Callable[[], None]]:
    from layerlens.instrument.adapters.protocols.a2a.adapter import A2AProtocolAdapter

    adapter = A2AProtocolAdapter(capture_config=cfg)

    def _fail(**kw: Any) -> Any:
        raise ValueError(f"downstream declined {SENTINEL}")

    target = SimpleNamespace(
        send_task=lambda **kw: {"task_id": "t1", "status": "completed"},
        get_task=_fail,
        cancel_task=lambda **kw: None,
        get_agent_card=lambda *a, **kw: {"id": "peer", "name": SENTINEL, "skills": [SENTINEL]},
        register_handler=lambda **kw: None,
    )
    adapter.connect(target=target)

    def go() -> None:
        target.get_agent_card("peer")
        # Topology ids (from_agent/to_agent) are METADATA that SURVIVES
        # no-content (A15) — so they must NOT carry the SENTINEL. The SENTINEL
        # rides only genuine content: the free-text skill_description.
        target.send_task(
            task_id="t-deleg",
            from_agent="orchestrator",
            to_agent="billing-agent",
            skill_description=SENTINEL,
        )
        try:
            # task_id is an opaque identifier (metadata); the SENTINEL rides the
            # exception text -> a2a.task.updated error: str(exc).
            target.get_task(task_id="task-err")
        except ValueError:
            pass

    return adapter, go


def _build_mcp(cfg: CaptureConfig) -> Tuple[Any, Callable[[], None]]:
    from layerlens.instrument.adapters.protocols.mcp.adapter import MCPProtocolAdapter

    adapter = MCPProtocolAdapter(capture_config=cfg)

    def _raise(name: str, arguments: Any = None, **kw: Any) -> Any:
        raise ValueError(f"charge failed {SENTINEL}")

    # Success then error: cover arguments, result, and error: str(exc).
    state = {"calls": 0}

    def _call(name: str, arguments: Any = None, **kw: Any) -> Any:
        state["calls"] += 1
        if state["calls"] == 1:
            return {"content": [{"type": "text", "text": SENTINEL}]}
        return _raise(name, arguments)

    target = SimpleNamespace(call_tool=_call, list_tools=lambda: {"tools": [{"name": SENTINEL}]})
    adapter.connect(target=target)

    def go() -> None:
        adapter._client.list_tools()
        adapter._client.call_tool("ok", {"q": SENTINEL})
        try:
            adapter._client.call_tool("charge", {"card": SENTINEL})
        except ValueError:
            pass

    return adapter, go


def _build_agui(cfg: CaptureConfig) -> Tuple[Any, Callable[[], None]]:
    from layerlens.instrument.adapters.protocols.agui.adapter import AGUIProtocolAdapter

    adapter = AGUIProtocolAdapter(capture_config=cfg)
    stream = [
        {"type": "TEXT_MESSAGE_CONTENT", "delta": SENTINEL},
        {"type": "TEXT_MESSAGE_END"},
        {"type": "TOOL_CALL_START", "toolCallId": "t", "toolCallName": "fn"},
        {"type": "TOOL_CALL_ARGS", "toolCallId": "t", "delta": json.dumps({"q": SENTINEL})},
        {"type": "TOOL_CALL_END", "toolCallId": "t"},
        {"type": "STATE_SNAPSHOT", "state": {"k": SENTINEL}},
        {"type": "MESSAGES_SNAPSHOT", "messages": [{"role": "user", "content": SENTINEL}]},
        {"type": "TOOL_CALL_RESULT", "toolCallId": "t", "content": SENTINEL},
    ]

    def go() -> None:
        for _ in adapter.wrap_stream(iter(stream)):
            pass

    return adapter, go


# NB: a2ui is intentionally NOT here. It emits only ids/counts and a KEYED
# HMAC of the context — it has no cleartext content surface to redact, so a
# capture_content=False sweep over it is vacuous (it passes even with redaction
# neutered). a2ui's privacy invariant (context hashed, key never reversible) is
# covered by TestA2UIHashing in test_protocol_redaction.py, which bites.
_BUILDERS = {
    "ap2": _build_ap2,
    "ucp": _build_ucp,
    "a2a": _build_a2a,
    "mcp": _build_mcp,
    "agui": _build_agui,
}


# Per-adapter metadata that MUST survive redaction (proves we redact without
# going blind — guards against an over-strip that swallows category/ids/status).
# Each predicate uses values that are NOT the SENTINEL, so it is independent of
# the absence check.
def _survives_metadata(name: str, events: List[dict]) -> str:
    payloads = [e["payload"] for e in events]
    if name == "ap2":
        # The blocked guardrail's reason_code survives on BOTH the policy.violation
        # and the payment.cart_mandate(status=blocked) events (category, not content).
        violations = [e for e in events if e["event_type"] == "policy.violation"]
        if not any(v["payload"].get("reason_code") == "MERCHANT_NOT_WHITELISTED" for v in violations):
            return "ap2: policy.violation reason_code did not survive"
        if not any(
            e["event_type"] == "payment.cart_mandate" and e["payload"].get("cart_id") == "cart1" for e in events
        ):
            return "ap2: cart_mandate cart_id metadata did not survive"
    elif name == "ucp":
        if not any(p.get("supplier_id") == "s1" for p in payloads):
            return "ucp: supplier_id metadata did not survive"
    elif name == "a2a":
        if not any(e["event_type"] == "a2a.task.updated" and e["payload"].get("status") for e in events):
            return "a2a: task.updated status did not survive"
        delegations = [e["payload"] for e in events if e["event_type"] == "a2a.delegation"]
        if not any(d.get("task_id") for d in delegations):
            return "a2a: delegation task_id did not survive"
        # A15: the delegation TOPOLOGY + fp must survive no-content (provenance).
        if not any(d.get("from_agent") == "orchestrator" and d.get("to_agent") == "billing-agent" for d in delegations):
            return "a2a: delegation topology (from_agent/to_agent) did not survive (A15 provenance loss)"
        if not any(str(d.get("delegation_fp", "")).startswith("sha256:") for d in delegations):
            return "a2a: delegation_fp did not survive (A15)"
    elif name == "mcp":
        if not any(e["event_type"] == "mcp.tool.call" and e["payload"].get("tool_name") for e in events):
            return "mcp: tool.call tool_name did not survive"
    elif name == "agui":
        if not any(e["payload"].get("agui_event") or e["payload"].get("tool_name") for e in events):
            return "agui: event-type/tool metadata did not survive"
    return ""


def _run(
    builder: Callable[[CaptureConfig], Tuple[Any, Callable[[], None]]],
    adapter_cfg: CaptureConfig,
    collector_cfg: CaptureConfig,
) -> List[dict]:
    _adapter, go = builder(adapter_cfg)
    collector = TraceCollector(object(), collector_cfg)
    token = _current_collector.set(collector)
    try:
        go()
    finally:
        _current_collector.reset(token)
    return collector.events


@pytest.mark.privacy_evidence
@pytest.mark.parametrize("name", sorted(_BUILDERS))
def test_no_sentinel_adapter_side(name: str) -> None:
    """Adapter constructed with capture_content=False — adapter-side gating."""
    events = _run(_BUILDERS[name], NO_CONTENT, CaptureConfig.standard())
    assert events, f"{name}: no events produced — workload not exercised"
    blob = json.dumps(events, default=str)
    assert SENTINEL not in blob, f"{name}: SENTINEL leaked under adapter capture_content=False:\n{blob}"
    over_strip = _survives_metadata(name, events)
    assert not over_strip, f"over-strip (observability blinded): {over_strip}"


@pytest.mark.privacy_evidence
@pytest.mark.parametrize("name", sorted(_BUILDERS))
def test_no_sentinel_collector_side(name: str) -> None:
    """Default adapter, no-content collector — the universal backstop."""
    events = _run(_BUILDERS[name], CaptureConfig.standard(), NO_CONTENT)
    assert events, f"{name}: no events produced — workload not exercised"
    blob = json.dumps(events, default=str)
    assert SENTINEL not in blob, f"{name}: SENTINEL leaked past collector backstop:\n{blob}"
    over_strip = _survives_metadata(name, events)
    assert not over_strip, f"over-strip (observability blinded): {over_strip}"
