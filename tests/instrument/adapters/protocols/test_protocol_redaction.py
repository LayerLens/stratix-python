"""Protocol content-gating + redaction (LAY-3578 / T4, fixes N7).

``capture_content=False`` must hold for protocol payloads in BOTH places:

* adapter-side — a protocol adapter constructed with a no-content
  ``CaptureConfig`` strips content keys at emit time, and
* collector-side (defense in depth, the accepted B1 pattern) — a no-content
  collector strips protocol content keys even from a default-constructed
  adapter.

Content vs metadata policy under test (flagged for team review in the PR):
message text, tool arguments/results, raw stream payloads, state snapshots,
catalog queries, elicitation titles, a2a request summaries, and — per the
plan-gate decision — payment ``amount``/``merchant``/``cumulative_spend``
and supplier names are CONTENT. Ids, counts, statuses, hashes, latencies,
currencies stay metadata.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from layerlens.instrument._context import _current_collector
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig

SECRET_TEXT = "SENTINEL-the user said something private"
SECRET_QUERY = "SENTINEL-embarrassing search query"
SECRET_MERCHANT = "SENTINEL-ACME Corp"
SECRET_AMOUNT = 1234.56


def _run_collected(
    mock_client: Any,
    fn: Any,
    collector_config: Optional[CaptureConfig] = None,
) -> List[Dict[str, Any]]:
    """Run *fn* with an ambient collector and return its raw events."""
    collector = TraceCollector(mock_client, collector_config or CaptureConfig())
    token = _current_collector.set(collector)
    try:
        fn()
    finally:
        _current_collector.reset(token)
    return collector.events


def _all_payload_text(events: List[Dict[str, Any]]) -> str:
    return json.dumps([e["payload"] for e in events], default=str)


_NO_CONTENT = CaptureConfig(capture_content=False)


# ---------------------------------------------------------------------------
# Constructor surface: every protocol adapter accepts capture_config
# ---------------------------------------------------------------------------


def _adapter_factories():
    from layerlens.instrument.adapters.protocols.ap2 import AP2ProtocolAdapter
    from layerlens.instrument.adapters.protocols.ucp import UCPProtocolAdapter
    from layerlens.instrument.adapters.protocols.a2ui import A2UIProtocolAdapter
    from layerlens.instrument.adapters.protocols.a2a.adapter import A2AProtocolAdapter
    from layerlens.instrument.adapters.protocols.mcp.adapter import MCPProtocolAdapter
    from layerlens.instrument.adapters.protocols.agui.adapter import AGUIProtocolAdapter

    return [
        ("mcp", MCPProtocolAdapter),
        ("agui", AGUIProtocolAdapter),
        ("ap2", AP2ProtocolAdapter),
        ("ucp", UCPProtocolAdapter),
        ("a2ui", A2UIProtocolAdapter),
        ("a2a", A2AProtocolAdapter),
    ]


class TestCaptureConfigSurface:
    @pytest.mark.parametrize("name,cls", _adapter_factories(), ids=lambda v: v if isinstance(v, str) else "")
    def test_adapter_accepts_capture_config(self, name: str, cls: Any) -> None:
        adapter = cls(capture_config=_NO_CONTENT)
        assert adapter is not None


# ---------------------------------------------------------------------------
# MCP — tool arguments/results
# ---------------------------------------------------------------------------


class TestMCPRedaction:
    def _drive(
        self, mock_client: Any, adapter_config: Optional[CaptureConfig], collector_config: Optional[CaptureConfig]
    ) -> List[Dict[str, Any]]:
        from layerlens.instrument.adapters.protocols.mcp.adapter import MCPProtocolAdapter

        adapter = (
            MCPProtocolAdapter(capture_config=adapter_config) if adapter_config is not None else MCPProtocolAdapter()
        )
        target = SimpleNamespace(call_tool=lambda name, arguments=None, **kw: {"content": SECRET_TEXT})
        adapter.connect(target=target)

        def go() -> None:
            target.call_tool(name="search", arguments={"q": SECRET_QUERY})

        return _run_collected(mock_client, go, collector_config)

    def test_content_present_by_default(self, mock_client: Any) -> None:
        events = self._drive(mock_client, None, None)
        text = _all_payload_text(events)
        assert SECRET_QUERY in text
        # Raw results never enter the payload — the adapter summarizes them
        # to a type descriptor by design; only the key's presence is asserted.
        assert SECRET_TEXT not in text
        assert any("result" in e["payload"] for e in events)

    def test_adapter_side_no_content_strips_args_and_result(self, mock_client: Any) -> None:
        events = self._drive(mock_client, _NO_CONTENT, None)
        text = _all_payload_text(events)
        assert SECRET_QUERY not in text, "tool arguments leaked despite adapter capture_content=False (N7)"
        # `result` is summarized to a descriptor before emit AND stripped by
        # redaction under no-content — assert the key is gone (bites on the
        # redaction fix; SECRET_TEXT-absence alone would be carried by the
        # summarization, not the redaction).
        calls = [e["payload"] for e in events if e["event_type"] == "mcp.tool.call"]
        assert calls and all("result" not in p for p in calls), "mcp.tool.call result not stripped under no-content"
        assert any(p.get("tool_name") == "search" for p in calls), "metadata over-stripped"

    def test_collector_side_no_content_strips_args_and_result(self, mock_client: Any) -> None:
        events = self._drive(mock_client, None, _NO_CONTENT)
        text = _all_payload_text(events)
        assert SECRET_QUERY not in text, "collector backstop missing for mcp.tool.call arguments (N7)"
        calls = [e["payload"] for e in events if e["event_type"] == "mcp.tool.call"]
        assert calls and all("result" not in p for p in calls), (
            "collector backstop missing for mcp.tool.call result (N7)"
        )

    # --- L3: the ERROR path puts str(exc) into 'error', bypassing args redaction ---

    def _drive_error(self, mock_client: Any, adapter_config: Optional[CaptureConfig]) -> List[Dict[str, Any]]:
        from layerlens.instrument.adapters.protocols.mcp.adapter import MCPProtocolAdapter

        def _raise(name: str, arguments: Any = None, **kw: Any) -> Any:
            # Real tool failures routinely echo the failing argument back.
            raise ValueError(f"charge failed for card {SECRET_QUERY}")

        adapter = MCPProtocolAdapter(capture_config=adapter_config)
        target = SimpleNamespace(call_tool=_raise)
        adapter.connect(target=target)

        def go() -> None:
            with pytest.raises(ValueError):
                target.call_tool(name="charge", arguments={"card": SECRET_QUERY})

        return _run_collected(mock_client, go, None)

    def test_error_path_no_content_strips_exception_string(self, mock_client: Any) -> None:
        events = self._drive_error(mock_client, _NO_CONTENT)
        text = _all_payload_text(events)
        assert SECRET_QUERY not in text, "mcp error path leaked str(exc) under capture_content=False (L3)"
        # Redact without going blind: the tool.call still carries the tool_name,
        # and the failure stays observable via the async-task lifecycle.
        calls = [e["payload"] for e in events if e["event_type"] == "mcp.tool.call"]
        assert calls and any(p.get("tool_name") == "charge" for p in calls), "tool_name over-stripped"
        assert any(e["event_type"] == "mcp.async_task" and e["payload"].get("status") == "failed" for e in events), (
            "failure no longer observable — over-stripped"
        )

    def test_error_path_default_captures_exception_string(self, mock_client: Any) -> None:
        # Sanity: under the default config the exception string IS captured, so
        # the no-content assertion above is meaningful (not vacuous).
        events = self._drive_error(mock_client, None)
        assert SECRET_QUERY in _all_payload_text(events), (
            "error str(exc) not captured by default — test would be vacuous"
        )


# ---------------------------------------------------------------------------
# AG-UI — message text + tool-call arguments + raw passthrough payloads
# ---------------------------------------------------------------------------

_AGUI_EVENTS = [
    {"type": "TEXT_MESSAGE_CONTENT", "delta": SECRET_TEXT},
    {"type": "TEXT_MESSAGE_END"},
    {"type": "TOOL_CALL_START", "toolCallId": "tc1", "toolCallName": "lookup"},
    {"type": "TOOL_CALL_ARGS", "toolCallId": "tc1", "delta": json.dumps({"q": SECRET_QUERY})},
    {"type": "TOOL_CALL_END", "toolCallId": "tc1"},
]


class TestAGUIRedaction:
    def _drive(
        self, mock_client: Any, adapter_config: Optional[CaptureConfig], collector_config: Optional[CaptureConfig]
    ) -> List[Dict[str, Any]]:
        from layerlens.instrument.adapters.protocols.agui.adapter import AGUIProtocolAdapter

        adapter = (
            AGUIProtocolAdapter(capture_config=adapter_config) if adapter_config is not None else AGUIProtocolAdapter()
        )

        def go() -> None:
            for _ in adapter.wrap_stream(iter(_AGUI_EVENTS)):
                pass

        return _run_collected(mock_client, go, collector_config)

    def test_content_present_by_default(self, mock_client: Any) -> None:
        text = _all_payload_text(self._drive(mock_client, None, None))
        assert SECRET_TEXT in text and SECRET_QUERY in text

    def test_adapter_side_no_content_strips_text_and_args(self, mock_client: Any) -> None:
        events = self._drive(mock_client, _NO_CONTENT, None)
        text = _all_payload_text(events)
        assert SECRET_TEXT not in text, "agui.message text leaked despite adapter capture_content=False (N7)"
        assert SECRET_QUERY not in text, "agui.tool_call arguments leaked (N7)"
        assert any(e["payload"].get("tool_name") == "lookup" for e in events), "metadata over-stripped"

    def test_collector_side_no_content_strips_text_and_args(self, mock_client: Any) -> None:
        text = _all_payload_text(self._drive(mock_client, None, _NO_CONTENT))
        assert SECRET_TEXT not in text, "collector backstop missing for agui.message text (N7)"
        assert SECRET_QUERY not in text, "collector backstop missing for agui.tool_call arguments (N7)"


# ---------------------------------------------------------------------------
# AP2 — payment amount / merchant (approved: financial details are CONTENT)
# ---------------------------------------------------------------------------


class TestAP2Redaction:
    def _drive(
        self, mock_client: Any, adapter_config: Optional[CaptureConfig], collector_config: Optional[CaptureConfig]
    ) -> List[Dict[str, Any]]:
        from layerlens.instrument.adapters.protocols.ap2 import AP2ProtocolAdapter

        adapter = (
            AP2ProtocolAdapter(capture_config=adapter_config) if adapter_config is not None else AP2ProtocolAdapter()
        )
        target = SimpleNamespace(
            create_intent_mandate=lambda **kw: {"ok": True},
            sign_payment_mandate=lambda **kw: {"ok": True},
            issue_receipt=lambda **kw: {"ok": True},
        )
        adapter.connect(target=target)

        def go() -> None:
            target.issue_receipt(
                mandate_id="m1",
                amount=SECRET_AMOUNT,
                merchant=SECRET_MERCHANT,
            )

        return _run_collected(mock_client, go, collector_config)

    def test_content_present_by_default(self, mock_client: Any) -> None:
        text = _all_payload_text(self._drive(mock_client, None, None))
        assert SECRET_MERCHANT in text and str(SECRET_AMOUNT) in text

    def test_adapter_side_no_content_strips_amount_and_merchant(self, mock_client: Any) -> None:
        events = self._drive(mock_client, _NO_CONTENT, None)
        text = _all_payload_text(events)
        assert SECRET_MERCHANT not in text, "merchant leaked despite adapter capture_content=False (N7)"
        assert str(SECRET_AMOUNT) not in text, "payment amount leaked despite adapter capture_content=False (N7)"
        assert any(e["payload"].get("mandate_id") == "m1" for e in events), "metadata over-stripped"

    def test_collector_side_no_content_strips_amount_and_merchant(self, mock_client: Any) -> None:
        text = _all_payload_text(self._drive(mock_client, None, _NO_CONTENT))
        assert SECRET_MERCHANT not in text, "collector backstop missing for payment merchant (N7)"
        assert str(SECRET_AMOUNT) not in text, "collector backstop missing for payment amount (N7)"

    # --- L1: the BLOCKED guardrail path (the riskier, previously-untested one) ---

    def _drive_blocked(
        self, mock_client: Any, adapter_config: Optional[CaptureConfig], guardrails: Any, **sign_kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Drive a guardrail-BLOCKED sign through the REAL adapter."""
        from layerlens.instrument.adapters.protocols.ap2 import AP2ProtocolAdapter

        adapter = AP2ProtocolAdapter(guardrails=guardrails, capture_config=adapter_config)
        target = SimpleNamespace(sign_payment_mandate=lambda **kw: {"ok": True})
        adapter.connect(target=target)

        def go() -> None:
            with pytest.raises(PermissionError):
                target.sign_payment_mandate(**sign_kwargs)

        return _run_collected(mock_client, go, None)

    def test_blocked_over_cap_default_captures_reason(self, mock_client: Any) -> None:
        from layerlens.instrument.adapters.protocols.ap2 import AP2Guardrails

        events = self._drive_blocked(
            mock_client, None, AP2Guardrails(max_transaction=10.0), mandate_id="m1", amount=SECRET_AMOUNT, merchant="ok"
        )
        blocked = [e for e in events if e["payload"].get("status") == "blocked"]
        assert blocked, "no blocked payment.mandate_signed event emitted"
        # Default (content captured): the amount detail AND the reason_code are present.
        assert str(SECRET_AMOUNT) in _all_payload_text(events), "default should still carry the reason detail"
        assert blocked[0]["payload"].get("reason_code") == "MAX_TRANSACTION_EXCEEDED"

    def test_blocked_over_cap_no_content_strips_amount_keeps_reason_code(self, mock_client: Any) -> None:
        from layerlens.instrument.adapters.protocols.ap2 import AP2Guardrails

        events = self._drive_blocked(
            mock_client,
            _NO_CONTENT,
            AP2Guardrails(max_transaction=10.0),
            mandate_id="m1",
            amount=SECRET_AMOUNT,
            merchant="ok",
        )
        text = _all_payload_text(events)
        assert str(SECRET_AMOUNT) not in text, "blocked-payment reason leaked amount under capture_content=False (L1)"
        blocked = [e for e in events if e["payload"].get("status") == "blocked"]
        assert blocked, "blocked event suppressed entirely — over-stripped"
        assert blocked[0]["payload"].get("reason_code") == "MAX_TRANSACTION_EXCEEDED", (
            "reason_code (why blocked) over-stripped — observability blinded"
        )

    def test_blocked_off_whitelist_no_content_strips_merchant_keeps_reason_code(self, mock_client: Any) -> None:
        from layerlens.instrument.adapters.protocols.ap2 import AP2Guardrails

        events = self._drive_blocked(
            mock_client,
            _NO_CONTENT,
            AP2Guardrails(merchant_whitelist=["ALLOWED-MERCHANT"]),
            mandate_id="m1",
            amount=5,
            merchant=SECRET_MERCHANT,
        )
        text = _all_payload_text(events)
        assert SECRET_MERCHANT not in text, "blocked-payment reason leaked merchant under capture_content=False (L1)"
        blocked = [e for e in events if e["payload"].get("status") == "blocked"]
        assert blocked and blocked[0]["payload"].get("reason_code") == "MERCHANT_NOT_WHITELISTED", (
            "reason_code (why blocked) over-stripped — observability blinded"
        )


# ---------------------------------------------------------------------------
# UCP — checkout amount + catalog query
# ---------------------------------------------------------------------------


class TestUCPRedaction:
    def _drive(
        self, mock_client: Any, adapter_config: Optional[CaptureConfig], collector_config: Optional[CaptureConfig]
    ) -> List[Dict[str, Any]]:
        from layerlens.instrument.adapters.protocols.ucp import UCPProtocolAdapter

        adapter = (
            UCPProtocolAdapter(capture_config=adapter_config) if adapter_config is not None else UCPProtocolAdapter()
        )
        target = SimpleNamespace(
            discover_suppliers=lambda **kw: [],
            browse_catalog=lambda **kw: [],
            start_checkout=lambda **kw: {"session_id": "s1"},
            complete_checkout=lambda **kw: {"ok": True},
            issue_refund=lambda **kw: {"ok": True},
        )
        adapter.connect(target=target)

        def go() -> None:
            target.browse_catalog(supplier_id="sup1", query=SECRET_QUERY)
            target.start_checkout(supplier_id="sup1")
            target.complete_checkout(supplier_id="sup1", amount=SECRET_AMOUNT)

        return _run_collected(mock_client, go, collector_config)

    def test_content_present_by_default(self, mock_client: Any) -> None:
        text = _all_payload_text(self._drive(mock_client, None, None))
        assert SECRET_QUERY in text and str(SECRET_AMOUNT) in text

    def test_adapter_side_no_content_strips_query_and_amount(self, mock_client: Any) -> None:
        events = self._drive(mock_client, _NO_CONTENT, None)
        text = _all_payload_text(events)
        assert SECRET_QUERY not in text, "catalog query leaked despite adapter capture_content=False (N7)"
        assert str(SECRET_AMOUNT) not in text, "checkout amount leaked despite adapter capture_content=False (N7)"
        assert any(e["payload"].get("supplier_id") == "sup1" for e in events), "metadata over-stripped"

    def test_collector_side_no_content_strips_query_and_amount(self, mock_client: Any) -> None:
        text = _all_payload_text(self._drive(mock_client, None, _NO_CONTENT))
        assert SECRET_QUERY not in text, "collector backstop missing for ucp catalog query (N7)"
        assert str(SECRET_AMOUNT) not in text, "collector backstop missing for ucp checkout amount (N7)"


# ---------------------------------------------------------------------------
# A2A — request summary on task creation
# ---------------------------------------------------------------------------


# A2A delegation: who delegates WHICH skill to WHOM is the core delegation
# signal (business-sensitive). The SENTINELs go on ``agent_id``/``skill`` (the
# fields the adapter actually emits as a2a.delegation target_agent/skill) — NOT
# on ``message``, which _summarize drops BEFORE redaction runs (the old test
# asserted ``message`` absent and so passed whether redaction worked or not —
# vacuous; LAY-3578 / L4).
SECRET_AGENT = "SENTINEL-billing-agent"
SECRET_SKILL = "SENTINEL-process-refund"


class TestA2ARedaction:
    def _drive(
        self, mock_client: Any, adapter_config: Optional[CaptureConfig], collector_config: Optional[CaptureConfig]
    ) -> List[Dict[str, Any]]:
        from layerlens.instrument.adapters.protocols.a2a.adapter import A2AProtocolAdapter

        adapter = (
            A2AProtocolAdapter(capture_config=adapter_config) if adapter_config is not None else A2AProtocolAdapter()
        )
        target = SimpleNamespace(
            send_task=lambda **kw: {"task_id": "t1"},
            get_task=lambda **kw: {"task_id": "t1"},
            cancel_task=lambda **kw: None,
            get_agent_card=lambda **kw: {"name": "agent"},
            register_handler=lambda **kw: None,
        )
        adapter.connect(target=target)

        def go() -> None:
            target.send_task(agent_id=SECRET_AGENT, skill=SECRET_SKILL, message=SECRET_TEXT)

        return _run_collected(mock_client, go, collector_config)

    def test_content_present_by_default(self, mock_client: Any) -> None:
        # Sanity: the delegation edge IS captured under the default config, so
        # the redaction assertions below are meaningful (not vacuous).
        events = self._drive(mock_client, None, None)
        delegations = [e for e in events if e["event_type"] == "a2a.delegation"]
        assert delegations, "no a2a.delegation event emitted — test would be vacuous"
        text = _all_payload_text(delegations)
        assert SECRET_AGENT in text and SECRET_SKILL in text

    def test_adapter_side_no_content_strips_delegation_target_and_skill(self, mock_client: Any) -> None:
        events = self._drive(mock_client, _NO_CONTENT, None)
        text = _all_payload_text(events)
        assert SECRET_AGENT not in text, "a2a.delegation target_agent leaked despite capture_content=False (L4)"
        assert SECRET_SKILL not in text, "a2a.delegation skill leaked despite capture_content=False (L4)"
        # metadata (task_id) must survive so the delegation edge is still visible
        assert any(e["payload"].get("task_id") for e in events if e["event_type"] == "a2a.delegation"), (
            "delegation task_id over-stripped"
        )

    def test_collector_side_no_content_strips_delegation_target_and_skill(self, mock_client: Any) -> None:
        text = _all_payload_text(self._drive(mock_client, None, _NO_CONTENT))
        assert SECRET_AGENT not in text, "collector backstop missing for a2a.delegation target_agent (L4)"
        assert SECRET_SKILL not in text, "collector backstop missing for a2a.delegation skill (L4)"


# ---------------------------------------------------------------------------
# AG-UI fallback + middleware — raw SSE event passthrough (#4/#5/#12)
# ---------------------------------------------------------------------------


class TestAGUIFallbackRedaction:
    """Un-handled AG-UI events fall through to a raw passthrough that rides
    agent.state.change (MESSAGES_SNAPSHOT) / tool.call (TOOL_CALL_RESULT) with
    the entire raw event under ``payload`` (adapter) / ``data`` (middleware).
    Under capture_content=False the raw event must not survive."""

    def _drive_stream(
        self, mock_client: Any, adapter_config: Optional[CaptureConfig], collector_config: Optional[CaptureConfig]
    ) -> List[Dict[str, Any]]:
        from layerlens.instrument.adapters.protocols.agui.adapter import AGUIProtocolAdapter

        stream = [
            {"type": "MESSAGES_SNAPSHOT", "messages": [{"role": "user", "content": SECRET_TEXT}]},
            {"type": "TOOL_CALL_RESULT", "toolCallId": "tc1", "content": SECRET_QUERY},
        ]
        adapter = (
            AGUIProtocolAdapter(capture_config=adapter_config) if adapter_config is not None else AGUIProtocolAdapter()
        )

        def go() -> None:
            for _ in adapter.wrap_stream(iter(stream)):
                pass

        return _run_collected(mock_client, go, collector_config)

    def test_content_present_by_default(self, mock_client: Any) -> None:
        text = _all_payload_text(self._drive_stream(mock_client, None, None))
        assert SECRET_TEXT in text and SECRET_QUERY in text, "raw passthrough not captured — test would be vacuous"

    def test_adapter_side_no_content_strips_raw_event(self, mock_client: Any) -> None:
        text = _all_payload_text(self._drive_stream(mock_client, _NO_CONTENT, None))
        assert SECRET_TEXT not in text, "agui MESSAGES_SNAPSHOT raw payload leaked under capture_content=False (#5)"
        assert SECRET_QUERY not in text, "agui TOOL_CALL_RESULT raw payload leaked under capture_content=False (#12)"

    def test_collector_side_no_content_strips_raw_event(self, mock_client: Any) -> None:
        text = _all_payload_text(self._drive_stream(mock_client, None, _NO_CONTENT))
        assert SECRET_TEXT not in text and SECRET_QUERY not in text, (
            "collector backstop missing for agui raw passthrough"
        )

    def test_middleware_no_content_strips_raw_data(self, mock_client: Any) -> None:
        from layerlens.instrument.adapters.protocols.agui.adapter import AGUIProtocolAdapter
        from layerlens.instrument.adapters.protocols.agui.middleware import _process_sse_chunk

        adapter = AGUIProtocolAdapter(capture_config=_NO_CONTENT)
        chunk = (
            'data: {"type": "MESSAGES_SNAPSHOT", "messages": [{"role": "user", "content": "' + SECRET_TEXT + '"}]}\n'
            'data: {"type": "TOOL_CALL_RESULT", "toolCallId": "tc1", "content": "' + SECRET_QUERY + '"}\n'
        ).encode()

        def go() -> None:
            _process_sse_chunk(adapter, chunk)

        text = _all_payload_text(_run_collected(mock_client, go, None))
        assert SECRET_TEXT not in text, "agui middleware raw 'data' leaked under capture_content=False (#4)"
        assert SECRET_QUERY not in text, "agui middleware raw 'data' leaked under capture_content=False (#4)"


# ---------------------------------------------------------------------------
# A2UI — the hash IS the privacy story; it must be keyed, not plain SHA-256 (P3)
# ---------------------------------------------------------------------------


class TestA2UIHashing:
    def _emit_hash(self, mock_client: Any, adapter: Any) -> str:
        def go() -> None:
            adapter.record_user_action(surface_id="cart-1", action_type="add_to_cart", context={"amount": "49.99"})

        events = _run_collected(mock_client, go, None)
        actions = [e for e in events if e["event_type"] == "commerce.ui.user_action"]
        assert actions, "no commerce.ui.user_action emitted"
        return actions[0]["payload"]["action_context_hash"]

    def test_hash_is_keyed_not_plain_sha256(self, mock_client: Any) -> None:
        import hashlib

        from layerlens.instrument.adapters.protocols.a2ui import A2UIProtocolAdapter

        ctx = {"amount": "49.99"}
        plain = "sha256:" + hashlib.sha256(str(ctx).encode()).hexdigest()
        emitted = self._emit_hash(mock_client, A2UIProtocolAdapter())
        assert emitted != plain, "action_context_hash is plain unsalted SHA-256 — trivially reversible (P3)"

    def test_hash_is_per_instance_keyed(self, mock_client: Any) -> None:
        from layerlens.instrument.adapters.protocols.a2ui import A2UIProtocolAdapter

        h_a = self._emit_hash(mock_client, A2UIProtocolAdapter())
        h_b = self._emit_hash(mock_client, A2UIProtocolAdapter())
        assert h_a != h_b, "same value -> same digest across instances: not keyed, rainbow-reversible (P3)"
