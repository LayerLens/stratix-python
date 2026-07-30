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
_CONTENT = CaptureConfig(capture_content=True)


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
        events = self._drive(mock_client, _CONTENT, _CONTENT)
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

        return _run_collected(mock_client, go, adapter_config)

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
        # Sanity: under a content-capturing config the exception string IS
        # captured, so the no-content assertion above is meaningful (not vacuous).
        events = self._drive_error(mock_client, _CONTENT)
        assert SECRET_QUERY in _all_payload_text(events), (
            "error str(exc) not captured under capture_content=True — test would be vacuous"
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
        text = _all_payload_text(self._drive(mock_client, _CONTENT, _CONTENT))
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
# AP2 redaction — moved to test_ap2_payment.py (real ap2 pydantic fixtures,
# LAY-3625). The adapter now operates on real IntentMandate/CartMandate/
# PaymentMandate objects from the pinned ap2 SDK, so its redaction tests live
# next to the real fixtures and importorskip("ap2"). The cross-adapter
# capture_content=False sweep in test_no_content_sweep.py still covers ap2.
# ---------------------------------------------------------------------------


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
        text = _all_payload_text(self._drive(mock_client, _CONTENT, _CONTENT))
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


# A2A delegation provenance (A15 / D3, user-approved 2026-06-25). OVERTURN of the
# old (mis-aimed) lock: the delegation TOPOLOGY — who delegated to whom — must
# SURVIVE capture_content=False so cross-agent provenance is auditable under
# privacy-on (mirrors agent.handoff keeping from_agent/to_agent). Only the
# free-text skill DESCRIPTION (what the skill does) is content. So the SENTINEL
# rides ``skill_description`` (must be stripped); the topology ids are OPAQUE
# non-SENTINEL ids whose SURVIVAL is asserted positively. The old test put the
# SENTINEL on the delegatee id and asserted it stripped — that locked in
# provenance-loss-under-no-content, the exact A15 bug we are fixing.
SECRET_SKILL = "SENTINEL-process-refunds-over-10k-no-approval"
DELEGATOR_ID = "orchestrator-1"  # opaque id (metadata) — must SURVIVE
DELEGATEE_ID = "billing-agent-7"  # opaque id (metadata) — must SURVIVE


class TestA2ADelegationProvenance:
    def _drive(
        self, mock_client: Any, adapter_config: Optional[CaptureConfig], collector_config: Optional[CaptureConfig]
    ) -> List[Dict[str, Any]]:
        from layerlens.instrument.adapters.protocols.a2a.adapter import A2AProtocolAdapter

        adapter = (
            A2AProtocolAdapter(capture_config=adapter_config) if adapter_config is not None else A2AProtocolAdapter()
        )
        target = SimpleNamespace(
            send_task=lambda **kw: {"status": "completed"},
            get_task=lambda **kw: {"status": "completed"},
            cancel_task=lambda **kw: None,
            get_agent_card=lambda **kw: {"name": "agent"},
            register_handler=lambda **kw: None,
        )
        adapter.connect(target=target)

        def go() -> None:
            target.send_task(
                task_id="t1",
                from_agent=DELEGATOR_ID,
                to_agent=DELEGATEE_ID,
                skill_description=SECRET_SKILL,
            )

        return _run_collected(mock_client, go, collector_config)

    def _delegation(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        deleg = [e["payload"] for e in events if e["event_type"] == "a2a.delegation"]
        assert deleg, "no a2a.delegation event emitted"
        return deleg[0]

    def test_content_present_by_default(self, mock_client: Any) -> None:
        # Sanity: the free-text skill IS captured under content-on, so the
        # no-content assertion below is meaningful (not vacuous).
        events = self._drive(mock_client, _CONTENT, _CONTENT)
        assert SECRET_SKILL in _all_payload_text([{"payload": self._delegation(events)}])

    def test_adapter_side_topology_survives_skill_stripped(self, mock_client: Any) -> None:
        events = self._drive(mock_client, _NO_CONTENT, None)
        d = self._delegation(events)
        text = _all_payload_text(events)
        # Free-text skill DESCRIPTION is content -> stripped.
        assert SECRET_SKILL not in text, "a2a.delegation skill description leaked despite capture_content=False"
        # Topology + fp are provenance metadata -> SURVIVE (A15 overturn).
        assert d.get("from_agent") == DELEGATOR_ID, "delegator id stripped under no-content (A15 provenance loss)"
        assert d.get("to_agent") == DELEGATEE_ID, "delegatee id stripped under no-content (A15 provenance loss)"
        assert d.get("target_agent") == DELEGATEE_ID
        assert str(d.get("delegation_fp", "")).startswith("sha256:"), "delegation fp stripped under no-content (A15)"

    def test_collector_side_topology_survives_skill_stripped(self, mock_client: Any) -> None:
        events = self._drive(mock_client, None, _NO_CONTENT)
        d = self._delegation(events)
        text = _all_payload_text(events)
        assert SECRET_SKILL not in text, "collector backstop leaked a2a.delegation skill description"
        assert d.get("from_agent") == DELEGATOR_ID, "collector backstop stripped the delegator id (A15)"
        assert d.get("to_agent") == DELEGATEE_ID, "collector backstop stripped the delegatee id (A15)"
        assert str(d.get("delegation_fp", "")).startswith("sha256:")


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
        text = _all_payload_text(self._drive_stream(mock_client, _CONTENT, _CONTENT))
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

    # --- the production connect()/on_user_action WRAP path (LAY-3572 / B14) ---
    # commerce.ui.user_action has NO _CONTENT_KEYS entry, so the keyed hash is the
    # SOLE privacy control on the action context. Only record_user_action (the
    # explicit API) was tested; the wrapped client method (what a real app calls)
    # was not. PAN/PII in the context must never reach the payload in cleartext.

    def test_wrap_path_user_action_hashes_context_no_cleartext(self, mock_client: Any) -> None:
        from layerlens.instrument.adapters.protocols.a2ui import A2UIProtocolAdapter

        adapter = A2UIProtocolAdapter()
        target = SimpleNamespace(
            on_user_action=lambda **kw: {"ok": True},
            on_surface_created=lambda **kw: {"ok": True},
        )
        adapter.connect(target=target)
        pan = "4111111111111111-SENTINEL"

        def go() -> None:
            target.on_user_action(surface_id="cart-1", action_type="add_to_cart", context={"pan": pan})

        events = _run_collected(mock_client, go)
        actions = [e for e in events if e["event_type"] == "commerce.ui.user_action"]
        assert actions, "wrap path emitted no commerce.ui.user_action"
        payload = actions[0]["payload"]
        blob = _all_payload_text(events)
        assert pan not in blob, "cleartext action context (PAN) leaked through the wrap path"
        assert payload["action_context_hash"].startswith("sha256:"), "context not hashed on the wrap path"
        assert payload.get("action_type") == "add_to_cart", "action_type metadata over-stripped"
        # the per-instance HMAC key must never be emitted
        assert adapter._hash_key.hex() not in blob, "the HMAC key leaked into telemetry"
