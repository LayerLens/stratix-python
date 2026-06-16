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
        assert SECRET_TEXT not in text, "tool result leaked despite adapter capture_content=False (N7)"
        assert any(e["payload"].get("tool_name") == "search" for e in events), "metadata over-stripped"

    def test_collector_side_no_content_strips_args_and_result(self, mock_client: Any) -> None:
        events = self._drive(mock_client, None, _NO_CONTENT)
        text = _all_payload_text(events)
        assert SECRET_QUERY not in text, "collector backstop missing for mcp.tool.call arguments (N7)"
        assert SECRET_TEXT not in text, "collector backstop missing for mcp.tool.call result (N7)"


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
            target.send_task(agent_id="a1", skill="answer", message=SECRET_TEXT)

        return _run_collected(mock_client, go, collector_config)

    def test_adapter_side_no_content_strips_request_summary(self, mock_client: Any) -> None:
        events = self._drive(mock_client, _NO_CONTENT, None)
        text = _all_payload_text(events)
        assert SECRET_TEXT not in text, "a2a request content leaked despite adapter capture_content=False (N7)"

    def test_collector_side_no_content_strips_request_summary(self, mock_client: Any) -> None:
        text = _all_payload_text(self._drive(mock_client, None, _NO_CONTENT))
        assert SECRET_TEXT not in text, "collector backstop missing for a2a request summary (N7)"
