"""Live workloads per protocol adapter.

Protocols are mostly LLM-free. Each runner mirrors the corresponding
``samples/adapters/protocols/*`` flow using an in-process fake client/stream, so
no external protocol server is required. Must run inside the harness's active
``TraceCollector``. Imports are lazy so collection never requires an optional
protocol package (``mcp``, ``a2a-sdk``) to be installed.
"""

from __future__ import annotations

import asyncio

from ._scenarios import SENTINEL


def run_agui(flow: str) -> None:  # noqa: ARG001
    from layerlens.instrument.adapters.protocols.agui import AGUIProtocolAdapter

    stream = [
        {"type": "TEXT_MESSAGE_CONTENT", "delta": "Hello "},
        {"type": "TEXT_MESSAGE_CONTENT", "delta": f"world {SENTINEL}"},
        {"type": "TEXT_MESSAGE_END"},
        {"type": "TOOL_CALL_START", "toolCallId": "tc1", "toolCallName": "lookup"},
        {"type": "TOOL_CALL_ARGS", "toolCallId": "tc1", "delta": '{"q": "x ' + SENTINEL + '"}'},
        {"type": "TOOL_CALL_END", "toolCallId": "tc1"},
        {"type": "STATE_SNAPSHOT", "state": {"turn": 1, "note": SENTINEL}},
        # un-handled events that fall through to the raw passthrough (#5/#12):
        {"type": "MESSAGES_SNAPSHOT", "messages": [{"role": "user", "content": f"snapshot {SENTINEL}"}]},
        {"type": "TOOL_CALL_RESULT", "toolCallId": "tc1", "content": f"result {SENTINEL}"},
    ]
    adapter = AGUIProtocolAdapter()
    for _ in adapter.wrap_stream(iter(stream)):
        pass


def run_a2ui(flow: str) -> None:  # noqa: ARG001
    from layerlens.instrument.adapters.protocols.a2ui import A2UIProtocolAdapter

    adapter = A2UIProtocolAdapter()
    adapter.record_surface_created(surface_id="cart-1", surface_type="cart", item_count=3)
    adapter.record_user_action(surface_id="cart-1", action_type="add_to_cart", context={"sku": "ABC-123"})


def run_ap2(flow: str) -> None:  # noqa: ARG001
    from layerlens.instrument.adapters.protocols.ap2 import (
        AP2Guardrails,
        instrument_ap2,
        uninstrument_ap2,
    )

    class _FakeAP2Client:
        def create_intent_mandate(self, *, mandate_id, amount, merchant, expires_at=None):
            return {"mandate_id": mandate_id}

        def sign_payment_mandate(self, *, mandate_id, amount, merchant):
            return {"mandate_id": mandate_id, "signature": "sig-xyz"}

        def issue_receipt(self, *, receipt_id, mandate_id, amount, merchant):
            return {"receipt_id": receipt_id}

    merchant = f"Bookstore {SENTINEL}"
    client = _FakeAP2Client()
    instrument_ap2(client, guardrails=AP2Guardrails(max_transaction=100.0, merchant_whitelist=[merchant]))
    try:
        client.create_intent_mandate(mandate_id="m-1", amount=50, merchant=merchant)
        client.sign_payment_mandate(mandate_id="m-1", amount=50, merchant=merchant)
        client.issue_receipt(receipt_id="r-1", mandate_id="m-1", amount=50, merchant=merchant)
        # BLOCKED path (L1): a non-whitelisted merchant -> the guardrail verdict
        # interpolates the merchant into the free-text reason. Under the
        # redaction variant only reason_code may reach the trace, never the
        # merchant string.
        try:
            client.sign_payment_mandate(mandate_id="m-2", amount=50, merchant=f"Evil {SENTINEL}")
        except PermissionError:
            pass
    finally:
        uninstrument_ap2()


def run_ucp(flow: str) -> None:  # noqa: ARG001
    from layerlens.instrument.adapters.protocols.ucp import instrument_ucp, uninstrument_ucp

    class _FakeUCPClient:
        def discover_suppliers(self, *, query):
            return [{"id": "acme", "name": "Acme"}]

        def browse_catalog(self, *, supplier_id, query):
            return [{"id": f"item-{i}"} for i in range(3)]

        def start_checkout(self, *, supplier_id, session_id):
            return {"session_id": session_id, "status": "started"}

        def complete_checkout(self, session_id, *, supplier_id, amount):
            return {"session_id": session_id, "status": "completed"}

        def issue_refund(self, *, session_id, amount, reason):
            return {"session_id": session_id, "status": "refunded"}

    client = _FakeUCPClient()
    instrument_ucp(client)
    try:
        client.discover_suppliers(query="books")
        client.browse_catalog(supplier_id="acme", query=f"novel {SENTINEL}")
        client.start_checkout(supplier_id="acme", session_id="s-1")
        client.complete_checkout("s-1", supplier_id="acme", amount=29.99)
        # refund reason is free text (commerce.refund_issued.reason) -> content.
        client.issue_refund(session_id="s-1", amount=29.99, reason=f"defective {SENTINEL}")
    finally:
        uninstrument_ucp()


def run_mcp(flow: str) -> None:  # noqa: ARG001
    from layerlens.instrument.adapters.protocols.mcp import instrument_mcp, uninstrument_mcp

    class _FakeMCPClient:
        async def call_tool(self, name: str, arguments: dict) -> dict:
            return {"content": [{"type": "text", "text": f"echo: {name}"}]}

        async def list_tools(self) -> dict:
            return {"tools": [{"name": "echo"}, {"name": "lookup"}]}

    client = _FakeMCPClient()
    instrument_mcp(client)
    try:

        async def _go() -> None:
            await client.list_tools()
            await client.call_tool("echo", {"msg": f"hello {SENTINEL}"})

        asyncio.run(_go())
    finally:
        uninstrument_mcp()
        # asyncio.run() leaves the main-thread loop closed/None. On Python <3.10,
        # protocol adapters construct asyncio primitives at __init__ (see report
        # B4), so restore a loop here to keep the suite order-independent.
        try:
            asyncio.set_event_loop(asyncio.new_event_loop())
        except Exception:
            pass


def run_a2a(flow: str) -> None:  # noqa: ARG001
    from layerlens.instrument.adapters.protocols.a2a import instrument_a2a, uninstrument_a2a

    class _FakeA2AClient:
        def send_task(self, *, agent_id: str, skill: str, payload: dict) -> dict:
            return {"status": "completed"}

        def get_agent_card(self, agent_id: str) -> dict:
            return {"id": agent_id, "name": f"researcher {SENTINEL}", "skills": [f"skill {SENTINEL}"]}

        def register_handler(self, handler, *, skill: str) -> None:
            pass

    client = _FakeA2AClient()
    instrument_a2a(client)
    try:
        # SENTINEL rides the DELEGATION signal (target_agent/skill) and the
        # discovered agent card name/skills — the fields that actually leak (L4).
        # (Putting it only on `payload` would be vacuous: _summarize drops it.)
        client.get_agent_card("agent-1")
        client.send_task(
            agent_id=f"agent {SENTINEL}", skill=f"summarize {SENTINEL}", payload={"text": f"hi {SENTINEL}"}
        )
    finally:
        uninstrument_a2a()
