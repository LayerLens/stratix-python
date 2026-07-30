"""ADP-W2 Family-B recorder for the ``mcp_extensions`` adapter (record-real-once).

``mcp_extensions`` is the MCP (Model Context Protocol) adapter's *extension*
surfaces — the real ``MCPProtocolAdapter`` (``PROTOCOL="mcp"``, spec 2025-11-25)
that instruments a real ``mcp.client.session.ClientSession``: tool discovery
(``list_tools`` -> ``mcp.tools.listed``), tool calls (``call_tool`` ->
``mcp.tool.call`` + fail-closed ``mcp.structured_output`` validation +
``mcp.async_task`` lifecycle), the server handshake (``initialize`` ->
``mcp.server.connected``), server-initiated **elicitation** (consent prompts ->
``mcp.elicitation`` request/response, form vs URL mode) and **sampling** (a nested
LLM round-trip the server asks the client to run -> ``mcp.sampling`` + a paired
``cost.record``).

This module records TWO real MCP sessions and writes each as a sealed real-trace
fixture under ``samples/data/traces/industry/``:

* ``generate_mcp_extensions_single`` -> ``healthcare_mcp_clinical.jsonl``: a
  clinical decision-support MCP client attaches to a ``clinical-records-mcp``
  server, negotiates the handshake, lists tools, calls ``get_patient_record``
  (whose ``structuredContent`` is validated CLOSED against the tool's real
  ``outputSchema`` -> ``validation_passed=True``), and handles ONE **form-mode**
  PHI-access consent elicitation the server raised (the clinician accepts, so the
  accepted form content carries a privacy-preserving ``content_hash``). This is
  the focused clinical tool-use + consent session.

* ``generate_mcp_extensions_multi`` -> ``financial_mcp_payment.jsonl``: the FULL
  MCP session — a payments MCP client attaches to a ``payments-mcp`` server,
  lists tools, calls ``authorize_payment`` (structured output validated closed),
  handles a **URL-mode** payment-authorization elicitation (a 3-D Secure browser
  consent — an out-of-band credential/payment flow whose ``ElicitResult.content``
  is absent, so it carries NO content hash), and runs a server-initiated
  **sampling** round-trip (the server asks the client's LLM to draft a receipt
  summary) that emits ``mcp.sampling`` + a priced ``cost.record``. This exercises
  the money + sensitive-consent paths end-to-end.

RENDER — HONEST EMPTY-STATE (NOT a DAG). MCP is a single-client protocol surface,
not a multi-agent system: there is no ``agent.identity`` / ``agent.handoff`` /
``agent.node``, so ``honest_agent_identity`` returns None and NO agent is
synthesized. Both traces render the honest empty-state Agent column (``—``) with a
``parent_span_id`` waterfall under one captured ``trace.root`` (the SDK's
``_synthesize_root_if_needed`` roots the session), Framework = ``mcp``, Status =
completed. ``genuinely_multi_agent`` is FALSE for both — the "multi" fixture is a
FULLER protocol session, not multiple agents.

FIDELITY / HONESTY. Every telemetry event flows through the REAL
``MCPProtocolAdapter.emit`` -> real ``TraceCollector`` -> real attestation hash
chain, and every protocol object driven through the adapter is a REAL ``mcp`` SDK
1.27 pydantic type (``InitializeResult`` / ``ListToolsResult`` / ``Tool`` /
``CallToolResult`` / ``ElicitRequest{Form,URL}Params`` / ``ElicitResult`` /
``CreateMessageRequestParams`` / ``CreateMessageResult``) attached to a real
``ClientSession`` — the same real-wire fidelity the bite-proven
``test_mcp_invariants`` fixtures use, and the callbacks are invoked exactly as the
live session dispatches a server-initiated elicitation/sampling request
(``session.py:567/577``). The server *responses* are recorded domain bodies
(non-sensitive synthetic clinical/payment data — there is no live MCP server
process), so the fixtures are captured against RECORDED server bodies, marked
honestly in ``metadata``. Nothing is fabricated: the MCP adapter is
privacy-preserving by design (``mcp.tool.call`` records only a result *shape*
``{content_items: N}`` + a structured-output *hash*, never the raw record), the
sampling token counts are ESTIMATED from text (the MCP wire carries no usage) and
flagged ``tokens_estimated=True``, and the sampling ``cost.record.cost_usd`` is a
real price the central price-on-emit chokepoint computed from the real model id +
those estimated tokens — never a claim of a metered paid call.

The recording reuses the ``_generate_fixtures`` capture seam: a per-session
``TraceCollector`` is made current under one pushed root span, the surfaces are
driven, the collector is flushed (``set_trace_observer`` + a no-op
``enqueue_upload`` capture the sealed payload but never upload during
generation), and the samples upload the captured fixtures themselves at run time.
``mcp`` is imported function-locally so this module imports in any venv (recorded
in ``.audit-venvs/sk`` — py3.11 + mcp 1.27 + the repo editable).
"""

from __future__ import annotations

import os
import sys
import uuid

# Reuse the record-real-once seam from the sibling ``_generate_fixtures`` module
# (single source of truth for the fixture writer + capture config).
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.dirname(_HERE)        # samples/data
_SAMPLES = os.path.dirname(_DATA)     # samples
_REPO = os.path.dirname(_SAMPLES)
for _p in (os.path.join(_REPO, "src"), _SAMPLES, _DATA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Drop this module's own directory from sys.path so the function-local
# ``import mcp`` always resolves to the installed ``mcp`` package (a no-op when
# imported as ``generators.mcp_extensions``; defensive when run directly).
sys.path[:] = [_p for _p in sys.path if os.path.abspath(_p or ".") != _HERE]

from layerlens import Stratix  # noqa: E402
import layerlens.instrument._collector as _collector_mod  # noqa: E402
from layerlens.instrument._collector import TraceCollector, set_trace_observer  # noqa: E402
from layerlens.instrument._context import (  # noqa: E402
    _current_collector,
    _push_span,
    _pop_span,
)

import _generate_fixtures as _gf  # noqa: E402

_write = _gf._write
_CAPTURE = _gf._CAPTURE

# Honest provenance marker (envelope-level, like ``tags`` — added AFTER the
# attestation chain is sealed, so it never perturbs the event hash chain that
# ``verify_chain`` checks). Records that the events are real MCP 1.27 wire types
# driven through the real adapter, against recorded (not live-server) responses.
_METADATA = {
    "capture": "real-mcp-1.27-types via real MCPProtocolAdapter + real ClientSession",
    "server_responses": "recorded synthetic domain bodies (no live MCP server process)",
    "sampling_tokens": "estimated from text (tokens_estimated=true), priced by the central chokepoint",
    "render": "honest empty-state (no agent DAG) + parent_span_id waterfall",
}


# --------------------------------------------------------------------------
# Capture seam: run an async drive body inside a live collector context rooted
# at ONE session span, then flush -> the SDK synthesizes a single ``trace.root``
# on that span (every protocol event parents to it) and no ``agent.identity`` is
# added (empty-state). Returns the sealed, attested payload.
# --------------------------------------------------------------------------
def _capture_session(client: Stratix, *, root_name: str, drive) -> dict:
    import anyio

    col = TraceCollector(client, _CAPTURE)
    root_id = uuid.uuid4().hex[:16]
    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    try:

        async def _run() -> None:
            tok = _current_collector.set(col)
            snap = _push_span(root_id, root_name)
            try:
                await drive()
            finally:
                _pop_span(snap)
                _current_collector.reset(tok)

        anyio.run(_run)
        col.flush()
    finally:
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig

    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for mcp session")
    return payload


def _client_session(*, elicit_cb=None, sample_cb=None):
    """Build a REAL ``mcp.client.session.ClientSession`` over anyio memory streams.

    We never run the transport — we attach the adapter and invoke the (wrapped)
    methods/callbacks directly, which is exactly how the live session dispatches
    a server-initiated elicitation/sampling request (session.py:567/577). This
    is the same construction the bite-proven ``test_mcp_invariants`` uses.
    """
    import anyio
    from mcp.client.session import ClientSession

    _w, _r = anyio.create_memory_object_stream(10)
    _w2, _r2 = anyio.create_memory_object_stream(10)
    return ClientSession(_r2, _w, elicitation_callback=elicit_cb, sampling_callback=sample_cb)


def _summary(payload: dict) -> dict:
    events = payload.get("events", [])
    kinds: dict[str, int] = {}
    for e in events:
        kinds[e.get("event_type")] = kinds.get(e.get("event_type"), 0) + 1
    costs = [
        (e.get("payload") or {})
        for e in events
        if e.get("event_type") == "cost.record"
    ]
    elic = [
        ((e.get("payload") or {}).get("phase"), (e.get("payload") or {}).get("mode"),
         (e.get("payload") or {}).get("action"),
         "hash" if (e.get("payload") or {}).get("content_hash") else "no-hash")
        for e in events
        if e.get("event_type") == "mcp.elicitation"
    ]
    so = [
        (e.get("payload") or {}).get("validation_passed")
        for e in events
        if e.get("event_type") == "mcp.structured_output"
    ]
    return {
        "n": len(events),
        "kinds": kinds,
        "has_agent_identity": any(e.get("event_type") == "agent.identity" for e in events),
        "has_trace_root": any(e.get("event_type") == "trace.root" for e in events),
        "structured_output": so,
        "elicitation": elic,
        "cost": [(c.get("model"), c.get("cost_usd"), c.get("total_tokens")) for c in costs],
    }


# --------------------------------------------------------------------------
# SINGLE — Healthcare: a clinical-records MCP session (tool-use + form consent)
# --------------------------------------------------------------------------
def generate_mcp_extensions_single(client: Stratix) -> dict:
    """Record a clinical decision-support MCP session: handshake + list_tools +
    ``get_patient_record`` (structured output validated closed) + a form-mode
    PHI-access consent the clinician accepts. Renders honest empty-state."""
    from mcp import types
    from layerlens.instrument.adapters.protocols.mcp.adapter import MCPProtocolAdapter

    # Real tool definitions the ``clinical-records-mcp`` server declares. The
    # ``get_patient_record`` outputSchema is what fail-closed validation checks
    # the structuredContent against (D4).
    get_record = types.Tool(
        name="get_patient_record",
        inputSchema={
            "type": "object",
            "properties": {"patient_id": {"type": "string"}},
            "required": ["patient_id"],
        },
        outputSchema={
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "triage_level": {"type": "string"},
            },
            "required": ["patient_id", "triage_level"],
        },
    )
    list_meds = types.Tool(
        name="list_medications",
        inputSchema={
            "type": "object",
            "properties": {"patient_id": {"type": "string"}},
            "required": ["patient_id"],
        },
    )

    async def elicit_cb(context, params):
        # The clinician consents to accessing PHI for a treatment purpose. The
        # submitted form content is hashed (privacy-preserving) under full capture.
        return types.ElicitResult(
            action="accept",
            content={"consent": True, "clinician_id": "DR-4471", "purpose": "treatment"},
        )

    session = _client_session(elicit_cb=elicit_cb)

    async def initialize(*a, **k):
        return types.InitializeResult(
            protocolVersion="2025-11-25",
            capabilities=types.ServerCapabilities(),
            serverInfo=types.Implementation(name="clinical-records-mcp", version="2.4.0"),
        )

    async def list_tools(*a, **k):
        return types.ListToolsResult(tools=[get_record, list_meds])

    async def call_tool(name, arguments=None, **k):
        # A real CallToolResult: privacy-preserving telemetry records only the
        # result SHAPE ({content_items}) + a hash of the structuredContent, never
        # the raw record. The structured content validates against the outputSchema.
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="patient record retrieved")],
            structuredContent={
                "patient_id": "PT-8842",
                "triage_level": "urgent",
                "primary_concern": "exertional_chest_pain",
            },
        )

    session.initialize = initialize  # type: ignore[method-assign]
    session.list_tools = list_tools  # type: ignore[method-assign]
    session.call_tool = call_tool  # type: ignore[method-assign]
    # Populate the tool-output-schema cache exactly as the real
    # ``ClientSession.list_tools`` does (session.py:537), so structured-output
    # validation runs against the REAL schema instead of falling to "unknown".
    session._tool_output_schemas["get_patient_record"] = get_record.outputSchema

    adapter = MCPProtocolAdapter(capture_config=_CAPTURE)
    adapter.connect(target=session)

    async def drive():
        from mcp import types as _t

        await session.initialize()
        await session.list_tools()
        await session.call_tool("get_patient_record", {"patient_id": "PT-8842"})
        # Server-initiated consent prompt (form mode): access PHI for treatment.
        await session._elicitation_callback(
            None,
            _t.ElicitRequestFormParams(
                message="Clinician consent required to access PHI for PT-8842 (treatment purpose). Confirm?",
                requestedSchema={
                    "type": "object",
                    "properties": {
                        "consent": {"type": "boolean"},
                        "clinician_id": {"type": "string"},
                    },
                    "required": ["consent", "clinician_id"],
                },
            ),
        )

    payload = _capture_session(client, root_name="clinical-mcp-session", drive=drive)
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "healthcare",
        "clinical-mcp",
        "protocol",
        "empty-state",
    ]
    payload["metadata"] = dict(_METADATA)
    s = _summary(payload)
    print(
        "  mcp_extensions single (clinical-records MCP, form consent)  "
        "events=%d kinds=%s structured_output=%s elicitation=%s agent_identity=%s trace_root=%s"
        % (s["n"], s["kinds"], s["structured_output"], s["elicitation"],
           s["has_agent_identity"], s["has_trace_root"])
    )
    print("  ->", _write([payload], "industry", "healthcare_mcp_clinical"), "\n")
    return payload


# --------------------------------------------------------------------------
# MULTI — Financial: the FULL MCP session (tool + URL consent + sampling + cost)
# --------------------------------------------------------------------------
def generate_mcp_extensions_multi(client: Stratix) -> dict:
    """Record the FULL payments MCP session: handshake + list_tools +
    ``authorize_payment`` (structured output validated closed) + a URL-mode
    payment-authorization consent (3-D Secure browser flow, no content hash) +
    a server-initiated sampling round-trip (mcp.sampling + priced cost.record).
    Renders honest empty-state (a fuller session, NOT multi-agent)."""
    from mcp import types
    from layerlens.instrument.adapters.protocols.mcp.adapter import MCPProtocolAdapter

    authorize = types.Tool(
        name="authorize_payment",
        inputSchema={
            "type": "object",
            "properties": {
                "amount_usd": {"type": "number"},
                "merchant": {"type": "string"},
                "card_last4": {"type": "string"},
            },
            "required": ["amount_usd", "merchant"],
        },
        outputSchema={
            "type": "object",
            "properties": {
                "authorization_id": {"type": "string"},
                "status": {"type": "string"},
                "amount_usd": {"type": "number"},
            },
            "required": ["authorization_id", "status"],
        },
    )
    lookup = types.Tool(
        name="lookup_transaction",
        inputSchema={
            "type": "object",
            "properties": {"transaction_id": {"type": "string"}},
            "required": ["transaction_id"],
        },
    )

    async def elicit_cb(context, params):
        # URL-mode payment authorization: the customer approves in their browser
        # (3-D Secure). The interaction is out-of-band, so ElicitResult.content is
        # absent — even an accept hashes NOTHING (a credential/payment flow never
        # fingerprints a payload the client never submitted).
        return types.ElicitResult(action="accept")

    async def sample_cb(context, params):
        # The server asks the CLIENT's LLM to draft a customer-facing receipt
        # summary (the agentic, money-burning path). A real CreateMessageResult
        # carrying the model — the wire has NO token usage, so the adapter
        # estimates tokens from text and the central chokepoint prices the model.
        return types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(
                type="text",
                text=(
                    "Receipt: your payment of $499.00 to Acme Cloud Services was "
                    "authorized (auth AUTH-77213). No further action is needed."
                ),
            ),
            model="claude-haiku-4-5-20251001",
            stopReason="endTurn",
        )

    session = _client_session(elicit_cb=elicit_cb, sample_cb=sample_cb)

    async def initialize(*a, **k):
        return types.InitializeResult(
            protocolVersion="2025-11-25",
            capabilities=types.ServerCapabilities(),
            serverInfo=types.Implementation(name="payments-mcp", version="3.1.0"),
        )

    async def list_tools(*a, **k):
        return types.ListToolsResult(tools=[authorize, lookup])

    async def call_tool(name, arguments=None, **k):
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="authorization created")],
            structuredContent={
                "authorization_id": "AUTH-77213",
                "status": "requires_3ds",
                "amount_usd": 499.00,
            },
        )

    session.initialize = initialize  # type: ignore[method-assign]
    session.list_tools = list_tools  # type: ignore[method-assign]
    session.call_tool = call_tool  # type: ignore[method-assign]
    session._tool_output_schemas["authorize_payment"] = authorize.outputSchema

    adapter = MCPProtocolAdapter(capture_config=_CAPTURE)
    adapter.connect(target=session)

    async def drive():
        from mcp import types as _t

        await session.initialize()
        await session.list_tools()
        await session.call_tool(
            "authorize_payment",
            {"amount_usd": 499.00, "merchant": "Acme Cloud Services", "card_last4": "4242"},
        )
        # URL-mode consent: approve the $499 charge via 3-D Secure in the browser.
        await session._elicitation_callback(
            None,
            _t.ElicitRequestURLParams(
                message="Approve this $499.00 payment via 3-D Secure in your browser.",
                url="https://pay.example.com/3ds/txn-77213",
                elicitationId="el-3ds-77213",
            ),
        )
        # Server-initiated sampling: draft the customer receipt summary (money path).
        await session._sampling_callback(
            None,
            _t.CreateMessageRequestParams(
                messages=[
                    _t.SamplingMessage(
                        role="user",
                        content=_t.TextContent(
                            type="text",
                            text=(
                                "Summarize this authorized transaction for the "
                                "customer receipt: $499.00 to Acme Cloud Services, "
                                "auth AUTH-77213."
                            ),
                        ),
                    )
                ],
                maxTokens=256,
                systemPrompt="Write a one-sentence customer receipt summary. Be terse.",
            ),
        )

    payload = _capture_session(client, root_name="payments-mcp-session", drive=drive)
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "financial-services",
        "payment-authorization",
        "protocol",
        "empty-state",
    ]
    payload["metadata"] = dict(_METADATA)
    s = _summary(payload)
    print(
        "  mcp_extensions multi (payments MCP, url consent + sampling)  "
        "events=%d kinds=%s structured_output=%s elicitation=%s cost=%s agent_identity=%s trace_root=%s"
        % (s["n"], s["kinds"], s["structured_output"], s["elicitation"], s["cost"],
           s["has_agent_identity"], s["has_trace_root"])
    )
    print("  ->", _write([payload], "industry", "financial_mcp_payment"), "\n")
    return payload


if __name__ == "__main__":  # pragma: no cover - manual regeneration entrypoint
    _client = Stratix()
    generate_mcp_extensions_single(_client)
    generate_mcp_extensions_multi(_client)
