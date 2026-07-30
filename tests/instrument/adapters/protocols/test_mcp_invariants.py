"""MCP consent + cost invariants — real ``mcp`` SDK 1.27 fixtures, bite-proven.

Every fixture here is a REAL pydantic object from the installed ``mcp`` SDK
(spec 2025-11-25) — ``ElicitResult`` / ``ElicitRequest{Form,URL}Params`` /
``CreateMessageResult`` / ``CreateMessageRequestParams`` / ``CallToolResult`` /
``Tool`` — NOT a hand-rolled ``SimpleNamespace``. A library upgrade that changes
a schema fails these fixtures loudly, so the tests can never drift from the real
wire shape (brief §3.5). The module ``importorskip("mcp")``, so it SKIPS in the
base py3.9 venv (which has no mcp) and runs in ``.audit-venvs/sk`` (py3.11 + mcp
1.27 + the repo editable).

Each invariant has a BITING test: revert/weaken the guard in the mcp adapter (or
the elicitation tracker / schema lock) and the test goes RED. The bites for the
three headline invariants (decline/cancel consent, sampling cost, structured-
output fail-closed) were confirmed by hand (break the guard, watch RED, restore).

The tests drive the adapter's REAL emit path (collector + redact backstop) and,
for the callback-surface invariant, the REAL ``ClientSession`` callback attach.
"""

from __future__ import annotations

from typing import Any, Dict, List

import anyio
import pytest

pytest.importorskip("mcp")

from mcp import types  # noqa: E402
from mcp.client.session import ClientSession  # noqa: E402

from layerlens.instrument._context import _current_collector  # noqa: E402
from layerlens.instrument._collector import TraceCollector  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.protocols.mcp.adapter import MCPProtocolAdapter  # noqa: E402

# Consent/cost controls belong in the fast Invariant Gates job (shift-left).
pytestmark = pytest.mark.invariant

_NO_CONTENT = CaptureConfig(capture_content=False)
_CONTENT = CaptureConfig(capture_content=True)


# ── real-mcp fixture builders (the library's OWN typings) ──────────────────


def form_params(
    message: str = "Enter card number to confirm $499", *, card: bool = True
) -> types.ElicitRequestFormParams:
    schema: Dict[str, Any] = {"type": "object", "properties": {}}
    if card:
        schema["properties"]["card"] = {"type": "string"}
    return types.ElicitRequestFormParams(message=message, requestedSchema=schema)


def url_params(
    message: str = "Approve this payment in your browser",
    url: str = "https://pay.example.com/checkout/abc",
    eid: str = "el-url-1",
) -> types.ElicitRequestURLParams:
    return types.ElicitRequestURLParams(message=message, url=url, elicitationId=eid)


def sampling_params(prompt: str = "summarize this very long document " * 20) -> types.CreateMessageRequestParams:
    return types.CreateMessageRequestParams(
        messages=[types.SamplingMessage(role="user", content=types.TextContent(type="text", text=prompt))],
        maxTokens=512,
        systemPrompt="be terse",
    )


def sampling_result(
    text: str = "Here is a concise summary of the document. " * 5,
    model: str = "claude-3-5-haiku-20241022",
) -> types.CreateMessageResult:
    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text=text),
        model=model,
        stopReason="endTurn",
    )


def call_tool_result(structured: Dict[str, Any]) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text="ok")],
        structuredContent=structured,
    )


def weather_tool() -> types.Tool:
    return types.Tool(
        name="weather",
        inputSchema={"type": "object"},
        outputSchema={
            "type": "object",
            "properties": {"temp": {"type": "number"}},
            "required": ["temp"],
        },
    )


# ── harness: collector + adapter ──────────────────────────────────────────


def _run(coro_fn: Any, config: CaptureConfig | None = None) -> List[Dict[str, Any]]:
    """Run an async body inside a live collector context, return emitted events."""
    collector = TraceCollector(object(), config or CaptureConfig())

    async def _wrapped() -> None:
        token = _current_collector.set(collector)
        try:
            await coro_fn()
        finally:
            _current_collector.reset(token)

    anyio.run(_wrapped)
    return collector.events


def _by_type(events: List[Dict[str, Any]], event_type: str) -> List[Dict[str, Any]]:
    return [e["payload"] for e in events if e["event_type"] == event_type]


def _all_text(events: List[Dict[str, Any]]) -> str:
    import json

    return json.dumps(events, default=str)


def _client_session(elicit_cb: Any = None, sample_cb: Any = None) -> ClientSession:
    """Build a REAL ClientSession over anyio memory streams.

    We never run the transport — we only attach the adapter and invoke the
    (wrapped) callbacks directly, which is exactly how the live session dispatches
    a server-initiated elicitation/sampling request (session.py:567/577)."""
    _w, _r = anyio.create_memory_object_stream(10)
    _w2, _r2 = anyio.create_memory_object_stream(10)
    return ClientSession(_r2, _w, elicitation_callback=elicit_cb, sampling_callback=sample_cb)


# ===========================================================================
# INVARIANT 1 (D1, headline) — ELICITATION decline/cancel is consent-faithful.
# The real ElicitResult.action is read and EMITTED; a refusal is distinguishable
# from an accept and carries NO hash of a payload the user never submitted.
# BITE: in adapter `_elicit_response`, hardcode `action="accept"` (or drop the
# `action` key) -> the action assertions + the schema-lock branch go RED. Hand-
# confirmed: setting action="accept" turned both decline/cancel tests RED.
# ===========================================================================


class TestElicitationConsent:
    def _drive_method(self, result: types.ElicitResult, params: Any, config: CaptureConfig) -> List[Dict[str, Any]]:
        adapter = MCPProtocolAdapter(capture_config=config)

        async def elicit(*_a: Any, **_k: Any) -> types.ElicitResult:
            return result

        target = _Server(elicit)
        adapter.connect(target=target)

        async def go() -> None:
            await target.elicit(message=params.message, schema=getattr(params, "requestedSchema", None))

        return _run(go, config)

    def test_decline_emits_action_decline(self) -> None:
        events = self._drive_method(types.ElicitResult(action="decline"), form_params(), _CONTENT)
        resp = [p for p in _by_type(events, "mcp.elicitation") if p.get("phase") == "response"]
        assert resp, "no elicitation response emitted"
        assert resp[0]["action"] == "decline", "a declined consent must be recorded as decline, not accept/submit"

    def test_cancel_emits_action_cancel(self) -> None:
        events = self._drive_method(types.ElicitResult(action="cancel"), form_params(), _CONTENT)
        resp = [p for p in _by_type(events, "mcp.elicitation") if p.get("phase") == "response"]
        assert resp and resp[0]["action"] == "cancel"

    def test_accept_emits_action_accept(self) -> None:
        events = self._drive_method(
            types.ElicitResult(action="accept", content={"card": "4111111111111111"}), form_params(), _CONTENT
        )
        resp = [p for p in _by_type(events, "mcp.elicitation") if p.get("phase") == "response"]
        assert resp and resp[0]["action"] == "accept"

    def test_decline_carries_no_content_hash(self) -> None:
        """A refusal hashes NOTHING — ElicitResult.content is None for decline.
        BITE: in `_elicit_response` hash `result` unconditionally (the OLD bug:
        `response_hash = hash_response(result)`) -> this goes RED via the
        schema-lock branch (decline + content_hash is rejected)."""
        events = self._drive_method(types.ElicitResult(action="decline"), form_params(), _CONTENT)
        resp = [p for p in _by_type(events, "mcp.elicitation") if p.get("phase") == "response"]
        assert resp and resp[0].get("content_hash") is None, (
            "declined elicitation must not carry a content-derived hash"
        )

    def test_cancel_carries_no_content_hash(self) -> None:
        events = self._drive_method(types.ElicitResult(action="cancel"), form_params(), _CONTENT)
        resp = [p for p in _by_type(events, "mcp.elicitation") if p.get("phase") == "response"]
        assert resp and resp[0].get("content_hash") is None

    def test_accept_with_content_carries_a_hash_only_under_content_capture(self) -> None:
        """An accepted FORM reply may carry a hash of the submitted data — but only
        when content capture is on; the hash is content-derived, so it is stripped
        under capture_content=False (it cannot leak a fingerprint of the card)."""
        evs_on = self._drive_method(
            types.ElicitResult(action="accept", content={"card": "4111111111111111"}), form_params(), _CONTENT
        )
        resp_on = [p for p in _by_type(evs_on, "mcp.elicitation") if p.get("phase") == "response"][0]
        assert isinstance(resp_on.get("content_hash"), str) and resp_on["content_hash"].startswith("sha256:")

        evs_off = self._drive_method(
            types.ElicitResult(action="accept", content={"card": "4111111111111111"}), form_params(), _NO_CONTENT
        )
        resp_off = [p for p in _by_type(evs_off, "mcp.elicitation") if p.get("phase") == "response"][0]
        assert "content_hash" not in resp_off, "content_hash survived under capture_content=False"


# ===========================================================================
# INVARIANT 2 (D2) — the elicitation MESSAGE is CONTENT (the prompt), not a
# metadata "title". Under capture_content=False it MUST be stripped.
# BITE: remove "message" from _CONTENT_KEYS["mcp.elicitation"] -> RED.
# ===========================================================================


class TestElicitationMessageIsContent:
    SECRET_PROMPT = "Enter SSN 078-05-1120 to continue"

    def _drive(self, config: CaptureConfig) -> List[Dict[str, Any]]:
        adapter = MCPProtocolAdapter(capture_config=config)

        async def elicit(*_a: Any, **_k: Any) -> types.ElicitResult:
            return types.ElicitResult(action="decline")

        target = _Server(elicit)
        adapter.connect(target=target)

        async def go() -> None:
            await target.elicit(message=self.SECRET_PROMPT, schema={"type": "object"})

        return _run(go, config)

    def test_message_present_under_content_capture(self) -> None:
        events = self._drive(_CONTENT)
        req = [p for p in _by_type(events, "mcp.elicitation") if p.get("phase") == "request"]
        assert req and req[0]["message"] == self.SECRET_PROMPT

    def test_message_stripped_under_no_content(self) -> None:
        events = self._drive(_NO_CONTENT)
        assert self.SECRET_PROMPT not in _all_text(events), (
            "elicitation prompt (message) leaked under capture_content=False"
        )
        # redact without going blind: the consent CATEGORY + ids survive.
        resp = [p for p in _by_type(events, "mcp.elicitation") if p.get("phase") == "response"]
        assert resp and resp[0]["action"] == "decline", "action over-stripped"
        req = [p for p in _by_type(events, "mcp.elicitation") if p.get("phase") == "request"]
        assert req and req[0].get("elicitation_id"), "elicitation_id over-stripped"


# ===========================================================================
# INVARIANT 3 (D3) — SAMPLING emits mcp.sampling + a cost.record carrying the
# real model + token counts; the central chokepoint prices it.
# BITE: in `_wrap_sampling_callback` / `_emit_sampling`, drop the COST_RECORD
# emit -> the cost assertions go RED. Hand-confirmed RED.
# ===========================================================================


class TestSamplingCost:
    def _drive(self, result: types.CreateMessageResult, config: CaptureConfig = _CONTENT) -> List[Dict[str, Any]]:
        adapter = MCPProtocolAdapter(capture_config=config)

        async def sample_cb(context: Any, params: Any) -> types.CreateMessageResult:
            return result

        session = _client_session(sample_cb=sample_cb)
        adapter.connect(target=session)

        async def go() -> None:
            # invoke the WRAPPED callback exactly as the live session dispatches it.
            await session._sampling_callback(None, sampling_params())

        return _run(go, config)

    def test_sampling_emits_event_with_model(self) -> None:
        events = self._drive(sampling_result())
        samp = _by_type(events, "mcp.sampling")
        assert samp, "no mcp.sampling event emitted for a sampling round-trip"
        assert samp[0]["model"] == "claude-3-5-haiku-20241022"
        assert samp[0]["status"] == "completed"
        assert samp[0]["prompt_tokens"] > 0 and samp[0]["completion_tokens"] > 0
        assert samp[0]["tokens_estimated"] is True, "token estimate must be flagged as derived, not metered"

    def test_sampling_emits_cost_record_with_tokens(self) -> None:
        events = self._drive(sampling_result())
        costs = _by_type(events, "cost.record")
        assert costs, "no cost.record emitted for a sampling round-trip (the money path is invisible)"
        cost = costs[0]
        assert cost["model"] == "claude-3-5-haiku-20241022"
        assert cost["prompt_tokens"] > 0 and cost["completion_tokens"] > 0
        assert cost["total_tokens"] == cost["prompt_tokens"] + cost["completion_tokens"]

    def test_cost_record_is_priced_by_the_central_chokepoint(self) -> None:
        """The collector's price-on-emit chokepoint fills cost_usd from model +
        tokens (claude-3-5-haiku is priced). A priced model with no cost_usd would
        be a dropped price (fail-closed schema-lock branch).
        BITE: change the sampling model to a priced one but drop the COST_RECORD
        emit -> RED (no cost). Or: confirm cost_usd is filled here."""
        # claude-haiku-4-5-20251001 is a real, bundled-priced Claude Haiku 4.5 id.
        events = self._drive(sampling_result(model="claude-haiku-4-5-20251001"))
        cost = _by_type(events, "cost.record")[0]
        assert cost.get("cost_usd") is not None and cost["cost_usd"] > 0, (
            "central chokepoint did not price the sampling cost.record"
        )

    def test_larger_prompt_costs_more(self) -> None:
        """The token estimate tracks text length: a bigger sampled completion
        bills more. BITE: make `_chars_to_tokens` return a constant -> RED."""
        small = self._drive(sampling_result(text="ok"))
        big = self._drive(sampling_result(text="x" * 4000))
        small_tokens = _by_type(small, "cost.record")[0]["completion_tokens"]
        big_tokens = _by_type(big, "cost.record")[0]["completion_tokens"]
        assert big_tokens > small_tokens

    def test_sampling_content_stripped_under_no_content(self) -> None:
        secret = "PATIENT SSN 078-05-1120 diagnosis confidential"
        events = self._drive(sampling_result(text=secret), config=_NO_CONTENT)
        assert secret not in _all_text(events), "sampled completion text leaked under capture_content=False"
        # cost survives (content-free) so the money path stays visible.
        assert _by_type(events, "cost.record"), "cost.record over-suppressed under no-content"


# ===========================================================================
# INVARIANT 4 (D4) — STRUCTURED-OUTPUT validation FAILS CLOSED. outputSchema is
# on the Tool definition, not CallToolResult; an unschema'd / non-validating
# structured output is never reported validation_passed=True.
# BITE: in `_emit_structured_output`, set the no-schema default to True (the OLD
# fail-open) -> the unknown-default test goes RED. Hand-confirmed RED.
# ===========================================================================


class TestStructuredOutputFailClosed:
    def _drive_with_session(
        self, structured: Dict[str, Any], tool: types.Tool | None, list_tools: bool
    ) -> List[Dict[str, Any]]:
        adapter = MCPProtocolAdapter()
        result = call_tool_result(structured)

        async def call_tool(name: str, arguments: Any = None, **kw: Any) -> types.CallToolResult:
            return result

        async def list_tools_fn() -> types.ListToolsResult:
            return types.ListToolsResult(tools=[tool] if tool else [])

        session = _client_session()
        # patch the two methods onto the real session object (real attach surface).
        session.call_tool = call_tool  # type: ignore[method-assign]
        session.list_tools = list_tools_fn  # type: ignore[method-assign]
        adapter.connect(target=session)

        async def go() -> None:
            if list_tools:
                await session.list_tools()  # populates _tool_output_schemas via the wrapper? no — see below
            await session.call_tool("weather", {})

        events = _run(go)
        return events

    def test_no_schema_available_is_unknown_not_true(self) -> None:
        """No Tool.outputSchema in the cache → validation_passed must be 'unknown',
        never True (the old fail-open hardcoded True)."""
        events = self._drive_with_session({"temp": 72}, tool=None, list_tools=False)
        so = _by_type(events, "mcp.structured_output")
        assert so, "no structured_output event emitted"
        assert so[0]["validation_passed"] == "unknown", "fail-OPEN: validation_passed defaulted True with no schema"

    def test_schema_in_session_cache_validates_closed(self) -> None:
        """When the real ClientSession._tool_output_schemas carries the schema, a
        structuredContent that VIOLATES it is validation_passed=False (fail closed).
        BITE: make `_lookup_output_schema` ignore the cache -> falls to 'unknown';
        flipping the no-schema default to True -> this would read True. RED either way."""
        adapter = MCPProtocolAdapter()
        bad = call_tool_result({"temp": "not-a-number"})  # violates {temp: number, required}

        async def call_tool(name: str, arguments: Any = None, **kw: Any) -> types.CallToolResult:
            return bad

        session = _client_session()
        session.call_tool = call_tool  # type: ignore[method-assign]
        # seed the real cache exactly as ClientSession.list_tools does (session.py:537)
        session._tool_output_schemas["weather"] = weather_tool().outputSchema
        adapter.connect(target=session)

        async def go() -> None:
            await session.call_tool("weather", {})

        events = _run(go)
        so = _by_type(events, "mcp.structured_output")
        assert so and so[0]["validation_passed"] is False, "a contract-violating structured output passed validation"

    def test_schema_in_cache_valid_passes(self) -> None:
        adapter = MCPProtocolAdapter()
        good = call_tool_result({"temp": 72})

        async def call_tool(name: str, arguments: Any = None, **kw: Any) -> types.CallToolResult:
            return good

        session = _client_session()
        session.call_tool = call_tool  # type: ignore[method-assign]
        session._tool_output_schemas["weather"] = weather_tool().outputSchema
        adapter.connect(target=session)

        async def go() -> None:
            await session.call_tool("weather", {})

        events = _run(go)
        so = _by_type(events, "mcp.structured_output")
        assert so and so[0]["validation_passed"] is True


# ===========================================================================
# INVARIANT 5 (D5) — the adapter attaches to the REAL ClientSession callback
# surface (_elicitation_callback / _sampling_callback), not a non-existent
# `elicit` method. Prove it against a real ClientSession.
# BITE: in `connect()`, delete the `_elicitation_callback` / `_sampling_callback`
# wrap branches -> the wrap assertions go RED.
# ===========================================================================


class TestRealCallbackSurface:
    def test_real_clientsession_has_no_elicit_method(self) -> None:
        """Documents the bug the rewrite fixes: the old `hasattr(target,'elicit')`
        wiring was DEAD against a real client."""
        session = _client_session()
        assert not hasattr(session, "elicit"), "a real ClientSession unexpectedly grew an elicit method"

    def test_connect_wraps_the_real_callbacks(self) -> None:
        async def elicit_cb(context: Any, params: Any) -> types.ElicitResult:
            return types.ElicitResult(action="accept", content={})

        async def sample_cb(context: Any, params: Any) -> types.CreateMessageResult:
            return sampling_result()

        session = _client_session(elicit_cb=elicit_cb, sample_cb=sample_cb)
        adapter = MCPProtocolAdapter()
        adapter.connect(target=session)
        # the callbacks are now the adapter's wrappers, not the originals.
        assert session._elicitation_callback is not elicit_cb, "elicitation callback not wrapped (dead instrumentation)"
        assert session._sampling_callback is not sample_cb, "sampling callback not wrapped (dead instrumentation)"

        adapter.disconnect()
        # disconnect restores the originals (real attach/restore round-trip).
        assert session._elicitation_callback is elicit_cb
        assert session._sampling_callback is sample_cb

    def test_wrapped_callback_returns_the_real_result_unchanged(self) -> None:
        """Instrumentation is OBSERVE-ONLY: the wrapped callback returns the exact
        ElicitResult the user callback produced (we never alter consent)."""
        sentinel = types.ElicitResult(action="decline")

        async def elicit_cb(context: Any, params: Any) -> types.ElicitResult:
            return sentinel

        session = _client_session(elicit_cb=elicit_cb)
        adapter = MCPProtocolAdapter()
        adapter.connect(target=session)

        async def go() -> types.ElicitResult:
            return await session._elicitation_callback(None, form_params())

        out: List[types.ElicitResult] = []

        async def wrapped() -> None:
            token = _current_collector.set(TraceCollector(object(), CaptureConfig()))
            try:
                out.append(await go())
            finally:
                _current_collector.reset(token)

        anyio.run(wrapped)
        assert out and out[0] is sentinel


# ===========================================================================
# INVARIANT 6 (D6) — URL-mode elicitation (credentials/OAuth/payment) is
# distinguished from form mode and its absent content is handled.
# BITE: in `_elicit_mode`, always return "form" -> the mode assertions go RED.
# ===========================================================================


class TestUrlModeElicitation:
    def _drive_callback(self, params: Any, result: types.ElicitResult) -> List[Dict[str, Any]]:
        async def elicit_cb(context: Any, p: Any) -> types.ElicitResult:
            return result

        session = _client_session(elicit_cb=elicit_cb)
        adapter = MCPProtocolAdapter(capture_config=_CONTENT)
        adapter.connect(target=session)

        async def go() -> None:
            await session._elicitation_callback(None, params)

        return _run(go, _CONTENT)

    def test_url_mode_is_distinguished(self) -> None:
        events = self._drive_callback(url_params(), types.ElicitResult(action="accept"))
        reqs = [p for p in _by_type(events, "mcp.elicitation") if p.get("phase") == "request"]
        resps = [p for p in _by_type(events, "mcp.elicitation") if p.get("phase") == "response"]
        assert reqs and reqs[0]["mode"] == "url", "URL-mode (sensitive credential/payment) flow not distinguished"
        assert resps and resps[0]["mode"] == "url"

    def test_form_mode_is_distinguished(self) -> None:
        events = self._drive_callback(form_params(), types.ElicitResult(action="accept", content={"card": "x"}))
        reqs = [p for p in _by_type(events, "mcp.elicitation") if p.get("phase") == "request"]
        assert reqs and reqs[0]["mode"] == "form"

    def test_url_mode_accept_carries_no_content_hash(self) -> None:
        """URL-mode ElicitResult.content is ALWAYS absent (the interaction is
        out-of-band) — even an accept hashes nothing."""
        events = self._drive_callback(url_params(), types.ElicitResult(action="accept"))
        resps = [p for p in _by_type(events, "mcp.elicitation") if p.get("phase") == "response"]
        assert resps and resps[0].get("content_hash") is None
        assert resps[0]["action"] == "accept"


# ===========================================================================
# INVARIANT 7 (D8) — the SCHEMA LOCK fails closed on a consent-faithless
# elicitation response (missing/invalid action; a refusal carrying a content
# hash). This is the population-complete net that catches a future emit path
# that forgets the action — independent of any single adapter test.
# BITE: delete the mcp.elicitation branch in _event_schema.validate_event -> RED.
# ===========================================================================


class TestSchemaLockConsentBranch:
    def _validate(self, payload: Dict[str, Any]) -> List[str]:
        # import the in-repo schema lock the same way the autouse net does.
        import sys

        if "tests" not in sys.path:
            sys.path.insert(0, "tests")
        from instrument._event_schema import validate_event

        return validate_event({"event_type": "mcp.elicitation", "payload": {"protocol": "mcp", **payload}})

    def test_response_without_action_is_rejected(self) -> None:
        assert self._validate({"phase": "response"}), "schema lock did not require an action on a response"

    def test_response_with_bogus_action_is_rejected(self) -> None:
        # "submit" is the OLD hardcoded value — not a real MCP action.
        assert self._validate({"phase": "response", "action": "submit"})

    def test_valid_actions_pass(self) -> None:
        for action in ("accept", "decline", "cancel"):
            assert not self._validate({"phase": "response", "action": action}), action

    def test_refusal_with_content_hash_is_rejected(self) -> None:
        assert self._validate({"phase": "response", "action": "decline", "content_hash": "sha256:x"})

    def test_request_phase_needs_no_action(self) -> None:
        assert not self._validate({"phase": "request"})


# ── a server-side double exposing an async `elicit` method (FastMCP-shaped) ──


class _Server:
    """A minimal stand-in for a FastMCP server context: the WRAP target is a real
    method-bearing object, but the ElicitResult it returns is a REAL mcp type."""

    def __init__(self, elicit_fn: Any) -> None:
        self.elicit = elicit_fn
