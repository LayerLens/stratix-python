"""Offline attestation + error + redaction + cost floor for the MCP protocol adapter.

Closes the census ◑/gap OFFLINE cells for ``mcp_extensions`` (the extension
surfaces of ``MCPProtocolAdapter``) so a regression fails in plain CI with NO
credentials and NO network. The net-new coverage the census flags is the
**offline attestation chain** over a real MCP session and a **recorded-shape**
full session; the redaction/error/cost cells are consolidated here into one
biting floor:

* **Attestation** (net-new; census ``attest: gap``) — a full, realistic MCP
  session (``list_tools`` → ``call_tool`` + structured-output → form-mode
  ``elicitation`` consent → ``sampling`` + paired ``cost.record``) is driven
  through the REAL ``mcp`` SDK types on a REAL ``ClientSession`` callback surface
  inside a flushing ``@trace``. The uploaded trace's attestation chain
  reconstructs and ``verify_chain(...)`` returns valid over EVERY event
  (money + consent events included), with a TAMPER control that breaks link 1 so
  the pass is not vacuous.
* **Real error-shape** — a REAL ``mcp.shared.exceptions.McpError`` (built from a
  real ``mcp.types.ErrorData``) is raised the real way through the wrapped
  ``call_tool`` and surfaces as ``agent.error`` with ``source == "mcp"``,
  ``error_type == "McpError"`` (the real SDK class name, not a synthetic
  ``RuntimeError``) and the exception message verbatim.
* **Redaction** — the same full session under ``capture_content=False`` keeps its
  structure (tool_name, consent action, elicitation id, the cost.record) but a
  SENTINEL riding the tool ARGUMENTS and the elicitation MESSAGE never survives
  into the stored trace; a ``capture_content=True`` vacuity control proves the
  SAME path DOES carry the SENTINEL otherwise.
* **Cost** (Group-B) — the ``mcp.sampling`` ``cost.record`` for a real,
  bundled-priced Claude model carries a non-null ``cost_usd`` filled by the
  central price-on-emit chokepoint (bites if the money path stops pricing).

The ONLY mock is the network boundary: no MCP transport is ever run — we attach
the adapter to a real ``ClientSession`` and invoke its (now-wrapped) callbacks /
tool methods directly, exactly how the live session dispatches a server-initiated
elicitation/sampling request and a client tool call. Every MCP object is a real
``mcp`` SDK 1.27 pydantic type. The module ``importorskip("mcp")`` (mcp is gated
on py>=3.10) so it runs in the py3.11 Invariant Gate + Run-Tests(3.10-3.12) and
in ``.audit-venvs/sk``.
"""

from __future__ import annotations

import json
import asyncio
from typing import Any, Dict

import anyio
import pytest

pytest.importorskip("mcp")

from mcp import types  # noqa: E402
from mcp.client.session import ClientSession  # noqa: E402
from mcp.shared.exceptions import McpError  # noqa: E402

from layerlens.instrument import trace  # noqa: E402
from layerlens.attestation._verify import verify_chain  # noqa: E402
from layerlens.instrument._context import _current_span_id, _current_collector  # noqa: E402
from layerlens.attestation._envelope import HashScope, AttestationEnvelope  # noqa: E402
from layerlens.instrument._collector import TraceCollector  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.protocols.mcp.adapter import MCPProtocolAdapter  # noqa: E402

from ...conftest import find_events  # noqa: E402

# The floor guards privacy / observability / attestation / cost contracts — it
# belongs in the fast, REQUIRED Invariant Gates job (shift-left), like the
# sibling test_mcp_invariants.py.
pytestmark = pytest.mark.invariant

SENTINEL = "LL-SENTINEL-7f3a9c2e"
_NO_CONTENT = CaptureConfig(capture_content=False)


# ── real-mcp fixture builders (the library's OWN typings) ──────────────────


def form_params(message: str = "Confirm the appointment") -> types.ElicitRequestFormParams:
    return types.ElicitRequestFormParams(
        message=message,
        requestedSchema={"type": "object", "properties": {"confirm": {"type": "boolean"}}},
    )


def sampling_params(prompt: str = "summarize this clinical note " * 20) -> types.CreateMessageRequestParams:
    return types.CreateMessageRequestParams(
        messages=[types.SamplingMessage(role="user", content=types.TextContent(type="text", text=prompt))],
        maxTokens=512,
        systemPrompt="be terse",
    )


def sampling_result(
    text: str = "A concise summary. ",
    model: str = "claude-haiku-4-5-20251001",
) -> types.CreateMessageResult:
    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text=text),
        model=model,
        stopReason="endTurn",
    )


def weather_tool() -> types.Tool:
    return types.Tool(
        name="weather",
        inputSchema={"type": "object"},
        outputSchema={"type": "object", "properties": {"temp": {"type": "number"}}, "required": ["temp"]},
    )


def call_tool_result(structured: Dict[str, Any]) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text="ok")],
        structuredContent=structured,
    )


def _session(*, elicit_cb: Any = None, sample_cb: Any = None) -> ClientSession:
    """A REAL ``ClientSession`` over anyio memory streams. The transport is never
    run — we only attach the adapter and invoke the wrapped callbacks/methods
    directly (exactly how the live session dispatches them, session.py:567/577)."""
    _w, _r = anyio.create_memory_object_stream(10)
    _w2, _r2 = anyio.create_memory_object_stream(10)
    return ClientSession(_r2, _w, elicitation_callback=elicit_cb, sampling_callback=sample_cb)


def _drive_full_session(
    mock_client: Any,
    capture_trace: Dict[str, Any],
    config: CaptureConfig,
    *,
    tool_args: Dict[str, Any] | None = None,
    elicit_message: str = "Confirm the appointment",
    sampling_model: str = "claude-haiku-4-5-20251001",
) -> Dict[str, Any]:
    """Drive a full MCP session inside a flushing ``@trace`` and return the
    uploaded trace payload (events + attestation).

    list_tools -> call_tool(+structured output) -> form-mode elicitation (accept)
    -> sampling + paired cost.record. Both the adapter and the trace collector run
    with *config* so redaction is exercised on both sides.
    """

    async def elicit_cb(context: Any, params: Any) -> types.ElicitResult:
        return types.ElicitResult(action="accept", content={"confirm": True})

    async def sample_cb(context: Any, params: Any) -> types.CreateMessageResult:
        return sampling_result(model=sampling_model)

    session = _session(elicit_cb=elicit_cb, sample_cb=sample_cb)

    async def call_tool(name: str, arguments: Any = None, **kw: Any) -> types.CallToolResult:
        return call_tool_result({"temp": 72})

    async def list_tools() -> types.ListToolsResult:
        return types.ListToolsResult(tools=[weather_tool()])

    # Patch the two real methods BEFORE connect so the adapter wraps THESE, and
    # seed the real ClientSession output-schema cache exactly as list_tools does.
    session.call_tool = call_tool  # type: ignore[method-assign]
    session.list_tools = list_tools  # type: ignore[method-assign]
    session._tool_output_schemas["weather"] = weather_tool().outputSchema

    adapter = MCPProtocolAdapter(capture_config=config)
    adapter.connect(target=session)

    @trace(mock_client, capture_config=config)
    async def clinical_mcp_agent() -> str:
        await session.list_tools()
        await session.call_tool("weather", tool_args if tool_args is not None else {"location": "clinic"})
        await session._elicitation_callback(None, form_params(elicit_message))
        await session._sampling_callback(None, sampling_params())
        return "done"

    asyncio.run(clinical_mcp_agent())
    return capture_trace


# ---------------------------------------------------------------------------
# Offline attestation-chain verification over a full real MCP session
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_over_full_session(self, mock_client, capture_trace):
        uploaded = _drive_full_session(mock_client, capture_trace, CaptureConfig.full())

        events = uploaded["events"]
        assert events, "the MCP session must flush a non-empty trace"
        # The realistic session actually exercised the money + consent paths, so
        # attestation is proven to cover them (not just @trace scaffolding).
        assert find_events(events, "mcp.tool.call"), "no mcp.tool.call in the session"
        assert find_events(events, "mcp.elicitation"), "no mcp.elicitation in the session"
        assert find_events(events, "mcp.sampling"), "no mcp.sampling in the session"
        assert find_events(events, "cost.record"), "no cost.record in the session"

        chain = (uploaded["attestation"] or {}).get("chain") or {}
        raw = chain.get("events") or []
        envelopes = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured for the MCP session"
        assert len(envelopes) == len(events), (
            f"attestation chain has {len(envelopes)} envelopes for {len(events)} events"
        )
        assert (uploaded["attestation"] or {}).get("root_hash") is not None

        result = verify_chain(envelopes)
        assert result.valid, f"attestation chain invalid: {result.error}"

        # Vacuity control: verify_chain must REJECT a broken interior link, proving
        # the pass above is not trivially true.
        assert len(envelopes) >= 2
        tampered = list(envelopes)
        tampered[1] = AttestationEnvelope(
            hash=tampered[1].hash,
            scope=tampered[1].scope,
            previous_hash="sha256:deadbeef-not-the-prior-hash",
        )
        broken = verify_chain(tampered)
        assert not broken.valid and broken.break_index == 1, "verify_chain failed to detect a broken link"


# ---------------------------------------------------------------------------
# Real error-shape floor (a real McpError raised through the wrapped call_tool)
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_real_mcp_error_surfaces_as_agent_error(self, mock_client, capture_trace):
        # A genuine mcp SDK exception — an McpError built from a real ErrorData, the
        # shape the mcp client raises on a JSON-RPC error. NOT a synthetic RuntimeError.
        err = McpError(types.ErrorData(code=-32603, message="MCP tool 'charge' failed: upstream service 502"))
        assert type(err).__name__ == "McpError"
        assert isinstance(err, Exception)
        real_message = str(err)

        session = _session()

        async def call_tool(name: str, arguments: Any = None, **kw: Any) -> Any:
            raise err

        session.call_tool = call_tool  # type: ignore[method-assign]
        adapter = MCPProtocolAdapter(capture_config=CaptureConfig.full())
        adapter.connect(target=session)

        @trace(mock_client, capture_config=CaptureConfig.full())
        async def agent() -> str:
            try:
                await session.call_tool("charge", {"amount": 499})
            except McpError:
                pass  # handled — the caller catches; the adapter already recorded it
            return "handled"

        asyncio.run(agent())

        events = capture_trace["events"]
        mcp_errors = [e["payload"] for e in find_events(events, "agent.error") if e["payload"].get("source") == "mcp"]
        assert len(mcp_errors) == 1, f"expected exactly one mcp agent.error, saw {find_events(events, 'agent.error')}"
        payload = mcp_errors[0]
        # The REAL SDK class name — not the synthetic RuntimeError the base suite uses.
        assert payload["error_type"] == "McpError"
        assert payload["source"] == "mcp"
        # The real exception message flows through verbatim (bite: dropped/mangled text).
        assert payload["error"] == real_message
        assert "502" in payload["error"]

        # The tool.call is recorded as an error and the async-task lifecycle shows
        # failed — the failure stays observable, not silently dropped.
        tool_calls = [e["payload"] for e in find_events(events, "mcp.tool.call")]
        assert any(p.get("status") == "error" and p.get("tool_name") == "charge" for p in tool_calls), (
            "the failing tool.call was not recorded status=error"
        )
        assert any(e["payload"].get("status") == "failed" for e in find_events(events, "mcp.async_task")), (
            "the async-task lifecycle did not record the failure"
        )


# ---------------------------------------------------------------------------
# Redaction content-absence over a real full MCP session (SENTINEL sweep)
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_content_present_when_capturing(self, mock_client, capture_trace):
        """Vacuity control: with capture_content=True the SAME session DOES carry
        the SENTINEL on the tool arguments and the elicitation message."""
        uploaded = _drive_full_session(
            mock_client,
            capture_trace,
            CaptureConfig.full(),
            tool_args={"q": SENTINEL},
            elicit_message=f"Approve access for {SENTINEL}",
        )
        events = uploaded["events"]
        assert SENTINEL in json.dumps(events), "control run must carry the SENTINEL when capturing content"
        # It rides real CONTENT keys: tool arguments and the elicitation prompt.
        calls = [e["payload"] for e in find_events(events, "mcp.tool.call")]
        assert any(SENTINEL in json.dumps(p.get("arguments"), default=str) for p in calls), "args not captured"
        reqs = [e["payload"] for e in find_events(events, "mcp.elicitation") if e["payload"].get("phase") == "request"]
        assert reqs and any(p.get("message") == f"Approve access for {SENTINEL}" for p in reqs), "prompt not captured"

    def test_content_absent_when_not_capturing(self, mock_client, capture_trace):
        """capture_content=False keeps the session's STRUCTURE but strips the
        SENTINEL out of every stored event (tool arguments + elicitation prompt)."""
        uploaded = _drive_full_session(
            mock_client,
            capture_trace,
            _NO_CONTENT,
            tool_args={"q": SENTINEL},
            elicit_message=f"Approve access for {SENTINEL}",
        )
        events = uploaded["events"]
        assert events, "the session must still emit structural events without content"

        # 1) SENTINEL sweep over the serialized trace.
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: SENTINEL survived capture_content=False"

        # 2) Content keys stripped from the payloads that would carry them.
        calls = [e["payload"] for e in find_events(events, "mcp.tool.call")]
        assert calls and all("arguments" not in p for p in calls), "tool arguments not stripped under no-content"
        for p in find_events(events, "mcp.elicitation"):
            assert "message" not in p["payload"], "elicitation prompt leaked under no-content"
            assert "title" not in p["payload"], "elicitation title (prompt echo) leaked under no-content"

        # 3) Redact without going blind: metadata + the money/consent structure survive.
        assert any(p.get("tool_name") == "weather" for p in calls), "tool_name metadata over-stripped"
        resp = [e["payload"] for e in find_events(events, "mcp.elicitation") if e["payload"].get("phase") == "response"]
        assert resp and resp[0].get("action") == "accept", "consent action over-stripped"
        assert resp[0].get("elicitation_id"), "elicitation_id over-stripped"
        assert find_events(events, "cost.record"), "cost.record over-suppressed under no-content"


# ---------------------------------------------------------------------------
# Cost floor (Group-B): the sampling cost.record is priced by the chokepoint
# ---------------------------------------------------------------------------
class TestSamplingCost:
    def test_cost_usd_present_on_real_token_shape(self, mock_client, capture_trace):
        # claude-haiku-4-5-20251001 is a real, bundled-priced Claude Haiku 4.5 id.
        uploaded = _drive_full_session(
            mock_client, capture_trace, CaptureConfig.full(), sampling_model="claude-haiku-4-5-20251001"
        )
        events = uploaded["events"]
        costs = [e["payload"] for e in find_events(events, "cost.record")]
        assert costs, "no cost.record emitted for the sampling round-trip (the money path is invisible)"
        cost = costs[0]
        assert cost["model"] == "claude-haiku-4-5-20251001"
        assert cost["prompt_tokens"] > 0 and cost["completion_tokens"] > 0
        assert cost["total_tokens"] == cost["prompt_tokens"] + cost["completion_tokens"]
        # The central price-on-emit chokepoint filled cost_usd from model + tokens.
        assert cost.get("cost_usd") is not None and cost["cost_usd"] > 0, (
            "central chokepoint did not price the sampling cost.record"
        )


# ---------------------------------------------------------------------------
# Concurrent async-task keying isolation ([D] fix)
# ---------------------------------------------------------------------------
class TestConcurrentAsyncTaskIsolation:
    """Two INTERLEAVED concurrent tool calls under ONE span each keep a distinct
    async_task lifecycle — no keying collision, no synthetic None-fallback.

    The bug: ``_emit_async_task_start`` used the originating ``parent_span_id``
    (== ``_current_span_id`` when a span is active) as BOTH the emitted
    ``async_task_id`` AND the ``AsyncTaskTracker`` dict key. Two concurrent tool
    calls under one span therefore collide: the second ``create()`` overwrites the
    first's ``_TaskState``, the first's ``end`` pops it, and the second's ``end``
    ``update()`` returns ``None`` → the synthetic fallback loses
    ``originating_span_id`` and every event shares one colliding ``async_task_id``.

    Per the ateam contract (``AsyncTaskEvent``: a UNIQUE ``async_task_id`` per
    async task, distinct from the ``originating_tool_call_span_id``) each call must
    get its own unique id while keeping the span as the originating id.
    """

    def _drive_interleaved(self) -> tuple[list, str]:
        adapter = MCPProtocolAdapter(capture_config=CaptureConfig.full())

        # Force interleaving: BOTH tool bodies enter (so BOTH `start` events are
        # already emitted by the wrapper's `_before`) before EITHER returns (so
        # neither `end` runs first) — the exact ordering that triggers the
        # collision. Sequential calls would clean up cleanly and hide the bug.
        a_in_body = anyio.Event()
        b_in_body = anyio.Event()

        async def call_tool(name: str, arguments: Any = None, **kw: Any) -> types.CallToolResult:
            if name == "tool_a":
                a_in_body.set()
                await b_in_body.wait()
                return call_tool_result({"a": 1})
            b_in_body.set()
            await a_in_body.wait()
            return call_tool_result({"b": 2})

        session = _session()
        session.call_tool = call_tool  # type: ignore[method-assign]
        adapter.connect(target=session)

        span = "one-shared-span-000001"
        collector = TraceCollector(object(), CaptureConfig.full())

        async def body() -> None:
            async with anyio.create_task_group() as tg:
                tg.start_soon(session.call_tool, "tool_a", {})
                tg.start_soon(session.call_tool, "tool_b", {})

        async def wrapped() -> None:
            ct = _current_collector.set(collector)
            st = _current_span_id.set(span)
            try:
                await body()
            finally:
                _current_span_id.reset(st)
                _current_collector.reset(ct)

        anyio.run(wrapped)
        return collector.events, span

    def test_concurrent_tool_calls_keep_distinct_async_tasks(self) -> None:
        events, span = self._drive_interleaved()
        tasks = [e["payload"] for e in find_events(events, "mcp.async_task")]

        # Two calls × (start + end) — both lifecycles emitted.
        assert len(tasks) == 4, f"expected 4 async_task events for 2 interleaved calls, saw {tasks}"

        ids = {p["async_task_id"] for p in tasks}
        # Collision check: two concurrent calls under one span → TWO distinct ids.
        assert len(ids) == 2, f"async_task keying COLLISION: {len(ids)} distinct id(s) for 2 concurrent calls: {tasks}"

        # The async_task_id is a UNIQUE task id, not the originating span (ateam
        # contract: async_task_id is distinct from originating_tool_call_span_id).
        assert span not in ids, f"async_task_id reused the originating span id (collision-prone): {ids}"

        # No synthetic None-fallback: every lifecycle event keeps its originating
        # span. The bug drops originating_span_id on the second call's end event.
        for p in tasks:
            assert p.get("originating_span_id") == span, (
                f"async_task lost originating_span_id (None-fallback collision): {p}"
            )

        # Each unique id owns a full start(running) → end(completed) lifecycle.
        for tid in ids:
            lifecycle = [p for p in tasks if p["async_task_id"] == tid]
            phases = {p["phase"] for p in lifecycle}
            assert phases == {"start", "end"}, f"async_task {tid} has an incomplete lifecycle: {lifecycle}"
            start = next(p for p in lifecycle if p["phase"] == "start")
            end = next(p for p in lifecycle if p["phase"] == "end")
            assert start["status"] == "running", f"start not running: {start}"
            assert end["status"] == "completed", f"end not completed: {end}"
