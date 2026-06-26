"""Behavioral unit tests for the LIVE MCP protocol adapter (LAY-3617).

The OLD ``test_mcp_tool_wrapper.py`` exercises ``mcp/tool_wrapper.py`` — DEAD
code the live adapter never imports. This file drives the real
:class:`MCPProtocolAdapter` the way production does: construct it, ``connect``
it to a target object exposing the patched methods, then invoke the now-wrapped
methods and assert the EMITTED events (exact ``event_type`` + key payload
fields).

MCP carries high-stakes content (tool args, structured outputs, elicitation
prompts), so we assert strong behavior: the tool-call lifecycle for BOTH the
sync and the async ``call_tool`` branch (the wrapper splits on
``inspect.iscoroutinefunction`` of the ORIGINAL), the ``mcp.async_task``
start/end framing around each call, ``mcp.tools.listed`` for both a bare list
and a ``.tools``-attr object, the ``mcp.structured_output`` validation path
(with and without an ``output_schema``), and the ``mcp.elicitation`` request +
response pair.

Every emitted event is flushed through the real upload path into the shared
``capture_trace`` fixture, so the autouse ``_enforce_schema_lock`` fixture
(LAY-3583) validates the real emitted payloads after each test — i.e. these
emits must be schema-valid, not just present. (Routing through ``flush()``
rather than reading ``collector.events`` directly is deliberate: the schema
lock's pending buffer lives in the pytest-loaded conftest module, so the only
reliable way to feed it is the fixture, not a re-imported ``record_for_schema_lock``.)
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from layerlens.instrument._events import (
    MCP_TOOL_CALL,
    MCP_ASYNC_TASK,
    MCP_ELICITATION,
    MCP_STRUCTURED_OUTPUT,
)
from layerlens.instrument._context import _current_collector
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.protocols.mcp.adapter import MCPProtocolAdapter
from layerlens.instrument.adapters.protocols.mcp.structured_output import (
    compute_output_hash,
    compute_schema_hash,
)

MCP_TOOLS_LISTED = "mcp.tools.listed"


def _run_collected(
    mock_client: Any,
    capture_trace: Dict[str, Any],
    fn: Any,
    config: Optional[CaptureConfig] = None,
) -> List[Dict[str, Any]]:
    """Drive *fn* under an ambient collector, flush, and return uploaded events.

    Flushing through the real upload path routes every event into the shared
    ``capture_trace`` fixture, which records them for the autouse schema lock —
    so the lock genuinely validates these payloads (verified to fail on an
    unregistered event type).
    """
    collector = TraceCollector(client=mock_client, config=config or CaptureConfig())
    token = _current_collector.set(collector)
    try:
        fn()
    finally:
        _current_collector.reset(token)
    collector.flush()
    return capture_trace["events"]


def find_events(events: List[Dict[str, Any]], event_type: str) -> List[Dict[str, Any]]:
    return [e for e in events if e["event_type"] == event_type]


def _payloads(events: List[Dict[str, Any]], event_type: str) -> List[Dict[str, Any]]:
    return [e["payload"] for e in find_events(events, event_type)]


# ---------------------------------------------------------------------------
# call_tool — sync branch + async_task lifecycle framing
# ---------------------------------------------------------------------------


class TestCallToolSync:
    def test_sync_tool_call_emits_call_and_async_task_lifecycle(self, mock_client, capture_trace) -> None:
        target = SimpleNamespace(
            call_tool=lambda name, arguments=None, **kw: {"content": [{"a": 1}, {"b": 2}, {"c": 3}]}
        )
        adapter = MCPProtocolAdapter()
        adapter.connect(target=target)

        def go() -> None:
            out = target.call_tool(name="search", arguments={"q": "hello"})
            assert out == {"content": [{"a": 1}, {"b": 2}, {"c": 3}]}, "wrapper must pass result through unchanged"

        events = _run_collected(mock_client, capture_trace, go)

        # mcp.tool.call: exact tool_name, arguments preserved, result SUMMARIZED to a shape.
        calls = _payloads(events, MCP_TOOL_CALL)
        assert len(calls) == 1
        call = calls[0]
        assert call["tool_name"] == "search"
        assert call["arguments"] == {"q": "hello"}
        assert call["result"] == {"content_items": 3}, "_summarize must collapse content list to a count"
        assert "error" not in call
        assert isinstance(call["latency_ms"], (int, float))
        assert call["protocol"] == "mcp"

        # mcp.async_task: exactly a start (running) and an end (completed) framing the call.
        tasks = _payloads(events, MCP_ASYNC_TASK)
        assert [t["phase"] for t in tasks] == ["start", "end"]
        start, end = tasks
        assert start["status"] == "running" and start["tool_name"] == "search"
        assert end["status"] == "completed" and end["tool_name"] == "search"
        # Lifecycle is correlated by a single async_task_id across both phases.
        assert start["async_task_id"] == end["async_task_id"]
        assert "error" not in end

    def test_sync_tool_call_error_path_emits_error_and_failed_task(self, mock_client, capture_trace) -> None:
        def boom(name, arguments=None, **kw):
            raise RuntimeError("tool exploded")

        target = SimpleNamespace(call_tool=boom)
        adapter = MCPProtocolAdapter()
        adapter.connect(target=target)

        def go() -> None:
            with pytest.raises(RuntimeError, match="tool exploded"):
                target.call_tool(name="search", arguments={"q": "hello"})

        events = _run_collected(mock_client, capture_trace, go)

        call = _payloads(events, MCP_TOOL_CALL)[0]
        assert call["tool_name"] == "search"
        assert call["error"] == "tool exploded"
        assert "result" not in call

        tasks = _payloads(events, MCP_ASYNC_TASK)
        assert [t["phase"] for t in tasks] == ["start", "end"]
        assert tasks[0]["status"] == "running"
        assert tasks[1]["status"] == "failed"
        assert tasks[1]["error"] == "tool exploded"


# ---------------------------------------------------------------------------
# call_tool — async branch (real `async def` hits wrapped_async)
# ---------------------------------------------------------------------------


class TestCallToolAsync:
    def test_async_tool_call_emits_same_lifecycle(self, mock_client, capture_trace) -> None:
        async def call_tool(name, arguments=None, **kw):
            return {"content": [{"x": 1}, {"y": 2}]}

        target = SimpleNamespace(call_tool=call_tool)
        adapter = MCPProtocolAdapter()
        adapter.connect(target=target)

        # connect() must have produced an async wrapper (sync callers must not
        # get a coroutine back, and async callers must still get one).
        assert asyncio.iscoroutinefunction(target.call_tool), "async original must keep an async wrapper"

        results: Dict[str, Any] = {}

        def go() -> None:
            results["out"] = asyncio.run(target.call_tool(name="fetch", arguments={"id": 7}))

        events = _run_collected(mock_client, capture_trace, go)
        assert results["out"] == {"content": [{"x": 1}, {"y": 2}]}

        call = _payloads(events, MCP_TOOL_CALL)[0]
        assert call["tool_name"] == "fetch"
        assert call["arguments"] == {"id": 7}
        assert call["result"] == {"content_items": 2}

        tasks = _payloads(events, MCP_ASYNC_TASK)
        assert [t["phase"] for t in tasks] == ["start", "end"]
        assert tasks[0]["status"] == "running"
        assert tasks[1]["status"] == "completed"

    def test_async_tool_call_error_path(self, mock_client, capture_trace) -> None:
        async def call_tool(name, arguments=None, **kw):
            raise ValueError("async boom")

        target = SimpleNamespace(call_tool=call_tool)
        adapter = MCPProtocolAdapter()
        adapter.connect(target=target)

        def go() -> None:
            with pytest.raises(ValueError, match="async boom"):
                asyncio.run(target.call_tool(name="fetch", arguments={"id": 7}))

        events = _run_collected(mock_client, capture_trace, go)

        call = _payloads(events, MCP_TOOL_CALL)[0]
        assert call["error"] == "async boom"
        end = _payloads(events, MCP_ASYNC_TASK)[1]
        assert end["status"] == "failed" and end["error"] == "async boom"


# ---------------------------------------------------------------------------
# list_tools — bare list AND a `.tools`-attr object
# ---------------------------------------------------------------------------


class TestListTools:
    def test_list_tools_returning_a_list(self, mock_client, capture_trace) -> None:
        tools = [SimpleNamespace(name="alpha"), SimpleNamespace(name="beta")]
        target = SimpleNamespace(list_tools=lambda: tools)
        adapter = MCPProtocolAdapter()
        adapter.connect(target=target)

        events = _run_collected(mock_client, capture_trace, lambda: target.list_tools())

        listed = _payloads(events, MCP_TOOLS_LISTED)
        assert len(listed) == 1
        assert listed[0]["tool_count"] == 2
        assert listed[0]["tool_names"] == ["alpha", "beta"]
        assert listed[0]["protocol"] == "mcp"

    def test_list_tools_returning_a_tools_attr_object(self, mock_client, capture_trace) -> None:
        # MCP ClientSession.list_tools() returns a result object with a .tools attr.
        result = SimpleNamespace(
            tools=[SimpleNamespace(name="one"), SimpleNamespace(name="two"), SimpleNamespace(name="three")]
        )
        target = SimpleNamespace(list_tools=lambda: result)
        adapter = MCPProtocolAdapter()
        adapter.connect(target=target)

        events = _run_collected(mock_client, capture_trace, lambda: target.list_tools())

        listed = _payloads(events, MCP_TOOLS_LISTED)[0]
        assert listed["tool_count"] == 3
        assert listed["tool_names"] == ["one", "two", "three"]

    def test_list_tools_async(self, mock_client, capture_trace) -> None:
        async def list_tools():
            return SimpleNamespace(tools=[SimpleNamespace(name="solo")])

        target = SimpleNamespace(list_tools=list_tools)
        adapter = MCPProtocolAdapter()
        adapter.connect(target=target)
        assert asyncio.iscoroutinefunction(target.list_tools)

        events = _run_collected(mock_client, capture_trace, lambda: asyncio.run(target.list_tools()))

        listed = _payloads(events, MCP_TOOLS_LISTED)[0]
        assert listed["tool_count"] == 1
        assert listed["tool_names"] == ["solo"]


# ---------------------------------------------------------------------------
# structured_output — structured_content / structuredContent + output_schema
# ---------------------------------------------------------------------------


class TestStructuredOutput:
    def test_structured_content_without_schema(self, mock_client, capture_trace) -> None:
        structured = {"answer": 42, "unit": "kg"}
        target = SimpleNamespace(
            call_tool=lambda name, arguments=None, **kw: {
                "content": [{"text": "ok"}],
                "structured_content": structured,
            }
        )
        adapter = MCPProtocolAdapter()
        adapter.connect(target=target)

        events = _run_collected(mock_client, capture_trace, lambda: target.call_tool(name="calc", arguments={}))

        so = _payloads(events, MCP_STRUCTURED_OUTPUT)
        assert len(so) == 1
        so = so[0]
        assert so["tool_name"] == "calc"
        # Output hashed (not dumped raw) — exact, deterministic hash of the value.
        assert so["output_hash"] == compute_output_hash(structured)
        assert so["validation_passed"] is True
        assert "schema_hash" not in so, "no output_schema → no schema_hash"

        # The matching mcp.tool.call summarizes the content list, never the raw struct.
        call = _payloads(events, MCP_TOOL_CALL)[0]
        assert call["result"] == {"content_items": 1}

    def test_structured_content_camelcase_with_passing_schema(self, mock_client, capture_trace) -> None:
        structured = {"answer": 42}
        schema = {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "number"}}}
        target = SimpleNamespace(
            call_tool=lambda name, arguments=None, **kw: {
                "structuredContent": structured,
                "outputSchema": schema,
            }
        )
        adapter = MCPProtocolAdapter()
        adapter.connect(target=target)

        events = _run_collected(mock_client, capture_trace, lambda: target.call_tool(name="calc", arguments={}))

        so = _payloads(events, MCP_STRUCTURED_OUTPUT)[0]
        assert so["output_hash"] == compute_output_hash(structured)
        assert so["schema_hash"] == compute_schema_hash(schema)
        assert so["validation_passed"] is True
        assert "validation_errors" not in so

    def test_structured_content_with_failing_schema_records_errors(self, mock_client, capture_trace) -> None:
        # `answer` is required but absent → validation must fail and surface errors.
        structured = {"wrong_key": 1}
        schema = {"type": "object", "required": ["answer"]}
        target = SimpleNamespace(
            call_tool=lambda name, arguments=None, **kw: {
                "structured_content": structured,
                "output_schema": schema,
            }
        )
        adapter = MCPProtocolAdapter()
        adapter.connect(target=target)

        events = _run_collected(mock_client, capture_trace, lambda: target.call_tool(name="calc", arguments={}))

        so = _payloads(events, MCP_STRUCTURED_OUTPUT)[0]
        assert so["validation_passed"] is False
        assert so["validation_errors"], "failing validation must record errors"
        assert so["schema_hash"] == compute_schema_hash(schema)

    def test_no_structured_output_when_result_has_none(self, mock_client, capture_trace) -> None:
        target = SimpleNamespace(call_tool=lambda name, arguments=None, **kw: {"content": []})
        adapter = MCPProtocolAdapter()
        adapter.connect(target=target)

        events = _run_collected(mock_client, capture_trace, lambda: target.call_tool(name="calc", arguments={}))
        assert _payloads(events, MCP_STRUCTURED_OUTPUT) == [], "no structured payload → no mcp.structured_output"


# ---------------------------------------------------------------------------
# elicitation — request + response phases
# ---------------------------------------------------------------------------


class TestElicitation:
    def test_elicit_emits_request_then_response(self, mock_client, capture_trace) -> None:
        schema = {"type": "object", "properties": {"confirm": {"type": "boolean"}}}
        target = SimpleNamespace(elicit=lambda title, schema=None, **kw: {"confirm": True})
        adapter = MCPProtocolAdapter()
        adapter.connect(target=target)

        def go() -> None:
            out = target.elicit("Approve the payment?", schema=schema)
            assert out == {"confirm": True}

        events = _run_collected(mock_client, capture_trace, go)

        elicits = _payloads(events, MCP_ELICITATION)
        assert [e["phase"] for e in elicits] == ["request", "response"]
        req, resp = elicits

        # Request phase: id, title, schema hashed (privacy-preserving).
        assert req["title"] == "Approve the payment?"
        from layerlens.instrument.adapters.protocols.mcp.elicitation import ElicitationTracker

        assert req["schema_hash"] == ElicitationTracker.hash_schema(schema)
        assert req["elicitation_id"]

        # Response phase: same id, response hashed, latency recorded.
        assert resp["elicitation_id"] == req["elicitation_id"]
        assert resp["response_hash"] == ElicitationTracker.hash_response({"confirm": True})
        assert isinstance(resp["latency_ms"], (int, float))

    def test_elicit_async(self, mock_client, capture_trace) -> None:
        async def elicit(title, schema=None, **kw):
            return {"confirm": False}

        target = SimpleNamespace(elicit=elicit)
        adapter = MCPProtocolAdapter()
        adapter.connect(target=target)
        assert asyncio.iscoroutinefunction(target.elicit)

        events = _run_collected(mock_client, capture_trace, lambda: asyncio.run(target.elicit("Async prompt?")))

        elicits = _payloads(events, MCP_ELICITATION)
        assert [e["phase"] for e in elicits] == ["request", "response"]
        assert elicits[0]["elicitation_id"] == elicits[1]["elicitation_id"]

    def test_elicit_error_does_not_emit_response(self, mock_client, capture_trace) -> None:
        def elicit(title, schema=None, **kw):
            raise RuntimeError("user cancelled")

        target = SimpleNamespace(elicit=elicit)
        adapter = MCPProtocolAdapter()
        adapter.connect(target=target)

        def go() -> None:
            with pytest.raises(RuntimeError, match="user cancelled"):
                target.elicit("Prompt?")

        events = _run_collected(mock_client, capture_trace, go)

        elicits = _payloads(events, MCP_ELICITATION)
        assert [e["phase"] for e in elicits] == ["request"], "an errored elicit must not emit a response phase"


# ---------------------------------------------------------------------------
# connect() wiring — only present methods patched
# ---------------------------------------------------------------------------


class TestConnectWiring:
    def test_connect_only_patches_present_methods(self) -> None:
        original = lambda name, arguments=None, **kw: {"content": []}
        target = SimpleNamespace(call_tool=original)  # no list_tools, no elicit
        adapter = MCPProtocolAdapter()
        adapter.connect(target=target)

        assert target.call_tool is not original, "call_tool must be wrapped"
        assert not hasattr(target, "list_tools")
        assert not hasattr(target, "elicit")
        # The original is preserved for disconnect().
        assert adapter._originals["call_tool"] is original
