"""S12/F4 — protocol adapters emit agent.error on terminal FAILURE states.

Before this fix, protocol failures were signalled only inside protocol-typed
events (a2a.task.updated status=failed/rejected; mcp.tool.call status=error) that
no server reader keys on — so the atlas A8 default derived Status="completed"
over a failed protocol run. Emitting agent.error lights up the atlas 'error'
derivation (and ateam's 'failed') with zero reader changes.

Guardrail 'blocked' (ap2/ucp) and a normal 'canceled' are NOT errors and must
not route here.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import MagicMock

from layerlens.instrument._events import AGENT_ERROR
from layerlens.instrument._context import _current_collector
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.protocols.a2a.client import A2AClientWrapper
from layerlens.instrument.adapters.protocols.a2a.server import A2AServerWrapper
from layerlens.instrument.adapters.protocols.a2a.adapter import A2AProtocolAdapter
from layerlens.instrument.adapters.protocols.mcp.adapter import MCPProtocolAdapter


def _run(fn: Any) -> List[Dict[str, Any]]:
    collector = TraceCollector(object(), CaptureConfig())
    token = _current_collector.set(collector)
    try:
        fn()
    finally:
        _current_collector.reset(token)
    return collector.events


def _run_async(coro_fn: Any) -> List[Dict[str, Any]]:
    collector = TraceCollector(object(), CaptureConfig())

    async def _wrapped() -> None:
        token = _current_collector.set(collector)
        try:
            await coro_fn()
        finally:
            _current_collector.reset(token)

    asyncio.run(_wrapped())
    return collector.events


def _errors(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [e["payload"] for e in events if e["event_type"] == AGENT_ERROR]


# --- a2a: client wrap path (the production instrument_a2a surface) -----------


def _wrap_target(**result):
    adapter = A2AProtocolAdapter()
    target = type("Cli", (), {"send_task": staticmethod(lambda **kw: dict(result))})()
    adapter.connect(target=target)
    return target


class TestA2AWrapPath:
    def test_failed_result_emits_agent_error(self):
        target = _wrap_target(task_id="t1", status="failed")
        events = _run(lambda: target.send_task(task_id="t1"))
        errs = _errors(events)
        assert len(errs) == 1
        assert errs[0]["source"] == "a2a"

    def test_rejected_result_emits_agent_error(self):
        target = _wrap_target(task_id="t1", status="rejected")
        events = _run(lambda: target.send_task(task_id="t1"))
        assert len(_errors(events)) == 1

    def test_completed_result_emits_no_agent_error(self):
        target = _wrap_target(task_id="t1", status="completed")
        events = _run(lambda: target.send_task(task_id="t1"))
        assert _errors(events) == []

    def test_raised_exception_emits_agent_error_with_type(self):
        adapter = A2AProtocolAdapter()

        def _boom(**kw):
            raise RuntimeError("peer down")

        target = type("Cli", (), {"send_task": staticmethod(_boom)})()
        adapter.connect(target=target)

        def go():
            try:
                target.send_task(task_id="t1")
            except RuntimeError:
                pass

        errs = _errors(_run(go))
        assert len(errs) == 1
        assert errs[0]["error_type"] == "RuntimeError"


# --- a2a: client-helper path (A2AClientWrapper) -----------------------------


class TestA2AClientHelper:
    def _names(self, adapter):
        return [c.args[0] for c in adapter.emit.call_args_list]

    def test_complete_failed_emits_agent_error(self):
        adapter = MagicMock()
        A2AClientWrapper(adapter, "https://peer").complete_task(
            "t1", "failed", error_code="E_TIMEOUT", error_message="timed out"
        )
        assert AGENT_ERROR in self._names(adapter)

    def test_complete_completed_emits_no_agent_error(self):
        adapter = MagicMock()
        A2AClientWrapper(adapter, "https://peer").complete_task("t1", "completed")
        assert AGENT_ERROR not in self._names(adapter)

    def test_cancel_emits_no_agent_error(self):
        adapter = MagicMock()
        A2AClientWrapper(adapter, "https://peer").cancel_task("t1")
        assert AGENT_ERROR not in self._names(adapter)


# --- a2a: server path (A2AServerWrapper) ------------------------------------


class TestA2AServer:
    def _names(self, adapter):
        return [c.args[0] for c in adapter.emit.call_args_list]

    def _resp(self, task_id, state):
        return {"jsonrpc": "2.0", "id": "r", "result": {"kind": "task", "id": task_id, "status": {"state": state}}}

    def test_handler_exception_emits_agent_error(self):
        adapter = MagicMock()

        def handler(_b):
            raise RuntimeError("500 internal")

        wrapper = A2AServerWrapper(adapter, original_handler=handler)
        try:
            wrapper.handle_request({"method": "message/send", "id": "r", "params": {"message": {"messageId": "m"}}})
        except RuntimeError:
            pass
        assert AGENT_ERROR in self._names(adapter)

    def test_failed_response_emits_agent_error(self):
        adapter = MagicMock()
        wrapper = A2AServerWrapper(adapter, original_handler=lambda _b: self._resp("t1", "failed"))
        wrapper.handle_request({"method": "message/send", "id": "r", "params": {"message": {"messageId": "m"}}})
        assert AGENT_ERROR in self._names(adapter)

    def test_completed_response_emits_no_agent_error(self):
        adapter = MagicMock()
        wrapper = A2AServerWrapper(adapter, original_handler=lambda _b: self._resp("t1", "completed"))
        wrapper.handle_request({"method": "message/send", "id": "r", "params": {"message": {"messageId": "m"}}})
        assert AGENT_ERROR not in self._names(adapter)


# --- mcp: tool-call exception -----------------------------------------------


class TestMCPToolFailure:
    def test_tool_exception_emits_agent_error(self):
        adapter = MCPProtocolAdapter()

        class _Session:
            async def call_tool(self, name: str, arguments: Any = None, **kw: Any) -> Any:
                raise RuntimeError("tool boom")

        session = _Session()
        adapter.connect(target=session)

        async def go():
            try:
                await session.call_tool("weather", {})
            except RuntimeError:
                pass

        errs = _errors(_run_async(go))
        assert len(errs) == 1
        assert errs[0]["source"] == "mcp"
        assert errs[0]["error_type"] == "RuntimeError"
