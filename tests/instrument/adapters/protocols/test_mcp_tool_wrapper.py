from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from layerlens.instrument._events import MCP_TOOL_CALL
from layerlens.instrument.adapters.protocols.mcp.tool_wrapper import (
    wrap_mcp_tool_call,
    wrap_mcp_tool_call_async,
)


def _last_payload(adapter):
    return adapter.emit.call_args.args[1]


class TestSyncWrapper:
    def test_emits_on_success(self):
        adapter = MagicMock()
        wrapped = wrap_mcp_tool_call(lambda **_kw: {"content": "ok"}, adapter)
        result = wrapped(name="search", arguments={"q": "hi"})
        assert result == {"content": "ok"}
        assert adapter.emit.call_args.args[0] == MCP_TOOL_CALL
        payload = _last_payload(adapter)
        assert payload["tool_name"] == "search"
        assert payload["arguments"] == {"q": "hi"}
        assert payload["result"] == {"content": "ok"}
        assert "error" not in payload
        assert payload["latency_ms"] >= 0

    def test_emits_on_error_and_reraises(self):
        adapter = MagicMock()

        def broken(**_kw):
            raise RuntimeError("kaboom")

        wrapped = wrap_mcp_tool_call(broken, adapter)
        with pytest.raises(RuntimeError, match="kaboom"):
            wrapped(name="search", arguments={})
        payload = _last_payload(adapter)
        assert payload["error"] == "kaboom"
        assert "result" not in payload

    def test_idempotent_wrapping(self):
        adapter = MagicMock()
        fn = lambda **_kw: None  # noqa: E731
        once = wrap_mcp_tool_call(fn, adapter)
        twice = wrap_mcp_tool_call(once, adapter)
        assert once is twice

    def test_extracts_tool_name_from_positional_arg(self):
        adapter = MagicMock()
        wrapped = wrap_mcp_tool_call(lambda *a, **_k: {"ok": True}, adapter)
        wrapped("search", {"q": "hi"})
        payload = _last_payload(adapter)
        assert payload["tool_name"] == "search"
        assert payload["arguments"] == {"q": "hi"}

    def test_coerces_model_dump_output(self):
        adapter = MagicMock()

        class Pydanticish:
            def model_dump(self):
                return {"value": 42}

        wrap_mcp_tool_call(lambda **_k: Pydanticish(), adapter)(name="x", arguments={})
        assert _last_payload(adapter)["result"] == {"value": 42}


class TestAsyncWrapper:
    def test_emits_on_success(self):
        adapter = MagicMock()

        async def coro(**_kw):
            return {"ok": True}

        wrapped = wrap_mcp_tool_call_async(coro, adapter)
        asyncio.run(wrapped(name="search", arguments={"q": "x"}))
        payload = _last_payload(adapter)
        assert payload["tool_name"] == "search"
        assert payload["result"] == {"ok": True}

    def test_emits_on_error(self):
        adapter = MagicMock()

        async def coro(**_kw):
            raise ValueError("bad")

        wrapped = wrap_mcp_tool_call_async(coro, adapter)
        with pytest.raises(ValueError):
            asyncio.run(wrapped(name="x", arguments={}))
        assert _last_payload(adapter)["error"] == "bad"

    def test_idempotent_wrapping(self):
        adapter = MagicMock()

        async def coro(**_k):
            return None

        once = wrap_mcp_tool_call_async(coro, adapter)
        twice = wrap_mcp_tool_call_async(once, adapter)
        assert once is twice
