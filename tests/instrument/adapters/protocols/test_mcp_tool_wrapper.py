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


# ---------------------------------------------------------------------------
# Proven-to-bite redaction (the existing tests use a MagicMock adapter, so they
# never exercise the collector backstop — nothing proved wrap_mcp_tool_call's
# args/result/error are actually stripped under capture_content=False).
# ---------------------------------------------------------------------------


class TestWrapMcpRealRedaction:
    SECRET_ARG = "SENTINEL-private-query"
    SECRET_RES = "SENTINEL-secret-result"
    SECRET_ERR = "SENTINEL-boom-with-args"

    def _run(self, fn, *, capture_content):
        from layerlens.instrument._context import _current_collector
        from layerlens.instrument._collector import TraceCollector
        from layerlens.instrument._capture_config import CaptureConfig
        from layerlens.instrument.adapters.protocols.mcp.adapter import MCPProtocolAdapter

        adapter = MCPProtocolAdapter(capture_config=CaptureConfig(capture_content=capture_content))
        collector = TraceCollector(object(), CaptureConfig(capture_content=capture_content))
        token = _current_collector.set(collector)
        try:
            fn(adapter)
        except RuntimeError:
            pass
        finally:
            _current_collector.reset(token)
        return [e for e in collector.events if e["event_type"] == MCP_TOOL_CALL]

    def test_arguments_and_result_stripped_under_no_content(self):
        def go(adapter):
            wrap_mcp_tool_call(lambda **_k: {"content": self.SECRET_RES}, adapter)(
                name="search", arguments={"q": self.SECRET_ARG}
            )

        calls = self._run(go, capture_content=False)
        assert calls, "no mcp.tool.call emitted"
        p = calls[0]["payload"]
        assert "arguments" not in p and "result" not in p, "args/result not stripped (real backstop)"
        assert p.get("tool_name") == "search" and "latency_ms" in p, "metadata over-stripped"
        blob = repr(p)
        assert self.SECRET_ARG not in blob and self.SECRET_RES not in blob

    def test_error_string_stripped_under_no_content(self):
        def go(adapter):
            def broken(**_k):
                raise RuntimeError(self.SECRET_ERR)

            wrap_mcp_tool_call(broken, adapter)(name="charge", arguments={"q": self.SECRET_ARG})

        calls = self._run(go, capture_content=False)
        assert calls and "error" not in calls[0]["payload"], "error str(exc) not stripped (real backstop)"
        assert self.SECRET_ERR not in repr(calls[0]["payload"])

    def test_content_kept_when_capture_content_true(self):
        # over-strip guard: default config keeps the content
        def go(adapter):
            wrap_mcp_tool_call(lambda **_k: {"content": self.SECRET_RES}, adapter)(
                name="s", arguments={"q": self.SECRET_ARG}
            )

        calls = self._run(go, capture_content=True)
        assert calls[0]["payload"]["arguments"] == {"q": self.SECRET_ARG}

    def test_async_path_stripped(self):
        import asyncio

        async def coro(**_k):
            return {"content": self.SECRET_RES}

        def go(adapter):
            asyncio.run(wrap_mcp_tool_call_async(coro, adapter)(name="s", arguments={"q": self.SECRET_ARG}))

        calls = self._run(go, capture_content=False)
        assert calls and "arguments" not in calls[0]["payload"]
        assert self.SECRET_ARG not in repr(calls[0]["payload"])
