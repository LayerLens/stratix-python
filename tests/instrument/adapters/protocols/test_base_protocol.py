"""Tests for BaseProtocolAdapter shared behavior (construction, pooling)."""

from __future__ import annotations

import asyncio
import threading

from layerlens.instrument._context import _current_collector
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.protocols.a2ui import A2UIProtocolAdapter
from layerlens.instrument.adapters.protocols._base_protocol import BaseProtocolAdapter


class _DummyProtocolAdapter(BaseProtocolAdapter):
    PROTOCOL = "dummy"
    PROTOCOL_VERSION = "1.0"

    def connect(self, target=None, **kwargs):
        self._client = target
        return target


class TestEmitStampsFramework:
    """Every protocol event must carry a ``framework`` tag (= PROTOCOL) so the
    traces Framework column/filter populate for protocol adapters (a2a, mcp, ucp,
    …), matching every framework/provider adapter. Regression for the dev
    finding that protocol traces left Framework blank."""

    def test_emit_stamps_framework_equal_to_protocol(self):
        adapter = _DummyProtocolAdapter()
        collector = TraceCollector(object(), CaptureConfig())
        token = _current_collector.set(collector)
        try:
            adapter.emit("dummy.event", {"foo": "bar"})
        finally:
            _current_collector.reset(token)
        assert len(collector.events) == 1
        p = collector.events[0]["payload"]
        assert p["framework"] == "dummy"
        assert p["protocol"] == "dummy"

    def test_explicit_payload_framework_wins(self):
        adapter = _DummyProtocolAdapter()
        collector = TraceCollector(object(), CaptureConfig())
        token = _current_collector.set(collector)
        try:
            adapter.emit("dummy.event", {"framework": "custom"})
        finally:
            _current_collector.reset(token)
        assert collector.events[0]["payload"]["framework"] == "custom"


class TestConstructionWithoutEventLoop:
    """On Python 3.8/3.9 ``asyncio.Semaphore()`` binds ``get_event_loop()`` at
    construction; building an adapter with no current loop must not crash
    (LAY-3567 B4)."""

    def test_construct_after_asyncio_run(self):
        # asyncio.run() leaves no current event loop behind on py<3.10
        async def _noop():
            return None

        asyncio.run(_noop())
        adapter = _DummyProtocolAdapter()
        assert adapter.adapter_info().adapter_type == "protocol"

    def test_construct_in_worker_thread(self):
        # worker threads never have a current event loop
        results = []

        def _build():
            try:
                results.append(A2UIProtocolAdapter())
            except Exception as exc:
                results.append(exc)

        t = threading.Thread(target=_build)
        t.start()
        t.join()
        assert len(results) == 1
        assert isinstance(results[0], A2UIProtocolAdapter), f"construction raised: {results[0]!r}"


class TestConnectionPool:
    def test_pool_still_limits_concurrency(self):
        adapter = _DummyProtocolAdapter(max_connections=2)

        async def _exercise():
            await adapter.acquire_connection()
            await adapter.acquire_connection()
            # pool exhausted
            assert adapter._connection_semaphore.locked()
            adapter.release_connection()
            assert not adapter._connection_semaphore.locked()
            adapter.release_connection()

        asyncio.run(_exercise())

    def test_release_before_acquire_without_loop_is_safe(self):
        async def _noop():
            return None

        asyncio.run(_noop())
        adapter = _DummyProtocolAdapter()
        adapter.release_connection()  # nothing acquired — must not crash
