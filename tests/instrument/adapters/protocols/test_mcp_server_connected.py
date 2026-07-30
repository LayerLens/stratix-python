"""S14/F7 — the MCP adapter captures the server identity at the handshake.

Before this fix nothing wrapped ClientSession.initialize, so the server's
serverInfo (name/version) and negotiated protocolVersion were dropped and the
ElicitationTracker was fed the literal "mcp" instead of a real server name. Wrap
initialize and emit a new mcp.server.connected event carrying that identity.
A server is NOT an agent — this does not touch agent identity.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List

from layerlens.instrument._events import MCP_SERVER_CONNECTED
from layerlens.instrument._context import _current_collector
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.protocols.mcp.adapter import MCPProtocolAdapter


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


def _init_result(name="filesystem", version="1.2.3", protocol="2025-11-25"):
    return SimpleNamespace(
        serverInfo=SimpleNamespace(name=name, version=version),
        protocolVersion=protocol,
    )


class _Session:
    def __init__(self, result):
        self._result = result

    async def initialize(self, *a, **k):
        return self._result


def test_initialize_emits_server_connected():
    adapter = MCPProtocolAdapter()
    session = _Session(_init_result())
    adapter.connect(target=session)

    events = _run_async(lambda: session.initialize())
    connected = [e["payload"] for e in events if e["event_type"] == MCP_SERVER_CONNECTED]
    assert len(connected) == 1
    p = connected[0]
    assert p["server_name"] == "filesystem"
    assert p["server_version"] == "1.2.3"
    assert p["protocol_version"] == "2025-11-25"


def test_initialize_passes_through_result():
    adapter = MCPProtocolAdapter()
    result = _init_result()
    session = _Session(result)
    adapter.connect(target=session)

    got: List[Any] = []

    async def go():
        got.append(await session.initialize())

    _run_async(go)
    assert got[0] is result  # instrumentation must not alter the caller's return


def test_no_server_info_emits_no_event():
    adapter = MCPProtocolAdapter()
    session = _Session(SimpleNamespace())  # no serverInfo / protocolVersion
    adapter.connect(target=session)

    events = _run_async(lambda: session.initialize())
    # Honest blank: nothing declared -> no fabricated event.
    assert not [e for e in events if e["event_type"] == MCP_SERVER_CONNECTED]
