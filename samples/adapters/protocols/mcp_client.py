"""Sample: instrument an MCP client session and capture tool-call telemetry."""

from __future__ import annotations

import os
import sys
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.protocols.mcp import instrument_mcp, uninstrument_mcp


class _FakeMCPClient:
    """Stand-in for mcp.ClientSession: lets the sample run without a live server."""

    async def call_tool(self, name: str, arguments: dict) -> dict:
        return {"content": [{"type": "text", "text": f"echo: {name} / {arguments}"}]}

    async def list_tools(self) -> dict:
        return {"tools": [{"name": "echo"}, {"name": "lookup"}]}


async def main() -> None:
    client = _FakeMCPClient()
    instrument_mcp(client)
    try:
        with capture_events("mcp"):
            await client.list_tools()
            await client.call_tool("echo", {"msg": "hello"})
    finally:
        uninstrument_mcp()


if __name__ == "__main__":
    asyncio.run(main())
