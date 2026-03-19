"""
MCP Extensions adapter test fixtures.
"""

import pytest
from typing import Any

from layerlens.instrument.adapters.protocols.mcp.adapter import MCPExtensionsAdapter
from layerlens.instrument.adapters._capture import CaptureConfig


class MockStratix:
    """Mock STRATIX instance for testing."""

    def __init__(self):
        self.events: list[Any] = []
        self.agent_id = "test-agent"
        self.framework = "mcp_extensions"
        self.is_policy_violated = False

    def __bool__(self):
        return True

    def emit(self, *args, **kwargs):
        self.events.append(args)


class MockMCPServer:
    """
    In-process mock MCP server for testing.

    Serves tool definitions and returns configurable structured outputs.
    """

    def __init__(self):
        self.tools = {
            "search": {
                "name": "search",
                "description": "Search the web",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                "outputSchema": {
                    "$id": "search-output",
                    "type": "object",
                    "properties": {"results": {"type": "array"}},
                    "required": ["results"],
                },
            }
        }
        self.call_count = 0
        self.fail_next = False

    def call_tool(self, name: str, arguments: dict) -> dict:
        self.call_count += 1
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("Tool execution failed")
        tool = self.tools.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")
        return {"results": [f"Result for {arguments.get('query', '')}"], "structuredContent": True}

    def get_tool_schema(self, name: str) -> dict | None:
        tool = self.tools.get(name)
        if tool:
            return tool.get("outputSchema")
        return None


@pytest.fixture
def mock_stratix():
    return MockStratix()


@pytest.fixture
def mock_server():
    return MockMCPServer()


@pytest.fixture
def mcp_adapter(mock_stratix):
    adapter = MCPExtensionsAdapter(stratix=mock_stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    return adapter
