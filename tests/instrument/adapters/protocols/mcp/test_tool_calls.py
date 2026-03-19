"""
Tests for MCP tool call interception and event emission.
"""

import pytest


class TestMCPToolCalls:
    def test_on_tool_call(self, mcp_adapter, mock_stratix):
        mcp_adapter.on_tool_call(
            tool_name="search",
            input_data={"query": "test"},
            output_data={"results": ["result1"]},
            latency_ms=42.5,
        )
        assert len(mock_stratix.events) == 1
        event = mock_stratix.events[0][0]
        assert event.event_type == "tool.call"
        assert event.tool.name == "search"
        assert event.latency_ms == 42.5

    def test_on_tool_call_with_error(self, mcp_adapter, mock_stratix):
        mcp_adapter.on_tool_call(
            tool_name="failing_tool",
            input_data={"arg": "value"},
            error="Tool execution failed",
        )
        assert len(mock_stratix.events) == 1
        event = mock_stratix.events[0][0]
        assert event.error == "Tool execution failed"
