"""
Tests for MCP Extensions adapter lifecycle.
"""

import pytest

from layerlens.instrument.adapters._base import AdapterCapability, AdapterStatus
from layerlens.instrument.adapters.protocols.mcp.adapter import MCPExtensionsAdapter


class TestMCPAdapterLifecycle:
    def test_connect(self):
        adapter = MCPExtensionsAdapter()
        adapter.connect()
        assert adapter.is_connected
        assert adapter.status == AdapterStatus.HEALTHY

    def test_disconnect(self):
        adapter = MCPExtensionsAdapter()
        adapter.connect()
        adapter.disconnect()
        assert not adapter.is_connected

    def test_get_adapter_info(self):
        adapter = MCPExtensionsAdapter()
        info = adapter.get_adapter_info()
        assert info.name == "MCPExtensionsAdapter"
        assert info.framework == "mcp_extensions"
        assert AdapterCapability.TRACE_PROTOCOL_EVENTS in info.capabilities
        assert AdapterCapability.TRACE_TOOLS in info.capabilities

    def test_serialize_for_replay(self):
        adapter = MCPExtensionsAdapter()
        replay = adapter.serialize_for_replay()
        assert replay.framework == "mcp_extensions"
