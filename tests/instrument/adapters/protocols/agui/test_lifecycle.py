"""
Tests for AG-UI adapter lifecycle.
"""

import pytest

from layerlens.instrument.adapters._base import AdapterCapability, AdapterStatus
from layerlens.instrument.adapters.protocols.agui.adapter import AGUIAdapter


class TestAGUIAdapterLifecycle:
    def test_connect(self):
        adapter = AGUIAdapter()
        adapter.connect()
        assert adapter.is_connected
        assert adapter.status == AdapterStatus.HEALTHY

    def test_disconnect(self):
        adapter = AGUIAdapter()
        adapter.connect()
        adapter.disconnect()
        assert not adapter.is_connected

    def test_get_adapter_info(self):
        adapter = AGUIAdapter()
        info = adapter.get_adapter_info()
        assert info.name == "AGUIAdapter"
        assert info.framework == "agui"
        assert AdapterCapability.TRACE_PROTOCOL_EVENTS in info.capabilities
        assert AdapterCapability.STREAMING in info.capabilities

    def test_serialize_for_replay(self):
        adapter = AGUIAdapter()
        replay = adapter.serialize_for_replay()
        assert replay.framework == "agui"
