"""
Tests for A2A adapter lifecycle: connect, disconnect, health_check, get_adapter_info.
"""

import pytest

from layerlens.instrument.adapters._base import AdapterCapability, AdapterStatus
from layerlens.instrument.adapters.protocols.a2a.adapter import A2AAdapter
from layerlens.instrument.adapters._capture import CaptureConfig


class TestA2AAdapterLifecycle:
    def test_connect(self):
        adapter = A2AAdapter()
        adapter.connect()
        assert adapter.is_connected
        assert adapter.status == AdapterStatus.HEALTHY

    def test_disconnect(self):
        adapter = A2AAdapter()
        adapter.connect()
        adapter.disconnect()
        assert not adapter.is_connected
        assert adapter.status == AdapterStatus.DISCONNECTED

    def test_get_adapter_info(self):
        adapter = A2AAdapter()
        info = adapter.get_adapter_info()
        assert info.name == "A2AAdapter"
        assert info.framework == "a2a"
        assert AdapterCapability.TRACE_PROTOCOL_EVENTS in info.capabilities
        assert AdapterCapability.TRACE_HANDOFFS in info.capabilities

    def test_health_check(self):
        adapter = A2AAdapter()
        adapter.connect()
        health = adapter.health_check()
        assert health.status == AdapterStatus.HEALTHY
        assert health.framework_name == "a2a"

    def test_serialize_for_replay(self):
        adapter = A2AAdapter()
        replay = adapter.serialize_for_replay()
        assert replay.adapter_name == "A2AAdapter"
        assert replay.framework == "a2a"

    def test_probe_health_connected(self):
        adapter = A2AAdapter()
        adapter.connect()
        result = adapter.probe_health()
        assert result["reachable"] is True
