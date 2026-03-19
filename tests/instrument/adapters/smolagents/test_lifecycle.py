"""Test SmolAgents adapter lifecycle methods."""

import pytest
from layerlens.instrument.adapters._base import AdapterStatus
from layerlens.instrument.adapters.smolagents.lifecycle import SmolAgentsAdapter
from layerlens.instrument.adapters._replay_models import ReplayableTrace


class TestSmolAgentsAdapterLifecycle:
    def test_adapter_initialization(self):
        adapter = SmolAgentsAdapter()
        assert adapter.FRAMEWORK == "smolagents"
        assert adapter.VERSION == "0.1.0"

    def test_adapter_initialization_with_stratix(self, mock_stratix):
        adapter = SmolAgentsAdapter(stratix=mock_stratix)
        assert adapter.has_stratix

    def test_adapter_initialization_legacy_param(self, mock_stratix):
        adapter = SmolAgentsAdapter(stratix_instance=mock_stratix)
        assert adapter.has_stratix

    def test_connect_sets_healthy(self):
        adapter = SmolAgentsAdapter()
        adapter.connect()
        assert adapter.is_connected
        assert adapter.status == AdapterStatus.HEALTHY

    def test_disconnect_sets_disconnected(self):
        adapter = SmolAgentsAdapter()
        adapter.connect()
        adapter.disconnect()
        assert not adapter.is_connected
        assert adapter.status == AdapterStatus.DISCONNECTED

    def test_health_check(self, adapter):
        health = adapter.health_check()
        assert health.status == AdapterStatus.HEALTHY
        assert health.framework_name == "smolagents"
        assert health.adapter_version == "0.1.0"
        assert health.error_count == 0
        assert not health.circuit_open

    def test_get_adapter_info(self, adapter):
        info = adapter.get_adapter_info()
        assert info.name == "SmolAgentsAdapter"
        assert info.framework == "smolagents"
        assert info.version == "0.1.0"

    def test_serialize_for_replay(self, adapter):
        trace = adapter.serialize_for_replay()
        assert isinstance(trace, ReplayableTrace)
        assert trace.adapter_name == "SmolAgentsAdapter"
        assert trace.framework == "smolagents"
        assert trace.trace_id is not None

    def test_null_stratix_pattern(self):
        adapter = SmolAgentsAdapter()
        adapter.connect()
        # Should not raise even without STRATIX
        adapter.emit_dict_event("agent.input", {"framework": "smolagents"})
