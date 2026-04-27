"""Test Bedrock Agents adapter lifecycle methods.

Ported from ``ateam/tests/adapters/bedrock_agents/test_lifecycle.py``.
"""

from __future__ import annotations

from typing import Any

from layerlens.instrument.adapters._base.adapter import (
    AdapterStatus,
    ReplayableTrace,
)
from layerlens.instrument.adapters.frameworks.bedrock_agents.lifecycle import (
    BedrockAgentsAdapter,
)


class TestBedrockAgentsAdapterLifecycle:
    def test_adapter_initialization(self) -> None:
        adapter = BedrockAgentsAdapter()
        assert adapter.FRAMEWORK == "bedrock_agents"
        assert adapter.VERSION == "0.1.0"

    def test_adapter_initialization_with_stratix(self, mock_stratix: Any) -> None:
        adapter = BedrockAgentsAdapter(stratix=mock_stratix)
        assert adapter.has_stratix

    def test_adapter_initialization_legacy_param(self, mock_stratix: Any) -> None:
        adapter = BedrockAgentsAdapter(stratix_instance=mock_stratix)
        assert adapter.has_stratix

    def test_connect_sets_healthy(self) -> None:
        adapter = BedrockAgentsAdapter()
        adapter.connect()
        assert adapter.is_connected
        assert adapter.status == AdapterStatus.HEALTHY

    def test_disconnect_sets_disconnected(self) -> None:
        adapter = BedrockAgentsAdapter()
        adapter.connect()
        adapter.disconnect()
        assert not adapter.is_connected
        assert adapter.status == AdapterStatus.DISCONNECTED

    def test_health_check(self, adapter: BedrockAgentsAdapter) -> None:
        health = adapter.health_check()
        assert health.status == AdapterStatus.HEALTHY
        assert health.framework_name == "bedrock_agents"
        assert health.adapter_version == "0.1.0"
        assert health.error_count == 0
        assert not health.circuit_open

    def test_get_adapter_info(self, adapter: BedrockAgentsAdapter) -> None:
        info = adapter.get_adapter_info()
        assert info.name == "BedrockAgentsAdapter"
        assert info.framework == "bedrock_agents"
        assert info.version == "0.1.0"

    def test_serialize_for_replay(self, adapter: BedrockAgentsAdapter) -> None:
        trace = adapter.serialize_for_replay()
        assert isinstance(trace, ReplayableTrace)
        assert trace.adapter_name == "BedrockAgentsAdapter"
        assert trace.framework == "bedrock_agents"
        assert trace.trace_id is not None

    def test_null_stratix_pattern(self) -> None:
        adapter = BedrockAgentsAdapter()
        adapter.connect()
        # Should not raise even without STRATIX
        adapter.emit_dict_event("agent.input", {"framework": "bedrock_agents"})
