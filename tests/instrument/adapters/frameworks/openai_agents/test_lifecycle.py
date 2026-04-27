"""Test OpenAI Agents adapter lifecycle methods.

Ported from ``ateam/tests/adapters/openai_agents/test_lifecycle.py``.

Renames:
- ``stratix.sdk.python.adapters.base.AdapterStatus`` ->
  ``layerlens.instrument.adapters._base.AdapterStatus``
- ``stratix.sdk.python.adapters.openai_agents.lifecycle.OpenAIAgentsAdapter``
  -> ``layerlens.instrument.adapters.frameworks.openai_agents.lifecycle.OpenAIAgentsAdapter``
- ``stratix.sdk.python.adapters.replay_models.ReplayableTrace`` ->
  ``layerlens.instrument.adapters._base.ReplayableTrace`` (the replay
  trace dataclass moved into the consolidated ``_base`` package).
"""

from __future__ import annotations

from layerlens.instrument.adapters._base import AdapterStatus, ReplayableTrace
from layerlens.instrument.adapters.frameworks.openai_agents.lifecycle import (
    OpenAIAgentsAdapter,
)

from .conftest import TEST_ORG_ID, MockStratix


class TestOpenAIAgentsAdapterLifecycle:
    def test_adapter_initialization(self) -> None:
        adapter = OpenAIAgentsAdapter(org_id=TEST_ORG_ID)
        assert adapter.FRAMEWORK == "openai_agents"
        assert adapter.VERSION == "0.1.0"

    def test_adapter_initialization_with_stratix(self, mock_stratix: MockStratix) -> None:
        adapter = OpenAIAgentsAdapter(stratix=mock_stratix)
        assert adapter.has_stratix

    def test_adapter_initialization_legacy_param(self, mock_stratix: MockStratix) -> None:
        adapter = OpenAIAgentsAdapter(stratix_instance=mock_stratix)
        assert adapter.has_stratix

    def test_connect_sets_healthy(self) -> None:
        adapter = OpenAIAgentsAdapter(org_id=TEST_ORG_ID)
        adapter.connect()
        assert adapter.is_connected
        assert adapter.status == AdapterStatus.HEALTHY

    def test_disconnect_sets_disconnected(self) -> None:
        adapter = OpenAIAgentsAdapter(org_id=TEST_ORG_ID)
        adapter.connect()
        adapter.disconnect()
        assert not adapter.is_connected
        assert adapter.status == AdapterStatus.DISCONNECTED

    def test_health_check(self, adapter: OpenAIAgentsAdapter) -> None:
        health = adapter.health_check()
        assert health.status == AdapterStatus.HEALTHY
        assert health.framework_name == "openai_agents"
        assert health.adapter_version == "0.1.0"
        assert health.error_count == 0
        assert not health.circuit_open

    def test_get_adapter_info(self, adapter: OpenAIAgentsAdapter) -> None:
        info = adapter.get_adapter_info()
        assert info.name == "OpenAIAgentsAdapter"
        assert info.framework == "openai_agents"
        assert info.version == "0.1.0"

    def test_serialize_for_replay(self, adapter: OpenAIAgentsAdapter) -> None:
        trace = adapter.serialize_for_replay()
        assert isinstance(trace, ReplayableTrace)
        assert trace.adapter_name == "OpenAIAgentsAdapter"
        assert trace.framework == "openai_agents"
        assert trace.trace_id is not None

    def test_null_stratix_pattern(self) -> None:
        adapter = OpenAIAgentsAdapter(org_id=TEST_ORG_ID)
        adapter.connect()
        # Should not raise even without STRATIX
        adapter.emit_dict_event("agent.input", {"framework": "openai_agents"})
