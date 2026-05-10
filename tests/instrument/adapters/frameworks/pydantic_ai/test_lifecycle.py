"""Test Pydantic AI adapter lifecycle methods.

Ported as-is from ``ateam/tests/adapters/pydantic_ai/test_lifecycle.py``.

Translation rules applied:
* ``stratix.sdk.python.adapters.pydantic_ai.lifecycle`` →
  ``layerlens.instrument.adapters.frameworks.pydantic_ai.lifecycle``
* ``stratix.sdk.python.adapters.base`` →
  ``layerlens.instrument.adapters._base``
* ``stratix.sdk.python.adapters.capture.CaptureConfig`` →
  ``layerlens.instrument.adapters._base.CaptureConfig``
* ``stratix.sdk.python.adapters.replay_models.ReplayableTrace`` →
  ``layerlens.instrument.adapters._base.ReplayableTrace``
* ``stratix.sdk.python.adapters.registry._ADAPTER_MODULES`` →
  ``layerlens.instrument.adapters._base.registry._ADAPTER_MODULES``
* The wrapper marker attribute renamed by the source from
  ``_stratix_original`` to ``_layerlens_original``.

Multi-tenancy: per the transitional "stratix attribute" pattern (see
migration doc §2.3 step 2 — keystone PR #118 still DRAFT), the
``MockStratix`` / ``EventCollector`` test stub gets an ``org_id``
attribute. The post-merge sweep PR will rebase to canonical kwarg once
#118 lands.
"""

from layerlens.instrument.adapters._base import AdapterStatus
from layerlens.instrument.adapters.frameworks.pydantic_ai.lifecycle import PydanticAIAdapter
from layerlens.instrument.adapters._base import ReplayableTrace


class TestPydanticAIAdapterLifecycle:
    def test_adapter_initialization(self):
        adapter = PydanticAIAdapter()
        assert adapter.FRAMEWORK == "pydantic_ai"
        assert adapter.VERSION == "0.1.0"

    def test_adapter_initialization_with_stratix(self, mock_stratix):
        adapter = PydanticAIAdapter(stratix=mock_stratix)
        assert adapter.has_stratix

    def test_adapter_initialization_legacy_param(self, mock_stratix):
        adapter = PydanticAIAdapter(stratix_instance=mock_stratix)
        assert adapter.has_stratix

    def test_connect_sets_healthy(self):
        adapter = PydanticAIAdapter()
        adapter.connect()
        assert adapter.is_connected
        assert adapter.status == AdapterStatus.HEALTHY

    def test_disconnect_sets_disconnected(self):
        adapter = PydanticAIAdapter()
        adapter.connect()
        adapter.disconnect()
        assert not adapter.is_connected
        assert adapter.status == AdapterStatus.DISCONNECTED

    def test_health_check(self, adapter):
        health = adapter.health_check()
        assert health.status == AdapterStatus.HEALTHY
        assert health.framework_name == "pydantic_ai"
        assert health.adapter_version == "0.1.0"
        assert health.error_count == 0
        assert not health.circuit_open

    def test_get_adapter_info(self, adapter):
        info = adapter.get_adapter_info()
        assert info.name == "PydanticAIAdapter"
        assert info.framework == "pydantic_ai"
        assert info.version == "0.1.0"

    def test_serialize_for_replay(self, adapter):
        trace = adapter.serialize_for_replay()
        assert isinstance(trace, ReplayableTrace)
        assert trace.adapter_name == "PydanticAIAdapter"
        assert trace.framework == "pydantic_ai"
        assert trace.trace_id is not None

    def test_null_stratix_pattern(self):
        adapter = PydanticAIAdapter()
        adapter.connect()
        # Should not raise even without STRATIX
        adapter.emit_dict_event("agent.input", {"framework": "pydantic_ai"})
