"""Test LlamaIndex adapter lifecycle methods.

Ported from ``ateam/tests/adapters/llama_index/test_lifecycle.py``.

Translation rules applied:
* ``stratix.sdk.python.adapters.base`` ->
  ``layerlens.instrument.adapters._base``
* ``stratix.sdk.python.adapters.llama_index.lifecycle`` ->
  ``layerlens.instrument.adapters.frameworks.llama_index.lifecycle``
* ``stratix.sdk.python.adapters.replay_models.ReplayableTrace`` ->
  ``layerlens.instrument.adapters._base.ReplayableTrace`` (consolidated
  in the ``_base`` package in stratix-python).
"""

from __future__ import annotations

from layerlens.instrument.adapters._base import (
    AdapterStatus,
    ReplayableTrace,
)
from layerlens.instrument.adapters.frameworks.llama_index.lifecycle import (
    LlamaIndexAdapter,
)

from .conftest import MockStratix


class TestLlamaIndexAdapterLifecycle:
    def test_adapter_initialization(self) -> None:
        adapter = LlamaIndexAdapter()
        assert adapter.FRAMEWORK == "llama_index"
        assert adapter.VERSION == "0.1.0"

    def test_adapter_initialization_with_stratix(self, mock_stratix: MockStratix) -> None:
        adapter = LlamaIndexAdapter(stratix=mock_stratix)
        assert adapter.has_stratix

    def test_adapter_initialization_legacy_param(self, mock_stratix: MockStratix) -> None:
        adapter = LlamaIndexAdapter(stratix_instance=mock_stratix)
        assert adapter.has_stratix

    def test_connect_sets_healthy(self) -> None:
        adapter = LlamaIndexAdapter()
        adapter.connect()
        assert adapter.is_connected
        assert adapter.status == AdapterStatus.HEALTHY

    def test_disconnect_sets_disconnected(self) -> None:
        adapter = LlamaIndexAdapter()
        adapter.connect()
        adapter.disconnect()
        assert not adapter.is_connected
        assert adapter.status == AdapterStatus.DISCONNECTED

    def test_health_check(self, adapter: LlamaIndexAdapter) -> None:
        health = adapter.health_check()
        assert health.status == AdapterStatus.HEALTHY
        assert health.framework_name == "llama_index"
        assert health.adapter_version == "0.1.0"
        assert health.error_count == 0
        assert not health.circuit_open

    def test_get_adapter_info(self, adapter: LlamaIndexAdapter) -> None:
        info = adapter.get_adapter_info()
        assert info.name == "LlamaIndexAdapter"
        assert info.framework == "llama_index"
        assert info.version == "0.1.0"

    def test_serialize_for_replay(self, adapter: LlamaIndexAdapter) -> None:
        trace = adapter.serialize_for_replay()
        assert isinstance(trace, ReplayableTrace)
        assert trace.adapter_name == "LlamaIndexAdapter"
        assert trace.framework == "llama_index"
        assert trace.trace_id is not None

    def test_null_stratix_pattern(self) -> None:
        adapter = LlamaIndexAdapter()
        adapter.connect()
        # Should not raise even without STRATIX
        adapter.emit_dict_event("agent.input", {"framework": "llama_index"})
