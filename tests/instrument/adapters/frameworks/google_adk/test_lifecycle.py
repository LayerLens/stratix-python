"""Test Google ADK adapter lifecycle methods.

Ported as-is from ``ateam/tests/adapters/google_adk/test_lifecycle.py``.

Translation rules applied:
* ``stratix.sdk.python.adapters.base`` →
  ``layerlens.instrument.adapters._base.adapter``
* ``stratix.sdk.python.adapters.google_adk.lifecycle`` →
  ``layerlens.instrument.adapters.frameworks.google_adk.lifecycle``
* ``stratix.sdk.python.adapters.replay_models`` →
  ``layerlens.instrument.adapters._base.adapter`` (``ReplayableTrace`` re-export)
"""

from __future__ import annotations

from typing import Any

from layerlens.instrument.adapters._base.adapter import (
    AdapterStatus,
    ReplayableTrace,
)
from layerlens.instrument.adapters.frameworks.google_adk.lifecycle import (
    GoogleADKAdapter,
)

from .conftest import MockStratix


class TestGoogleADKAdapterLifecycle:
    def test_adapter_initialization(self) -> None:
        adapter: GoogleADKAdapter = GoogleADKAdapter(org_id="test-org")
        assert adapter.FRAMEWORK == "google_adk"
        assert adapter.VERSION == "0.1.0"

    def test_adapter_initialization_with_stratix(self, mock_stratix: MockStratix) -> None:
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix=mock_stratix)
        assert adapter.has_stratix

    def test_adapter_initialization_legacy_param(self, mock_stratix: MockStratix) -> None:
        adapter: GoogleADKAdapter = GoogleADKAdapter(stratix_instance=mock_stratix)
        assert adapter.has_stratix

    def test_connect_sets_healthy(self) -> None:
        adapter: GoogleADKAdapter = GoogleADKAdapter(org_id="test-org")
        adapter.connect()
        assert adapter.is_connected
        assert adapter.status == AdapterStatus.HEALTHY

    def test_disconnect_sets_disconnected(self) -> None:
        adapter: GoogleADKAdapter = GoogleADKAdapter(org_id="test-org")
        adapter.connect()
        adapter.disconnect()
        assert not adapter.is_connected
        assert adapter.status == AdapterStatus.DISCONNECTED

    def test_health_check(self, adapter: GoogleADKAdapter) -> None:
        health = adapter.health_check()
        assert health.status == AdapterStatus.HEALTHY
        assert health.framework_name == "google_adk"
        assert health.adapter_version == "0.1.0"
        assert health.error_count == 0
        assert not health.circuit_open

    def test_get_adapter_info(self, adapter: GoogleADKAdapter) -> None:
        info = adapter.get_adapter_info()
        assert info.name == "GoogleADKAdapter"
        assert info.framework == "google_adk"
        assert info.version == "0.1.0"

    def test_serialize_for_replay(self, adapter: GoogleADKAdapter) -> None:
        trace: ReplayableTrace = adapter.serialize_for_replay()
        assert isinstance(trace, ReplayableTrace)
        assert trace.adapter_name == "GoogleADKAdapter"
        assert trace.framework == "google_adk"
        assert trace.trace_id is not None

    def test_null_stratix_pattern(self) -> None:
        adapter: GoogleADKAdapter = GoogleADKAdapter(org_id="test-org")
        adapter.connect()
        # Should not raise even without STRATIX
        payload: dict[str, Any] = {"framework": "google_adk"}
        adapter.emit_dict_event("agent.input", payload)
