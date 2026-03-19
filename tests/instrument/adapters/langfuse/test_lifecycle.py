"""Tests for Langfuse adapter lifecycle."""

import pytest
from unittest.mock import patch, MagicMock

from layerlens.instrument.adapters._base import (
    AdapterCapability,
    AdapterStatus,
    ReplayableTrace,
)
from layerlens.instrument.adapters._capture import CaptureConfig
from layerlens.instrument.adapters.langfuse.lifecycle import LangfuseAdapter
from layerlens.instrument.adapters.langfuse.config import (
    LangfuseConfig,
    SyncDirection,
    SyncState,
)


class MockStratix:
    """Mock STRATIX instance for testing."""

    def __init__(self):
        self.events = []

    def __bool__(self):
        return True

    def emit(self, event_type: str, payload: dict):
        self.events.append({"type": event_type, "payload": payload})

    def get_events(self, event_type: str = None):
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


class TestLangfuseAdapter:
    """Tests for LangfuseAdapter lifecycle."""

    def test_adapter_initialization(self):
        adapter = LangfuseAdapter()
        assert adapter.FRAMEWORK == "langfuse"
        assert adapter.VERSION == "0.1.0"
        assert not adapter.is_connected

    def test_adapter_with_stratix(self):
        stratix = MockStratix()
        adapter = LangfuseAdapter(stratix=stratix)
        assert adapter.has_stratix

    def test_adapter_without_stratix(self):
        adapter = LangfuseAdapter()
        assert not adapter.has_stratix

    def test_connect_without_config(self):
        """Connect without config succeeds — adapter is usable standalone."""
        adapter = LangfuseAdapter()
        adapter.connect()
        assert adapter.is_connected
        assert adapter.status == AdapterStatus.HEALTHY

    @patch("layerlens.instrument.adapters.langfuse.lifecycle.LangfuseAPIClient")
    def test_connect_with_config(self, MockClient):
        """Connect with config creates API client and checks health."""
        mock_client = MagicMock()
        mock_client.health_check.return_value = {"status": "OK"}
        MockClient.return_value = mock_client

        config = LangfuseConfig(public_key="pk", secret_key="sk")
        adapter = LangfuseAdapter(config=config)
        adapter.connect()

        assert adapter.is_connected
        assert adapter.status == AdapterStatus.HEALTHY
        mock_client.health_check.assert_called_once()

    @patch("layerlens.instrument.adapters.langfuse.lifecycle.LangfuseAPIClient")
    def test_connect_health_check_fails(self, MockClient):
        """Connect with failed health check sets DEGRADED status."""
        from layerlens.instrument.adapters.langfuse.client import LangfuseAPIError

        mock_client = MagicMock()
        mock_client.health_check.side_effect = LangfuseAPIError("Connection refused")
        MockClient.return_value = mock_client

        config = LangfuseConfig(public_key="pk", secret_key="sk")
        adapter = LangfuseAdapter(config=config)
        adapter.connect()

        assert adapter.is_connected
        assert adapter.status == AdapterStatus.DEGRADED

    @patch("layerlens.instrument.adapters.langfuse.lifecycle.LangfuseAPIClient")
    def test_connect_with_config_arg(self, MockClient):
        """Config can be passed to connect() directly."""
        mock_client = MagicMock()
        mock_client.health_check.return_value = {"status": "OK"}
        MockClient.return_value = mock_client

        adapter = LangfuseAdapter()
        config = LangfuseConfig(public_key="pk", secret_key="sk")
        adapter.connect(config=config)

        assert adapter.is_connected
        assert adapter.config is config

    def test_disconnect(self):
        adapter = LangfuseAdapter()
        adapter.connect()
        adapter.disconnect()

        assert not adapter.is_connected
        assert adapter.status == AdapterStatus.DISCONNECTED

    def test_disconnect_clears_client(self):
        adapter = LangfuseAdapter()
        adapter.connect()
        adapter.disconnect()

        assert adapter._client is None
        assert adapter._importer is None
        assert adapter._exporter is None
        assert adapter._sync is None

    def test_health_check_no_config(self):
        adapter = LangfuseAdapter()
        adapter.connect()
        health = adapter.health_check()

        assert health.status == AdapterStatus.HEALTHY
        assert health.framework_name == "langfuse"
        assert "No Langfuse config" in health.message

    @patch("layerlens.instrument.adapters.langfuse.lifecycle.LangfuseAPIClient")
    def test_health_check_connected(self, MockClient):
        mock_client = MagicMock()
        mock_client.health_check.return_value = {"status": "OK"}
        MockClient.return_value = mock_client

        config = LangfuseConfig(public_key="pk", secret_key="sk")
        adapter = LangfuseAdapter(config=config)
        adapter.connect()

        health = adapter.health_check()
        assert health.status == AdapterStatus.HEALTHY
        assert "reachable" in health.message

    def test_get_adapter_info(self):
        adapter = LangfuseAdapter()
        info = adapter.get_adapter_info()

        assert info.name == "LangfuseAdapter"
        assert info.framework == "langfuse"
        assert AdapterCapability.TRACE_TOOLS in info.capabilities
        assert AdapterCapability.TRACE_MODELS in info.capabilities
        assert AdapterCapability.REPLAY in info.capabilities
        assert "Bidirectional" in info.description

    def test_serialize_for_replay(self):
        adapter = LangfuseAdapter()
        trace = adapter.serialize_for_replay()

        assert isinstance(trace, ReplayableTrace)
        assert trace.adapter_name == "LangfuseAdapter"
        assert trace.framework == "langfuse"
        assert trace.trace_id

    def test_serialize_includes_sync_state(self):
        adapter = LangfuseAdapter()
        adapter._sync_state.imported_trace_ids.add("t1")
        adapter._sync_state.exported_trace_ids.add("t2")

        trace = adapter.serialize_for_replay()
        meta = trace.metadata
        assert meta["sync_state"]["imported"] == 1
        assert meta["sync_state"]["exported"] == 1

    def test_get_status(self):
        adapter = LangfuseAdapter()
        adapter.connect()
        status = adapter.get_status()

        assert status["connected"] is True
        assert status["imported_traces"] == 0
        assert status["exported_traces"] == 0

    @patch("layerlens.instrument.adapters.langfuse.lifecycle.LangfuseAPIClient")
    def test_get_status_with_config(self, MockClient):
        mock_client = MagicMock()
        mock_client.health_check.return_value = {"status": "OK"}
        MockClient.return_value = mock_client

        config = LangfuseConfig(public_key="pk", secret_key="sk", mode="bidirectional")
        adapter = LangfuseAdapter(config=config)
        adapter.connect()

        status = adapter.get_status()
        assert status["host"] == "https://cloud.langfuse.com"
        assert status["mode"] == "bidirectional"
        assert status["langfuse_healthy"] is True

    def test_import_without_connection(self):
        adapter = LangfuseAdapter()
        adapter.connect()  # No config, so no importer
        result = adapter.import_traces()
        assert result.direction == SyncDirection.IMPORT
        assert len(result.errors) > 0

    def test_export_without_connection(self):
        adapter = LangfuseAdapter()
        adapter.connect()
        result = adapter.export_traces()
        assert result.direction == SyncDirection.EXPORT
        assert len(result.errors) > 0

    def test_sync_without_connection(self):
        adapter = LangfuseAdapter()
        adapter.connect()
        result = adapter.sync()
        assert len(result.errors) > 0

    def test_sync_state_property(self):
        adapter = LangfuseAdapter()
        assert isinstance(adapter.sync_state, SyncState)

    def test_config_property(self):
        config = LangfuseConfig(public_key="pk", secret_key="sk")
        adapter = LangfuseAdapter(config=config)
        assert adapter.config is config

    def test_config_property_none(self):
        adapter = LangfuseAdapter()
        assert adapter.config is None
