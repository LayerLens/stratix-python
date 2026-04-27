"""Unit tests for the Langfuse framework adapter.

Mocked at the SDK shape level — no real Langfuse API calls.

Unlike runtime-wrapping adapters, the Langfuse adapter is a data
import/export pipeline. Tests focus on:

  * lifecycle (with and without config)
  * connect-without-config edge case (adapter still healthy, no client)
  * import/export return SyncResult with appropriate error message when
    no client is configured
  * SyncState tracking semantics
  * health_check / get_status structural correctness
  * serialize_for_replay returns proper ReplayableTrace
"""

from __future__ import annotations

from typing import Any, Dict, List

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.langfuse import (
    ADAPTER_CLASS,
    LangfuseAdapter,
)
from layerlens.instrument.adapters.frameworks.langfuse.config import (
    SyncDirection,
    LangfuseConfig,
)


class _RecordingStratix:
    # Multi-tenant test stand-in: every recording client carries an
    # org_id so adapters constructed with this stratix pass the
    # BaseAdapter fail-fast check. Tests asserting cross-tenant
    # isolation override this default.
    org_id: str = "test-org"

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is LangfuseAdapter


def test_lifecycle_no_config() -> None:
    """Adapter is usable without a Langfuse config — connects HEALTHY but no client."""
    a = LangfuseAdapter(org_id="test-org")
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    assert a.is_connected is True
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_adapter_info_and_health() -> None:
    a = LangfuseAdapter(org_id="test-org")
    a.connect()
    info = a.get_adapter_info()
    assert info.framework == "langfuse"
    assert info.name == "LangfuseAdapter"
    health = a.health_check()
    assert health.framework_name == "langfuse"
    assert "No Langfuse config" in (health.message or "")


def test_import_returns_error_result_when_not_connected() -> None:
    """Without a Langfuse client, import_traces returns an errored SyncResult."""
    a = LangfuseAdapter(org_id="test-org")
    a.connect()
    result = a.import_traces()
    assert result.direction == SyncDirection.IMPORT
    assert result.errors
    assert "not connected" in result.errors[0].lower()


def test_export_returns_error_result_when_not_connected() -> None:
    """Without a Langfuse client, export_traces returns an errored SyncResult."""
    a = LangfuseAdapter(org_id="test-org")
    a.connect()
    result = a.export_traces(events_by_trace={"trace-1": []})
    assert result.direction == SyncDirection.EXPORT
    assert result.errors
    assert "not connected" in result.errors[0].lower()


def test_sync_returns_error_result_when_not_connected() -> None:
    """Without a Langfuse client, sync() returns an errored SyncResult."""
    a = LangfuseAdapter(org_id="test-org")
    a.connect()
    result = a.sync()
    assert result.errors
    assert "not connected" in result.errors[0].lower()


def test_sync_state_tracking() -> None:
    """SyncState records imports/exports and updates cursors."""
    a = LangfuseAdapter(org_id="test-org")
    a.connect()
    state = a.sync_state

    from datetime import datetime, timezone

    UTC = timezone.utc  # Python 3.11+ has datetime.UTC; alias for 3.9/3.10 compat.

    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    t1 = datetime(2024, 1, 2, tzinfo=UTC)
    state.record_import("trace-1", t0)
    state.record_import("trace-2", t1)
    assert "trace-1" in state.imported_trace_ids
    assert "trace-2" in state.imported_trace_ids
    assert state.last_import_cursor == t1


def test_get_status_structure() -> None:
    """get_status returns a complete status dict."""
    a = LangfuseAdapter(org_id="test-org")
    a.connect()
    status = a.get_status()
    assert "connected" in status
    assert "langfuse_healthy" in status
    assert "host" in status
    assert "imported_traces" in status
    assert "exported_traces" in status
    assert "quarantined_traces" in status


def test_config_property_default_none() -> None:
    """When no config provided, config property is None."""
    a = LangfuseAdapter(org_id="test-org")
    a.connect()
    assert a.config is None


def test_config_property_returns_provided_config() -> None:
    """When config provided to constructor, it is exposed via the config property.

    This avoids ``connect(config=cfg)`` which would attempt an HTTP health
    check against the (fake) host. The constructor path stores the config
    without networking; ``connect()`` then runs without the health check
    when ``self._config`` was set on a prior call we skip here.
    """
    cfg = LangfuseConfig(public_key="pk-test", secret_key="sk-test", host="https://api/")
    a = LangfuseAdapter(config=cfg, org_id="test-org")
    assert a.config is cfg
    # Trailing slash is stripped by the validator.
    assert a.config.host == "https://api"


def test_capture_config_passes_through() -> None:
    """The standard capture_config kwarg is accepted and stored."""
    cfg = CaptureConfig.full()
    a = LangfuseAdapter(capture_config=cfg, org_id="test-org")
    assert a.capture_config is cfg


def test_serialize_for_replay() -> None:
    a = LangfuseAdapter(stratix=_RecordingStratix(), capture_config=CaptureConfig.full())
    a.connect()
    rt = a.serialize_for_replay()
    assert rt.framework == "langfuse"
    assert rt.adapter_name == "LangfuseAdapter"
    assert "sync_state" in rt.metadata
