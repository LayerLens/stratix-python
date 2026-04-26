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
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is LangfuseAdapter


def test_lifecycle_no_config() -> None:
    """Adapter is usable without a Langfuse config — connects HEALTHY but no client."""
    a = LangfuseAdapter()
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    assert a.is_connected is True
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_adapter_info_and_health() -> None:
    a = LangfuseAdapter()
    a.connect()
    info = a.get_adapter_info()
    assert info.framework == "langfuse"
    assert info.name == "LangfuseAdapter"
    health = a.health_check()
    assert health.framework_name == "langfuse"
    assert "No Langfuse config" in (health.message or "")


def test_import_returns_error_result_when_not_connected() -> None:
    """Without a Langfuse client, import_traces returns an errored SyncResult."""
    a = LangfuseAdapter()
    a.connect()
    result = a.import_traces()
    assert result.direction == SyncDirection.IMPORT
    assert result.errors
    assert "not connected" in result.errors[0].lower()


def test_export_returns_error_result_when_not_connected() -> None:
    """Without a Langfuse client, export_traces returns an errored SyncResult."""
    a = LangfuseAdapter()
    a.connect()
    result = a.export_traces(events_by_trace={"trace-1": []})
    assert result.direction == SyncDirection.EXPORT
    assert result.errors
    assert "not connected" in result.errors[0].lower()


def test_sync_returns_error_result_when_not_connected() -> None:
    """Without a Langfuse client, sync() returns an errored SyncResult."""
    a = LangfuseAdapter()
    a.connect()
    result = a.sync()
    assert result.errors
    assert "not connected" in result.errors[0].lower()


def test_sync_state_tracking() -> None:
    """SyncState records imports/exports and updates cursors."""
    a = LangfuseAdapter()
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
    a = LangfuseAdapter()
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
    a = LangfuseAdapter()
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
    a = LangfuseAdapter(config=cfg)
    assert a.config is cfg
    # Trailing slash is stripped by the validator.
    assert a.config.host == "https://api"


def test_capture_config_passes_through() -> None:
    """The standard capture_config kwarg is accepted and stored."""
    cfg = CaptureConfig.full()
    a = LangfuseAdapter(capture_config=cfg)
    assert a.capture_config is cfg


def test_serialize_for_replay() -> None:
    a = LangfuseAdapter(stratix=_RecordingStratix(), capture_config=CaptureConfig.full())
    a.connect()
    rt = a.serialize_for_replay()
    assert rt.framework == "langfuse"
    assert rt.adapter_name == "LangfuseAdapter"
    assert "sync_state" in rt.metadata


# ---------------------------------------------------------------------------
# Importer-hardening (PR ``feat/instrument-importer-hardening``):
# verify the refactored importer/exporter use the shared base helpers.
# ---------------------------------------------------------------------------


class TestImporterHardening:
    """Verify refactored TraceImporter/TraceExporter use shared helpers.

    These tests intentionally mock at the client level — no real HTTP —
    so the unit boundary is the importer/exporter logic rather than the
    Langfuse API itself.
    """

    def _make_state(self) -> Any:
        from layerlens.instrument.adapters.frameworks.langfuse.config import SyncState

        return SyncState()

    def test_importer_skips_invalid_trace_ids(self) -> None:
        """Trace summaries with non-UUID ids are skipped, not fetched."""
        from layerlens.instrument.adapters.frameworks.langfuse.importer import TraceImporter

        class FakeClient:
            def __init__(self) -> None:
                self.fetched: List[str] = []

            def get_all_traces(self, **_kw: Any) -> List[Dict[str, Any]]:
                # Two malformed ids and one valid UUID.
                return [
                    {"id": "not-a-uuid"},
                    {"id": ""},
                    {"id": "550e8400-e29b-41d4-a716-446655440000"},
                ]

            def get_trace(self, trace_id: str) -> Dict[str, Any]:
                self.fetched.append(trace_id)
                return {"id": trace_id, "timestamp": "2026-04-25T00:00:00Z"}

        client = FakeClient()
        state = self._make_state()
        importer = TraceImporter(client, state)  # type: ignore[arg-type]
        result = importer.import_traces(stratix=None)
        # Only the valid UUID was fetched (and skipped because no events
        # came out of the empty trace).
        assert client.fetched == ["550e8400-e29b-41d4-a716-446655440000"]
        # Two skips for invalid ids, plus one for the empty-events trace.
        assert result.skipped_count >= 2
        # Two errors recorded (one per invalid id).
        invalid_errors = [e for e in result.errors if "Invalid trace id" in e]
        assert len(invalid_errors) == 2

    def test_importer_retries_transient_fetch_failures(self) -> None:
        """A transient 503 on get_trace is retried, not quarantined."""
        from layerlens.instrument.adapters.frameworks.langfuse.client import LangfuseAPIError
        from layerlens.instrument.adapters.frameworks.langfuse.importer import TraceImporter

        class FakeClient:
            def __init__(self) -> None:
                self.attempts = 0

            def get_all_traces(self, **_kw: Any) -> List[Dict[str, Any]]:
                return [{"id": "550e8400-e29b-41d4-a716-446655440000"}]

            def get_trace(self, trace_id: str) -> Dict[str, Any]:
                self.attempts += 1
                if self.attempts < 2:
                    raise LangfuseAPIError("transient", status_code=503)
                return {"id": trace_id, "timestamp": "2026-04-25T00:00:00Z"}

        client = FakeClient()
        state = self._make_state()
        importer = TraceImporter(
            client,  # type: ignore[arg-type]
            state,
            max_retries=3,
            base_delay=0.001,
            max_delay=0.001,
        )
        result = importer.import_traces(stratix=None)
        # Even though first attempt failed, retry recovered → no failure.
        assert result.failed_count == 0
        assert client.attempts == 2

    def test_importer_does_not_retry_terminal_4xx(self) -> None:
        """A 404 from get_trace is terminal — no retry, marked as failed."""
        from layerlens.instrument.adapters.frameworks.langfuse.client import LangfuseAPIError
        from layerlens.instrument.adapters.frameworks.langfuse.importer import TraceImporter

        class FakeClient:
            def __init__(self) -> None:
                self.attempts = 0

            def get_all_traces(self, **_kw: Any) -> List[Dict[str, Any]]:
                return [{"id": "550e8400-e29b-41d4-a716-446655440000"}]

            def get_trace(self, trace_id: str) -> Dict[str, Any]:
                self.attempts += 1
                raise LangfuseAPIError("not found", status_code=404)

        client = FakeClient()
        state = self._make_state()
        importer = TraceImporter(
            client,  # type: ignore[arg-type]
            state,
            max_retries=5,
            base_delay=0.001,
            max_delay=0.001,
        )
        result = importer.import_traces(stratix=None)
        # 404 is terminal — exactly one attempt.
        assert client.attempts == 1
        assert result.failed_count == 1

    def test_exporter_skips_invalid_trace_ids(self) -> None:
        """Exporter rejects malformed trace ids before push."""
        from layerlens.instrument.adapters.frameworks.langfuse.exporter import TraceExporter

        class FakeClient:
            def __init__(self) -> None:
                self.pushed: List[List[Dict[str, Any]]] = []

            def ingestion_batch(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
                self.pushed.append(events)
                return {}

        client = FakeClient()
        state = self._make_state()
        exporter = TraceExporter(client, state)  # type: ignore[arg-type]
        result = exporter.export_traces(
            events_by_trace={
                "not-a-uuid": [{"event_type": "trace.start", "payload": {}}],
                "550e8400-e29b-41d4-a716-446655440000": [
                    {"event_type": "trace.start", "payload": {}}
                ],
            }
        )
        # Only the valid UUID was pushed.
        assert len(client.pushed) == 1
        # The invalid id was reported.
        assert any("Invalid trace id" in e for e in result.errors)

    def test_exporter_retries_transient_push_failures(self) -> None:
        """Transient 502 during push is retried."""
        from layerlens.instrument.adapters.frameworks.langfuse.client import LangfuseAPIError
        from layerlens.instrument.adapters.frameworks.langfuse.exporter import TraceExporter

        class FakeClient:
            def __init__(self) -> None:
                self.attempts = 0

            def ingestion_batch(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
                self.attempts += 1
                if self.attempts < 2:
                    raise LangfuseAPIError("transient", status_code=502)
                return {}

        client = FakeClient()
        state = self._make_state()
        exporter = TraceExporter(
            client,  # type: ignore[arg-type]
            state,
            max_retries=3,
            base_delay=0.001,
            max_delay=0.001,
        )
        result = exporter.export_traces(
            events_by_trace={
                "550e8400-e29b-41d4-a716-446655440000": [
                    {"event_type": "trace.start", "payload": {}}
                ],
            }
        )
        assert client.attempts == 2
        assert result.failed_count == 0
        assert result.exported_count == 1

    def test_client_pagination_uses_shared_paginate(self) -> None:
        """LangfuseAPIClient.get_all_traces walks via the shared paginate helper.

        We verify behaviour rather than internals by stubbing
        list_traces and confirming the iteration pattern matches what
        :func:`paginate` produces.
        """
        from layerlens.instrument.adapters.frameworks.langfuse.client import LangfuseAPIClient

        client = LangfuseAPIClient(public_key="pk", secret_key="sk")

        pages = [
            {"data": [{"id": "a"}, {"id": "b"}], "meta": {"page": 1, "totalPages": 3}},
            {"data": [{"id": "c"}], "meta": {"page": 2, "totalPages": 3}},
            {"data": [{"id": "d"}], "meta": {"page": 3, "totalPages": 3}},
        ]
        idx = {"i": 0}

        def fake_list_traces(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
            page = pages[idx["i"]]
            idx["i"] += 1
            return page

        client.list_traces = fake_list_traces  # type: ignore[method-assign]

        result = client.get_all_traces(limit=2)
        assert [t["id"] for t in result] == ["a", "b", "c", "d"]

    def test_client_warns_on_high_rate_limit_usage(
        self, caplog: Any
    ) -> None:
        """Client emits a warning when usage_ratio >= 80%."""
        import logging

        from layerlens.instrument.adapters._base.importer import parse_rate_limit_headers
        from layerlens.instrument.adapters.frameworks.langfuse.client import LangfuseAPIClient

        client = LangfuseAPIClient(public_key="pk", secret_key="sk")
        # Simulate a response with 95% usage.
        client._last_rate_limit = parse_rate_limit_headers(
            {"X-RateLimit-Remaining": "5", "X-RateLimit-Limit": "100"}
        )

        with caplog.at_level(logging.WARNING):
            client._warn_if_throttle_imminent()

        assert any(
            "rate-limit warning" in rec.message for rec in caplog.records
        )
