"""Tests for Langfuse bidirectional sync."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from layerlens.instrument.adapters.langfuse.client import LangfuseAPIClient, LangfuseAPIError
from layerlens.instrument.adapters.langfuse.config import SyncDirection, SyncState
from layerlens.instrument.adapters.langfuse.importer import TraceImporter
from layerlens.instrument.adapters.langfuse.exporter import TraceExporter
from layerlens.instrument.adapters.langfuse.sync import BidirectionalSync


class MockStratix:
    def __init__(self):
        self.events = []

    def __bool__(self):
        return True

    def emit(self, event_type, payload):
        self.events.append({"type": event_type, "payload": payload})


def _mock_client(traces=None, full_traces=None):
    client = MagicMock(spec=LangfuseAPIClient)
    client.get_all_traces.return_value = traces or []
    if full_traces:
        client.get_trace.side_effect = lambda tid: full_traces.get(tid, {})
    else:
        client.get_trace.return_value = {}
    client.ingestion_batch.return_value = {"successes": []}
    return client


def _sample_trace(trace_id="lf-1"):
    return {
        "id": trace_id,
        "name": "test",
        "input": "Hello",
        "output": "World",
        "timestamp": "2024-06-01T10:00:00+00:00",
        "updatedAt": "2024-06-01T10:00:05+00:00",
        "tags": [],
        "observations": [],
    }


def _sample_events(trace_id="t1"):
    return [
        {
            "event_type": "agent.input",
            "trace_id": trace_id,
            "timestamp": "2024-06-01T10:00:00+00:00",
            "payload": {"input_text": "Hello"},
        },
        {
            "event_type": "agent.output",
            "trace_id": trace_id,
            "timestamp": "2024-06-01T10:00:01+00:00",
            "payload": {"output_text": "World"},
        },
    ]


class TestBidirectionalSync:

    def test_import_only(self):
        """Import-only sync only imports, doesn't export."""
        summaries = [{"id": "lf-1", "tags": []}]
        full_traces = {"lf-1": _sample_trace("lf-1")}
        client = _mock_client(traces=summaries, full_traces=full_traces)
        state = SyncState()
        importer = TraceImporter(client, state)
        exporter = TraceExporter(client, state)
        sync = BidirectionalSync(importer, exporter, state)
        stratix = MockStratix()

        result = sync.run(stratix=stratix, direction=SyncDirection.IMPORT)

        assert result.imported_count == 1
        assert result.exported_count == 0
        assert result.direction == SyncDirection.IMPORT

    def test_export_only(self):
        """Export-only sync only exports, doesn't import."""
        client = _mock_client()
        state = SyncState()
        importer = TraceImporter(client, state)
        exporter = TraceExporter(client, state)
        sync = BidirectionalSync(importer, exporter, state)

        events_by_trace = {"t1": _sample_events("t1")}
        result = sync.run(direction=SyncDirection.EXPORT, events_by_trace=events_by_trace)

        assert result.exported_count == 1
        assert result.imported_count == 0
        assert result.direction == SyncDirection.EXPORT

    def test_bidirectional(self):
        """Bidirectional sync imports and exports."""
        summaries = [{"id": "lf-1", "tags": []}]
        full_traces = {"lf-1": _sample_trace("lf-1")}
        client = _mock_client(traces=summaries, full_traces=full_traces)
        state = SyncState()
        importer = TraceImporter(client, state)
        exporter = TraceExporter(client, state)
        sync = BidirectionalSync(importer, exporter, state)
        stratix = MockStratix()

        events_by_trace = {"t1": _sample_events("t1")}
        result = sync.run(
            stratix=stratix,
            direction=SyncDirection.BIDIRECTIONAL,
            events_by_trace=events_by_trace,
        )

        assert result.imported_count == 1
        assert result.exported_count == 1

    def test_dry_run(self):
        """Dry run counts but doesn't make changes."""
        summaries = [{"id": "lf-1", "tags": []}]
        client = _mock_client(traces=summaries)
        state = SyncState()
        importer = TraceImporter(client, state)
        exporter = TraceExporter(client, state)
        sync = BidirectionalSync(importer, exporter, state)

        events_by_trace = {"t1": _sample_events("t1")}
        result = sync.run(
            direction=SyncDirection.BIDIRECTIONAL,
            dry_run=True,
            events_by_trace=events_by_trace,
        )

        assert result.dry_run is True
        assert result.imported_count == 1
        assert result.exported_count == 1
        assert "lf-1" not in state.imported_trace_ids
        assert "t1" not in state.exported_trace_ids

    def test_cursor_based_incremental_import(self):
        """Sync uses last_import_cursor for incremental import."""
        client = _mock_client(traces=[])
        state = SyncState()
        cursor_time = datetime(2024, 6, 1, tzinfo=timezone.utc)
        state.last_import_cursor = cursor_time

        importer = TraceImporter(client, state)
        exporter = TraceExporter(client, state)
        sync = BidirectionalSync(importer, exporter, state)

        sync.run(direction=SyncDirection.IMPORT)

        client.get_all_traces.assert_called_once_with(
            tags=None,
            from_timestamp=cursor_time,
        )

    def test_since_override(self):
        """Explicit since overrides the cursor."""
        client = _mock_client(traces=[])
        state = SyncState()
        state.last_import_cursor = datetime(2024, 1, 1, tzinfo=timezone.utc)

        importer = TraceImporter(client, state)
        exporter = TraceExporter(client, state)
        sync = BidirectionalSync(importer, exporter, state)

        override = datetime(2024, 6, 1, tzinfo=timezone.utc)
        sync.run(direction=SyncDirection.IMPORT, since=override)

        client.get_all_traces.assert_called_once_with(
            tags=None,
            from_timestamp=override,
        )

    def test_export_without_events(self):
        """Export phase is skipped when no events are provided."""
        client = _mock_client(traces=[])
        state = SyncState()
        importer = TraceImporter(client, state)
        exporter = TraceExporter(client, state)
        sync = BidirectionalSync(importer, exporter, state)

        result = sync.run(direction=SyncDirection.BIDIRECTIONAL)

        assert result.exported_count == 0

    def test_errors_aggregated(self):
        """Errors from both import and export are aggregated."""
        client = MagicMock(spec=LangfuseAPIClient)
        client.get_all_traces.side_effect = LangfuseAPIError("Import error")
        client.ingestion_batch.side_effect = LangfuseAPIError("Export error")

        state = SyncState()
        importer = TraceImporter(client, state)
        exporter = TraceExporter(client, state)
        sync = BidirectionalSync(importer, exporter, state)

        events_by_trace = {"t1": _sample_events("t1")}
        result = sync.run(
            direction=SyncDirection.BIDIRECTIONAL,
            events_by_trace=events_by_trace,
        )

        assert result.failed_count >= 1
        assert len(result.errors) >= 1

    def test_tags_passed_to_importer(self):
        """Tags are forwarded to the import call."""
        client = _mock_client(traces=[])
        state = SyncState()
        importer = TraceImporter(client, state)
        exporter = TraceExporter(client, state)
        sync = BidirectionalSync(importer, exporter, state)

        sync.run(direction=SyncDirection.IMPORT, tags=["v2", "production"])

        client.get_all_traces.assert_called_once_with(
            tags=["v2", "production"],
            from_timestamp=None,
        )

    def test_loop_prevention_in_bidirectional(self):
        """Imported traces are not re-exported in bidirectional mode."""
        summaries = [{"id": "lf-1", "tags": []}]
        full_traces = {"lf-1": _sample_trace("lf-1")}
        client = _mock_client(traces=summaries, full_traces=full_traces)
        state = SyncState()
        importer = TraceImporter(client, state)
        exporter = TraceExporter(client, state)
        sync = BidirectionalSync(importer, exporter, state)
        stratix = MockStratix()

        # Try to export the same trace that was just imported
        events_by_trace = {"lf-1": _sample_events("lf-1")}
        result = sync.run(
            stratix=stratix,
            direction=SyncDirection.BIDIRECTIONAL,
            events_by_trace=events_by_trace,
        )

        # Import succeeds, export of same trace is skipped
        assert result.imported_count == 1
        assert result.exported_count == 0
        assert result.skipped_count >= 1
