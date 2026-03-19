"""Tests for Langfuse trace exporter."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from layerlens.instrument.adapters.langfuse.client import LangfuseAPIClient, LangfuseAPIError
from layerlens.instrument.adapters.langfuse.config import SyncDirection, SyncState
from layerlens.instrument.adapters.langfuse.exporter import TraceExporter


def _mock_client():
    """Create a mock LangfuseAPIClient."""
    client = MagicMock(spec=LangfuseAPIClient)
    client.ingestion_batch.return_value = {"successes": []}
    return client


def _sample_events(trace_id="t1"):
    """Create sample STRATIX events for export."""
    return [
        {
            "event_type": "agent.input",
            "trace_id": trace_id,
            "timestamp": "2024-06-01T10:00:00+00:00",
            "payload": {"input_text": "Hello", "agent_id": "bot"},
        },
        {
            "event_type": "model.invoke",
            "trace_id": trace_id,
            "timestamp": "2024-06-01T10:00:01+00:00",
            "payload": {"model": "gpt-4", "tokens_prompt": 10, "tokens_completion": 5},
        },
        {
            "event_type": "agent.output",
            "trace_id": trace_id,
            "timestamp": "2024-06-01T10:00:02+00:00",
            "payload": {"output_text": "World"},
        },
    ]


class TestTraceExporter:

    def test_export_empty(self):
        """Export with no events returns zero counts."""
        client = _mock_client()
        state = SyncState()
        exporter = TraceExporter(client, state)
        result = exporter.export_traces(events_by_trace={})
        assert result.exported_count == 0

    def test_export_single_trace(self):
        """Export a single trace."""
        client = _mock_client()
        state = SyncState()
        exporter = TraceExporter(client, state)

        events_by_trace = {"t1": _sample_events("t1")}
        result = exporter.export_traces(events_by_trace=events_by_trace)

        assert result.exported_count == 1
        assert "t1" in state.exported_trace_ids
        client.ingestion_batch.assert_called_once()

    def test_export_multiple_traces(self):
        """Export multiple traces."""
        client = _mock_client()
        state = SyncState()
        exporter = TraceExporter(client, state)

        events_by_trace = {
            "t1": _sample_events("t1"),
            "t2": _sample_events("t2"),
            "t3": _sample_events("t3"),
        }
        result = exporter.export_traces(events_by_trace=events_by_trace)

        assert result.exported_count == 3
        assert len(state.exported_trace_ids) == 3

    def test_export_with_trace_ids_filter(self):
        """Export only specified trace IDs."""
        client = _mock_client()
        state = SyncState()
        exporter = TraceExporter(client, state)

        events_by_trace = {
            "t1": _sample_events("t1"),
            "t2": _sample_events("t2"),
            "t3": _sample_events("t3"),
        }
        result = exporter.export_traces(
            events_by_trace=events_by_trace,
            trace_ids=["t1", "t3"],
        )

        assert result.exported_count == 2

    def test_export_skips_imported_traces(self):
        """Loop prevention: imported traces are not re-exported."""
        client = _mock_client()
        state = SyncState()
        state.imported_trace_ids.add("t1")

        exporter = TraceExporter(client, state)
        result = exporter.export_traces(events_by_trace={"t1": _sample_events("t1")})

        assert result.exported_count == 0
        assert result.skipped_count == 1

    def test_export_skips_already_exported(self):
        """Already exported traces are skipped."""
        client = _mock_client()
        state = SyncState()
        state.exported_trace_ids.add("t1")

        exporter = TraceExporter(client, state)
        result = exporter.export_traces(events_by_trace={"t1": _sample_events("t1")})

        assert result.exported_count == 0
        assert result.skipped_count == 1

    def test_export_skips_empty_events(self):
        """Traces with no events are skipped."""
        client = _mock_client()
        state = SyncState()
        exporter = TraceExporter(client, state)
        result = exporter.export_traces(events_by_trace={"t1": []})

        assert result.exported_count == 0
        assert result.skipped_count == 1

    def test_export_dry_run(self):
        """Dry run counts but doesn't export."""
        client = _mock_client()
        state = SyncState()
        exporter = TraceExporter(client, state)

        result = exporter.export_traces(
            events_by_trace={"t1": _sample_events("t1")},
            dry_run=True,
        )

        assert result.exported_count == 1
        assert result.dry_run is True
        assert "t1" not in state.exported_trace_ids
        client.ingestion_batch.assert_not_called()

    def test_export_api_failure(self):
        """API failure increments failure count."""
        client = _mock_client()
        client.ingestion_batch.side_effect = LangfuseAPIError("Server error")
        state = SyncState()
        exporter = TraceExporter(client, state)

        result = exporter.export_traces(events_by_trace={"t1": _sample_events("t1")})

        assert result.exported_count == 0
        assert result.failed_count == 1
        assert len(result.errors) == 1

    def test_export_partial_failure(self):
        """Some traces export, some fail."""
        client = _mock_client()
        call_count = 0

        def side_effect(batch):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise LangfuseAPIError("Error on second trace")
            return {"successes": []}

        client.ingestion_batch.side_effect = side_effect
        state = SyncState()
        exporter = TraceExporter(client, state)

        events_by_trace = {
            "t1": _sample_events("t1"),
            "t2": _sample_events("t2"),
            "t3": _sample_events("t3"),
        }
        result = exporter.export_traces(events_by_trace=events_by_trace)

        assert result.exported_count == 2
        assert result.failed_count == 1

    def test_export_batch_contains_trace_and_observations(self):
        """Verify the ingestion batch has trace + observation events."""
        client = _mock_client()
        state = SyncState()
        exporter = TraceExporter(client, state)

        events = _sample_events("t1")
        exporter.export_traces(events_by_trace={"t1": events})

        batch_call = client.ingestion_batch.call_args
        batch = batch_call[0][0]

        # Should have: 1 trace-create + 1 generation-create = 2 events
        types = [e["type"] for e in batch]
        assert "trace-create" in types
        assert "generation-create" in types

    def test_export_updates_cursor(self):
        """Successful export updates the export cursor."""
        client = _mock_client()
        state = SyncState()
        exporter = TraceExporter(client, state)

        exporter.export_traces(events_by_trace={"t1": _sample_events("t1")})

        assert state.last_export_cursor is not None

    def test_export_missing_trace_id(self):
        """Export with trace_id not in events_by_trace skips it."""
        client = _mock_client()
        state = SyncState()
        exporter = TraceExporter(client, state)

        result = exporter.export_traces(
            events_by_trace={"t1": _sample_events("t1")},
            trace_ids=["t1", "t2"],  # t2 has no events
        )

        assert result.exported_count == 1
        assert result.skipped_count == 1
