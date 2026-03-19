"""Tests for Langfuse trace importer."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from layerlens.instrument.adapters.langfuse.client import LangfuseAPIClient, LangfuseAPIError
from layerlens.instrument.adapters.langfuse.config import SyncDirection, SyncState
from layerlens.instrument.adapters.langfuse.importer import TraceImporter


class MockStratix:
    def __init__(self):
        self.events = []

    def __bool__(self):
        return True

    def emit(self, event_type, payload):
        self.events.append({"type": event_type, "payload": payload})


def _mock_client(traces=None, full_traces=None):
    """Create a mock LangfuseAPIClient."""
    client = MagicMock(spec=LangfuseAPIClient)
    client.get_all_traces.return_value = traces or []
    if full_traces:
        client.get_trace.side_effect = lambda tid: full_traces.get(tid, {})
    else:
        client.get_trace.return_value = {}
    return client


def _sample_trace(trace_id="lf-1", tags=None, input_val="Hello"):
    return {
        "id": trace_id,
        "name": "test",
        "input": input_val,
        "output": "World",
        "timestamp": "2024-06-01T10:00:00+00:00",
        "updatedAt": "2024-06-01T10:00:05+00:00",
        "tags": tags or [],
        "observations": [],
    }


class TestTraceImporter:

    def test_import_empty(self):
        """Import with no traces returns zero counts."""
        client = _mock_client(traces=[])
        state = SyncState()
        importer = TraceImporter(client, state)
        result = importer.import_traces()
        assert result.imported_count == 0
        assert result.skipped_count == 0

    def test_import_single_trace(self):
        """Import a single trace."""
        trace_summary = {"id": "lf-1", "tags": []}
        full_trace = _sample_trace("lf-1")
        client = _mock_client(
            traces=[trace_summary],
            full_traces={"lf-1": full_trace},
        )
        state = SyncState()
        stratix = MockStratix()
        importer = TraceImporter(client, state)
        result = importer.import_traces(stratix=stratix)

        assert result.imported_count == 1
        assert "lf-1" in state.imported_trace_ids
        assert len(stratix.events) > 0

    def test_import_deduplicates(self):
        """Already imported traces are skipped."""
        trace_summary = {"id": "lf-1", "tags": []}
        client = _mock_client(traces=[trace_summary])
        state = SyncState()
        state.imported_trace_ids.add("lf-1")
        importer = TraceImporter(client, state)
        result = importer.import_traces()

        assert result.imported_count == 0
        assert result.skipped_count == 1

    def test_import_skips_stratix_exported(self):
        """Traces tagged with 'stratix-exported' are skipped (loop prevention)."""
        trace_summary = {"id": "lf-1", "tags": ["stratix-exported"]}
        client = _mock_client(traces=[trace_summary])
        state = SyncState()
        importer = TraceImporter(client, state)
        result = importer.import_traces()

        assert result.imported_count == 0
        assert result.skipped_count == 1

    def test_import_skips_quarantined(self):
        """Quarantined traces are skipped."""
        trace_summary = {"id": "lf-1", "tags": []}
        client = _mock_client(traces=[trace_summary])
        state = SyncState()
        state.quarantined_trace_ids["lf-1"] = 3  # Already quarantined
        importer = TraceImporter(client, state)
        result = importer.import_traces()

        assert result.imported_count == 0
        assert result.quarantined_count == 1

    def test_import_with_limit(self):
        """Limit restricts number of imported traces."""
        traces = [{"id": f"lf-{i}", "tags": []} for i in range(5)]
        full_traces = {f"lf-{i}": _sample_trace(f"lf-{i}") for i in range(5)}
        client = _mock_client(traces=traces, full_traces=full_traces)
        state = SyncState()
        stratix = MockStratix()
        importer = TraceImporter(client, state)
        result = importer.import_traces(stratix=stratix, limit=2)

        assert result.imported_count == 2

    def test_import_dry_run(self):
        """Dry run counts but doesn't import."""
        trace_summary = {"id": "lf-1", "tags": []}
        client = _mock_client(traces=[trace_summary])
        state = SyncState()
        importer = TraceImporter(client, state)
        result = importer.import_traces(dry_run=True)

        assert result.imported_count == 1
        assert result.dry_run is True
        assert "lf-1" not in state.imported_trace_ids
        # get_trace should NOT have been called
        client.get_trace.assert_not_called()

    def test_import_fetch_failure(self):
        """Failed trace fetch increments failure count."""
        trace_summary = {"id": "lf-1", "tags": []}
        client = _mock_client(traces=[trace_summary])
        client.get_trace.side_effect = LangfuseAPIError("Not found", status_code=404)
        state = SyncState()
        importer = TraceImporter(client, state)
        result = importer.import_traces()

        assert result.imported_count == 0
        assert result.failed_count == 1
        assert len(result.errors) == 1

    def test_import_quarantines_after_failures(self):
        """Trace is quarantined after 3 failures."""
        trace_summary = {"id": "lf-1", "tags": []}
        client = _mock_client(traces=[trace_summary])
        client.get_trace.side_effect = LangfuseAPIError("Error")

        state = SyncState()
        state.quarantined_trace_ids["lf-1"] = 2  # Already failed twice
        importer = TraceImporter(client, state)

        # Third failure — should quarantine (but trace is already at 2, need fresh state)
        state2 = SyncState()
        importer2 = TraceImporter(client, state2)

        # Fail 3 times
        for _ in range(3):
            state2_copy = SyncState()
            state2_copy.quarantined_trace_ids = dict(state2.quarantined_trace_ids)
            state2.record_failure("lf-1")

        assert state2.is_quarantined("lf-1")

    def test_import_mapping_failure(self):
        """Failed mapping increments failure count."""
        trace_summary = {"id": "lf-1", "tags": []}
        # Return a trace that will cause a mapping issue
        bad_trace = {"id": "lf-1", "observations": "not-a-list"}
        client = _mock_client(traces=[trace_summary], full_traces={"lf-1": bad_trace})
        state = SyncState()
        importer = TraceImporter(client, state)
        result = importer.import_traces()

        assert result.failed_count == 1

    def test_import_with_since(self):
        """Since parameter is passed to the client."""
        client = _mock_client(traces=[])
        state = SyncState()
        importer = TraceImporter(client, state)
        since = datetime(2024, 1, 1, tzinfo=timezone.utc)
        importer.import_traces(since=since)

        client.get_all_traces.assert_called_once_with(
            tags=None,
            from_timestamp=since,
        )

    def test_import_with_tags(self):
        """Tags parameter is passed to the client."""
        client = _mock_client(traces=[])
        state = SyncState()
        importer = TraceImporter(client, state)
        importer.import_traces(tags=["v2", "production"])

        client.get_all_traces.assert_called_once_with(
            tags=["v2", "production"],
            from_timestamp=None,
        )

    def test_import_updates_cursor(self):
        """Successful import updates the import cursor."""
        trace_summary = {"id": "lf-1", "tags": []}
        full_trace = _sample_trace("lf-1")
        full_trace["updatedAt"] = "2024-06-15T12:00:00+00:00"
        client = _mock_client(
            traces=[trace_summary],
            full_traces={"lf-1": full_trace},
        )
        state = SyncState()
        stratix = MockStratix()
        importer = TraceImporter(client, state)
        importer.import_traces(stratix=stratix)

        assert state.last_import_cursor is not None

    def test_import_emits_events(self):
        """Imported trace events are emitted to STRATIX."""
        trace_summary = {"id": "lf-1", "tags": []}
        full_trace = _sample_trace("lf-1")
        client = _mock_client(
            traces=[trace_summary],
            full_traces={"lf-1": full_trace},
        )
        state = SyncState()
        stratix = MockStratix()
        importer = TraceImporter(client, state)
        importer.import_traces(stratix=stratix)

        types = [e["type"] for e in stratix.events]
        assert "agent.input" in types
        assert "agent.output" in types

    def test_import_without_stratix(self):
        """Import without STRATIX instance succeeds (no emit)."""
        trace_summary = {"id": "lf-1", "tags": []}
        full_trace = _sample_trace("lf-1")
        client = _mock_client(
            traces=[trace_summary],
            full_traces={"lf-1": full_trace},
        )
        state = SyncState()
        importer = TraceImporter(client, state)
        result = importer.import_traces(stratix=None)

        assert result.imported_count == 1

    def test_import_api_error(self):
        """API error on list_traces returns error in result."""
        client = MagicMock(spec=LangfuseAPIClient)
        client.get_all_traces.side_effect = LangfuseAPIError("Connection refused")
        state = SyncState()
        importer = TraceImporter(client, state)
        result = importer.import_traces()

        assert result.failed_count == 1
        assert len(result.errors) == 1

    def test_import_multiple_traces(self):
        """Import multiple traces in one run."""
        summaries = [{"id": f"lf-{i}", "tags": []} for i in range(3)]
        full_traces = {f"lf-{i}": _sample_trace(f"lf-{i}") for i in range(3)}
        client = _mock_client(traces=summaries, full_traces=full_traces)
        state = SyncState()
        stratix = MockStratix()
        importer = TraceImporter(client, state)
        result = importer.import_traces(stratix=stratix)

        assert result.imported_count == 3
        assert len(state.imported_trace_ids) == 3

    def test_import_partial_failure(self):
        """Some traces succeed, some fail — partial success reported."""
        summaries = [{"id": "ok-1", "tags": []}, {"id": "bad-1", "tags": []}, {"id": "ok-2", "tags": []}]
        client = _mock_client(traces=summaries)

        def get_trace_side_effect(tid):
            if tid == "bad-1":
                raise LangfuseAPIError("Not found")
            return _sample_trace(tid)

        client.get_trace.side_effect = get_trace_side_effect
        state = SyncState()
        stratix = MockStratix()
        importer = TraceImporter(client, state)
        result = importer.import_traces(stratix=stratix)

        assert result.imported_count == 2
        assert result.failed_count == 1
