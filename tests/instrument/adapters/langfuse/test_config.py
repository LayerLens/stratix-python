"""Tests for Langfuse adapter configuration models."""

import pytest
from datetime import datetime, timezone

from layerlens.instrument.adapters.langfuse.config import (
    ConflictStrategy,
    LangfuseConfig,
    SyncDirection,
    SyncResult,
    SyncState,
)


class TestLangfuseConfig:
    """Tests for LangfuseConfig model."""

    def test_minimal_config(self):
        config = LangfuseConfig(public_key="pk-test", secret_key="sk-test")
        assert config.public_key == "pk-test"
        assert config.secret_key == "sk-test"
        assert config.host == "https://cloud.langfuse.com"
        assert config.mode == SyncDirection.IMPORT

    def test_full_config(self):
        config = LangfuseConfig(
            public_key="pk-test",
            secret_key="sk-test",
            host="https://custom.langfuse.com",
            mode=SyncDirection.BIDIRECTIONAL,
            sync_interval_seconds=1800,
            project_filter="production",
            tag_filter=["v2", "deployed"],
            since=datetime(2024, 1, 1, tzinfo=timezone.utc),
            conflict_strategy=ConflictStrategy.MANUAL,
        )
        assert config.host == "https://custom.langfuse.com"
        assert config.mode == SyncDirection.BIDIRECTIONAL
        assert config.tag_filter == ["v2", "deployed"]
        assert config.conflict_strategy == ConflictStrategy.MANUAL

    def test_trailing_slash_stripped(self):
        config = LangfuseConfig(
            public_key="pk-test",
            secret_key="sk-test",
            host="https://cloud.langfuse.com/",
        )
        assert config.host == "https://cloud.langfuse.com"

    def test_defaults(self):
        config = LangfuseConfig(public_key="pk", secret_key="sk")
        assert config.max_retries == 3
        assert config.page_size == 50
        assert config.sync_interval_seconds == 3600
        assert config.project_filter is None
        assert config.tag_filter is None
        assert config.since is None


class TestSyncState:
    """Tests for SyncState tracking."""

    def test_empty_state(self):
        state = SyncState()
        assert state.last_import_cursor is None
        assert state.last_export_cursor is None
        assert len(state.imported_trace_ids) == 0
        assert len(state.exported_trace_ids) == 0

    def test_record_import(self):
        state = SyncState()
        ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        state.record_import("trace-1", ts)
        assert "trace-1" in state.imported_trace_ids
        assert state.last_import_cursor == ts

    def test_record_import_updates_cursor(self):
        state = SyncState()
        ts1 = datetime(2024, 6, 1, tzinfo=timezone.utc)
        ts2 = datetime(2024, 7, 1, tzinfo=timezone.utc)
        state.record_import("trace-1", ts1)
        state.record_import("trace-2", ts2)
        assert state.last_import_cursor == ts2

    def test_record_import_does_not_regress_cursor(self):
        state = SyncState()
        ts1 = datetime(2024, 7, 1, tzinfo=timezone.utc)
        ts2 = datetime(2024, 6, 1, tzinfo=timezone.utc)
        state.record_import("trace-1", ts1)
        state.record_import("trace-2", ts2)
        assert state.last_import_cursor == ts1

    def test_record_export(self):
        state = SyncState()
        ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        state.record_export("trace-1", ts)
        assert "trace-1" in state.exported_trace_ids
        assert state.last_export_cursor == ts

    def test_quarantine_after_failures(self):
        state = SyncState()
        assert not state.is_quarantined("trace-1")
        state.record_failure("trace-1")
        assert not state.is_quarantined("trace-1")
        state.record_failure("trace-1")
        assert not state.is_quarantined("trace-1")
        state.record_failure("trace-1")
        assert state.is_quarantined("trace-1")

    def test_record_failure_returns_quarantine_status(self):
        state = SyncState()
        assert state.record_failure("trace-1") is False
        assert state.record_failure("trace-1") is False
        assert state.record_failure("trace-1") is True

    def test_clear_quarantine_specific(self):
        state = SyncState()
        for _ in range(3):
            state.record_failure("trace-1")
            state.record_failure("trace-2")
        assert state.is_quarantined("trace-1")
        assert state.is_quarantined("trace-2")
        state.clear_quarantine("trace-1")
        assert not state.is_quarantined("trace-1")
        assert state.is_quarantined("trace-2")

    def test_clear_quarantine_all(self):
        state = SyncState()
        for _ in range(3):
            state.record_failure("trace-1")
            state.record_failure("trace-2")
        state.clear_quarantine()
        assert not state.is_quarantined("trace-1")
        assert not state.is_quarantined("trace-2")

    def test_import_clears_quarantine(self):
        state = SyncState()
        for _ in range(3):
            state.record_failure("trace-1")
        assert state.is_quarantined("trace-1")
        state.record_import("trace-1", datetime.now(timezone.utc))
        assert not state.is_quarantined("trace-1")


class TestSyncResult:
    """Tests for SyncResult model."""

    def test_empty_result(self):
        result = SyncResult(direction=SyncDirection.IMPORT)
        assert result.imported_count == 0
        assert result.exported_count == 0
        assert result.skipped_count == 0
        assert result.failed_count == 0
        assert result.errors == []
        assert result.dry_run is False

    def test_result_with_counts(self):
        result = SyncResult(
            direction=SyncDirection.BIDIRECTIONAL,
            imported_count=10,
            exported_count=5,
            skipped_count=3,
            failed_count=1,
            quarantined_count=1,
            errors=["Trace xyz failed"],
            duration_ms=1234.5,
            dry_run=True,
        )
        assert result.imported_count == 10
        assert result.exported_count == 5
        assert result.dry_run is True
        assert len(result.errors) == 1
