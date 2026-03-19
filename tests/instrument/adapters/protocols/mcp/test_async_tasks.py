"""
Tests for MCP async task lifecycle events.
"""

import pytest
import time

from layerlens.instrument.adapters.protocols.mcp.async_task_tracker import AsyncTaskTracker


class TestAsyncTaskTracker:
    def setup_method(self):
        self.tracker = AsyncTaskTracker(default_timeout_ms=5000)

    def test_create_task(self):
        self.tracker.create("task-001", timeout_ms=10000)
        assert self.tracker.active_count == 1
        task = self.tracker.get_task("task-001")
        assert task is not None
        assert task["status"] == "created"
        assert task["timeout_ms"] == 10000

    def test_update_running(self):
        self.tracker.create("task-001")
        result = self.tracker.update("task-001", "running", progress_pct=25.0)
        assert result is not None
        assert result["status"] == "running"
        assert result["progress_pct"] == 25.0
        assert result["elapsed_ms"] >= 0

    def test_update_completed(self):
        self.tracker.create("task-001")
        result = self.tracker.update("task-001", "completed")
        assert result["status"] == "completed"
        assert self.tracker.active_count == 0

    def test_update_failed(self):
        self.tracker.create("task-001")
        result = self.tracker.update("task-001", "failed")
        assert result["status"] == "failed"
        assert self.tracker.active_count == 0

    def test_update_unknown_task(self):
        result = self.tracker.update("unknown", "running")
        assert result is None

    def test_default_timeout(self):
        self.tracker.create("task-001")
        task = self.tracker.get_task("task-001")
        assert task["timeout_ms"] == 5000

    def test_check_timeouts_none(self):
        self.tracker.create("task-001", timeout_ms=999999)
        timed_out = self.tracker.check_timeouts()
        assert len(timed_out) == 0

    def test_originating_span_id(self):
        self.tracker.create("task-001", originating_span_id="span-abc")
        result = self.tracker.update("task-001", "running")
        assert result["originating_span_id"] == "span-abc"


class TestAsyncTaskEvents:
    def test_async_task_created_event(self, mcp_adapter, mock_stratix):
        mcp_adapter.on_async_task(
            async_task_id="async-001",
            status="created",
            timeout_ms=30000,
        )
        assert len(mock_stratix.events) == 1
        event = mock_stratix.events[0][0]
        assert event.event_type == "protocol.async_task"
        assert event.status == "created"
        assert event.protocol == "mcp"

    def test_async_task_completed_event(self, mcp_adapter, mock_stratix):
        mcp_adapter.on_async_task(async_task_id="async-002", status="created")
        mcp_adapter.on_async_task(async_task_id="async-002", status="completed")
        assert len(mock_stratix.events) == 2
        completed = mock_stratix.events[1][0]
        assert completed.status == "completed"
        assert completed.elapsed_ms is not None

    def test_async_task_with_progress(self, mcp_adapter, mock_stratix):
        mcp_adapter.on_async_task(async_task_id="async-003", status="created")
        mcp_adapter.on_async_task(
            async_task_id="async-003",
            status="running",
            progress_pct=50.0,
        )
        event = mock_stratix.events[1][0]
        assert event.progress_pct == 50.0
