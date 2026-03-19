"""
MCP Async Task Tracker

Tracks long-running MCP tool executions, detecting timeouts and
emitting protocol.async_task events for lifecycle transitions.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class AsyncTaskTracker:
    """
    Tracks the lifecycle of MCP async tasks.

    Monitors created → running → completed/failed/timeout transitions
    and computes elapsed time.
    """

    def __init__(self, default_timeout_ms: int = 300_000) -> None:
        self._default_timeout_ms = default_timeout_ms
        self._tasks: dict[str, _TaskState] = {}

    def create(
        self,
        task_id: str,
        originating_span_id: str | None = None,
        timeout_ms: int | None = None,
    ) -> None:
        """Record creation of an async task."""
        self._tasks[task_id] = _TaskState(
            task_id=task_id,
            originating_span_id=originating_span_id,
            timeout_ms=timeout_ms or self._default_timeout_ms,
            start_time=time.monotonic(),
            status="created",
        )

    def update(
        self,
        task_id: str,
        status: str,
        progress_pct: float | None = None,
    ) -> dict[str, Any] | None:
        """
        Update an async task's status.

        Args:
            task_id: Task identifier.
            status: New status (running, completed, failed, timeout).
            progress_pct: Optional progress percentage.

        Returns:
            Task state dict for event emission, or None if task not found.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return None

        task.status = status
        if progress_pct is not None:
            task.progress_pct = progress_pct

        elapsed_ms = (time.monotonic() - task.start_time) * 1000

        result = {
            "async_task_id": task_id,
            "status": status,
            "originating_span_id": task.originating_span_id,
            "progress_pct": task.progress_pct,
            "timeout_ms": task.timeout_ms,
            "elapsed_ms": elapsed_ms,
        }

        if status in ("completed", "failed", "timeout"):
            self._tasks.pop(task_id, None)

        return result

    def check_timeouts(self) -> list[str]:
        """
        Check for tasks that have exceeded their timeout.

        Returns:
            List of task IDs that have timed out.
        """
        now = time.monotonic()
        timed_out: list[str] = []
        for task_id, task in list(self._tasks.items()):
            elapsed_ms = (now - task.start_time) * 1000
            if elapsed_ms > task.timeout_ms:
                timed_out.append(task_id)
        return timed_out

    @property
    def active_count(self) -> int:
        return len(self._tasks)

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        return {
            "task_id": task.task_id,
            "status": task.status,
            "elapsed_ms": (time.monotonic() - task.start_time) * 1000,
            "timeout_ms": task.timeout_ms,
            "progress_pct": task.progress_pct,
        }


class _TaskState:
    """Internal task state tracker."""
    __slots__ = (
        "task_id", "originating_span_id", "timeout_ms",
        "start_time", "status", "progress_pct",
    )

    def __init__(
        self,
        task_id: str,
        originating_span_id: str | None,
        timeout_ms: int,
        start_time: float,
        status: str,
    ) -> None:
        self.task_id = task_id
        self.originating_span_id = originating_span_id
        self.timeout_ms = timeout_ms
        self.start_time = start_time
        self.status = status
        self.progress_pct: float | None = None
