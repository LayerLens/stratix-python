"""Track MCP long-running async task lifecycle.

``AsyncTaskTracker`` records per-task timestamps and timeouts so the MCP
adapter can emit ``mcp.async_task`` events for ``created → running →
completed/failed/timeout`` transitions. ``check_timeouts()`` returns tasks
that have exceeded their configured timeout so the adapter can emit a
timeout event even without further progress updates from the server.
"""

from __future__ import annotations

import time
import logging
from typing import Any, Optional
from dataclasses import field, dataclass

log = logging.getLogger(__name__)


@dataclass
class _TaskState:
    task_id: str
    originating_span_id: Optional[str]
    timeout_ms: int
    start_time: float
    status: str
    progress_pct: Optional[float] = field(default=None)


class AsyncTaskTracker:
    """State tracker for long-running MCP tool executions."""

    def __init__(self, default_timeout_ms: int = 300_000) -> None:
        self._default_timeout_ms = default_timeout_ms
        self._tasks: dict[str, _TaskState] = {}

    def create(
        self,
        task_id: str,
        originating_span_id: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> None:
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
        progress_pct: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        """Record a status change. Returns an event-ready payload, or None if unknown task."""
        task = self._tasks.get(task_id)
        if task is None:
            return None

        task.status = status
        if progress_pct is not None:
            task.progress_pct = progress_pct

        elapsed_ms = (time.monotonic() - task.start_time) * 1000
        result: dict[str, Any] = {
            "async_task_id": task_id,
            "status": status,
            "originating_span_id": task.originating_span_id,
            "progress_pct": task.progress_pct,
            "timeout_ms": task.timeout_ms,
            "elapsed_ms": elapsed_ms,
        }
        if status in {"completed", "failed", "timeout"}:
            self._tasks.pop(task_id, None)
        return result

    def check_timeouts(self) -> list[str]:
        now = time.monotonic()
        return [
            task_id for task_id, task in list(self._tasks.items()) if (now - task.start_time) * 1000 > task.timeout_ms
        ]

    def get_task(self, task_id: str) -> Optional[dict[str, Any]]:
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

    @property
    def active_count(self) -> int:
        return len(self._tasks)
