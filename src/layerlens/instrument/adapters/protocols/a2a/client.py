"""Client-side helper for emitting A2A lifecycle events.

Thin wrapper around :class:`A2AProtocolAdapter` that exposes a stable,
typed surface for code that submits A2A tasks without going through a
fully-instrumented client object. Callers that already have an SDK
instance should use ``instrument_a2a`` instead.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from ...._events import A2A_DELEGATION, A2A_TASK_CREATED, A2A_TASK_UPDATED
from .task_lifecycle import TaskState


class A2AClientWrapper:
    """Emit A2A client-side events against an :class:`A2AProtocolAdapter`."""

    def __init__(self, adapter: Any, target_url: str) -> None:
        self._adapter = adapter
        self._target_url = target_url
        self._task_starts: Dict[str, float] = {}

    def send_task(
        self,
        task_id: str,
        messages: List[Dict[str, Any]],
        *,
        task_type: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> str:
        parent = uuid.uuid4().hex[:16]
        self._task_starts[task_id] = time.time()
        self._adapter.emit(
            A2A_TASK_CREATED,
            {
                "task_id": task_id,
                "receiver_url": self._target_url,
                "task_type": task_type,
                "message_count": len(messages),
                "submitter_agent_id": agent_id,
            },
            parent_span_id=parent,
        )
        if agent_id is not None:
            self._adapter.emit(
                A2A_DELEGATION,
                {
                    "task_id": task_id,
                    "target_agent": agent_id,
                    "target_url": self._target_url,
                },
                parent_span_id=parent,
            )
        return parent

    def complete_task(
        self,
        task_id: str,
        status: str,
        *,
        artifacts: Optional[List[Dict[str, Any]]] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        start = self._task_starts.pop(task_id, None)
        latency_ms = (time.time() - start) * 1000 if start is not None else None
        payload: Dict[str, Any] = {
            "task_id": task_id,
            "status": status,
            "artifact_count": len(artifacts) if artifacts else 0,
        }
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        if error_code is not None:
            payload["error_code"] = error_code
        if error_message is not None:
            payload["error"] = error_message
        self._adapter.emit(A2A_TASK_UPDATED, payload)

    def delegate_task(
        self,
        from_agent: str,
        to_agent: str,
        *,
        task_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._adapter.emit(
            A2A_DELEGATION,
            {
                "task_id": task_id or uuid.uuid4().hex[:16],
                "from_agent": from_agent,
                "target_agent": to_agent,
                "context_keys": sorted(context.keys()) if context else [],
            },
        )

    def cancel_task(self, task_id: str) -> None:
        self.complete_task(task_id, status=TaskState.CANCELLED.value)
