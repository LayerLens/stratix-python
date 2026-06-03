"""Server-side helper for tracing incoming A2A JSON-RPC requests.

Complements :class:`A2AProtocolAdapter` for servers that dispatch raw
JSON-RPC payloads rather than calling a typed SDK method — e.g. an
ASGI route handler that forwards ``tasks/send`` envelopes directly.
"""

from __future__ import annotations

import time
import uuid
import logging
from typing import Any, Dict, Optional
from collections.abc import Callable

from ...._events import A2A_TASK_CREATED, A2A_TASK_UPDATED
from .task_lifecycle import TaskState, TaskStateMachine

log = logging.getLogger(__name__)

_TASK_METHODS = frozenset(
    {
        "tasks/send",
        "tasks/sendSubscribe",
        "tasks/get",
        "tasks/cancel",
        "tasks/pushNotification/set",
        "tasks/pushNotification/get",
    }
)


class A2AServerWrapper:
    """Intercept A2A JSON-RPC envelopes and emit lifecycle events."""

    def __init__(
        self,
        adapter: Any,
        original_handler: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._adapter = adapter
        self._original_handler = original_handler
        self._fsms: Dict[str, TaskStateMachine] = {}
        self._task_starts: Dict[str, float] = {}

    def handle_request(
        self,
        request_body: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        method = request_body.get("method", "")
        params = request_body.get("params") or {}

        task_id: Optional[str] = None
        parent = uuid.uuid4().hex[:16]

        if method in {"tasks/send", "tasks/sendSubscribe"}:
            task = params.get("task", params) or {}
            task_id = str(task.get("id") or request_body.get("id") or uuid.uuid4().hex[:16])
            self._fsms[task_id] = TaskStateMachine(task_id)
            self._task_starts[task_id] = time.time()
            self._adapter.emit(
                A2A_TASK_CREATED,
                {
                    "task_id": task_id,
                    "source": "server",
                    "method": method,
                    "headers_present": sorted((headers or {}).keys()),
                },
                parent_span_id=parent,
            )
        elif method == "tasks/cancel":
            task_id = str(params.get("id") or request_body.get("id") or "")
            if task_id:
                self._record_transition(task_id, TaskState.CANCELLED)
                self._emit_update(task_id, TaskState.CANCELLED.value, parent=parent)
        elif method and method not in _TASK_METHODS:
            log.debug("A2A server: ignoring non-task method %s", method)

        if self._original_handler is None:
            return None
        try:
            response = self._original_handler(request_body)
        except Exception as exc:
            if task_id:
                self._record_transition(task_id, TaskState.FAILED)
                self._emit_update(task_id, "failed", parent=parent, error=str(exc))
            raise

        if task_id and method in {"tasks/send", "tasks/sendSubscribe"}:
            status = _status_from(response) or TaskState.COMPLETED.value
            self._record_transition(task_id, status)
            self._emit_update(task_id, status, parent=parent)
        return response

    def handle_agent_card_request(self) -> Optional[Dict[str, Any]]:
        self._adapter.emit("a2a.agent.card.served", {})
        return None

    def _record_transition(self, task_id: str, new_state: Any) -> None:
        fsm = self._fsms.get(task_id)
        if fsm is None:
            return
        fsm.transition(new_state)
        if fsm.is_terminal:
            self._fsms.pop(task_id, None)

    def _emit_update(
        self,
        task_id: str,
        status: str,
        *,
        parent: str,
        error: Optional[str] = None,
    ) -> None:
        start = self._task_starts.pop(task_id, None)
        payload: Dict[str, Any] = {"task_id": task_id, "status": status}
        if start is not None:
            payload["latency_ms"] = (time.time() - start) * 1000
        if error is not None:
            payload["error"] = error
        self._adapter.emit(A2A_TASK_UPDATED, payload, parent_span_id=parent)


def _status_from(response: Any) -> Optional[str]:
    if response is None:
        return None
    if isinstance(response, dict):
        result = response.get("result") or {}
        if isinstance(result, dict):
            status = result.get("status")
            if isinstance(status, dict):
                return status.get("state")
            if isinstance(status, str):
                return status
    return None
