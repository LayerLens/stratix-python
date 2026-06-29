"""Server-side helper for tracing incoming A2A JSON-RPC requests.

Complements :class:`A2AProtocolAdapter` for servers that dispatch raw JSON-RPC
payloads rather than calling a typed SDK method — e.g. an ASGI route handler
that forwards ``message/send`` envelopes directly.

Keys on the REAL a2a-sdk 1.1.0 method vocabulary (jsonrpc_adapter.py:53-64):
``message/send``, ``message/stream``, ``tasks/get``, ``tasks/cancel``,
``tasks/resubscribe``, ``tasks/pushNotificationConfig/{set,get,list,delete}``,
``agent/getAuthenticatedExtendedCard``. The v0.1 ``tasks/send`` /
``tasks/sendSubscribe`` strings are ABSENT from a2a-sdk 1.1.0 — keying on them
made the adapter silent against every modern A2A server (D1). A task is now
created implicitly when an agent responds to ``message/send`` with a ``Task``
(spec §9.4.1), so the task id is read from the response, falling back to the
request message's ``taskId`` (a continuation) or the JSON-RPC id.
"""

from __future__ import annotations

import time
import uuid
import logging
from typing import Any, Dict, Optional
from collections.abc import Callable

from ...._events import A2A_TASK_CREATED, A2A_TASK_UPDATED, A2A_AGENT_CARD_SERVED
from .agent_card import summarize_signatures
from .task_lifecycle import TaskState, TaskStateMachine

log = logging.getLogger(__name__)

# Methods that CREATE/continue a task and trigger lifecycle tracking. The real
# 1.1.0 names — message/send (non-streaming) + message/stream (SSE).
_TASK_SEND_METHODS = frozenset({"message/send", "message/stream"})
# The full real method vocabulary we recognize (anything else is a non-task
# method we don't track). Push-config + resubscribe are observed-as-known so a
# future emit can hang off them (D6).
_KNOWN_METHODS = frozenset(
    {
        "message/send",
        "message/stream",
        "tasks/get",
        "tasks/cancel",
        "tasks/resubscribe",
        "tasks/pushNotificationConfig/set",
        "tasks/pushNotificationConfig/get",
        "tasks/pushNotificationConfig/list",
        "tasks/pushNotificationConfig/delete",
        "agent/getAuthenticatedExtendedCard",
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
        message = params.get("message") if isinstance(params, dict) else None
        if not isinstance(message, dict):
            message = {}

        task_id: Optional[str] = None
        parent = uuid.uuid4().hex[:16]
        start = time.time()
        is_send = method in _TASK_SEND_METHODS

        if method == "tasks/cancel":
            task_id = str(params.get("id") or params.get("taskId") or request_body.get("id") or "")
            if task_id:
                self._fsms.setdefault(task_id, TaskStateMachine(task_id))
                self._record_transition(task_id, TaskState.CANCELED)
                self._emit_update(task_id, TaskState.CANCELED.value, parent=parent)
        elif not is_send and method and method not in _KNOWN_METHODS:
            log.debug("A2A server: ignoring non-task method %s", method)

        if self._original_handler is None:
            return None
        try:
            response = self._original_handler(request_body)
        except Exception as exc:
            if is_send:
                # The send failed before the server assigned a task id — track it
                # under the request id so the failure is still observable.
                fallback = str(
                    message.get("taskId") or message.get("task_id") or request_body.get("id") or uuid.uuid4().hex[:16]
                )
                self._emit_created(fallback, method, message, headers, parent, start)
                self._fsms.setdefault(fallback, TaskStateMachine(fallback)).transition(TaskState.WORKING)
                self._record_transition(fallback, TaskState.FAILED)
                self._emit_update(fallback, TaskState.FAILED.value, parent=parent, error=str(exc))
            raise

        if is_send:
            # A task is created implicitly when the server responds with a Task
            # (spec §9.4.1); the id is SERVER-ASSIGNED in the response, falling
            # back to a continuation taskId or the request id. Emit created+updated
            # with the real id now that it is known.
            task_id = _task_id_from_response(response) or str(
                message.get("taskId") or message.get("task_id") or request_body.get("id") or uuid.uuid4().hex[:16]
            )
            self._fsms[task_id] = TaskStateMachine(task_id)
            self._task_starts[task_id] = start
            self._emit_created(task_id, method, message, headers, parent, start)
            self._record_transition(task_id, TaskState.WORKING)
            status = _status_from(response) or TaskState.UNKNOWN.value
            self._record_transition(task_id, status)
            self._emit_update(task_id, status, parent=parent)
        return response

    def _emit_created(
        self,
        task_id: str,
        method: str,
        message: Dict[str, Any],
        headers: Optional[Dict[str, str]],
        parent: str,
        start: float,  # noqa: ARG002
    ) -> None:
        self._adapter.emit(
            A2A_TASK_CREATED,
            {
                "task_id": task_id,
                "source": "server",
                "method": method,
                # Provenance chain (spec §2.5/§2.6): context_id groups related
                # tasks; reference_task_ids is the delegation/provenance chain.
                "context_id": message.get("contextId") or message.get("context_id"),
                "reference_task_ids": message.get("referenceTaskIds") or message.get("reference_task_ids"),
                "headers_present": sorted((headers or {}).keys()),
            },
            parent_span_id=parent,
        )

    def handle_agent_card_request(self, card: Any = None) -> Optional[Dict[str, Any]]:
        """Emit ``a2a.agent.card.served`` with the served card's signature
        provenance (presence + keyed-HMAC fp; never the raw JWS, D2)."""
        payload: Dict[str, Any] = {}
        if card is not None:
            payload.update(summarize_signatures(card, self._adapter._fingerprint))
        self._adapter.emit(A2A_AGENT_CARD_SERVED, payload)
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


def _result_dict(response: Any) -> Optional[Dict[str, Any]]:
    if isinstance(response, dict):
        result = response.get("result")
        return result if isinstance(result, dict) else None
    return None


def _task_id_from_response(response: Any) -> Optional[str]:
    """The server-assigned task id from a message/send response Task."""
    result = _result_dict(response)
    if result and result.get("kind") == "task":
        tid = result.get("id")
        return str(tid) if tid else None
    if result:
        tid = result.get("id")
        return str(tid) if tid else None
    return None


def _status_from(response: Any) -> Optional[str]:
    result = _result_dict(response)
    if not result:
        return None
    status = result.get("status")
    if isinstance(status, dict):
        return status.get("state")
    if isinstance(status, str):
        return status
    return None
