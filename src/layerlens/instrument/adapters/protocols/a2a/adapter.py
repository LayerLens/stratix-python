"""A2A (Agent-to-Agent) protocol adapter.

Instruments both sides of an A2A interaction:

* Server side: wraps ``serve()`` to emit ``a2a.task.created`` / ``a2a.task.updated``
  from inbound task lifecycle events.
* Client side: wraps ``client()`` / ``get_agent_card()`` / ``send_task()`` to
  emit ``a2a.agent.discovered`` and ``a2a.delegation`` events.

Works against any object exposing the standard a2a-sdk surface; missing
methods are silently skipped so the adapter is compatible with partial
implementations and test doubles.
"""

from __future__ import annotations

import time
import uuid
import logging
from typing import Any, Dict, Callable

from ...._events import (
    A2A_DELEGATION,
    A2A_TASK_CREATED,
    A2A_TASK_UPDATED,
    A2A_AGENT_DISCOVERED,
)
from .agent_card import parse_agent_card
from .acp_normalizer import ACPNormalizer
from .task_lifecycle import TaskState, TaskStateMachine
from .._base_protocol import BaseProtocolAdapter

log = logging.getLogger(__name__)


class A2AProtocolAdapter(BaseProtocolAdapter):
    PROTOCOL = "a2a"
    PROTOCOL_VERSION = "0.3.0"

    def __init__(self) -> None:
        super().__init__()
        self._tasks: Dict[str, float] = {}
        self._agent_cards: Dict[str, Any] = {}
        self._task_fsms: Dict[str, TaskStateMachine] = {}
        self._acp_normalizer = ACPNormalizer()

    def connect(self, target: Any = None, **kwargs: Any) -> Any:  # noqa: ARG002
        self._client = target

        for method in ("send_task", "get_task", "cancel_task"):
            if hasattr(target, method):
                orig = getattr(target, method)
                self._originals[method] = orig
                setattr(target, method, self._wrap_client_method(orig, method))

        if hasattr(target, "get_agent_card"):
            orig = target.get_agent_card
            self._originals["get_agent_card"] = orig
            target.get_agent_card = self._wrap_discovery(orig)

        if hasattr(target, "register_handler"):
            orig = target.register_handler
            self._originals["register_handler"] = orig
            target.register_handler = self._wrap_register_handler(orig)

        return target

    def _wrap_client_method(self, original: Callable[..., Any], method: str) -> Callable[..., Any]:
        adapter = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            task_id = kwargs.get("task_id") or (args[0] if args else None) or uuid.uuid4().hex[:16]
            parent = uuid.uuid4().hex[:16]
            start = time.time()
            if method == "send_task":
                adapter._tasks[task_id] = start
                adapter._task_fsms[task_id] = TaskStateMachine(task_id)
                adapter.emit(
                    A2A_TASK_CREATED,
                    {"task_id": task_id, "method": method, "request": _summarize(kwargs)},
                    parent_span_id=parent,
                )
                adapter.emit(
                    A2A_DELEGATION,
                    {
                        "task_id": task_id,
                        "target_agent": kwargs.get("agent_id"),
                        "skill": kwargs.get("skill"),
                    },
                    parent_span_id=parent,
                )
            # Enter WORKING state before invoking the handler so the FSM
            # transitions submitted → working → completed / failed validly.
            if method == "send_task":
                adapter._record_transition(task_id, TaskState.WORKING)
            try:
                result = original(*args, **kwargs)
            except Exception as exc:
                adapter._record_transition(task_id, TaskState.FAILED)
                adapter.emit(
                    A2A_TASK_UPDATED,
                    {
                        "task_id": task_id,
                        "status": "failed",
                        "error": str(exc),
                        "latency_ms": (time.time() - start) * 1000,
                    },
                    parent_span_id=parent,
                )
                raise
            status = _task_status(result)
            adapter._record_transition(task_id, status)
            adapter.emit(
                A2A_TASK_UPDATED,
                {
                    "task_id": task_id,
                    "status": status,
                    "latency_ms": (time.time() - start) * 1000,
                },
                parent_span_id=parent,
            )
            return result

        return wrapped

    def _record_transition(self, task_id: str, new_state: TaskState | str) -> None:
        """Advance the state machine; logs a warning on invalid transitions."""
        fsm = self._task_fsms.get(task_id)
        if fsm is None:
            return
        fsm.transition(new_state)
        if fsm.is_terminal:
            self._task_fsms.pop(task_id, None)

    def _wrap_discovery(self, original: Callable[..., Any]) -> Callable[..., Any]:
        adapter = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            agent_id = _extract_agent_id(result)
            if agent_id is not None:
                adapter._agent_cards[agent_id] = result
            # If the result is a dict or JSON string, normalize via parse_agent_card.
            normalized: Dict[str, Any] | None = None
            if isinstance(result, (dict, str)):
                try:
                    normalized = parse_agent_card(result)
                except ValueError:
                    normalized = None
            adapter.emit(
                A2A_AGENT_DISCOVERED,
                {
                    "agent_id": agent_id,
                    "name": (normalized or {}).get("name") or getattr(result, "name", None),
                    "skills": (normalized or {}).get("skills") or _extract_skills(result),
                    "authScheme": (normalized or {}).get("authScheme"),
                    "protocolVersion": (normalized or {}).get("protocolVersion"),
                },
            )
            return result

        return wrapped

    def _wrap_register_handler(self, original: Callable[..., Any]) -> Callable[..., Any]:
        adapter = self

        def wrapped(handler: Any, *args: Any, **kwargs: Any) -> Any:
            wrapped_handler = adapter._wrap_server_handler(handler)
            return original(wrapped_handler, *args, **kwargs)

        return wrapped

    def _wrap_server_handler(self, handler: Callable[..., Any]) -> Callable[..., Any]:
        adapter = self

        def on_task(task: Any, *args: Any, **kwargs: Any) -> Any:
            # Normalize ACP-origin payloads into A2A canonical form before dispatch.
            if isinstance(task, dict):
                task, is_acp = adapter._acp_normalizer.detect_and_normalize(task)
                if is_acp:
                    log.debug("A2A adapter normalized ACP-origin payload")

            task_id = _task_id_from(task)
            parent = uuid.uuid4().hex[:16]
            start = time.time()
            adapter._task_fsms[task_id] = TaskStateMachine(task_id)
            adapter.emit(
                A2A_TASK_CREATED,
                {"task_id": task_id, "source": "server", "skill": _skill_from(task)},
                parent_span_id=parent,
            )
            # Advance submitted → working before handler runs so the final
            # completed / failed transition is valid.
            adapter._record_transition(task_id, TaskState.WORKING)
            try:
                result = handler(task, *args, **kwargs)
            except Exception as exc:
                adapter._record_transition(task_id, TaskState.FAILED)
                adapter.emit(
                    A2A_TASK_UPDATED,
                    {
                        "task_id": task_id,
                        "status": "failed",
                        "error": str(exc),
                        "latency_ms": (time.time() - start) * 1000,
                    },
                    parent_span_id=parent,
                )
                raise
            status = _task_status(result)
            adapter._record_transition(task_id, status)
            adapter.emit(
                A2A_TASK_UPDATED,
                {
                    "task_id": task_id,
                    "status": status,
                    "latency_ms": (time.time() - start) * 1000,
                },
                parent_span_id=parent,
            )
            return result

        return on_task


def _extract_agent_id(card: Any) -> str | None:
    for attr in ("id", "agent_id", "name"):
        val = getattr(card, attr, None)
        if val is not None:
            return str(val)
    if isinstance(card, dict):
        return card.get("id") or card.get("agent_id") or card.get("name")
    return None


def _extract_skills(card: Any) -> list[str]:
    skills = getattr(card, "skills", None)
    if isinstance(card, dict):
        skills = card.get("skills")
    if isinstance(skills, list):
        return [getattr(s, "name", str(s)) for s in skills]
    return []


def _task_status(result: Any) -> str:
    status = getattr(result, "status", None)
    if status is None and isinstance(result, dict):
        status = result.get("status")
    if isinstance(status, dict):
        status = status.get("state")
    return status or "completed"


def _task_id_from(task: Any) -> str:
    tid = getattr(task, "id", None)
    if tid is None and isinstance(task, dict):
        tid = (
            task.get("id") or (task.get("task") or {}).get("id")
            if isinstance(task.get("task"), dict)
            else task.get("id")
        )
    return tid or uuid.uuid4().hex[:16]


def _skill_from(task: Any) -> Any:
    skill = getattr(task, "skill", None)
    if skill is None and isinstance(task, dict):
        skill = task.get("skill")
        if skill is None and isinstance(task.get("task"), dict):
            skill = task["task"].get("skill")
    return skill


def _summarize(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ("agent_id", "skill", "task_id", "priority"):
        if key in kwargs:
            out[key] = kwargs[key]
    return out


def instrument_a2a(target: Any) -> A2AProtocolAdapter:
    from ..._registry import get, register

    existing = get("a2a")
    if existing is not None:
        existing.disconnect()
    adapter = A2AProtocolAdapter()
    adapter.connect(target)
    register("a2a", adapter)
    return adapter


def uninstrument_a2a() -> None:
    from ..._registry import unregister

    unregister("a2a")
