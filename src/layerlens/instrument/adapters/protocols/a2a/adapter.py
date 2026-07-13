"""A2A (Agent-to-Agent) protocol adapter — real a2a-sdk 1.1.0 surface.

Instruments both sides of an A2A interaction:

* Server side: :class:`A2AServerWrapper` keys on the REAL JSON-RPC method
  vocabulary (``message/send`` / ``message/stream`` / ``tasks/*`` — the v0.1
  ``tasks/send``/``tasks/sendSubscribe`` strings are absent from a2a-sdk 1.1.0).
* Client side: wraps ``send_message`` (the real a2a Client method) and the
  legacy ``send_task``/``get_task``/``cancel_task`` duck-typed surface to emit
  ``a2a.task.created`` / ``a2a.task.updated`` / ``a2a.delegation``.
* Discovery: wraps ``get_agent_card`` to emit ``a2a.agent.discovered`` with the
  card-signature PROVENANCE (presence + a keyed-HMAC fingerprint — never the raw
  JWS).

Delegation provenance (A15 / D3): ``a2a.delegation`` keeps the delegation
TOPOLOGY (delegator ``from_agent`` + delegatee ``to_agent``/``target_agent`` +
``task_id``) and a keyed-HMAC ``delegation_fp`` of (target+skill) as METADATA
that SURVIVES ``capture_content=False`` (mirrors ``agent.handoff``); the
free-text ``skill_description`` stays content. See ``_capture_config._CONTENT_KEYS``.

Works against any object exposing the standard a2a-sdk surface; missing methods
are silently skipped so the adapter is compatible with partial implementations
and test doubles.
"""

from __future__ import annotations

import hmac
import time
import uuid
import hashlib
import logging
import secrets
from typing import Any, Dict, Callable

from ...._events import (
    AGENT_ERROR,
    A2A_DELEGATION,
    A2A_TASK_CREATED,
    A2A_TASK_UPDATED,
    A2A_AGENT_DISCOVERED,
)
from .agent_card import parse_agent_card, summarize_signatures
from ...._context import _current_span_id
from .acp_normalizer import ACPNormalizer
from .task_lifecycle import TaskState, TaskStateMachine
from .._base_protocol import BaseProtocolAdapter

log = logging.getLogger(__name__)

# The real a2a Client method (base_client.py:50) + the legacy duck-typed names a
# partial implementation / test double may expose. ``send_message`` is the
# canonical 1.1.0 entry point; ``send_task`` is the v0.1 name we still wrap when
# present so a caller using the older surface is observed too.
_CLIENT_SEND_METHODS = ("send_message", "send_task")
_CLIENT_OTHER_METHODS = ("get_task", "cancel_task")

# a2a terminal states that are real failures. A normal completion, a still-running
# state, or a caller-initiated cancel is NOT an error and must not route here.
_A2A_ERROR_STATES = frozenset({TaskState.FAILED.value, TaskState.REJECTED.value})


def _maybe_emit_task_error(
    adapter: Any,
    task_id: str,
    status: str,
    *,
    parent: str | None = None,
    error: Any = None,
    error_type: str | None = None,
) -> None:
    """Emit ``agent.error`` for a terminal a2a FAILURE (failed/rejected) so the
    trace's derived status is ``error``, not ``completed`` (S12/F4).

    The task-lifecycle status the a2a spec defines is read by no downstream
    engine, so a failed/rejected run would otherwise be mislabelled completed by
    the atlas default. No invented detail: ``error``/``error_type`` are only what
    the failure actually carried; when the terminal status arrived without an
    exception the honest ``error_type`` is the state itself (``a2a_task_<status>``).
    Shared by the client-wrap, server, and client-helper paths.
    """
    if status not in _A2A_ERROR_STATES:
        return
    payload: Dict[str, Any] = {
        "task_id": task_id,
        "source": "a2a",
        "error_type": error_type or f"a2a_task_{status}",
    }
    if error is not None:
        payload["error"] = str(error)
    adapter.emit(AGENT_ERROR, payload, parent_span_id=parent)


class A2AProtocolAdapter(BaseProtocolAdapter):
    PROTOCOL = "a2a"
    PROTOCOL_VERSION = "0.3.0"

    def __init__(self, *, capture_config: Any = None) -> None:
        super().__init__(capture_config=capture_config)
        self._tasks: Dict[str, float] = {}
        self._agent_cards: Dict[str, Any] = {}
        self._task_fsms: Dict[str, TaskStateMachine] = {}
        self._acp_normalizer = ACPNormalizer()
        # Per-instance HMAC key (a2ui.py / ap2.py P3 pattern): the delegation +
        # card-signature fingerprints are KEYED HMACs so a low-entropy
        # (target+skill) pair or a JWS can't be brute-forced from the emitted
        # fingerprint; the key is never emitted.
        self._hash_key = secrets.token_bytes(32)

    def connect(self, target: Any = None, **kwargs: Any) -> Any:  # noqa: ARG002
        self._client = target

        for method in (*_CLIENT_SEND_METHODS, *_CLIENT_OTHER_METHODS):
            if hasattr(target, method):
                orig = getattr(target, method)
                self._originals[method] = orig
                is_send = method in _CLIENT_SEND_METHODS
                setattr(target, method, self._wrap_client_method(orig, method, is_send=is_send))

        if hasattr(target, "get_agent_card"):
            orig = target.get_agent_card
            self._originals["get_agent_card"] = orig
            target.get_agent_card = self._wrap_discovery(orig)

        if hasattr(target, "register_handler"):
            orig = target.register_handler
            self._originals["register_handler"] = orig
            target.register_handler = self._wrap_register_handler(orig)

        return target

    # -- fingerprinting --

    def _fingerprint(self, value: Any) -> str:
        """Keyed-HMAC fingerprint (a2ui/ap2 P3). One-way without the per-instance
        key, which is never emitted; the raw value is NEVER emitted."""
        return "sha256:" + hmac.new(self._hash_key, str(value).encode(), hashlib.sha256).hexdigest()

    # -- client wrap --

    def _wrap_client_method(self, original: Callable[..., Any], method: str, *, is_send: bool) -> Callable[..., Any]:
        adapter = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            task_id = kwargs.get("task_id") or (args[0] if args else None) or uuid.uuid4().hex[:16]
            task_id = str(task_id)
            parent = _current_span_id.get() or uuid.uuid4().hex[:16]
            start = time.time()
            if is_send:
                adapter._tasks[task_id] = start
                adapter._task_fsms[task_id] = TaskStateMachine(task_id)
                created_payload: Dict[str, Any] = {
                    "task_id": task_id,
                    "method": method,
                    "request": _summarize(kwargs),
                }
                # Node-identity parity with A2AClientWrapper.send_task (client.py):
                # stamp the submitter when the caller declared one (S13/F6). It is
                # topology, not content — omitted honestly when absent.
                from_agent = kwargs.get("from_agent")
                if from_agent is not None:
                    created_payload["submitter_agent_id"] = str(from_agent)
                adapter.emit(A2A_TASK_CREATED, created_payload, parent_span_id=parent)
                adapter._emit_delegation(kwargs, task_id=task_id, parent=parent)
                # Enter WORKING before invoking the handler so the FSM transitions
                # submitted → working → completed/failed/rejected validly.
                adapter._record_transition(task_id, TaskState.WORKING)
            try:
                result = original(*args, **kwargs)
            except Exception as exc:
                if is_send:
                    adapter._record_transition(task_id, TaskState.FAILED)
                    adapter.emit(
                        A2A_TASK_UPDATED,
                        {
                            "task_id": task_id,
                            "status": TaskState.FAILED.value,
                            "error": str(exc),
                            "latency_ms": (time.time() - start) * 1000,
                        },
                        parent_span_id=parent,
                    )
                    _maybe_emit_task_error(
                        adapter,
                        task_id,
                        TaskState.FAILED.value,
                        parent=parent,
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
                raise
            if is_send:
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
                _maybe_emit_task_error(adapter, task_id, status, parent=parent)
            return result

        return wrapped

    def _emit_delegation(self, kwargs: Dict[str, Any], *, task_id: str, parent: str) -> None:
        """Emit ``a2a.delegation`` IFF a delegatee is named. The TOPOLOGY ids +
        a keyed-HMAC fp of (target+skill) survive no-content (A15); the free-text
        skill description is content."""
        target_agent = kwargs.get("to_agent") or kwargs.get("target_agent") or kwargs.get("agent_id")
        if target_agent is None:
            return
        from_agent = kwargs.get("from_agent")
        skill_desc = kwargs.get("skill_description") or kwargs.get("skill")
        payload: Dict[str, Any] = {
            "task_id": task_id,
            "target_agent": str(target_agent),
            "to_agent": str(target_agent),
            # The fp binds the delegated skill to the target for server-anchored
            # verification and SURVIVES redaction (the raw skill does not).
            "delegation_fp": self._fingerprint(str(target_agent) + (str(skill_desc) if skill_desc else "")),
        }
        if from_agent is not None:
            payload["from_agent"] = str(from_agent)
        if kwargs.get("target_url") is not None:
            payload["target_url"] = kwargs["target_url"]
        if skill_desc is not None:
            payload["skill_description"] = skill_desc
        self.emit(A2A_DELEGATION, payload, parent_span_id=parent)

    def _record_transition(self, task_id: str, new_state: TaskState | str) -> None:
        """Advance the state machine; logs a warning on invalid transitions."""
        fsm = self._task_fsms.get(task_id)
        if fsm is None:
            return
        fsm.transition(new_state)
        if fsm.is_terminal:
            self._task_fsms.pop(task_id, None)

    # -- discovery wrap (card-signature provenance, D2) --

    def _wrap_discovery(self, original: Callable[..., Any]) -> Callable[..., Any]:
        adapter = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            agent_id = _extract_agent_id(result)
            if agent_id is not None:
                adapter._agent_cards[agent_id] = result
            normalized: Dict[str, Any] | None = None
            if isinstance(result, (dict, str)):
                try:
                    normalized = parse_agent_card(result)
                except ValueError:
                    normalized = None
            sig = summarize_signatures(result, adapter._fingerprint)
            payload: Dict[str, Any] = {
                "agent_id": agent_id,
                "name": (normalized or {}).get("name") or getattr(result, "name", None),
                "skills": (normalized or {}).get("skills") or _extract_skills(result),
                "authScheme": (normalized or {}).get("authScheme"),
                "protocolVersion": (normalized or {}).get("protocolVersion"),
            }
            payload.update(sig)
            adapter.emit(A2A_AGENT_DISCOVERED, payload)
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
            parent = _current_span_id.get() or uuid.uuid4().hex[:16]
            start = time.time()
            adapter._task_fsms[task_id] = TaskStateMachine(task_id)
            adapter.emit(
                A2A_TASK_CREATED,
                {"task_id": task_id, "source": "server", "skill_description": _skill_from(task)},
                parent_span_id=parent,
            )
            adapter._record_transition(task_id, TaskState.WORKING)
            try:
                result = handler(task, *args, **kwargs)
            except Exception as exc:
                adapter._record_transition(task_id, TaskState.FAILED)
                adapter.emit(
                    A2A_TASK_UPDATED,
                    {
                        "task_id": task_id,
                        "status": TaskState.FAILED.value,
                        "error": str(exc),
                        "latency_ms": (time.time() - start) * 1000,
                    },
                    parent_span_id=parent,
                )
                _maybe_emit_task_error(
                    adapter,
                    task_id,
                    TaskState.FAILED.value,
                    parent=parent,
                    error=str(exc),
                    error_type=type(exc).__name__,
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
            _maybe_emit_task_error(adapter, task_id, status, parent=parent)
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
    """Read the terminal status from a task/result. Defaults to UNKNOWN (NOT
    completed) so a result with no parseable status is never mislabeled as a
    successful completion (D5)."""
    status = getattr(result, "status", None)
    if status is None and isinstance(result, dict):
        status = result.get("status")
    # a2a Task: status is a TaskStatus with a .state (an enum or a string).
    state = getattr(status, "state", None)
    if state is not None:
        return getattr(state, "value", str(state))
    if isinstance(status, dict):
        state = status.get("state")
        if state is not None:
            return getattr(state, "value", str(state))
    if isinstance(status, str):
        return status
    return TaskState.UNKNOWN.value


def _task_id_from(task: Any) -> str:
    tid = getattr(task, "id", None)
    if tid is None and isinstance(task, dict):
        tid = (
            task.get("id") or (task.get("task") or {}).get("id")
            if isinstance(task.get("task"), dict)
            else task.get("id")
        )
    return str(tid) if tid else uuid.uuid4().hex[:16]


def _skill_from(task: Any) -> Any:
    skill = getattr(task, "skill", None)
    if skill is None and isinstance(task, dict):
        skill = task.get("skill")
        if skill is None and isinstance(task.get("task"), dict):
            skill = task["task"].get("skill")
    return skill


def _summarize(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ("skill", "skill_description", "task_id", "priority"):
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
