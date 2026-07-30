"""A2A autonomy invariants — real ``a2a-sdk`` 1.1.0 fixtures, bite-proven.

Every fixture here is a REAL object/wire-shape from the installed ``a2a-sdk``
(v1.1.0) — the v0.3 JSON-RPC envelopes (``message/send`` / ``message/stream``),
``AgentCard`` + ``AgentCardSignature`` (the JWS card-signing surface),
``Message`` (with ``reference_task_ids`` / ``context_id``), and
``TaskStatusUpdateEvent`` / ``TaskStatus`` / ``TaskState`` — NOT a hand-rolled
``SimpleNamespace``. The bodies are produced via ``model_dump(by_alias=True)`` so
they carry the exact camelCase the adapter sees on the wire; a library upgrade
that changes a schema fails these fixtures loudly (brief §3.5). The module
``importorskip("a2a")``, so it SKIPS in the base py3.9 venv (no a2a) and runs in
``.audit-venvs/sk`` (py3.11 + a2a-sdk 1.1.0 + the repo editable).

The five invariants (each @pytest.mark.invariant, each with a confirmed bite):

1. Real method vocabulary — a ``message/send`` / ``message/stream`` JSON-RPC
   envelope emits ``a2a.task.*`` (the obsolete ``tasks/send`` wiring emitted
   nothing against a modern server). BITE: rename the server's method branch
   back to ``tasks/send`` -> no a2a.task.* -> RED.
2. Delegation provenance survives no-content (A15) — under capture_content=False
   ``a2a.delegation`` KEEPS from_agent/to_agent ids + a keyed-HMAC fingerprint of
   (target+skill); the free-text skill DESCRIPTION is stripped. BITE: add
   ``to_agent``/``delegation_fp`` to ``_CONTENT_KEYS['a2a.delegation']`` (revert
   the reclassification) -> provenance vanishes -> RED.
3. Agent-card signature — emitted as PRESENCE + keyed-HMAC fp, never the raw JWS,
   even under capture_content=True. BITE: emit the raw ``signature`` -> the
   raw-JWS-absence + secret-scan assertions go RED.
4. Late SSE attribution (A8) — an SSE event arriving after stream-open lands on
   the stream's trace_id (not None / not a later trace). BITE: drop the
   snapshot/re-establish in A2ASSEHandler -> the late event has no collector ->
   RED.
5. TaskState terminal correctness — a real ``failed``/``rejected`` TaskState maps
   to a failed/rejected a2a.task.* status, never completed. BITE: change the
   server's status default back to ``completed`` (or break the FSM rejected edge)
   -> a rejected task reads completed -> RED.
"""

from __future__ import annotations

import hmac
import json
import hashlib
from typing import Any, Dict, List, Optional

import pytest

pytest.importorskip("a2a")

from a2a.compat.v0_3 import types as a2a_types  # noqa: E402

from layerlens.instrument._context import _current_collector  # noqa: E402
from layerlens.instrument._collector import TraceCollector  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.protocols.a2a.server import A2AServerWrapper  # noqa: E402
from layerlens.instrument.adapters.protocols.a2a.adapter import A2AProtocolAdapter  # noqa: E402
from layerlens.instrument.adapters.protocols.a2a.sse_handler import A2ASSEHandler  # noqa: E402

# A2A autonomy controls belong in the fast Invariant Gates job (shift-left).
pytestmark = pytest.mark.invariant

_NO_CONTENT = CaptureConfig(capture_content=False)
_CONTENT = CaptureConfig(capture_content=True)

# Free-text card/signature/skill canaries — must NEVER survive redaction or
# (for the raw JWS) appear at all.
SECRET_SKILL_DESC = "SENTINEL-process-refunds-over-10k-no-approval"
RAW_JWS_SIG = "SENTINELrawjwssignaturesegmentABCDEF1234567890xyz"
RAW_JWS_PROTECTED = "eyJhbGciOiJFUzI1NiIsImtpZCI6InNlbnRpbmVsLWtleSJ9"


# ── real-a2a fixture builders (the library's OWN typings) ──────────────────


def _agent_skill(desc: str = SECRET_SKILL_DESC) -> a2a_types.AgentSkill:
    return a2a_types.AgentSkill(id="refund", name="refund", description=desc, tags=["billing"])


def signed_agent_card(*, sig: str = RAW_JWS_SIG, protected: str = RAW_JWS_PROTECTED) -> a2a_types.AgentCard:
    """A real AgentCard carrying a real AgentCardSignature (JWS protected.signature)."""
    return a2a_types.AgentCard(
        name="Billing Agent",
        description="Processes refunds",
        url="https://billing.example.com/a2a/v1",
        version="1.0.0",
        protocol_version="0.3.0",
        capabilities=a2a_types.AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[_agent_skill()],
        signatures=[a2a_types.AgentCardSignature(protected=protected, signature=sig)],
    )


def message_send_body(
    *,
    text: str = "please refund order 123",
    context_id: str = "ctx-7",
    reference_task_ids: Optional[List[str]] = None,
    streaming: bool = False,
) -> Dict[str, Any]:
    """A real ``message/send`` (or ``message/stream``) JSON-RPC body, camelCase."""
    msg = a2a_types.Message(
        message_id="m1",
        role=a2a_types.Role.user,
        parts=[a2a_types.Part(root=a2a_types.TextPart(text=text))],
        context_id=context_id,
        reference_task_ids=reference_task_ids if reference_task_ids is not None else ["task-prev-1"],
    )
    cls = a2a_types.SendStreamingMessageRequest if streaming else a2a_types.SendMessageRequest
    req = cls(id="req-1", params=a2a_types.MessageSendParams(message=msg))
    return req.model_dump(mode="json", by_alias=True, exclude_none=True)


def task_response_body(task_id: str, state: a2a_types.TaskState, context_id: str = "ctx-7") -> Dict[str, Any]:
    """A real ``message/send`` SUCCESS response carrying a Task at *state*."""
    task = a2a_types.Task(id=task_id, context_id=context_id, status=a2a_types.TaskStatus(state=state))
    resp = a2a_types.SendMessageResponse(root=a2a_types.SendMessageSuccessResponse(id="req-1", result=task))
    return resp.model_dump(mode="json", by_alias=True, exclude_none=True)


def status_update_event(
    task_id: str, state: a2a_types.TaskState, *, context_id: str = "ctx-7", final: bool = True
) -> Dict[str, Any]:
    """A real TaskStatusUpdateEvent (the body of a message/stream SSE frame)."""
    ev = a2a_types.TaskStatusUpdateEvent(
        task_id=task_id,
        context_id=context_id,
        final=final,
        status=a2a_types.TaskStatus(state=state),
    )
    return ev.model_dump(mode="json", by_alias=True, exclude_none=True)


# ── harness ────────────────────────────────────────────────────────────────


def _run(fn: Any, config: Optional[CaptureConfig] = None) -> List[Dict[str, Any]]:
    """Run *fn* inside a live collector context, return emitted events."""
    collector = TraceCollector(object(), config or CaptureConfig())
    token = _current_collector.set(collector)
    try:
        fn()
    finally:
        _current_collector.reset(token)
    return collector.events


def _by_type(events: List[Dict[str, Any]], event_type: str) -> List[Dict[str, Any]]:
    return [e["payload"] for e in events if e["event_type"] == event_type]


def _all_text(events: List[Dict[str, Any]]) -> str:
    return json.dumps(events, default=str)


def _server(adapter: A2AProtocolAdapter, handler: Any = None) -> A2AServerWrapper:
    return A2AServerWrapper(adapter, original_handler=handler)


# ===========================================================================
# INVARIANT 1 (D1) — REAL METHOD VOCABULARY. message/send + message/stream are
# observed; the obsolete tasks/send wiring would emit nothing here.
# ===========================================================================


class TestRealMethodVocabulary:
    def test_message_send_emits_task_created_and_updated(self) -> None:
        adapter = A2AProtocolAdapter()
        body = message_send_body()
        resp = task_response_body("task-9", a2a_types.TaskState.completed)
        srv = _server(adapter, handler=lambda _b: resp)

        events = _run(lambda: srv.handle_request(body, headers={"authorization": "Bearer x"}))
        created = _by_type(events, "a2a.task.created")
        updated = _by_type(events, "a2a.task.updated")
        assert created, "message/send did not emit a2a.task.created (obsolete tasks/send wiring is silent here)"
        assert updated, "message/send did not emit a2a.task.updated"
        # The task id is the server-assigned id from the Task in the response.
        assert created[0]["task_id"] == "task-9"
        assert created[0]["method"] == "message/send"

    def test_message_stream_is_recognized_as_a_task_method(self) -> None:
        adapter = A2AProtocolAdapter()
        body = message_send_body(streaming=True)
        resp = task_response_body("task-stream", a2a_types.TaskState.working)
        srv = _server(adapter, handler=lambda _b: resp)

        events = _run(lambda: srv.handle_request(body))
        assert _by_type(events, "a2a.task.created"), "message/stream was not recognized as a task method (D1)"
        assert _by_type(events, "a2a.task.created")[0]["method"] == "message/stream"

    def test_reference_task_ids_provenance_carried(self) -> None:
        adapter = A2AProtocolAdapter()
        body = message_send_body(reference_task_ids=["task-prev-1", "task-prev-2"], context_id="ctx-xyz")
        resp = task_response_body("task-9", a2a_types.TaskState.completed, context_id="ctx-xyz")
        srv = _server(adapter, handler=lambda _b: resp)

        events = _run(lambda: srv.handle_request(body))
        created = _by_type(events, "a2a.task.created")
        assert created, "no task.created"
        # The provenance chain (reference_task_ids + context_id) is the spec's
        # delegation/provenance signal (§2.5/§2.6) and survives no-content.
        assert created[0].get("reference_task_ids") == ["task-prev-1", "task-prev-2"]
        assert created[0].get("context_id") == "ctx-xyz"

    def test_obsolete_tasks_send_emits_nothing(self) -> None:
        # The whole point of D1: the OLD vocabulary must NOT be the trigger.
        adapter = A2AProtocolAdapter()
        srv = _server(adapter, handler=lambda _b: task_response_body("t", a2a_types.TaskState.completed))
        events = _run(lambda: srv.handle_request({"method": "tasks/send", "id": "r", "params": {}}))
        assert not _by_type(events, "a2a.task.created"), (
            "tasks/send (a v0.1 method absent from a2a-sdk 1.1.0) still triggers task tracking — "
            "the adapter must key on the real message/send vocabulary, not the obsolete one"
        )


# ===========================================================================
# INVARIANT 2 (A15, D3) — DELEGATION PROVENANCE SURVIVES NO-CONTENT.
# from_agent/to_agent/target_agent ids + a keyed-HMAC fp of (target+skill)
# survive capture_content=False; the free-text skill DESCRIPTION is stripped.
# ===========================================================================


class TestDelegationProvenance:
    def _drive(self, config: CaptureConfig) -> List[Dict[str, Any]]:
        adapter = A2AProtocolAdapter(capture_config=config)
        # Real send via the duck-typed client wrap — the production path.
        target = type(
            "Cli",
            (),
            {
                "send_task": staticmethod(lambda **kw: {"task_id": "t1", "status": "completed"}),
            },
        )()
        adapter.connect(target=target)

        def go() -> None:
            target.send_task(
                task_id="t1",
                from_agent="orchestrator-1",
                to_agent="billing-agent-7",
                skill_description=SECRET_SKILL_DESC,
            )

        return _run(go, config)

    def test_content_present_by_default(self) -> None:
        # Sanity: the free-text skill IS captured under content-on, so the
        # no-content assertion below is meaningful (not vacuous).
        events = self._drive(_CONTENT)
        delegations = _by_type(events, "a2a.delegation")
        assert delegations, "no a2a.delegation event emitted — test would be vacuous"
        assert SECRET_SKILL_DESC in _all_text([{"payload": p} for p in delegations])

    def test_topology_and_fp_survive_no_content(self) -> None:
        events = self._drive(_NO_CONTENT)
        delegations = _by_type(events, "a2a.delegation")
        assert delegations, "no a2a.delegation event emitted"
        d = delegations[0]
        text = _all_text(events)
        # Free-text skill description is CONTENT -> stripped.
        assert SECRET_SKILL_DESC not in text, "delegation skill description leaked under capture_content=False"
        # Topology ids are METADATA -> survive (mirror agent.handoff from/to).
        assert d.get("from_agent") == "orchestrator-1", "delegator id (from_agent) did not survive no-content (A15)"
        assert d.get("to_agent") == "billing-agent-7", "delegatee id (to_agent) did not survive no-content (A15)"
        assert d.get("target_agent") == "billing-agent-7", "target_agent id did not survive no-content (A15)"
        # The keyed-HMAC fingerprint of (target+skill) survives for server-anchored
        # verification — provenance auditable WITHOUT the free-text skill.
        fp = d.get("delegation_fp")
        assert isinstance(fp, str) and fp.startswith("sha256:"), (
            "delegation_fp (keyed-HMAC of target+skill) did not survive no-content (A15)"
        )

    def test_fp_is_keyed_not_plain_sha256(self) -> None:
        # The fp must be a KEYED HMAC, not a reversible plain SHA-256 of a
        # low-entropy (target+skill) pair.
        events = self._drive(_NO_CONTENT)
        d = _by_type(events, "a2a.delegation")[0]
        plain = "sha256:" + hashlib.sha256(("billing-agent-7" + SECRET_SKILL_DESC).encode()).hexdigest()
        assert d["delegation_fp"] != plain, "delegation_fp is plain unsalted SHA-256 — trivially reversible"

    def test_fp_binds_target_and_skill(self) -> None:
        # The fp must change when the (target, skill) pair changes — it is not a
        # constant. We recompute against the adapter's own key to prove binding.
        adapter = A2AProtocolAdapter(capture_config=_NO_CONTENT)
        target = type("Cli", (), {"send_task": staticmethod(lambda **kw: {"status": "completed"})})()
        adapter.connect(target=target)

        def go() -> None:
            target.send_task(task_id="t1", from_agent="a", to_agent="billing-agent-7", skill_description="skill-A")

        events = _run(go, _NO_CONTENT)
        fp = _by_type(events, "a2a.delegation")[0]["delegation_fp"]
        expect = (
            "sha256:"
            + hmac.new(adapter._hash_key, ("billing-agent-7" + "skill-A").encode(), hashlib.sha256).hexdigest()
        )
        assert fp == expect, "delegation_fp is not HMAC(target+skill) — it does not bind the delegated skill"


# ===========================================================================
# INVARIANT 3 (D2) — AGENT-CARD SIGNATURE: PRESENCE + keyed-HMAC fp, NEVER raw.
# The raw JWS (protected/signature) must never appear in any payload, even under
# capture_content=True.
# ===========================================================================


class TestCardSignatureProvenance:
    def _drive(self, config: CaptureConfig, card: a2a_types.AgentCard) -> List[Dict[str, Any]]:
        adapter = A2AProtocolAdapter(capture_config=config)
        target = type("Cli", (), {"get_agent_card": staticmethod(lambda *a, **kw: card)})()
        adapter.connect(target=target)
        return _run(lambda: target.get_agent_card(), config)

    def test_signature_presence_and_fp_emitted(self) -> None:
        events = self._drive(_CONTENT, signed_agent_card())
        discovered = _by_type(events, "a2a.agent.discovered")
        assert discovered, "no a2a.agent.discovered emitted"
        d = discovered[0]
        assert d.get("signature_present") is True, "signed card not reported as signature_present (D2)"
        assert d.get("signature_count") == 1
        fp = d.get("signature_fp")
        assert isinstance(fp, str) and fp.startswith("sha256:"), "card signature fp (keyed-HMAC) not emitted (D2)"

    def test_unsigned_card_reports_absence(self) -> None:
        card = signed_agent_card()
        card.signatures = None
        d = self._drive(_CONTENT, card)
        discovered = _by_type(d, "a2a.agent.discovered")
        assert discovered and discovered[0].get("signature_present") is False, (
            "an UNSIGNED card must report signature_present=False (so a missing card-auth is auditable)"
        )

    def test_raw_jws_never_emitted_even_with_content_on(self) -> None:
        # The single most security-relevant assertion: the raw JWS segments are
        # NEVER in any payload, regardless of capture_content.
        for cfg in (_CONTENT, _NO_CONTENT):
            events = self._drive(cfg, signed_agent_card())
            text = _all_text(events)
            assert RAW_JWS_SIG not in text, (
                f"raw JWS signature leaked into telemetry (capture_content={cfg.capture_content})"
            )
            assert RAW_JWS_PROTECTED not in text, (
                f"raw JWS protected header leaked into telemetry (capture_content={cfg.capture_content})"
            )

    def test_fp_is_keyed_not_plain_sha256(self) -> None:
        events = self._drive(_CONTENT, signed_agent_card())
        fp = _by_type(events, "a2a.agent.discovered")[0]["signature_fp"]
        plain = "sha256:" + hashlib.sha256((RAW_JWS_PROTECTED + "." + RAW_JWS_SIG).encode()).hexdigest()
        assert fp != plain, "signature_fp is plain unsalted SHA-256 — not a keyed HMAC"

    def test_signature_fp_survives_no_content(self) -> None:
        # Provenance must be auditable under privacy-on.
        events = self._drive(_NO_CONTENT, signed_agent_card())
        d = _by_type(events, "a2a.agent.discovered")[0]
        assert d.get("signature_present") is True
        assert isinstance(d.get("signature_fp"), str) and d["signature_fp"].startswith("sha256:"), (
            "card signature fp did not survive capture_content=False (D2)"
        )


# ===========================================================================
# INVARIANT 4 (A8, D4) — LATE SSE ATTRIBUTION. An SSE event arriving after
# stream-open lands on the stream's trace_id (snapshot at open, re-established
# per event), not None and not a later trace.
# ===========================================================================


class TestLateSSEAttribution:
    def test_late_sse_event_lands_on_stream_trace(self) -> None:
        adapter = A2AProtocolAdapter()
        stream_collector = TraceCollector(object(), CaptureConfig())

        # Stream OPENS with the stream's collector ambient — the handler snapshots it.
        open_token = _current_collector.set(stream_collector)
        try:
            handler = A2ASSEHandler(task_id="task-stream", adapter=adapter)
        finally:
            _current_collector.reset(open_token)

        # ... later, a DIFFERENT (or no) collector is ambient when the SSE frame
        # actually arrives (a worker thread / a later trace). The event must still
        # land on the stream's trace, not this one.
        later_collector = TraceCollector(object(), CaptureConfig())
        later_token = _current_collector.set(later_collector)
        try:
            handler.process_event(status_update_event("task-stream", a2a_types.TaskState.completed))
        finally:
            _current_collector.reset(later_token)

        stream_events = stream_collector.events
        later_events = later_collector.events
        assert stream_events, "late SSE event was DROPPED — it never landed on the stream's trace (A8/D4)"
        assert all(e["trace_id"] == stream_collector.trace_id for e in stream_events)
        assert not later_events, "late SSE event mis-attributed to the later/ambient trace (A8/D4)"

    def test_late_sse_event_with_no_ambient_collector(self) -> None:
        # The hard case: NO collector ambient at emit time (the default-None bug).
        adapter = A2AProtocolAdapter()
        stream_collector = TraceCollector(object(), CaptureConfig())
        open_token = _current_collector.set(stream_collector)
        try:
            handler = A2ASSEHandler(task_id="task-stream", adapter=adapter)
        finally:
            _current_collector.reset(open_token)

        # Emit with NO ambient collector (context default None).
        handler.process_event(status_update_event("task-stream", a2a_types.TaskState.completed))
        assert stream_collector.events, (
            "with no ambient collector the late SSE event was dropped — the snapshot at stream-open was not re-established"
        )

    def test_sse_final_event_drives_terminal_state(self) -> None:
        adapter = A2AProtocolAdapter()
        stream_collector = TraceCollector(object(), CaptureConfig())
        open_token = _current_collector.set(stream_collector)
        try:
            handler = A2ASSEHandler(task_id="task-stream", adapter=adapter)
            # working (non-final) then completed (final)
            handler.process_event(status_update_event("task-stream", a2a_types.TaskState.working, final=False))
            handler.process_event(status_update_event("task-stream", a2a_types.TaskState.completed, final=True))
        finally:
            _current_collector.reset(open_token)
        updates = _by_type(stream_collector.events, "a2a.task.updated")
        assert any(u.get("status") == "completed" and u.get("final") is True for u in updates), (
            "the final SSE status-update did not record the terminal completed state (D4)"
        )


# ===========================================================================
# INVARIANT 5 (D5) — TaskState TERMINAL CORRECTNESS. A real failed/rejected
# TaskState maps to a failed/rejected a2a.task.* status, NEVER completed.
# ===========================================================================


class TestTaskStateTerminalCorrectness:
    def _drive_server_with_state(self, state: a2a_types.TaskState) -> List[Dict[str, Any]]:
        adapter = A2AProtocolAdapter()
        body = message_send_body()
        resp = task_response_body("task-9", state)
        srv = _server(adapter, handler=lambda _b: resp)
        return _run(lambda: srv.handle_request(body))

    def test_rejected_task_is_not_completed(self) -> None:
        events = self._drive_server_with_state(a2a_types.TaskState.rejected)
        updates = _by_type(events, "a2a.task.updated")
        assert updates, "no a2a.task.updated for a rejected task"
        last = updates[-1]
        assert last["status"] == "rejected", f"a rejected task was labeled {last['status']!r}, not 'rejected' (D5)"
        assert last["status"] != "completed", "a rejected task must NEVER be mislabeled completed (D5)"

    def test_failed_task_is_not_completed(self) -> None:
        events = self._drive_server_with_state(a2a_types.TaskState.failed)
        last = _by_type(events, "a2a.task.updated")[-1]
        assert last["status"] == "failed", f"a failed task was labeled {last['status']!r} (D5)"

    def test_canceled_single_l_spelling_accepted(self) -> None:
        # The real spec spelling is single-L 'canceled' — it must drive the FSM
        # (the old double-L 'cancelled' would be dropped as unknown).
        from layerlens.instrument.adapters.protocols.a2a.task_lifecycle import (
            TaskState,
            TaskStateMachine,
        )

        assert TaskState("canceled") is TaskState.CANCELED
        assert TaskState("input-required") is TaskState.INPUT_REQUIRED
        assert TaskState("auth-required") is TaskState.AUTH_REQUIRED
        assert TaskState("rejected") is TaskState.REJECTED
        fsm = TaskStateMachine("t")
        assert fsm.transition(TaskState.WORKING)
        assert fsm.transition(TaskState.REJECTED), "working -> rejected must be a valid terminal transition (D5)"
        assert fsm.is_terminal

    def test_real_canceled_state_from_wire(self) -> None:
        events = self._drive_server_with_state(a2a_types.TaskState.canceled)
        last = _by_type(events, "a2a.task.updated")[-1]
        assert last["status"] == "canceled", f"canceled task labeled {last['status']!r} (single-L spelling, D5)"
