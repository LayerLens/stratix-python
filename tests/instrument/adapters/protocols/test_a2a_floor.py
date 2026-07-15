"""Offline attestation + real-error-shape + redaction floor for the A2A protocol adapter.

a2a already has strong offline coverage — real ``a2a-sdk`` 1.1.0 method-vocabulary
+ delegation-provenance + card-signature + late-SSE + terminal-state invariants
(``test_a2a_invariants.py``), a terminal-failure error suite
(``test_protocol_terminal_failures.py``), and a redaction sweep
(``test_protocol_redaction.py`` / ``test_no_content_sweep.py``). Its ONE true
offline gap is **attestation**: no test reconstructs the sealed trace's hash
chain and calls ``verify_chain`` over a real a2a event chain. This floor closes
that, and consolidates the other two protocol-floor pillars so a regression in
any of them fails in plain CI (no creds, no network):

* Attestation — a REAL a2a delegation lifecycle (real ``send_task`` client-wrap
                path: ``a2a.task.created`` -> ``a2a.delegation`` -> ``a2a.task.updated``)
                is flushed through the REAL collector/upload seam; the uploaded
                trace's attestation chain reconstructs and ``verify_chain`` returns
                valid, one envelope per event, root_hash present. A TAMPER control
                breaks link 1 and proves the check is not vacuous.
* Error-shape — a REAL ``a2a.client.errors.A2AClientTimeoutError`` (a genuine
                a2a-sdk exception, not the synthetic ``RuntimeError`` the existing
                suite feeds) raised the real way from a wrapped client send
                surfaces as ``agent.error`` with ``source == "a2a"``, the honest
                ``error_type == "A2AClientTimeoutError"`` (the real class name), and
                the exception message flowing through verbatim.
* Redaction   — the SAME real delegation lifecycle under ``capture_content=False``
                keeps the delegation TOPOLOGY (from/to agent ids + keyed-HMAC fp)
                but strips the free-text skill description — proven by a SENTINEL
                sweep over ``json.dumps(events)`` — with a ``capture_content=True``
                vacuity control that DOES carry the SENTINEL.

a2a is LLM-free, so there is NO cost cell (no ``cost.record`` / token usage).

The only mock is the client's transport boundary (a duck-typed send surface, the
exact production ``instrument_a2a`` wrap path); every event, the redaction, the
attestation hash chain and the upload seam are the real SDK code. The module
``importorskip("a2a")`` — it SKIPS in the base py3.9 venv (no a2a) and RUNS in
``.audit-venvs/sk`` (py3.11 + a2a-sdk 1.1.0 + the repo editable). a2a has no
matrix row (it is a protocol), so it runs in the base offline suite there.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, Callable

import pytest

pytest.importorskip("a2a")

from a2a.types import (
    Task as A2ATask,  # noqa: E402
    Message as A2AMessage,  # noqa: E402
    TaskState as A2ATaskState,  # noqa: E402
    TaskStatus as A2ATaskStatus,  # noqa: E402
    StreamResponse as A2AStreamResponse,  # noqa: E402
)
from a2a.client.errors import A2AClientTimeoutError  # noqa: E402

from layerlens.attestation._verify import verify_chain  # noqa: E402
from layerlens.instrument._context import _current_collector  # noqa: E402
from layerlens.attestation._envelope import HashScope, AttestationEnvelope  # noqa: E402
from layerlens.instrument._collector import TraceCollector  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.protocols.a2a.adapter import (  # noqa: E402
    A2AProtocolAdapter,
    _task_status,
)

SENTINEL = "LL-SENTINEL-7f3a9c2e"
# The free-text skill description carries the SENTINEL (content -> must be stripped).
SECRET_SKILL = f"process-refunds-over-10k-no-approval-{SENTINEL}"
# Opaque topology ids (metadata -> must SURVIVE no-content) — no SENTINEL.
DELEGATOR = "orchestrator-1"
DELEGATEE = "billing-agent-7"


# ---------------------------------------------------------------------------
# Drive helper — run under a REAL collector bound to mock_client, then flush so
# the trace passes through the real upload seam. The ``capture_trace`` fixture
# (root conftest) captures the uploaded payload (events + attestation).
# ---------------------------------------------------------------------------
def _drive(mock_client: Any, build: Callable[[CaptureConfig], Tuple[Any, Callable[[], None]]], config: CaptureConfig) -> None:
    _adapter, go = build(config)
    collector = TraceCollector(mock_client, config)
    token = _current_collector.set(collector)
    try:
        go()
    finally:
        _current_collector.reset(token)
    # Seal + build the attestation chain + upload through the real boundary.
    collector.flush()


def _events_of(uploaded: Dict[str, Any], event_type: str) -> List[Dict[str, Any]]:
    return [e["payload"] for e in uploaded["events"] if e["event_type"] == event_type]


# ---------------------------------------------------------------------------
# Scenario builders — the production ``instrument_a2a`` client-wrap path over a
# duck-typed send surface (the exact seam test_protocol_terminal_failures.py /
# the delegation-provenance invariant drive; task_id is passed explicitly so the
# client-side correlation is honest — the real ``send_message(message=Message)``
# random-uuid correlation gap is NOT exercised here).
# ---------------------------------------------------------------------------
def _delegation_lifecycle(config: CaptureConfig) -> Tuple[Any, Callable[[], None]]:
    """orchestrator delegates a refund task to a specialist billing agent.

    Emits a2a.task.created -> a2a.delegation -> a2a.task.updated(completed).
    """
    adapter = A2AProtocolAdapter(capture_config=config)
    target = type("Cli", (), {"send_task": staticmethod(lambda **kw: {"task_id": "t1", "status": "completed"})})()
    adapter.connect(target=target)

    def go() -> None:
        target.send_task(
            task_id="t1",
            from_agent=DELEGATOR,
            to_agent=DELEGATEE,
            skill_description=SECRET_SKILL,
        )

    return adapter, go


# ---------------------------------------------------------------------------
# Offline attestation-chain verification over a real a2a delegation lifecycle
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_over_real_a2a_lifecycle(self, mock_client, capture_trace):
        _drive(mock_client, _delegation_lifecycle, CaptureConfig.full())

        events = capture_trace["events"]
        assert events, "a real a2a delegation lifecycle must flush a non-empty trace"
        chain = (capture_trace["attestation"] or {}).get("chain") or {}
        raw = chain.get("events") or []
        envelopes = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured for the real a2a trace"
        assert len(envelopes) == len(events), (
            f"attestation chain has {len(envelopes)} envelopes for {len(events)} events"
        )
        assert (capture_trace["attestation"] or {}).get("root_hash") is not None

        result = verify_chain(envelopes)
        assert result.valid, f"attestation chain invalid: {result.error}"

        # Vacuity control: verify_chain must REJECT a broken interior link, proving
        # the pass above is not trivially true.
        assert len(envelopes) >= 2, "need >= 2 envelopes to break an interior link"
        tampered = list(envelopes)
        tampered[1] = AttestationEnvelope(
            hash=tampered[1].hash,
            scope=tampered[1].scope,
            previous_hash="sha256:deadbeef-not-the-prior-hash",
        )
        broken = verify_chain(tampered)
        assert not broken.valid and broken.break_index == 1, "verify_chain failed to detect a broken link"


# ---------------------------------------------------------------------------
# Real error-shape floor (a real a2a-sdk exception, raised the real way)
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_real_a2a_client_error_surfaces_as_agent_error(self, mock_client, capture_trace):
        # A genuine a2a-sdk exception — the shape a real A2A client raises when a
        # peer is unreachable. NOT the synthetic RuntimeError the existing suite feeds.
        real_message = "peer billing-agent-7 timed out after 30s"
        err = A2AClientTimeoutError(real_message)
        # Prove it is the real class, not a hand-rolled stand-in.
        assert type(err).__name__ == "A2AClientTimeoutError"
        assert isinstance(err, Exception)
        assert str(err) == real_message

        def build(config: CaptureConfig) -> Tuple[Any, Callable[[], None]]:
            adapter = A2AProtocolAdapter(capture_config=config)

            def _boom(**kw: Any) -> Any:
                raise err

            target = type("Cli", (), {"send_task": staticmethod(_boom)})()
            adapter.connect(target=target)

            def go() -> None:
                with pytest.raises(A2AClientTimeoutError):
                    target.send_task(task_id="t1", from_agent=DELEGATOR, to_agent=DELEGATEE)

            return adapter, go

        _drive(mock_client, build, CaptureConfig.full())

        errors = _events_of(capture_trace, "agent.error")
        assert len(errors) == 1, f"expected exactly one agent.error, saw {[e['event_type'] for e in capture_trace['events']]}"
        payload = errors[0]
        # source stamps the protocol so the atlas 'error' derivation lights up (S12/F4).
        assert payload["source"] == "a2a"
        # The honest error_type IS the real SDK class name (bite: a synthetic/absent
        # class name, or a mislabeled a2a_task_failed sentinel, fails here).
        assert payload["error_type"] == "A2AClientTimeoutError"
        # The REAL exception message flows through verbatim (bite: dropped/mangled).
        assert payload["error"] == real_message

        # And the terminal task update honestly records the failure (never completed).
        updated = _events_of(capture_trace, "a2a.task.updated")
        assert updated, "no a2a.task.updated emitted for the failed send"
        assert updated[-1]["status"] == "failed"
        assert updated[-1]["status"] != "completed"


# ---------------------------------------------------------------------------
# Redaction content-absence over the real delegation lifecycle
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_content_present_when_capturing(self, mock_client, capture_trace):
        """Vacuity control: with capture_content=True the SAME real lifecycle DOES
        carry the SENTINEL and the free-text skill it rides on."""
        _drive(mock_client, _delegation_lifecycle, CaptureConfig(capture_content=True))

        events = capture_trace["events"]
        assert SENTINEL in json.dumps(events), "control run must carry the SENTINEL when capturing content"
        deleg = _events_of(capture_trace, "a2a.delegation")
        assert deleg, "no a2a.delegation event emitted — test would be vacuous"
        assert deleg[0].get("skill_description") == SECRET_SKILL

    def test_content_absent_when_not_capturing(self, mock_client, capture_trace):
        """capture_content=False strips the free-text skill description — and the
        SENTINEL — but the delegation TOPOLOGY + keyed-HMAC fp SURVIVE (provenance
        stays auditable under privacy-on; mirrors agent.handoff from/to)."""
        _drive(mock_client, _delegation_lifecycle, CaptureConfig(capture_content=False))

        events = capture_trace["events"]
        assert events, "the lifecycle must still emit structural events without content"

        # 1) SENTINEL sweep over the serialized trace (rides skill_description in
        #    both a2a.task.created.request and a2a.delegation).
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: skill description SENTINEL survived capture_content=False"

        # 2) The free-text skill description key is gone from the delegation payload.
        deleg = _events_of(capture_trace, "a2a.delegation")
        assert deleg, "delegation topology must still be emitted without content"
        d = deleg[0]
        assert "skill_description" not in d, "a2a.delegation leaked 'skill_description' under capture_content=False"

        # 3) Topology + fp are provenance metadata -> they SURVIVE (A15 overturn).
        assert d.get("from_agent") == DELEGATOR, "delegator id stripped under no-content (A15 provenance loss)"
        assert d.get("to_agent") == DELEGATEE, "delegatee id stripped under no-content (A15 provenance loss)"
        assert d.get("target_agent") == DELEGATEE
        assert str(d.get("delegation_fp", "")).startswith("sha256:"), "delegation fp stripped under no-content (A15)"

        # 4) And the created event dropped its whole content 'request' summary.
        created = _events_of(capture_trace, "a2a.task.created")
        assert created, "no a2a.task.created emitted"
        assert "request" not in created[0], "a2a.task.created leaked the content 'request' summary under no-content"


# ---------------------------------------------------------------------------
# Client-side task correlation over a REAL a2a-sdk send_message(message=Message)
# ---------------------------------------------------------------------------
# The delegation lifecycle above passes task_id explicitly (the duck-typed
# send_task helper), so it never exercises the send_message correlation gap. A
# real a2a-sdk send_message carries a protobuf ``Message`` and NO task_id kwarg,
# and the server assigns the Task.id in the RESPONSE. This floor drives that real
# shape end-to-end through the collector/upload seam and pins the correlation
# contract ateam's normalizer depends on (a single a2a.task.id):
#   * a2a.task.created carries a NON-random id derived from the Message (the
#     Message's own task id when CONTINUING, else its messageId) + request_id.
#   * a2a.task.updated carries the SERVER-assigned Task.id from the response, with
#     request_id bridging back to created.
# BITE (proven by reverting only adapter.py): on the buggy code the created id is
# a random uuid4().hex[:16] and the updated id is that same random id — never the
# server Task.id — so every assertion below goes RED.

# Real, human-legible ids (NOT 16-hex uuids) so a random-uuid fallback can never
# accidentally match, and created (client-derived) vs updated (server) differ.
CLIENT_MESSAGE_ID = "msg-client-abc123"
SERVER_TASK_ID = "srv-task-9f2b1d"
CLIENT_MESSAGE_ID_2 = "msg-client-def456"
CONTINUING_TASK_ID = "task-prior-7a1c"
SERVER_TASK_ID_2 = "srv-task-c4e8aa"


def _send_message_new_task(config: CaptureConfig) -> Tuple[Any, Callable[[], None]]:
    """A REAL a2a-sdk ``send_message(message=Message)`` for a NEW task.

    The client knows no server id yet (the Message carries only a messageId); the
    response is a bare a2a ``Task`` carrying the server-assigned id (exercises the
    ``result.id`` extraction branch).
    """
    adapter = A2AProtocolAdapter(capture_config=config)
    msg = A2AMessage(message_id=CLIENT_MESSAGE_ID)
    server_task = A2ATask(id=SERVER_TASK_ID)
    target = type("Cli", (), {"send_message": staticmethod(lambda message: server_task)})()
    adapter.connect(target=target)

    def go() -> None:
        target.send_message(message=msg)

    return adapter, go


def _send_message_continuing_task(config: CaptureConfig) -> Tuple[Any, Callable[[], None]]:
    """A REAL ``send_message`` CONTINUING an existing task.

    The Message carries its OWN task id (it is a follow-up turn on a live server
    task); the response is a ``StreamResponse`` wrapping a Task (the shape the real
    client yields — exercises the ``result.task.id`` extraction branch).
    """
    adapter = A2AProtocolAdapter(capture_config=config)
    msg = A2AMessage(message_id=CLIENT_MESSAGE_ID_2, task_id=CONTINUING_TASK_ID)
    server_resp = A2AStreamResponse(task=A2ATask(id=SERVER_TASK_ID_2))
    target = type("Cli", (), {"send_message": staticmethod(lambda message: server_resp)})()
    adapter.connect(target=target)

    def go() -> None:
        target.send_message(message=msg)

    return adapter, go


class TestClientSendMessageCorrelation:
    def test_new_task_derives_message_id_and_reconciles_server_id(self, mock_client, capture_trace):
        # Sanity: the test is only meaningful if the three ids are distinct.
        assert CLIENT_MESSAGE_ID != SERVER_TASK_ID

        _drive(mock_client, _send_message_new_task, CaptureConfig.full())

        created = _events_of(capture_trace, "a2a.task.created")
        assert created, "send_message(message=Message) emitted no a2a.task.created"
        c = created[0]
        # The provisional id is DERIVED from the Message (its messageId for a new
        # task), NEVER a random uuid4 (the bug: no task_id kwarg → random id).
        assert c["task_id"] == CLIENT_MESSAGE_ID, (
            f"created.task_id={c['task_id']!r} is not the message-derived id — the "
            "client-side correlation fell back to a random uuid (the correlation bug)"
        )
        # request_id (the messageId) is stamped so created↔updated can bridge.
        assert c.get("request_id") == CLIENT_MESSAGE_ID, (
            "a2a.task.created did not stamp request_id=<messageId> (created↔updated bridge missing)"
        )

        updated = _events_of(capture_trace, "a2a.task.updated")
        assert updated, "send_message emitted no a2a.task.updated"
        u = updated[-1]
        # The terminal event carries the SERVER-assigned Task.id from the response,
        # so the client trace correlates with the server trace (ateam keys on a
        # single a2a.task.id). BITE: buggy code emits the random provisional id here.
        assert u["task_id"] == SERVER_TASK_ID, (
            f"updated.task_id={u['task_id']!r} is not the server Task.id — the terminal "
            "event never carried the server-assigned id (client↔server correlation gap)"
        )
        # request_id bridges the (server-id) updated event back to the created event.
        assert u.get("request_id") == CLIENT_MESSAGE_ID, (
            "a2a.task.updated did not carry request_id bridging back to created"
        )
        # And the two events legitimately carry DIFFERENT task ids (provisional vs
        # server) — reconciliation actually happened, it is not a no-op coincidence.
        assert c["task_id"] != u["task_id"]

    def test_continuing_task_prefers_message_task_id(self, mock_client, capture_trace):
        assert CONTINUING_TASK_ID != SERVER_TASK_ID_2 != CLIENT_MESSAGE_ID_2

        _drive(mock_client, _send_message_continuing_task, CaptureConfig.full())

        created = _events_of(capture_trace, "a2a.task.created")
        assert created, "send_message emitted no a2a.task.created"
        c = created[0]
        # CONTINUING a task: the Message's OWN task id is the provisional id (a new
        # random uuid would break the follow-up turn's correlation to its own task).
        assert c["task_id"] == CONTINUING_TASK_ID, (
            f"created.task_id={c['task_id']!r} did not prefer the Message's own task_id "
            "when continuing an existing server task"
        )
        assert c.get("request_id") == CLIENT_MESSAGE_ID_2

        updated = _events_of(capture_trace, "a2a.task.updated")
        assert updated, "send_message emitted no a2a.task.updated"
        u = updated[-1]
        # Server id pulled from result.task.id (StreamResponse wrap).
        assert u["task_id"] == SERVER_TASK_ID_2, (
            f"updated.task_id={u['task_id']!r} is not the server Task.id from result.task.id"
        )
        assert u.get("request_id") == CLIENT_MESSAGE_ID_2


def _send_message_completed(config: CaptureConfig) -> Tuple[Any, Callable[[], None]]:
    """A REAL ``send_message`` whose response Task carries a real protobuf
    ``TaskStatus(state=TASK_STATE_COMPLETED)`` — the shape a real a2a server
    returns. On protobuf, ``status.state`` is the INT enum ``3``, not a string."""
    adapter = A2AProtocolAdapter(capture_config=config)
    msg = A2AMessage(message_id=CLIENT_MESSAGE_ID)
    server_task = A2ATask(id=SERVER_TASK_ID, status=A2ATaskStatus(state=A2ATaskState.TASK_STATE_COMPLETED))
    target = type("Cli", (), {"send_message": staticmethod(lambda message: server_task)})()
    adapter.connect(target=target)

    def go() -> None:
        target.send_message(message=msg)

    return adapter, go


class TestProtobufTaskStatusMapping:
    """A real a2a-sdk Task is protobuf: ``status.state`` is an INT enum (e.g. 3),
    which ``str()`` turned into the meaningless ``"3"`` — a real run emitted
    ``a2a.task.updated.status == "3"`` (never matching the ``completed``/``failed``
    the FSM + downstream readers expect). The duck-typed doubles (string statuses)
    masked it. The adapter must map the protobuf enum to the canonical status."""

    def test_task_status_maps_protobuf_enum_to_canonical_string(self) -> None:
        # The exact real shape: TaskStatus(state=<int enum>). Bite: str(3) == "3".
        completed = A2ATask(id="t", status=A2ATaskStatus(state=A2ATaskState.TASK_STATE_COMPLETED))
        assert _task_status(completed) == "completed", (
            f"protobuf TASK_STATE_COMPLETED must map to 'completed', got "
            f"{_task_status(completed)!r} (the raw int-enum string is the bug)"
        )
        failed = A2ATask(id="t", status=A2ATaskStatus(state=A2ATaskState.TASK_STATE_FAILED))
        assert _task_status(failed) == "failed", f"got {_task_status(failed)!r}"
        working = A2ATask(id="t", status=A2ATaskStatus(state=A2ATaskState.TASK_STATE_WORKING))
        assert _task_status(working) == "working", f"got {_task_status(working)!r}"

    def test_send_message_emits_canonical_terminal_status(self, mock_client, capture_trace):
        """End-to-end: a completed real-protobuf Task response yields
        ``a2a.task.updated.status == "completed"`` (not ``"3"``)."""
        _drive(mock_client, _send_message_completed, CaptureConfig.full())
        updated = _events_of(capture_trace, "a2a.task.updated")
        assert updated, "send_message emitted no a2a.task.updated"
        u = updated[-1]
        assert u["status"] == "completed", (
            f"a2a.task.updated.status={u['status']!r} — a real protobuf Task's int "
            "enum leaked as a raw string instead of the canonical 'completed'"
        )
