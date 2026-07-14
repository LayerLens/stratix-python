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

from a2a.client.errors import A2AClientTimeoutError  # noqa: E402

from layerlens.attestation._verify import verify_chain  # noqa: E402
from layerlens.instrument._context import _current_collector  # noqa: E402
from layerlens.attestation._envelope import HashScope, AttestationEnvelope  # noqa: E402
from layerlens.instrument._collector import TraceCollector  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.protocols.a2a.adapter import A2AProtocolAdapter  # noqa: E402

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
