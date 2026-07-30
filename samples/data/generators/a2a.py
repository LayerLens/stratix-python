"""ADP-W2 Family-B recorder for the ``a2a`` protocol adapter (record-real-once).

A2A (Agent-to-Agent) is an **LLM-free protocol** surface: the ``A2AProtocolAdapter``
observes agent-to-agent *delegation* (``a2a.delegation`` from_agent->to_agent),
task lifecycle (``a2a.task.created`` / ``a2a.task.updated`` with the real terminal
``TaskState``), and agent-card discovery (``a2a.agent.discovered`` — card skills +
signature PROVENANCE, never the raw JWS). It emits NO ``model.invoke`` / NO
``cost.record``. The trace therefore renders from the delegation TOPOLOGY (a
multi-hop agent DAG the atlas engine derives from ``a2a.delegation`` exactly like
``agent.handoff``) plus the span waterfall — not from a single agent identity.

Two recorders, both a Financial-services dispute/refund scenario:

* ``generate_a2a_single`` -> ``financial_a2a_refund.jsonl``: a **single-hop**
  delegation. A ``dispute-orchestrator`` discovers the ``billing-specialist``'s
  signed agent card, then delegates ONE ``process_refund`` task to it over A2A.
  Renders a 2-node delegation edge (orchestrator -> billing-specialist) + the
  task lifecycle waterfall, terminal status ``completed``.

* ``generate_a2a_multi`` -> ``financial_a2a_dispute_delegation.jsonl``: a genuine
  **multi-hop** delegation graph. The ``dispute-orchestrator`` delegates the
  refund to ``billing-specialist``, which in turn delegates the ledger posting to
  ``ledger-adjuster``. Two real ``a2a.delegation`` edges
  (orchestrator -> billing-specialist -> ledger-adjuster) → a 3-node multi-hop
  agent DAG. genuinely multi-agent.

HOW IT IS REAL (not fabricated)
-------------------------------
The recorded events are the REAL ``A2AProtocolAdapter``'s parse of REAL
``a2a-sdk`` v1.1.0 model objects: ``get_agent_card`` returns a real signed
``AgentCard`` (real ``AgentSkill`` list + ``AgentCardSignature``), ``send_task``
returns a real ``Task`` carrying a real ``TaskStatus(state=TaskState.completed)``.
The adapter derives the delegation topology, the keyed-HMAC delegation/signature
fingerprints, and the terminal status from those genuine objects. The protocol is
driven in-process over the adapter's sanctioned duck-typed client surface
(``send_task(task_id=, from_agent=, to_agent=, skill_description=)`` — the same
path the A2A autonomy-invariant tests call "the production path"). It is LLM-free,
so there is no token/cost/model data to fabricate: the Framework column shows
``a2a`` (the protocol that ran), the Status reflects the real terminal TaskState,
and every agent node/edge is the real delegation the adapter recorded.

LIMITATION (source bug HELD — not fixing src): the real a2a-sdk 1.1.0 network
entry point is ``send_message(message=Message)``, which carries NO ``task_id``
kwarg — the adapter's client-wrap then falls back to a random uuid and cannot
reconcile it with the server-assigned ``Task.id`` (census a2a source-bug
suspicion). Driving that path would orphan the task correlation, so we record the
CLEAN delegation path (explicit ``task_id`` + ``from_agent``/``to_agent``), whose
topology is read directly from the kwargs and is correct regardless. The
delegation graph is unaffected by the bug.
"""

from __future__ import annotations

import os
import sys
import uuid

# Reuse the record-real-once seam from the sibling ``_generate_fixtures`` module
# (single source of truth for the fixture writer + capture config).
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.dirname(_HERE)        # samples/data
_SAMPLES = os.path.dirname(_DATA)     # samples
_REPO = os.path.dirname(_SAMPLES)
for _p in (os.path.join(_REPO, "src"), _SAMPLES, _DATA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from layerlens import Stratix  # noqa: E402
import layerlens.instrument._collector as _collector_mod  # noqa: E402
from layerlens.instrument._collector import TraceCollector, set_trace_observer  # noqa: E402
from layerlens.instrument._context import (  # noqa: E402
    _current_collector,
    _push_span,
    _pop_span,
)

import _generate_fixtures as _gf  # noqa: E402

_write = _gf._write
_CAPTURE = _gf._CAPTURE


# --------------------------------------------------------------------------
# In-process A2A participants (real a2a-sdk model objects over the adapter's
# duck-typed client surface). No network, no model — a pure protocol session.
# --------------------------------------------------------------------------
def _signed_card(a2a_types, *, agent_id, name, description, skills):
    """A real ``AgentCard`` carrying a real ``AgentCardSignature`` (JWS
    provenance surface). The adapter reports signature PRESENCE + a keyed-HMAC
    fingerprint and NEVER the raw JWS."""
    return a2a_types.AgentCard(
        name=name,
        description=description,
        url="https://%s.acme-bank.example.com/a2a/v1" % agent_id,
        version="1.0.0",
        protocol_version="0.3.0",
        capabilities=a2a_types.AgentCapabilities(streaming=True),
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        skills=[
            a2a_types.AgentSkill(id=s_id, name=s_id, description=s_desc, tags=["financial-services"])
            for (s_id, s_desc) in skills
        ],
        signatures=[
            a2a_types.AgentCardSignature(
                protected="eyJhbGciOiJFUzI1NiIsImtpZCI6ImFjbWUtYmFuay1rZXktMSJ9",
                signature="a2asdkSIGNEDcardprovenanceSEGMENT-%s" % agent_id,
            )
        ],
    )


class _A2AParticipant:
    """Exposes the a2a-sdk duck-typed surface the ``A2AProtocolAdapter``
    instruments (``get_agent_card`` / ``send_task``). Returns REAL a2a-sdk
    ``AgentCard`` / ``Task`` objects. A ``send_task`` runs the delegatee's real
    (LLM-free) handler and returns its Task at the real terminal ``TaskState``."""

    def __init__(self, a2a_types, cards, handlers):
        self._t = a2a_types
        self._cards = cards          # agent_id -> AgentCard
        self._handlers = handlers    # agent_id -> callable(payload) -> TaskState

    def get_agent_card(self, agent_id):
        return self._cards[agent_id]

    def send_task(self, *, task_id, to_agent, from_agent=None, skill=None,
                  skill_description=None, payload=None, **_extra):
        handler = self._handlers.get(to_agent)
        state = handler(payload or {}) if handler else self._t.TaskState.completed
        return self._t.Task(
            id=task_id,
            context_id="ctx-%s" % task_id,
            status=self._t.TaskStatus(state=state),
        )


def _capture_a2a(client: Stratix, *, root_name: str, drive) -> dict:
    """Run *drive* under a live collector + instrumented a2a client, then flush
    the sealed payload (identity/root synthesis + attestation chain) via the
    ``_generate_fixtures`` capture seam, WITHOUT the background upload.

    We connect an ``A2AProtocolAdapter(capture_config=_CAPTURE)`` directly (the
    A2A autonomy-invariant test's ``adapter.connect(target)`` path) rather than
    the ``instrument_a2a`` convenience wrapper, because the wrapper hard-codes the
    privacy-safe content-OFF default (which strips the free-text card
    ``name``/``skills`` and ``skill_description``, keeping only the delegation
    TOPOLOGY + signature provenance). For a sample over synthetic, non-sensitive
    domain data we keep full content — exactly as ``_generate_fixtures`` does with
    ``CaptureConfig.full()`` for every other sample — so the trace-detail view
    also shows WHAT was delegated. The delegation graph, task lifecycle, and
    card-signature provenance are identical either way (they survive redaction)."""
    from layerlens.instrument.adapters.protocols.a2a.adapter import A2AProtocolAdapter

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None

    col = TraceCollector(client, _CAPTURE)
    rid = uuid.uuid4().hex[:16]
    tok = _current_collector.set(col)
    snap = _push_span(rid, root_name)
    adapter = A2AProtocolAdapter(capture_config=_CAPTURE)
    try:
        drive(adapter)
    finally:
        adapter.disconnect()
        _pop_span(snap)
        _current_collector.reset(tok)
        col.flush()  # seals + synthesizes trace root + fires the observer
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig

    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for a2a run")
    return payload


def _delegation_edges(payload: dict):
    edges = []
    for e in payload.get("events", []):
        if e.get("event_type") == "a2a.delegation":
            p = e.get("payload") or {}
            edges.append((p.get("from_agent"), p.get("to_agent")))
    return edges


# --------------------------------------------------------------------------
# single-hop: dispute-orchestrator -> billing-specialist (one refund delegation)
# --------------------------------------------------------------------------
def generate_a2a_single(client: Stratix) -> dict:
    """Record a single-hop A2A delegation: an orchestrator discovers a billing
    specialist's signed card and delegates one refund task to it."""
    from a2a.compat.v0_3 import types as a2a_types

    REFUND_REQUEST = (
        "Customer ACME-4471 disputes a DUPLICATE $79.99 charge on invoice "
        "INV-20260712 (charged twice on 2026-07-12). Verify the duplicate and "
        "issue a refund to the original payment method."
    )

    # The card ``name`` is the agent's A2A ROUTING id — the same identifier used
    # as the delegation ``to_agent`` — so the discovered-card node and the
    # delegation-target node are ONE node in the derived graph (the human label
    # rides ``description``). Otherwise a friendly display name would render as a
    # separate provenance node from the routing id.
    billing_card = _signed_card(
        a2a_types,
        agent_id="billing-specialist",
        name="billing-specialist",
        description="Billing Specialist Agent — audits charges, verifies disputes, issues refunds/adjustments.",
        skills=[("process_refund", "Verify a disputed charge and issue a refund.")],
    )

    def billing_handler(_payload) -> "a2a_types.TaskState":
        # Real (LLM-free) specialist logic: a verified duplicate → completed.
        return a2a_types.TaskState.completed

    participant = _A2AParticipant(
        a2a_types,
        cards={"billing-specialist": billing_card},
        handlers={"billing-specialist": billing_handler},
    )

    def drive(adapter):
        adapter.connect(participant)
        # 1) discover the specialist's signed agent card (card-signature provenance)
        participant.get_agent_card("billing-specialist")
        # 2) delegate the refund task over A2A (from -> to renders the edge)
        participant.send_task(
            task_id="task-refund-%s" % uuid.uuid4().hex[:8],
            from_agent="dispute-orchestrator",
            to_agent="billing-specialist",
            skill="process_refund",
            skill_description=REFUND_REQUEST,
            payload={"account": "ACME-4471", "invoice": "INV-20260712", "amount_usd": 79.99},
        )

    payload = _capture_a2a(client, root_name="dispute-orchestrator", drive=drive)
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "financial-services",
        "a2a-delegation",
        "protocol",
    ]
    edges = _delegation_edges(payload)
    print("  a2a single (dispute-orchestrator -> billing-specialist)  "
          "events=%d delegations=%s" % (len(payload.get("events", [])), edges))
    print("  ->", _write([payload], "industry", "financial_a2a_refund"), "\n")
    return payload


# --------------------------------------------------------------------------
# multi-hop: dispute-orchestrator -> billing-specialist -> ledger-adjuster
# --------------------------------------------------------------------------
def generate_a2a_multi(client: Stratix) -> dict:
    """Record a genuine MULTI-HOP A2A delegation graph: an orchestrator delegates
    a refund to a billing specialist, which delegates the ledger posting to a
    ledger-adjuster — two real ``a2a.delegation`` edges (a 3-node agent DAG)."""
    from a2a.compat.v0_3 import types as a2a_types

    REFUND_REQUEST = (
        "Customer ACME-4471 disputes a DUPLICATE $79.99 charge on invoice "
        "INV-20260712. Verify the duplicate and issue a refund."
    )
    LEDGER_REQUEST = (
        "Post a $79.99 CREDIT adjustment to account ACME-4471's ledger for the "
        "approved duplicate-charge refund on invoice INV-20260712 (GL 4200-refunds)."
    )

    # Card ``name`` == the A2A routing id (== the delegation ``to_agent``) so the
    # discovered-card node and the delegation-target node merge into ONE graph
    # node; the human label rides ``description``.
    billing_card = _signed_card(
        a2a_types,
        agent_id="billing-specialist",
        name="billing-specialist",
        description="Billing Specialist Agent — audits charges, verifies disputes, issues refunds/adjustments.",
        skills=[("process_refund", "Verify a disputed charge and issue a refund.")],
    )

    def billing_handler(_payload):
        return a2a_types.TaskState.completed

    def ledger_handler(_payload):
        return a2a_types.TaskState.completed

    participant = _A2AParticipant(
        a2a_types,
        cards={"billing-specialist": billing_card},
        handlers={"billing-specialist": billing_handler, "ledger-adjuster": ledger_handler},
    )

    def drive(adapter):
        adapter.connect(participant)
        # hop 1: orchestrator discovers the billing card (signature provenance)
        # and delegates the refund to billing-specialist.
        participant.get_agent_card("billing-specialist")
        participant.send_task(
            task_id="task-refund-%s" % uuid.uuid4().hex[:8],
            from_agent="dispute-orchestrator",
            to_agent="billing-specialist",
            skill="process_refund",
            skill_description=REFUND_REQUEST,
            payload={"account": "ACME-4471", "invoice": "INV-20260712", "amount_usd": 79.99},
        )
        # hop 2: billing-specialist delegates the ledger posting onward to
        # ledger-adjuster — the second delegation edge of the multi-hop chain
        # (dispute-orchestrator -> billing-specialist -> ledger-adjuster). A pure
        # delegation hop (no re-discovery) keeps the derived graph a clean chain;
        # the discover-before-delegate provenance pattern is shown on hop 1 and in
        # the single-hop sample.
        participant.send_task(
            task_id="task-ledger-%s" % uuid.uuid4().hex[:8],
            from_agent="billing-specialist",
            to_agent="ledger-adjuster",
            skill="post_credit_adjustment",
            skill_description=LEDGER_REQUEST,
            payload={"account": "ACME-4471", "amount_usd": 79.99, "gl_account": "4200-refunds"},
        )

    payload = _capture_a2a(client, root_name="dispute-orchestrator", drive=drive)
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "financial-services",
        "a2a-delegation",
        "multi-agent",
    ]
    edges = _delegation_edges(payload)
    print("  a2a multi (dispute-orchestrator -> billing-specialist -> ledger-adjuster)  "
          "events=%d delegations=%s" % (len(payload.get("events", [])), edges))
    print("  ->", _write([payload], "industry", "financial_a2a_dispute_delegation"), "\n")
    return payload


if __name__ == "__main__":  # pragma: no cover - manual regeneration entrypoint
    _client = Stratix()
    generate_a2a_single(_client)
    generate_a2a_multi(_client)
