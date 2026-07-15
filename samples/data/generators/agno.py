"""ADP-W2 Family-B recorder for the ``agno`` adapter (record-real-once).

Records TWO real, fully-instrumented ``agno`` runs and writes each as a sealed
real-trace fixture under ``samples/data/traces/industry/``:

* ``generate_agno_single`` -> ``insurance_claims_agno.jsonl``: a single agno
  ``claims-intake-agent`` (an ``agno.agent.Agent`` backed by ``OpenAIChat``) that
  runs a real tool-use turn — it calls the ``lookup_policy`` function tool to
  fetch the policy's coverage/deductible/exclusions, then answers whether the
  claim is covered. Renders a single honest agent node (Agent column =
  ``claims-intake-agent``) with the real ``model.invoke`` / ``cost.record`` /
  ``tool.call`` / ``tool.result`` events of the turn.

* ``generate_agno_multi`` -> ``insurance_underwriting_agno_team.jsonl``: a genuine
  multi-agent run — an ``underwriting-team`` (an ``agno.team.Team``) whose leader
  delegates the risk assessment to a ``risk-analyst`` member and the compliance
  check to a ``compliance-checker`` member. agno's ``delegate_task_to_member``
  tool really invokes ``member_agent.run()`` (agno 2.6.x
  ``team/_default_tools.py``), and the adapter recursively instruments the two
  members, so the trace carries THREE distinct honest agent identities — each
  with its OWN real ``model.invoke`` / ``cost.record`` — plus one
  ``agent.handoff`` per delegation (underwriting-team -> risk-analyst,
  underwriting-team -> compliance-checker). It renders as a multi-agent DAG
  (3 agent nodes + 2 handoff edges, Agent column ``multi-agent``).

Both are recorded through the REAL ``AgnoAdapter`` (wraps ``Agent.run`` /
``arun`` with a per-run ContextVar collector that flushes on run-end); the flush
is observed via the ``_generate_fixtures`` capture seam (``set_trace_observer`` +
a no-op ``enqueue_upload``) so the sealed payload — real per-node
``model.invoke``/``cost.record`` (gpt-4o-mini pricing is applied by the shared
price-on-emit chokepoint, so ``cost.record.cost_usd`` is a real non-None figure)
+ an intact attestation chain — is captured but never uploaded during
generation. The samples upload the captured fixtures themselves at run time.

Nothing is fabricated: the Framework column shows ``agno`` (the framework that
really ran), the token/cost fields are real, and the multi-agent nodes / handoff
edges are the real declared members / real ``delegate_task_to_member``
delegations the framework emitted (``to_agent`` is the real ``member_id`` agno
passed to the delegation tool).

ADAPTER/VERSION-DRIFT SHIM (multi only) — PING the SDK owner:
    The AgnoAdapter classifies a team delegation into an ``agent.handoff`` edge
    only when the delegation tool name matches ``_is_transfer_tool`` — which today
    matches ``transfer``/``forward`` variants (``transfer_task_to_member`` /
    ``forward_task_to_member``). Installed agno 2.6.x names its delegation tool
    ``delegate_task_to_member`` / ``delegate_task_to_members``, which
    ``_is_transfer_tool`` does NOT match, so an un-shimmed real run buries the
    delegation as an ordinary ``tool.call`` and emits ZERO handoff edges. The
    ``agno.py`` unit tests inject the (now-stale) ``transfer_``/``forward_`` names
    so they pass, masking the drift. This recorder installs a GENERATOR-TIME shim
    (``_delegation_aware_transfer``, restored in a ``finally``) that broadens the
    classifier to also match ``delegate_*_member`` so the adapter's OWN
    ``_emit_handoff`` / ``_parse_handoff_target`` code path records the REAL
    delegation as an honest handoff — nothing is invented (the delegation genuinely
    occurred; the member ran and produced a real ``model.invoke``). The correct
    source fix is a one-line broadening of ``_is_transfer_tool`` to recognize
    ``delegate``; this recorder does NOT edit src.
"""

from __future__ import annotations

import json
import os
import sys

# Reuse the record-real-once seam from the sibling ``_generate_fixtures`` module
# (single source of truth for the fixture writer + capture config + model name).
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.dirname(_HERE)        # samples/data
_SAMPLES = os.path.dirname(_DATA)     # samples
_REPO = os.path.dirname(_SAMPLES)
for _p in (os.path.join(_REPO, "src"), _SAMPLES, _DATA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# This module is named ``agno.py`` (to match the adapter). When the file is run
# directly, Python inserts its own directory at ``sys.path[0]``, which would
# shadow the real ``agno`` package for the function-local ``import agno``. Drop
# this module's own directory from the path so the framework import always
# resolves to the installed package (a no-op when imported as ``generators.agno``,
# since the package dir is not on the path then).
sys.path[:] = [_p for _p in sys.path if os.path.abspath(_p or ".") != _HERE]

from layerlens import Stratix  # noqa: E402
import layerlens.instrument._collector as _collector_mod  # noqa: E402
from layerlens.instrument._collector import set_trace_observer  # noqa: E402

import _generate_fixtures as _gf  # noqa: E402

_write = _gf._write
_CAPTURE = _gf._CAPTURE
OPENAI_MODEL = _gf.OPENAI_MODEL


# --------------------------------------------------------------------------
# A small, realistic (non-sensitive) auto-policy book the ``lookup_policy``
# tool reads from, so the single agent's tool-use turn returns genuine grounded
# coverage terms for the claim question.
# --------------------------------------------------------------------------
_POLICY_BOOK = {
    "AUTO-2024-8891": {
        "policy_id": "AUTO-2024-8891",
        "type": "auto",
        "deductible_usd": 500,
        "collision_covered": True,
        "comprehensive_covered": True,
        "rental_reimbursement": True,
        "coverage_limit_usd": 50000,
        "exclusions": ["racing", "commercial_use", "intentional_damage"],
        "status": "active",
    },
    "AUTO-2023-4410": {
        "policy_id": "AUTO-2023-4410",
        "type": "auto",
        "deductible_usd": 1000,
        "collision_covered": False,
        "comprehensive_covered": True,
        "rental_reimbursement": False,
        "coverage_limit_usd": 25000,
        "exclusions": ["racing", "off_road", "commercial_use"],
        "status": "active",
    },
}


# --------------------------------------------------------------------------
# Adapter-driven capture: AgnoAdapter wraps ``Agent.run`` with a per-run
# collector that flushes on run-end; we register it (plus any Team members),
# drive a REAL ``.run(...)``, and observe the flushed payload.
# --------------------------------------------------------------------------
def _capture_agno(client: Stratix, target, task: str, *, members=()) -> dict:
    from layerlens.instrument.adapters.frameworks.agno import AgnoAdapter

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    adapter = AgnoAdapter(client, capture_config=_CAPTURE)
    adapter.connect(target=target)
    # Instrument each declared Team member so its delegated ``run()`` emits its
    # own honest agent node (real per-member model.invoke/cost.record) into the
    # SAME shared collector (nested ``_begin_run`` reuses the active collector).
    for m in members:
        adapter._instrument_agent(m)
    try:
        target.run(task)
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for agno run")
    return payload


def _delegation_aware_transfer(name):
    """Generator-time widening of the adapter's ``_is_transfer_tool`` classifier
    so agno 2.6.x's real ``delegate_task_to_member`` delegation tool is recorded
    as an ``agent.handoff`` (see the module docstring — PING the source fix)."""
    if not name:
        return False
    low = name.lower()
    return ("transfer" in low or "forward" in low or "delegate" in low) and (
        "member" in low or "agent" in low or "task" in low
    )


# --------------------------------------------------------------------------
# Single agent + a real ``lookup_policy`` tool turn (insurance claims intake)
# --------------------------------------------------------------------------
def generate_agno_single(client: Stratix) -> dict:
    """Record a single ``claims-intake-agent`` running a real tool-use turn."""
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat

    def lookup_policy(policy_id: str) -> str:
        """Look up an auto-insurance policy's coverage terms, deductible, and
        exclusions by its policy_id.

        Args:
            policy_id: The policy identifier (e.g. "AUTO-2024-8891").
        """
        rec = _POLICY_BOOK.get((policy_id or "").strip())
        if rec is None:
            return json.dumps({"policy_id": policy_id, "found": False})
        return json.dumps(rec)

    agent = Agent(
        name="claims-intake-agent",
        model=OpenAIChat(id=OPENAI_MODEL),
        tools=[lookup_policy],
        instructions=(
            "You are an auto-insurance claims intake assistant. For a claim, FIRST "
            "call lookup_policy with the stated policy_id to fetch the coverage "
            "terms, THEN tell the customer whether the loss is covered, their "
            "deductible, and any relevant exclusions — grounded ONLY in the policy "
            "the tool returned. Answer concisely (under 120 words)."
        ),
        markdown=False,
    )

    task = (
        "Claim for policy AUTO-2024-8891: my car was damaged in a parking-lot "
        "collision, the repair estimate is $4,200. Is this covered, what is my "
        "deductible, and are there any exclusions I should know about?"
    )
    payload = _capture_agno(client, agent, task)
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "insurance",
        "claims-intake",
        "tool-use",
    ]
    events = payload.get("events", [])
    tools = sorted(
        {(e.get("payload") or {}).get("tool_name") for e in events
         if e.get("event_type") == "tool.call"}
        - {None}
    )
    mi = [e for e in events if e.get("event_type") == "model.invoke"]
    cr = [e for e in events if e.get("event_type") == "cost.record"]
    print("  agno single (claims-intake-agent, tool-use)  "
          "events=%d tools=%s model.invoke=%d cost.record=%d"
          % (len(events), tools, len(mi), len(cr)))
    print("  ->", _write([payload], "industry", "insurance_claims_agno"), "\n")
    return payload


# --------------------------------------------------------------------------
# Multi-agent: an agno Team leader delegates to two named members (underwriting)
# --------------------------------------------------------------------------
def generate_agno_multi(client: Stratix) -> dict:
    """Record a genuine multi-agent agno Team: an ``underwriting-team`` leader
    delegates risk assessment to ``risk-analyst`` and the compliance check to
    ``compliance-checker`` (real ``delegate_task_to_member`` -> real member runs
    + honest handoff edges)."""
    import layerlens.instrument.adapters.frameworks.agno as _agno_mod
    from agno.agent import Agent
    from agno.team.team import Team
    from agno.models.openai import OpenAIChat

    risk_analyst = Agent(
        name="risk-analyst",
        model=OpenAIChat(id=OPENAI_MODEL),
        role="Assess the applicant's default and collateral risk.",
        instructions=(
            "In 2 sentences, assess default risk (from FICO, DTI, income) and "
            "collateral risk (loan-to-value). Be brief and specific."
        ),
        markdown=False,
    )
    compliance_checker = Agent(
        name="compliance-checker",
        model=OpenAIChat(id=OPENAI_MODEL),
        role="Confirm fair-lending / ECOA compliance.",
        instructions=(
            "In 2 sentences, confirm the underwriting decision relies only on "
            "permissible factors and complies with fair-lending / ECOA rules "
            "(no protected attributes). Be brief."
        ),
        markdown=False,
    )
    team = Team(
        name="underwriting-team",
        members=[risk_analyst, compliance_checker],
        model=OpenAIChat(id=OPENAI_MODEL),
        instructions=(
            "You lead a loan-underwriting team. Delegate the risk assessment to "
            "risk-analyst and the fair-lending compliance check to "
            "compliance-checker, then return a one-line "
            "APPROVE / CONDITIONAL / DECLINE decision with a brief rationale."
        ),
        markdown=False,
    )

    application = json.dumps(
        {
            "applicant_id": "APP-7781",
            "loan_type": "auto",
            "amount_usd": 32000,
            "applicant": {"fico": 712, "annual_income_usd": 84000, "dti_ratio": 0.28,
                          "employment_years": 5},
            "collateral": {"item": "2024 sedan", "appraised_value_usd": 30000},
        }
    )
    task = (
        "Underwrite this auto-loan application and return a decision. Delegate the "
        "risk assessment and the compliance check to your team members, then give "
        "the final decision:\n" + application
    )

    # See module docstring: broaden the adapter's handoff classifier so agno's
    # real ``delegate_task_to_member`` records via the adapter's own _emit_handoff.
    _orig_transfer = _agno_mod._is_transfer_tool
    _agno_mod._is_transfer_tool = _delegation_aware_transfer
    try:
        payload = _capture_agno(
            client, team, task, members=(risk_analyst, compliance_checker)
        )
    finally:
        _agno_mod._is_transfer_tool = _orig_transfer

    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "insurance",
        "underwriting",
        "multi-agent",
    ]
    events = payload.get("events", [])
    idents = sorted(
        {(e.get("payload") or {}).get("agent_name") for e in events
         if (e.get("payload") or {}).get("agent_name")}
        - {None}
    )
    handoffs = [
        (
            (e.get("payload") or {}).get("from_agent"),
            (e.get("payload") or {}).get("to_agent"),
        )
        for e in events
        if e.get("event_type") == "agent.handoff"
    ]
    mi = [e for e in events if e.get("event_type") == "model.invoke"]
    cr = [e for e in events if e.get("event_type") == "cost.record"]
    print("  agno multi (underwriting-team -> risk-analyst/compliance-checker)  "
          "events=%d agents=%s handoffs=%s model.invoke=%d cost.record=%d"
          % (len(events), idents, handoffs, len(mi), len(cr)))
    print("  ->", _write([payload], "industry", "insurance_underwriting_agno_team"), "\n")
    return payload


if __name__ == "__main__":  # pragma: no cover - manual regeneration entrypoint
    _client = Stratix()
    generate_agno_single(_client)
    generate_agno_multi(_client)
