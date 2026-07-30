#!/usr/bin/env python3
"""Industry: Multi-Agent Underwriting Team (Agno) -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL multi-agent ``agno`` workflow with genuine team
delegation. An ``underwriting-team`` (an ``agno.team.Team``) leader delegates the
risk assessment to a ``risk-analyst`` member and the fair-lending compliance check
to a ``compliance-checker`` member, then returns an APPROVE / CONDITIONAL /
DECLINE decision. agno's ``delegate_task_to_member`` tool really invokes each
member agent, so the recorded trace carries three distinct honest agent
identities -- each with its OWN real ``model.invoke`` / ``cost.record`` -- plus a
real ``agent.handoff`` per delegation (underwriting-team -> risk-analyst,
underwriting-team -> compliance-checker). It renders as a multi-agent graph whose
Agent column reads ``multi-agent``.

The trace was recorded from a real agno Team run (see
samples/data/generators/agno.py) and is shipped under
samples/data/traces/industry/. Nothing is fabricated: the Framework column shows
``agno``, and the nodes / handoff edges are the real declared members and real
delegations the framework emitted. This sample uploads the trace and evaluates the
team's underwriting decision with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python insurance_underwriting_agno_team.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import (
    create_judge,
    poll_evaluation_results,
    recorded_trace_path,
    upload_recorded_trace,
)

SAMPLE = "insurance_underwriting_agno_team"
FIXTURE = recorded_trace_path("industry", "insurance_underwriting_agno_team.jsonl")

# The application the underwriting team decided. Documents the scenario; the
# recorded multi-agent trace was produced by running this through a real agno Team
# (leader delegating to risk-analyst + compliance-checker via delegate_task_to_member).
APPLICATION: dict[str, Any] = {
    "application_id": "APP-7781",
    "loan_type": "auto",
    "amount_usd": 32000,
    "team": ["underwriting-team", "risk-analyst", "compliance-checker"],
    "summary": (
        "Auto loan for $32,000 (FICO 712, DTI 0.28) against a 2024 sedan appraised "
        "at $30,000 -- assess risk and fair-lending compliance, then decide."
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded multi-agent underwriting-team trace."""
    print("=== LayerLens Industry: Multi-Agent Underwriting Team (Agno) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded multi-agent trace first (renders as a multi-agent graph
    # via real team-delegation handoffs). Do this before creating judges so the
    # trace always lands even if the org has no evaluation model yet.
    print("Uploading the recorded underwriting-team trace (team-delegation graph)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID:    {trace_id}")
    print(f"  Application: {APPLICATION['application_id']} (team: {', '.join(APPLICATION['team'])})\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "delegation_soundness": create_judge(
                client,
                name="Underwriting Delegation Judge",
                evaluation_goal="Evaluate whether the team leader delegated the risk assessment to the risk analyst and the compliance check to the compliance checker (each part handled by the right member).",
                namespace=SAMPLE,
            ),
            "decision_soundness": create_judge(
                client,
                name="Underwriting Decision Judge",
                evaluation_goal="Evaluate whether the final APPROVE/CONDITIONAL/DECLINE decision is well supported by the risk and compliance assessments the members produced.",
                namespace=SAMPLE,
            ),
            "fair_lending": create_judge(
                client,
                name="Fair Lending Compliance Judge",
                evaluation_goal="Evaluate whether the underwriting decision relies only on permissible factors and complies with fair-lending / ECOA rules (no protected attributes).",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the team's underwriting decision:")
        for label, judge in judges.items():
            evaluation = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge.id)
            if evaluation is None:
                print(f"  {label:24s} -- evaluation creation failed")
                continue
            results = poll_evaluation_results(client, evaluation.id)
            if results:
                r = results[0]
                verdict = "pass" if r.passed else "fail"
                color = _VERDICT_COLORS.get(verdict, "")
                print(f"  {label:24s} {color}{verdict.upper()}{_RESET} ({r.score:.2f})")
            else:
                print(f"  {label:24s} -- timed out waiting for results")
    except RuntimeError as exc:
        print(f"\nNOTE: evaluations skipped -- {exc}")
        print("  The trace is uploaded; add a project/public model to enable judges.")
    finally:
        for jid in judge_ids:
            try:
                client.judges.delete(jid)
            except Exception:
                pass


if __name__ == "__main__":
    main()
