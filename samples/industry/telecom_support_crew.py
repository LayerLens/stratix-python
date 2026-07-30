#!/usr/bin/env python3
"""Industry: Multi-Agent Telecom Support Crew -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL multi-agent CrewAI workflow with genuine
delegation. A telecom customer-support manager (CrewAI's hierarchical process)
delegates a mixed billing + connectivity complaint to two specialists --
billing-specialist and network-specialist -- via CrewAI's built-in
"Delegate work to coworker" tool. The CrewAIAdapter records the real
agent-to-agent handoffs, so the recorded trace renders as a multi-agent graph
(Crew Manager -> billing-specialist, Crew Manager -> network-specialist) whose
Agent column reads ``multi-agent``.

The trace was recorded from a real hierarchical crew run (see
samples/data/_generate_fixtures.py) and is shipped under
samples/data/traces/industry/. This sample uploads it and evaluates the crew's
resolution with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python telecom_support_crew.py
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

SAMPLE = "telecom_support_crew"
FIXTURE = recorded_trace_path("industry", "telecom_support_crew.jsonl")

# The customer complaint the support crew resolved. Documents the scenario; the
# recorded multi-agent trace was produced by running this through a real CrewAI
# hierarchical crew (manager delegating to billing-specialist + network-specialist).
COMPLAINT: dict[str, Any] = {
    "ticket_id": "TKT-88213",
    "channel": "chat",
    "issues": ["billing_dispute", "connectivity"],
    "summary": (
        "Double-charged $79.99 on the last bill AND home internet keeps dropping "
        "every evening around 8pm."
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded multi-agent telecom-support-crew trace."""
    print("=== LayerLens Industry: Multi-Agent Telecom Support Crew ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded multi-agent trace first (renders as a multi-agent graph
    # via real delegation handoffs). Do this before creating judges so the trace
    # always lands even if the org has no evaluation model yet.
    print("Uploading the recorded telecom-support-crew trace (multi-agent delegation graph)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Ticket:   {COMPLAINT['ticket_id']} ({', '.join(COMPLAINT['issues'])})\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "resolution_quality": create_judge(
                client,
                name="Support Resolution Judge",
                evaluation_goal="Evaluate whether the crew fully resolved BOTH the billing dispute and the connectivity issue with concrete, correct actions.",
                namespace=SAMPLE,
            ),
            "delegation_soundness": create_judge(
                client,
                name="Delegation Soundness Judge",
                evaluation_goal="Evaluate whether the manager delegated each part of the complaint to the appropriate specialist (billing vs network).",
                namespace=SAMPLE,
            ),
            "customer_empathy": create_judge(
                client,
                name="Customer Empathy Judge",
                evaluation_goal="Evaluate whether the final response is clear, empathetic, and gives the customer a concrete next step.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the crew's resolution:")
        for label, judge in judges.items():
            evaluation = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge.id)
            if evaluation is None:
                print(f"  {label:22s} -- evaluation creation failed")
                continue
            results = poll_evaluation_results(client, evaluation.id)
            if results:
                r = results[0]
                verdict = "pass" if r.passed else "fail"
                color = _VERDICT_COLORS.get(verdict, "")
                print(f"  {label:22s} {color}{verdict.upper()}{_RESET} ({r.score:.2f})")
            else:
                print(f"  {label:22s} -- timed out waiting for results")
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
