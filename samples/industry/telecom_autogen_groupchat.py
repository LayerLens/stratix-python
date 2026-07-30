#!/usr/bin/env python3
"""Industry: Multi-Agent Telecom Support Panel (AutoGen group chat) -- LayerLens Sample.

Demonstrates evaluating a REAL multi-agent AutoGen ``RoundRobinGroupChat``. Three
named telecom support agents -- ``triage_agent`` -> ``billing_specialist`` ->
``network_specialist`` -- collaborate on a single mixed billing + connectivity
complaint: the triage agent splits the complaint, the billing specialist resolves
the duplicate charge, and the network specialist resolves the connectivity issue
and gives the combined next step. Each agent reads the running conversation and
contributes its own turn.

Because the trace was recorded from an actual AutoGen group-chat run, it carries
genuine per-agent ``agent.input`` / ``model.invoke`` / ``cost.record`` events and
renders as a multi-agent trace (Agent column ``multi-agent``) built from the three
distinct honest agent nodes. (AutoGen routes inter-agent messages through the
group-chat manager, so it emits no agent-to-agent handoff EDGE -- the multi-agent
topology is the real per-agent nodes, exactly as the LayerLens graph engine
expects for AutoGen; nothing is fabricated.)

The recorded trace is shipped under samples/data/traces/industry/ (produced by
samples/data/generators/autogen.py); this sample uploads it and evaluates the
panel's resolution with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python telecom_autogen_groupchat.py
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

SAMPLE = "telecom_autogen_groupchat"
FIXTURE = recorded_trace_path("industry", "telecom_autogen_groupchat.jsonl")

# The complaint the support panel resolved. Documents the scenario; the recorded
# multi-agent trace was produced by running this through the real AutoGen group
# chat (triage_agent -> billing_specialist -> network_specialist).
COMPLAINT: dict[str, Any] = {
    "ticket_id": "TKT-77401",
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
    """Evaluate the recorded multi-agent AutoGen telecom-support-panel trace."""
    print("=== LayerLens Industry: Multi-Agent Telecom Support Panel (AutoGen) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded multi-agent trace first (renders as a multi-agent graph
    # from the three real per-agent nodes). Do this before creating judges so the
    # trace always lands even if the org has no evaluation model yet.
    print("Uploading the recorded telecom-support-panel trace (triage/billing/network group chat)...\n")
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
                name="Panel Resolution Judge",
                evaluation_goal="Evaluate whether the panel fully resolved BOTH the billing dispute and the connectivity issue with concrete, correct actions.",
                namespace=SAMPLE,
            ),
            "role_specialization": create_judge(
                client,
                name="Role Specialization Judge",
                evaluation_goal="Evaluate whether each agent (triage, billing, network) contributed its own role and the billing vs connectivity parts were handled by the right specialist.",
                namespace=SAMPLE,
            ),
            "customer_empathy": create_judge(
                client,
                name="Customer Empathy Judge",
                evaluation_goal="Evaluate whether the final combined response is clear, empathetic, and gives the customer a single concrete next step.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the panel's resolution:")
        for label, judge in judges.items():
            evaluation = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge.id)
            if evaluation is None:
                print(f"  {label:20s} -- evaluation creation failed")
                continue
            results = poll_evaluation_results(client, evaluation.id)
            if results:
                r = results[0]
                verdict = "pass" if r.passed else "fail"
                color = _VERDICT_COLORS.get(verdict, "")
                print(f"  {label:20s} {color}{verdict.upper()}{_RESET} ({r.score:.2f})")
            else:
                print(f"  {label:20s} -- timed out waiting for results")
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
