#!/usr/bin/env python3
"""Industry: Salesforce Agentforce Service Cloud -- Billing Escalation (SEALED).

Demonstrates evaluating a REAL multi-topic Salesforce Agentforce customer-service
session imported from the Session Tracing Data Model (STDM / ``ssot__*__dlm``
Data Cloud objects). In one session the SAME Agentforce Service Agent handles two
turns: turn 1 resolves an order-status question; turn 2 is a duplicate-charge
billing dispute the agent cannot settle automatically, so it emits an escalation
step. The AgentforceAdapter maps that escalation to an ``agent.handoff`` whose
``from_agent`` is the service agent.

HONEST SINGLE-AGENT ESCALATION (not a multi-agent DAG): the STDM step schema
carries NO target-agent field for an escalation, so ``to_agent`` is deliberately
ABSENT -- the adapter never guesses it. The trace therefore renders a single
Agentforce agent (Agent column = ``Order_Support_Service_Agent``,
Framework = ``agentforce``) with a handoff ORIGIN over a multi-turn span
waterfall -- NOT a fabricated two-node graph. It is a multi-turn single-agent
session, mirroring exactly how real Agentforce STDM records a human escalation.
The STDM has no token fields, so there is NO ``cost.record``.

SEALED FIXTURE: no Salesforce Agentforce + Data Cloud org exists on the build
machines (``SF_CLIENT_ID`` / ``SF_CLIENT_SECRET`` / ``SF_INSTANCE_URL`` unset),
so the trace was recorded by driving the REAL ``AgentforceAdapter`` (its real
OAuth + SOQL connection and full ``import_sessions`` STDM parser) against an
``httpx.MockTransport`` serving documented/synthetic ``ssot__*`` rows -- only the
Salesforce network is sealed (see ``samples/data/generators/salesforce_agentforce.py``,
``metadata.sealed = true``). Every parse, classification and attestation is real;
nothing is fabricated as a live billed call. Provision a real org and re-run the
recorder to replace the sealed rows with a genuine capture.

This sample uploads the recorded trace and evaluates the agent's handling with
domain judges -- it runs with ONLY a LayerLens API key.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python salesforce_agentforce_billing_escalation.py
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

SAMPLE = "salesforce_agentforce_billing_escalation"
FIXTURE = recorded_trace_path("industry", "salesforce_agentforce_billing_escalation.jsonl")

# The multi-topic Service Cloud session the recorded STDM trace captured.
SESSION: dict[str, Any] = {
    "channel": "MessagingForWeb",
    "agent": "Order_Support_Service_Agent",
    "topics": ["Order_Status_And_Tracking", "Billing_Dispute_Resolution"],
    "order_number": "ORD-5583991",
    "end_type": "EscalatedToHuman",
    "dispute": (
        "I was charged $84.98 twice for order ORD-5583991. I want the duplicate "
        "charge refunded now."
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded Salesforce Agentforce billing-escalation session."""
    print("=== LayerLens Industry: Salesforce Agentforce -- Billing Escalation ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded multi-topic Agentforce STDM trace (renders the single
    # service agent + a handoff origin over a multi-turn waterfall). Do this before
    # creating judges so the trace always lands even without an evaluation model.
    print("Uploading the recorded Agentforce billing-escalation trace (sealed STDM import)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Agent:    {SESSION['agent']} (topics: {', '.join(SESSION['topics'])})")
    print(f"  Outcome:  {SESSION['end_type']} (order {SESSION['order_number']})\n")

    judge_ids: list[str] = []
    try:
        judges = {
            "escalation_soundness": create_judge(
                client,
                name="Escalation Soundness Judge",
                evaluation_goal="Evaluate whether escalating the duplicate-charge refund to a human billing specialist was the correct action, given that the refund exceeds the agent's automated policy limit.",
                namespace=SAMPLE,
            ),
            "dispute_handling": create_judge(
                client,
                name="Dispute Handling Judge",
                evaluation_goal="Evaluate whether the agent correctly confirmed the duplicate charge from the payment-lookup action and clearly communicated what happens next before escalating.",
                namespace=SAMPLE,
            ),
            "customer_empathy": create_judge(
                client,
                name="Customer Empathy Judge",
                evaluation_goal="Evaluate whether the final response is empathetic, sets a clear expectation about the billing team's follow-up, and leaves the customer confident their issue is being handled.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the agent's escalation handling:")
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
