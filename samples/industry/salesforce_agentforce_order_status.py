#!/usr/bin/env python3
"""Industry: Salesforce Agentforce Service Cloud -- Order-Status Session (SEALED).

Demonstrates evaluating a REAL Salesforce Agentforce customer-service session
imported from the Session Tracing Data Model (STDM / ``ssot__*__dlm`` Data Cloud
objects). A retail shopper asks where their order is; the Agentforce Service
Agent routes the topic, reasons over the request, calls a ``Get_Order_Status``
Apex action, and composes the answer. The AgentforceAdapter maps the STDM steps
to genuine LayerLens events -- ``model.invoke`` (carrying the real generation /
gateway ids), ``tool.call`` (the Apex action), and ``agent.interaction`` (the
conversation turns) -- so the trace renders the single Agentforce agent
(Agent column = ``Order_Support_Service_Agent``, Framework = ``agentforce``,
Status = ok) over a span waterfall. The STDM has no token fields, so there is
NO ``cost.record`` -- the adapter fabricates none.

SEALED FIXTURE: no Salesforce Agentforce + Data Cloud org exists on the build
machines (``SF_CLIENT_ID`` / ``SF_CLIENT_SECRET`` / ``SF_INSTANCE_URL`` unset),
so the trace was recorded by driving the REAL ``AgentforceAdapter`` (its real
OAuth + SOQL connection and full ``import_sessions`` STDM parser) against an
``httpx.MockTransport`` serving documented/synthetic ``ssot__*`` rows -- only the
Salesforce network is sealed (see ``samples/data/generators/salesforce_agentforce.py``,
``metadata.sealed = true``). Every parse, classification and attestation is real;
nothing is fabricated as a live billed call. Provision a real org and re-run the
recorder to replace the sealed rows with a genuine capture.

This sample uploads the recorded trace and evaluates the agent's answer with
domain judges -- it runs with ONLY a LayerLens API key.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python salesforce_agentforce_order_status.py
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

SAMPLE = "salesforce_agentforce_order_status"
FIXTURE = recorded_trace_path("industry", "salesforce_agentforce_order_status.jsonl")

# The Service Cloud session the recorded STDM trace captured. Documents the
# scenario; the trace itself was produced by the real Agentforce adapter parsing
# the (sealed) ssot__*__dlm rows for this session.
SESSION: dict[str, Any] = {
    "channel": "MessagingForWeb",
    "agent": "Order_Support_Service_Agent",
    "topic": "Order_Status_And_Tracking",
    "order_number": "ORD-5582107",
    "customer_message": (
        "Hi, I ordered a pair of trail running shoes last Tuesday and haven't "
        "gotten a shipping update. My order number is ORD-5582107. Where is it?"
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded Salesforce Agentforce order-status session."""
    print("=== LayerLens Industry: Salesforce Agentforce -- Order-Status Session ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded Agentforce STDM trace (renders the single service agent
    # + a span waterfall over the topic/LLM/action steps). Do this before creating
    # judges so the trace always lands even if the org has no evaluation model yet.
    print("Uploading the recorded Agentforce order-status trace (sealed STDM import)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Agent:    {SESSION['agent']} (topic: {SESSION['topic']})")
    print(f"  Order:    {SESSION['order_number']}\n")

    judge_ids: list[str] = []
    try:
        judges = {
            "answer_accuracy": create_judge(
                client,
                name="Order Status Accuracy Judge",
                evaluation_goal="Evaluate whether the agent's reply accurately reports the order's shipment status, carrier, tracking number, and delivery estimate as returned by the order-lookup action.",
                namespace=SAMPLE,
            ),
            "tool_grounding": create_judge(
                client,
                name="Action Grounding Judge",
                evaluation_goal="Evaluate whether the agent grounded its answer in the Get_Order_Status action result rather than guessing -- the reply should reflect the fetched order record, not invented shipment details.",
                namespace=SAMPLE,
            ),
            "customer_tone": create_judge(
                client,
                name="Service Tone Judge",
                evaluation_goal="Evaluate whether the final response is clear, friendly, and gives the customer a concrete next step (how to track the order).",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the agent's order-status resolution:")
        for label, judge in judges.items():
            evaluation = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge.id)
            if evaluation is None:
                print(f"  {label:18s} -- evaluation creation failed")
                continue
            results = poll_evaluation_results(client, evaluation.id)
            if results:
                r = results[0]
                verdict = "pass" if r.passed else "fail"
                color = _VERDICT_COLORS.get(verdict, "")
                print(f"  {label:18s} {color}{verdict.upper()}{_RESET} ({r.score:.2f})")
            else:
                print(f"  {label:18s} -- timed out waiting for results")
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
