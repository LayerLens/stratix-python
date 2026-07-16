#!/usr/bin/env python3
"""Industry: Retail Support Triage + Handoff (OpenAI Agents SDK) -- LayerLens Sample.

Demonstrates evaluating a REAL multi-agent OpenAI Agents SDK workflow with a
genuine handoff and an input guardrail. A ``triage-agent`` (guarded by a real
``@input_guardrail`` that screens for prompt-injection / off-topic input) routes
a product-return request by handing off to a ``returns-specialist``, which calls
a real ``check_return_eligibility`` tool and tells the customer whether they can
return the item, the refund amount, and how.

Because the trace was captured through the real ``OpenAIAgentsAdapter`` (the
SDK's own ``TracingProcessor``), the handoff span records a real ``agent.handoff``
(triage-agent -> returns-specialist) and the guardrail records a real
``evaluation.result`` -- so the recorded trace renders as a multi-agent graph
(Agent column ``multi-agent``: triage-agent -> returns-specialist) with the real
per-agent ``model.invoke`` / ``cost.record`` / ``tool.call`` events. Nothing is
fabricated. The recorded trace is shipped under samples/data/traces/industry/
(produced by samples/data/generators/openai_agents.py); this sample uploads it
and evaluates the resolution with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python retail_openai_agents_triage.py
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

SAMPLE = "retail_openai_agents_triage"
FIXTURE = recorded_trace_path("industry", "retail_openai_agents_triage.jsonl")

# The support request the crew resolved. Documents the scenario; the recorded
# multi-agent trace was produced by running this through a real OpenAI Agents SDK
# run (triage-agent guardrail -> handoff -> returns-specialist tool call).
REQUEST: dict[str, Any] = {
    "ticket_id": "TKT-90731",
    "channel": "chat",
    "order_id": "ORD-88120",
    "intent": "product_return",
    "text": (
        "I received a defective blender (order ORD-88120) and want to return it "
        "for a refund."
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded multi-agent retail-support-triage trace."""
    print("=== LayerLens Industry: Retail Support Triage + Handoff (OpenAI Agents) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded multi-agent trace first (renders a triage-agent ->
    # returns-specialist handoff graph via real handoff + guardrail events). Do
    # this before creating judges so the trace always lands even if the org has
    # no evaluation model yet.
    print("Uploading the recorded triage trace (handoff + guardrail, multi-agent)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Ticket:   {REQUEST['ticket_id']} ({REQUEST['intent']})\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "handoff_correctness": create_judge(
                client,
                name="Handoff Correctness Judge",
                evaluation_goal="Evaluate whether the triage-agent correctly handed the product-return request off to the returns-specialist rather than answering it itself.",
                namespace=SAMPLE,
            ),
            "resolution_quality": create_judge(
                client,
                name="Return Resolution Judge",
                evaluation_goal="Evaluate whether the returns-specialist gave a correct, complete resolution (return eligibility, refund amount, and return method) grounded in the eligibility tool result.",
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

        print("Evaluating the triage + handoff resolution:")
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
