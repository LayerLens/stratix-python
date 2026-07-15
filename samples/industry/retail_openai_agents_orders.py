#!/usr/bin/env python3
"""Industry: Retail Order Support (OpenAI Agents SDK, tool-use) -- LayerLens Sample.

Demonstrates evaluating a REAL single-agent tool-use workflow built on the OpenAI
Agents SDK. A retail ``order-support-agent`` (an ``agents.Agent`` driven by
``Runner.run_sync``) answers a shopper's order question by calling one real
``@function_tool`` (``lookup_order``) to fetch live order status, carrier, ETA
and whether the shipping address can still be changed, then replies grounded in
that record.

Because the trace was captured through the real ``OpenAIAgentsAdapter`` (which is
the SDK's own ``TracingProcessor``) from a genuine two-step tool loop, it renders
a single honest agent node (Agent column ``order-support-agent``, Framework
``openai-agents``) plus the real ``model.invoke`` / ``cost.record`` /
``tool.call`` / ``tool.result`` events -- no fabrication. The recorded trace is
shipped under samples/data/traces/industry/ (produced by
samples/data/generators/openai_agents.py); this sample uploads it and evaluates
the answer with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python retail_openai_agents_orders.py
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

SAMPLE = "retail_openai_agents_orders"
FIXTURE = recorded_trace_path("industry", "retail_openai_agents_orders.jsonl")

# The order question the agent answered. Documents the scenario; the recorded
# tool-use trace was produced by running this through a real OpenAI Agents SDK
# run (the agent called lookup_order, then answered from the returned record).
QUESTION: dict[str, Any] = {
    "ticket_id": "ORD-CHAT-4471",
    "channel": "chat",
    "order_id": "ORD-10432",
    "text": (
        "Where is my order ORD-10432 and can I still change the shipping address?"
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded retail order-support tool-use trace."""
    print("=== LayerLens Industry: Retail Order Support (OpenAI Agents, tool-use) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded tool-use trace first (renders a single order-support-
    # agent node with real model.invoke / tool.call / tool.result events). Do
    # this before creating judges so the trace always lands even if the org has
    # no evaluation model yet.
    print("Uploading the recorded order-support trace (lookup_order tool loop)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Order:    {QUESTION['order_id']} ({QUESTION['channel']})\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "answer_accuracy": create_judge(
                client,
                name="Order Answer Accuracy Judge",
                evaluation_goal="Evaluate whether the agent's answer correctly reports the order's status, carrier, ETA, and whether the shipping address can still be changed.",
                namespace=SAMPLE,
            ),
            "tool_grounding": create_judge(
                client,
                name="Tool Grounding Judge",
                evaluation_goal="Evaluate whether the agent grounded its answer in the record returned by the lookup_order tool rather than inventing order details.",
                namespace=SAMPLE,
            ),
            "customer_clarity": create_judge(
                client,
                name="Customer Clarity Judge",
                evaluation_goal="Evaluate whether the response is clear, concise, and gives the customer a correct, actionable answer.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the order-support answer:")
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
