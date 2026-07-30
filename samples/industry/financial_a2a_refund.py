#!/usr/bin/env python3
"""Industry: Financial-services A2A refund delegation (single hop) -- LayerLens SDK Sample.

Demonstrates observing a REAL **Agent-to-Agent (A2A) protocol** delegation. A
``dispute-orchestrator`` discovers a ``billing-specialist``'s signed agent card
over A2A and delegates a single customer refund-dispute task to it. A2A is an
**LLM-free protocol** surface, so the trace carries NO model/token/cost data --
instead it renders the delegation itself:

* an ``a2a.agent.discovered`` event with the specialist's card skills and its
  signature PROVENANCE (signature present + a keyed-HMAC fingerprint -- never the
  raw JWS),
* an ``a2a.delegation`` edge ``dispute-orchestrator -> billing-specialist`` (the
  Agent column renders the orchestrator; the delegation edge renders the graph),
* the ``a2a.task.created`` / ``a2a.task.updated`` lifecycle with the real
  terminal ``TaskState`` (Status column = ``completed``).

The recorded trace was captured under the real ``A2AProtocolAdapter`` from
genuine ``a2a-sdk`` v1.1.0 model objects (a real signed ``AgentCard`` + a real
``Task`` at ``TaskState.completed``) -- nothing is fabricated. It is shipped under
samples/data/traces/industry/ (produced by samples/data/_generate_fixtures.py);
this sample uploads it and evaluates whether the dispute was delegated to the
right specialist.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python financial_a2a_refund.py
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

SAMPLE = "financial_a2a_refund"
FIXTURE = recorded_trace_path("industry", "financial_a2a_refund.jsonl")

# The dispute the orchestrator delegated to the billing specialist over A2A.
# Documents the scenario; the recorded delegation trace was produced by driving
# this through the real A2AProtocolAdapter (get_agent_card + send_task).
DISPUTE: dict[str, Any] = {
    "dispute_id": "DSP-20260712-4471",
    "account": "ACME-4471",
    "invoice": "INV-20260712",
    "amount_usd": 79.99,
    "reason": "duplicate charge (billed twice on 2026-07-12)",
    "delegated_to": "billing-specialist",
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Upload and evaluate the recorded single-hop A2A refund delegation trace."""
    print("=== LayerLens Industry: Financial-services A2A refund delegation ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded delegation trace first (renders the orchestrator ->
    # billing-specialist delegation edge + card provenance + completed status).
    # Do this before creating judges so the trace always lands even if the org
    # has no evaluation model yet.
    print("Uploading the recorded A2A refund-delegation trace "
          "(discover billing card -> delegate refund)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID:   {trace_id}")
    print(f"  Dispute:    {DISPUTE['dispute_id']} (${DISPUTE['amount_usd']} on {DISPUTE['invoice']})")
    print(f"  Delegation: dispute-orchestrator -> {DISPUTE['delegated_to']} (completed)\n")

    # Judges evaluate the delegation itself (A2A is LLM-free -- there is no model
    # answer to grade; the delegation topology + skill descriptions are the
    # signal). Scoped to this sample (namespace avoids cross-sample name clashes).
    judge_ids: list[str] = []
    try:
        judges = {
            "delegation_routing": create_judge(
                client,
                name="A2A Delegation Routing Judge",
                evaluation_goal="Evaluate whether the customer refund-dispute task was delegated to the appropriate specialist agent (a billing specialist), given the disputed duplicate charge described in the delegated task.",
                namespace=SAMPLE,
            ),
            "task_completion": create_judge(
                client,
                name="A2A Task Completion Judge",
                evaluation_goal="Evaluate whether the delegated A2A task reached a terminal successful state (the task lifecycle ends in a completed status rather than failed or rejected).",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the delegation:")
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
