#!/usr/bin/env python3
"""Legal: Multi-Agent Contract-Review Workflow -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL LlamaIndex ``AgentWorkflow`` with genuine
agent-to-agent handoff. Two ``FunctionAgent``s collaborate on a contract review:
``contract-intake`` (the root agent) states the contract type and the clauses
present, then HANDS OFF to ``clause-risk`` -- via AgentWorkflow's built-in
``handoff`` tool -- to assess the legal risks and recommend changes. The
LlamaIndex adapter records the real per-agent turns and the handoff, so the
recorded trace renders as a genuine multi-agent DAG (contract-intake ->
clause-risk) whose Agent column reads ``multi-agent``.

The trace was recorded from a real AgentWorkflow run (see
samples/data/_generate_fixtures.py) and is shipped under
samples/data/traces/industry/. This sample uploads it and evaluates the
review with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python legal_agentworkflow.py
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

SAMPLE = "legal_agentworkflow"
FIXTURE = recorded_trace_path("industry", "legal_agentworkflow.jsonl")

# The contract-review request the workflow handled. Documents the scenario; the
# recorded multi-agent trace was produced by running this through a real
# LlamaIndex AgentWorkflow (contract-intake handing off to clause-risk).
REVIEW_REQUEST: dict[str, Any] = {
    "matter_id": "MAT-70318",
    "document": "SaaS Master Agreement (Acme Corp / Widget Inc)",
    "agents": ["contract-intake", "clause-risk"],
    "summary": (
        "Review a SaaS master agreement: auto-renewal with 180-day notice, net-45 payment, "
        "UNLIMITED liability for data breaches, GDPR data-processing, confidentiality, and "
        "indemnification. Flag the legal risks and recommend changes."
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded multi-agent contract-review workflow trace."""
    print("=== LayerLens Legal: Multi-Agent Contract-Review Workflow ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded multi-agent trace first (renders as a multi-agent graph
    # via the real contract-intake -> clause-risk handoff). Do this before
    # creating judges so the trace always lands even if the org has no
    # evaluation model yet.
    print("Uploading the recorded contract-review workflow trace (multi-agent handoff graph)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Matter:   {REVIEW_REQUEST['matter_id']} -- {REVIEW_REQUEST['document']}")
    print(f"  Agents:   {' -> '.join(REVIEW_REQUEST['agents'])}\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "risk_analysis_quality": create_judge(
                client,
                name="Clause Risk Analysis Judge",
                evaluation_goal="Evaluate whether the clause-risk agent correctly identifies and rates the highest-risk clauses (e.g. unlimited data-breach liability, auto-renewal lock-in).",
                namespace=SAMPLE,
            ),
            "handoff_soundness": create_judge(
                client,
                name="Handoff Soundness Judge",
                evaluation_goal="Evaluate whether the contract-intake agent correctly summarized the contract and appropriately handed off the risk analysis to the clause-risk specialist.",
                namespace=SAMPLE,
            ),
            "recommendation_quality": create_judge(
                client,
                name="Recommendation Quality Judge",
                evaluation_goal="Evaluate whether the workflow's final recommendations are concrete, actionable, and address the flagged legal risks.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the workflow's contract review:")
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
