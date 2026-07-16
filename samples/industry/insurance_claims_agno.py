#!/usr/bin/env python3
"""Industry: Insurance Claims Intake (Agno tool-use) -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL single-agent tool-use workflow built with the
``agno`` framework. An auto-insurance ``claims-intake-agent`` (an ``agno.agent.Agent``
backed by OpenAI) reviews a claim, calls a real ``lookup_policy`` function tool to
fetch the policy's coverage terms, deductible, and exclusions, then tells the
customer whether the loss is covered -- grounded only in the policy the tool
returned.

Because the trace was captured through the real ``AgnoAdapter`` (which wraps
``Agent.run``), it renders a single honest agent node (Agent column =
``claims-intake-agent``) plus the real ``model.invoke`` / ``cost.record`` /
``tool.call`` / ``tool.result`` events of the tool-use turn -- no fabrication. The
recorded trace is shipped under samples/data/traces/industry/ (produced by
samples/data/generators/agno.py); this sample uploads it and evaluates the claims
decision with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python insurance_claims_agno.py
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

SAMPLE = "insurance_claims_agno"
FIXTURE = recorded_trace_path("industry", "insurance_claims_agno.jsonl")

# The claim the intake agent handled. Documents the scenario; the recorded
# tool-use trace was produced by running this through a real agno Agent that
# called lookup_policy, then answered against the returned policy terms.
CLAIM: dict[str, Any] = {
    "claim_id": "CLM-77120",
    "policy_id": "AUTO-2024-8891",
    "loss_type": "collision",
    "repair_estimate_usd": 4200,
    "description": (
        "Car damaged in a parking-lot collision; asking whether it is covered, the "
        "deductible, and any exclusions."
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded agno claims-intake tool-use trace."""
    print("=== LayerLens Industry: Insurance Claims Intake (Agno tool-use) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded tool-use trace first (renders a single claims-intake-agent
    # node with real model.invoke / tool.call / tool.result events). Do this before
    # creating judges so the trace always lands even if the org has no evaluation
    # model yet.
    print("Uploading the recorded claims-intake trace (lookup_policy tool turn)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Claim:    {CLAIM['claim_id']} (policy {CLAIM['policy_id']}, {CLAIM['loss_type']})\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "coverage_accuracy": create_judge(
                client,
                name="Coverage Accuracy Judge",
                evaluation_goal="Evaluate whether the coverage decision (covered/not covered, deductible, exclusions) correctly applies the policy terms returned by the lookup_policy tool.",
                namespace=SAMPLE,
            ),
            "policy_grounding": create_judge(
                client,
                name="Policy Grounding Judge",
                evaluation_goal="Evaluate whether the agent grounded its answer in the policy the lookup_policy tool actually returned rather than inventing coverage terms.",
                namespace=SAMPLE,
            ),
            "customer_clarity": create_judge(
                client,
                name="Customer Clarity Judge",
                evaluation_goal="Evaluate whether the response clearly and correctly states the deductible and next step in plain language for the customer.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the claims-intake decision:")
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
