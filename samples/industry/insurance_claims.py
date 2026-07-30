#!/usr/bin/env python3
"""Insurance: Claims Processing -- LayerLens Python SDK Sample.

Evaluates AI claims adjudication for coverage determination accuracy,
state regulatory compliance, and settlement fairness.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python insurance_claims.py
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

# This sample uploads RECORDED REAL traces: each was captured from a genuine
# instrumented ``claims-adjudicator`` run over the claims below (see
# ``samples/data/_generate_fixtures.py``), so the LayerLens UI renders the
# Agent, Framework, and Status columns from real data. The claims remain
# here as documentation of what was analyzed and to label the evaluation output.
SAMPLE = "insurance_claims"
FIXTURE = recorded_trace_path("industry", "insurance_claims.jsonl")

CLAIMS: list[dict[str, Any]] = [
    {
        "id": "claim-001",
        "type": "Auto collision",
        "description": "Rear-end accident at intersection. Claimant not at fault.",
        "claimed_amount": 8500.00,
        "policy": {"type": "comprehensive", "deductible": 500, "max_coverage": 50000},
        "decision": {
            "approved": True,
            "amount": 8000.00,
            "reasoning": "Liability clearly established. Less $500 deductible.",
        },
    },
    {
        "id": "claim-002",
        "type": "Property damage",
        "description": "Water damage from burst pipe during winter freeze",
        "claimed_amount": 25000.00,
        "policy": {
            "type": "homeowners",
            "deductible": 1000,
            "max_coverage": 300000,
            "exclusions": ["flood"],
        },
        "decision": {
            "approved": True,
            "amount": 22000.00,
            "reasoning": "Burst pipe covered. Adjusted to $23,000 less $1,000 deductible.",
        },
    },
    {
        "id": "claim-003",
        "type": "Health insurance",
        "description": "Emergency room visit for chest pain, CT scan, overnight observation",
        "claimed_amount": 15000.00,
        "policy": {
            "type": "health_ppo",
            "deductible": 2000,
            "copay_percent": 20,
            "max_oop": 8000,
        },
        "decision": {
            "approved": True,
            "amount": 10400.00,
            "reasoning": "ER visit medically necessary. Insurance pays: $10,400.",
        },
    },
]

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m", "uncertain": "\033[93m"}
_RESET = "\033[0m"


def main() -> None:
    """Run claims processing evaluation."""
    print("=== LayerLens Insurance: Claims Processing ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded real traces first. Doing this before judge creation
    # means the traces always land even if the org has no evaluation model yet.
    print(f"Uploading {len(CLAIMS)} recorded claims-adjudicator traces...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no traces uploaded (fixture missing or rejected).")
        sys.exit(1)

    # Create judges. If the org has no models available, judge creation raises
    # RuntimeError -- we skip the evaluations (the traces are already uploaded)
    # rather than crash.
    judge_ids: list[str] = []
    try:
        judges = {
            "coverage_determination": create_judge(
                client,
                name="Coverage Determination Judge",
                evaluation_goal="Evaluate whether the coverage determination correctly applies policy terms, deductibles, and exclusions to the claim.",
                namespace=SAMPLE,
            ),
            "regulatory_compliance": create_judge(
                client,
                name="Regulatory Compliance Judge",
                evaluation_goal="Evaluate whether the claims decision complies with state insurance regulations and fair claims practices.",
                namespace=SAMPLE,
            ),
            "settlement_fairness": create_judge(
                client,
                name="Settlement Fairness Judge",
                evaluation_goal="Evaluate whether the settlement amount is fair and reasonable given the claim details and policy terms.",
                namespace=SAMPLE,
            ),
        }
        judge_labels = {
            "coverage_determination": "Coverage",
            "regulatory_compliance": "Compliance",
            "settlement_fairness": "Fairness",
        }
        judge_ids = [j.id for j in judges.values()]

        print(f"Evaluating {len(CLAIMS)} claims decisions...\n")

        for claim, trace_id in zip(CLAIMS, trace_ids):
            print(
                f"Claim: {claim['type']} - {claim['description'][:40]}... (${claim['claimed_amount']:,.2f})"
            )
            for judge_key, judge_obj in judges.items():
                label = judge_labels[judge_key]
                evaluation = client.trace_evaluations.create(
                    trace_id=trace_id, judge_id=judge_obj.id
                )
                results = poll_evaluation_results(client, evaluation.id)
                score = 0.0
                passed = False
                reasoning = ""
                if results:
                    r = results[0]
                    score = r.score
                    passed = r.passed
                    reasoning = r.reasoning
                verdict = "pass" if passed else "fail"
                color = _VERDICT_COLORS.get(verdict, "")
                print(
                    f"  {label:12s}  {color}{verdict.upper()}{_RESET} ({score:.2f}) - {reasoning}"
                )
            print()

    except RuntimeError as exc:
        print(f"\nNOTE: evaluations skipped -- {exc}")
        print("  Traces are uploaded; add a project/public model to enable judges.")
    finally:
        for jid in judge_ids:
            try:
                client.judges.delete(jid)
            except Exception:
                pass


if __name__ == "__main__":
    main()
