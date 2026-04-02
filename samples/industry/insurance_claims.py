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
from _helpers import create_judge, upload_trace_dict, poll_evaluation_results

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
        "policy": {"type": "homeowners", "deductible": 1000, "max_coverage": 300000, "exclusions": ["flood"]},
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
        "policy": {"type": "health_ppo", "deductible": 2000, "copay_percent": 20, "max_oop": 8000},
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

    # Create judges up front
    judges = {
        "coverage_determination": create_judge(
            client,
            name="Coverage Determination Judge",
            evaluation_goal="Evaluate whether the coverage determination correctly applies policy terms, deductibles, and exclusions to the claim.",
        ),
        "regulatory_compliance": create_judge(
            client,
            name="Regulatory Compliance Judge",
            evaluation_goal="Evaluate whether the claims decision complies with state insurance regulations and fair claims practices.",
        ),
        "settlement_fairness": create_judge(
            client,
            name="Settlement Fairness Judge",
            evaluation_goal="Evaluate whether the settlement amount is fair and reasonable given the claim details and policy terms.",
        ),
    }
    judge_labels = {
        "coverage_determination": "Coverage",
        "regulatory_compliance": "Compliance",
        "settlement_fairness": "Fairness",
    }
    judge_ids = [j.id for j in judges.values()]

    try:
        print(f"Evaluating {len(CLAIMS)} claims decisions...\n")

        for claim in CLAIMS:
            trace_result = upload_trace_dict(
                client,
                input_text=f"{claim['type']}: {claim['description']}",
                output_text=str(claim["decision"]),
                metadata={"policy": claim["policy"], "claimed_amount": claim["claimed_amount"]},
            )
            trace_id = trace_result.trace_ids[0] if trace_result.trace_ids else claim["id"]

            print(f"Claim: {claim['type']} - {claim['description'][:40]}... (${claim['claimed_amount']:,.2f})")
            for judge_key, judge_obj in judges.items():
                label = judge_labels[judge_key]
                evaluation = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge_obj.id)
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
                print(f"  {label:12s}  {color}{verdict.upper()}{_RESET} ({score:.2f}) - {reasoning}")
            print()

    finally:
        for jid in judge_ids:
            try:
                client.judges.delete(jid)
            except Exception:
                pass


if __name__ == "__main__":
    main()
