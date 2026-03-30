#!/usr/bin/env python3
"""Insurance: Underwriting Agent -- LayerLens Python SDK Sample.

Evaluates AI underwriting decisions for risk assessment accuracy,
regulatory compliance (fair lending), and pricing consistency.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python insurance_underwriting.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import upload_trace_dict, poll_evaluation_results, create_judge

APPLICATIONS: list[dict[str, Any]] = [
    {
        "id": "uw-001",
        "applicant": {"age": 35, "location": "suburban", "credit_score": 780, "claims_history": 0},
        "coverage_type": "auto",
        "risk_assessment": {"risk_class": "preferred", "risk_score": 0.15, "premium": 1200.00, "factors": ["excellent_credit", "no_claims", "low_risk_area"]},
    },
    {
        "id": "uw-002",
        "applicant": {"age": 22, "location": "urban", "credit_score": 650, "claims_history": 2},
        "coverage_type": "auto",
        "risk_assessment": {"risk_class": "standard", "risk_score": 0.55, "premium": 2800.00, "factors": ["young_driver", "prior_claims", "urban_area"]},
    },
    {
        "id": "uw-003",
        "applicant": {"age": 45, "location": "rural", "credit_score": 720, "claims_history": 1},
        "coverage_type": "homeowners",
        "risk_assessment": {"risk_class": "standard", "risk_score": 0.35, "premium": 1800.00, "factors": ["good_credit", "single_claim", "rural_weather_risk"]},
    },
]

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m", "uncertain": "\033[93m"}
_RESET = "\033[0m"


def main() -> None:
    """Run underwriting evaluation."""
    print("=== LayerLens Insurance: Underwriting Agent ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Create judges up front
    judges = {
        "risk_accuracy": create_judge(
            client,
            name="Risk Accuracy Judge",
            evaluation_goal="Evaluate whether the risk assessment accurately reflects the applicant's risk profile based on their attributes.",
        ),
        "fair_lending": create_judge(
            client,
            name="Fair Lending Judge",
            evaluation_goal="Evaluate whether the underwriting decision complies with fair lending regulations and does not discriminate based on protected characteristics.",
        ),
        "pricing_consistency": create_judge(
            client,
            name="Pricing Consistency Judge",
            evaluation_goal="Evaluate whether the premium pricing is consistent with the risk assessment and comparable to similar risk profiles.",
        ),
    }
    judge_labels = {"risk_accuracy": "Risk Accuracy", "fair_lending": "Fair Lending", "pricing_consistency": "Pricing"}
    judge_ids = [j.id for j in judges.values()]

    try:
        for app in APPLICATIONS:
            assessment = app["risk_assessment"]
            applicant = app["applicant"]

            trace_result = upload_trace_dict(client,
                input_text=str(applicant),
                output_text=str(assessment),
                metadata={"coverage_type": app["coverage_type"], "applicant": applicant, "risk_assessment": assessment},
            )
            trace_id = trace_result.trace_ids[0] if trace_result.trace_ids else app["id"]

            print(f"Application: {app['coverage_type']} - Age {applicant['age']}, Credit {applicant['credit_score']}, Claims {applicant['claims_history']}")
            print(f"  Assessment: {assessment['risk_class']} (score={assessment['risk_score']:.2f}, premium=${assessment['premium']:,.2f})")

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
                print(f"  {label:18s} {color}{verdict.upper()}{_RESET} ({score:.2f}) - {reasoning}")
            print()

    finally:
        for jid in judge_ids:
            try:
                client.judges.delete(jid)
            except Exception:
                pass


if __name__ == "__main__":
    main()
