#!/usr/bin/env python3
"""Government: Citizen Services -- LayerLens Python SDK Sample.

Evaluates government AI agent for regulatory accuracy, accessibility
(plain language, reading level), and equity (consistent across demographics).

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python government_citizen.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import create_judge, upload_trace_dict, poll_evaluation_results

CITIZEN_INQUIRIES: list[dict[str, Any]] = [
    {
        "id": "citizen-001",
        "program": "SNAP",
        "inquiry": "Am I eligible for SNAP benefits? I'm a single parent with 2 children, income $2,200/month.",
        "response": "Based on the information provided, you may be eligible for SNAP benefits. For a household of 3, the gross monthly income limit is $2,311. Your income of $2,200 is below this threshold. To apply, visit your local SNAP office or apply online. Processing typically takes 30 days.",
    },
    {
        "id": "citizen-002",
        "program": "Medicaid",
        "inquiry": "I just lost my job. Can I get health insurance through Medicaid?",
        "response": "You may qualify for Medicaid depending on your state and income. In expansion states, adults under 65 with income up to 138% of FPL are eligible. Job loss is a qualifying life event for Marketplace insurance within 60 days. Visit Healthcare.gov to check eligibility.",
    },
    {
        "id": "citizen-003",
        "program": "Veterans Benefits",
        "inquiry": "How do I apply for VA disability compensation?",
        "response": "To apply for VA disability compensation, file online at VA.gov, by mail using VA Form 21-526EZ, or in person. You'll need your DD214, medical records, and evidence connecting your disability to service. Processing averages 3-4 months.",
    },
]

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m", "uncertain": "\033[93m"}
_RESET = "\033[0m"


def main() -> None:
    """Run citizen services evaluation."""
    print("=== LayerLens Government: Citizen Services Agent ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Create judges up front
    judges = {
        "regulatory_accuracy": create_judge(
            client,
            name="Regulatory Accuracy Judge",
            evaluation_goal="Evaluate whether the response accurately reflects current regulations, eligibility criteria, and program requirements.",
        ),
        "accessibility": create_judge(
            client,
            name="Accessibility Judge",
            evaluation_goal="Evaluate whether the response uses plain language at an appropriate reading level and is accessible to all citizens.",
        ),
        "equity": create_judge(
            client,
            name="Equity Judge",
            evaluation_goal="Evaluate whether the response provides equitable treatment and consistent information regardless of demographics.",
        ),
    }
    judge_labels = {"regulatory_accuracy": "Accuracy", "accessibility": "Accessibility", "equity": "Equity"}
    judge_ids = [j.id for j in judges.values()]

    try:
        print(f"Evaluating {len(CITIZEN_INQUIRIES)} citizen interactions...\n")

        for inquiry in CITIZEN_INQUIRIES:
            trace_result = upload_trace_dict(
                client,
                input_text=inquiry["inquiry"],
                output_text=inquiry["response"],
                metadata={"program": inquiry["program"]},
            )
            trace_id = trace_result.trace_ids[0] if trace_result.trace_ids else inquiry["id"]

            print(f"Inquiry: {inquiry['program']} - {inquiry['inquiry'][:50]}...")
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
                print(f"  {label:16s} {color}{verdict.upper()}{_RESET} ({score:.2f}) - {reasoning}")
            print()

    finally:
        for jid in judge_ids:
            try:
                client.judges.delete(jid)
            except Exception:
                pass


if __name__ == "__main__":
    main()
