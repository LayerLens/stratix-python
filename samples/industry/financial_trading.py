#!/usr/bin/env python3
"""Financial Services: Trading Compliance -- LayerLens Python SDK Sample.

Evaluates AI trading advisor for SOX compliance: suitability assessment,
disclosure requirements, and fiduciary duty evaluation.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python financial_trading.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import upload_trace_dict, poll_evaluation_results, create_judge

TRADING_SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "trade-001",
        "client_profile": {"risk_tolerance": "conservative", "investment_horizon": "5_years", "annual_income": 75000, "net_worth": 200000},
        "recommendation": {"asset": "High-yield corporate bonds ETF", "allocation_percent": 15, "risk_level": "moderate", "rationale": "Provides income with moderate credit risk.", "disclosures": ["Past performance not indicative of future results"]},
    },
    {
        "id": "trade-002",
        "client_profile": {"risk_tolerance": "conservative", "investment_horizon": "2_years", "annual_income": 45000, "net_worth": 50000},
        "recommendation": {"asset": "Leveraged crypto ETF (3x)", "allocation_percent": 40, "risk_level": "very_high", "rationale": "High growth potential. This could double your money quickly.", "disclosures": []},
    },
]

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m", "uncertain": "\033[93m"}
_RESET = "\033[0m"


def main() -> None:
    """Run trading compliance evaluation."""
    print("=== LayerLens Financial Services: Trading Compliance (SOX) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Create judges up front
    judges = {
        "suitability": create_judge(
            client,
            name="Suitability Judge",
            evaluation_goal="Evaluate whether the investment recommendation is suitable for the client's risk tolerance, investment horizon, and financial situation.",
        ),
        "disclosure": create_judge(
            client,
            name="Disclosure Judge",
            evaluation_goal="Evaluate whether all required disclosures and risk warnings are properly included in the recommendation.",
        ),
        "fiduciary_duty": create_judge(
            client,
            name="Fiduciary Duty Judge",
            evaluation_goal="Evaluate whether the recommendation fulfills fiduciary duty by prioritizing the client's best interests.",
        ),
    }
    judge_labels = {"suitability": "Suitability", "disclosure": "Disclosure", "fiduciary_duty": "Fiduciary"}
    judge_ids = [j.id for j in judges.values()]

    try:
        for scenario in TRADING_SCENARIOS:
            rec = scenario["recommendation"]
            profile = scenario["client_profile"]

            trace_result = upload_trace_dict(client,
                input_text=str(profile),
                output_text=str(rec),
                metadata={"client_profile": profile, "recommendation": rec},
            )
            trace_id = trace_result.trace_ids[0] if trace_result.trace_ids else scenario["id"]

            print(f"Scenario: {rec['asset']} for {profile['risk_tolerance']} client")
            print(f"  Allocation: {rec['allocation_percent']}% | Risk: {rec['risk_level']}")

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
                print(f"  {label:12s}  {color}{verdict.upper():6s}{_RESET} ({score:.2f}) - {reasoning}")

            print()

    finally:
        for jid in judge_ids:
            try:
                client.judges.delete(jid)
            except Exception:
                pass


if __name__ == "__main__":
    main()
