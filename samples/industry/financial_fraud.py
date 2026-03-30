#!/usr/bin/env python3
"""Financial Services: Fraud Detection -- LayerLens Python SDK Sample.

Evaluates transaction analysis AI for fraud risk scoring accuracy,
financial guardrail compliance, and AML pattern detection.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python financial_fraud.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import upload_trace_dict, poll_evaluation_results, create_judge

TRANSACTIONS: list[dict[str, Any]] = [
    {"id": "txn-001", "amount": 45.99, "merchant": "Office Depot", "category": "office_supplies", "description": "Routine office supply purchase", "risk_factors": []},
    {"id": "txn-002", "amount": 12500.00, "merchant": "Offshore Holdings Ltd", "category": "wire_transfer", "description": "Wire transfer to offshore account", "risk_factors": ["large_amount", "offshore_destination", "first_time_recipient"]},
    {"id": "txn-003", "amount": 9999.00, "merchant": "Currency Exchange", "category": "currency_exchange", "description": "Cash purchase just below reporting threshold", "risk_factors": ["structuring_pattern", "cash_transaction", "near_threshold"]},
    {"id": "txn-004", "amount": 299.99, "merchant": "Amazon", "category": "retail", "description": "Online purchase matching user profile", "risk_factors": []},
]

_RISK_COLORS = {"low": "\033[92m", "medium": "\033[93m", "high": "\033[91m"}
_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Run fraud detection analysis."""
    print("=== LayerLens Financial Services: Fraud Detection ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Create judges up front
    judges = {
        "fraud_risk": create_judge(
            client,
            name="Fraud Risk Judge",
            evaluation_goal="Evaluate the fraud risk score of the transaction based on amount, merchant, and risk factors.",
        ),
        "financial_guardrail": create_judge(
            client,
            name="Financial Guardrail Judge",
            evaluation_goal="Evaluate whether the transaction complies with financial guardrails and regulatory limits.",
        ),
        "aml_compliance": create_judge(
            client,
            name="AML Compliance Judge",
            evaluation_goal="Evaluate whether the transaction shows patterns consistent with anti-money laundering (AML) violations such as structuring or suspicious activity.",
        ),
    }
    judge_ids = [j.id for j in judges.values()]

    try:
        print(f"Analyzing {len(TRANSACTIONS)} transactions...\n")

        for txn in TRANSACTIONS:
            trace_result = upload_trace_dict(client,
                input_text=str(txn),
                output_text=f"Risk assessment for {txn['merchant']}: {txn['description']}",
                metadata={"amount": txn["amount"], "merchant": txn["merchant"], "category": txn["category"], "risk_factors": txn["risk_factors"]},
            )
            trace_id = trace_result.trace_ids[0] if trace_result.trace_ids else txn["id"]

            # Evaluate with all judges and collect results
            eval_results: dict[str, Any] = {}
            for judge_key, judge_obj in judges.items():
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
                eval_results[judge_key] = {"score": score, "passed": passed, "reasoning": reasoning}

            print(f"Transaction: ${txn['amount']:,.2f} at {txn['merchant']} ({txn['description'][:40]})")

            fraud = eval_results["fraud_risk"]
            score = fraud["score"]
            risk_level = "HIGH" if score > 0.7 else "MEDIUM" if score > 0.3 else "LOW"
            print(f"  Fraud Score:  {score:.2f} ({_RISK_COLORS.get(risk_level.lower(), '')}{risk_level} RISK{_RESET})")

            guardrail = eval_results["financial_guardrail"]
            verdict = "pass" if guardrail["passed"] else "fail"
            reasoning = guardrail["reasoning"]
            color = _VERDICT_COLORS.get(verdict, "")
            print(f"  Guardrail:    {color}{verdict.upper()}{_RESET} - {reasoning}")

            aml = eval_results["aml_compliance"]
            if not aml["passed"]:
                print(f"  AML Flag:     {aml['reasoning']}")

            print()

    finally:
        for jid in judge_ids:
            try:
                client.judges.delete(jid)
            except Exception:
                pass


if __name__ == "__main__":
    main()
