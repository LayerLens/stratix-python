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
from _helpers import (
    create_judge,
    poll_evaluation_results,
    recorded_trace_path,
    upload_recorded_trace,
)

# This sample uploads RECORDED REAL traces: each was captured from a genuine
# instrumented ``fraud-risk-analyzer`` run over the scenarios below (see
# ``samples/data/_generate_fixtures.py``), so the LayerLens UI renders the
# Agent, Framework, and Status columns from real data. The scenarios remain
# here as documentation of what was analyzed and to label the evaluation output.
SAMPLE = "financial_fraud"
FIXTURE = recorded_trace_path("industry", "financial_fraud.jsonl")

TRANSACTIONS: list[dict[str, Any]] = [
    {
        "id": "txn-001",
        "amount": 45.99,
        "merchant": "Office Depot",
        "category": "office_supplies",
        "description": "Routine office supply purchase",
        "risk_factors": [],
    },
    {
        "id": "txn-002",
        "amount": 12500.00,
        "merchant": "Offshore Holdings Ltd",
        "category": "wire_transfer",
        "description": "Wire transfer to offshore account",
        "risk_factors": [
            "large_amount",
            "offshore_destination",
            "first_time_recipient",
        ],
    },
    {
        "id": "txn-003",
        "amount": 9999.00,
        "merchant": "Currency Exchange",
        "category": "currency_exchange",
        "description": "Cash purchase just below reporting threshold",
        "risk_factors": ["structuring_pattern", "cash_transaction", "near_threshold"],
    },
    {
        "id": "txn-004",
        "amount": 299.99,
        "merchant": "Amazon",
        "category": "retail",
        "description": "Online purchase matching user profile",
        "risk_factors": [],
    },
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

    # Upload the recorded real traces first. Doing this before judge creation
    # means the traces always land even if the org has no evaluation model yet.
    print(f"Uploading {len(TRANSACTIONS)} recorded fraud-analysis traces...\n")
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
            "fraud_risk": create_judge(
                client,
                name="Fraud Risk Judge",
                evaluation_goal="Evaluate the fraud risk score of the transaction based on amount, merchant, and risk factors.",
                namespace=SAMPLE,
            ),
            "financial_guardrail": create_judge(
                client,
                name="Financial Guardrail Judge",
                evaluation_goal="Evaluate whether the transaction complies with financial guardrails and regulatory limits.",
                namespace=SAMPLE,
            ),
            "aml_compliance": create_judge(
                client,
                name="AML Compliance Judge",
                evaluation_goal="Evaluate whether the transaction shows patterns consistent with anti-money laundering (AML) violations such as structuring or suspicious activity.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print(f"Analyzing {len(trace_ids)} transactions...\n")

        for txn, trace_id in zip(TRANSACTIONS, trace_ids):
            # Evaluate with all judges and collect results
            eval_results: dict[str, Any] = {}
            for judge_key, judge_obj in judges.items():
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
                eval_results[judge_key] = {
                    "score": score,
                    "passed": passed,
                    "reasoning": reasoning,
                }

            print(
                f"Transaction: ${txn['amount']:,.2f} at {txn['merchant']} ({txn['description'][:40]})"
            )

            fraud = eval_results["fraud_risk"]
            score = fraud["score"]
            risk_level = "HIGH" if score > 0.7 else "MEDIUM" if score > 0.3 else "LOW"
            print(
                f"  Fraud Score:  {score:.2f} ({_RISK_COLORS.get(risk_level.lower(), '')}{risk_level} RISK{_RESET})"
            )

            guardrail = eval_results["financial_guardrail"]
            verdict = "pass" if guardrail["passed"] else "fail"
            reasoning = guardrail["reasoning"]
            color = _VERDICT_COLORS.get(verdict, "")
            print(f"  Guardrail:    {color}{verdict.upper()}{_RESET} - {reasoning}")

            aml = eval_results["aml_compliance"]
            if not aml["passed"]:
                print(f"  AML Flag:     {aml['reasoning']}")

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
