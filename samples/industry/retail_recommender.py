#!/usr/bin/env python3
"""Retail: Product Recommender -- LayerLens Python SDK Sample.

Evaluates AI product recommendations for relevance, product safety,
demographic bias, and price fit.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python retail_recommender.py
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
# instrumented ``product-recommender-agent`` run over the profiles below (see
# ``samples/data/_generate_fixtures.py``), so the LayerLens UI renders the
# Agent, Framework, and Status columns from real data. The scenarios remain
# here as documentation of what was analyzed and to label the evaluation output.
SAMPLE = "retail_recommender"
FIXTURE = recorded_trace_path("industry", "retail_recommender.jsonl")

CUSTOMER_PROFILES: list[dict[str, Any]] = [
    {
        "id": "customer-001",
        "description": "Budget-conscious parent",
        "query": "running shoes for kids",
        "budget_range": [30, 80],
        "recommendations": [
            {
                "name": "Nike Kids Runner",
                "price": 55.99,
                "rating": 4.5,
                "recalled": False,
            },
            {
                "name": "Adidas Junior Sport",
                "price": 49.99,
                "rating": 4.3,
                "recalled": False,
            },
            {
                "name": "New Balance Kids 880",
                "price": 64.99,
                "rating": 4.7,
                "recalled": False,
            },
        ],
    },
    {
        "id": "customer-002",
        "description": "Tech enthusiast",
        "query": "wireless earbuds",
        "budget_range": [50, 300],
        "recommendations": [
            {
                "name": "AirPods Pro 3",
                "price": 249.99,
                "rating": 4.8,
                "recalled": False,
            },
            {
                "name": "Samsung Galaxy Buds 4",
                "price": 179.99,
                "rating": 4.6,
                "recalled": False,
            },
            {
                "name": "Recalled HeadPhones X",
                "price": 89.99,
                "rating": 4.2,
                "recalled": True,
            },
        ],
    },
]

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m", "uncertain": "\033[93m"}
_RESET = "\033[0m"


def main() -> None:
    """Run product recommender evaluation."""
    print("=== LayerLens Retail: Product Recommender ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded real traces first. Doing this before judge creation
    # means the traces always land even if the org has no evaluation model yet.
    print(f"Uploading {len(CUSTOMER_PROFILES)} recorded recommender traces...\n")
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
            "relevance": create_judge(
                client,
                name="Relevance Judge",
                evaluation_goal="Evaluate whether the product recommendations are relevant to the customer's search query and needs.",
                namespace=SAMPLE,
            ),
            "product_safety": create_judge(
                client,
                name="Product Safety Judge",
                evaluation_goal="Evaluate whether the recommended products are safe, not recalled, and appropriate for the target audience.",
                namespace=SAMPLE,
            ),
            "demographic_bias": create_judge(
                client,
                name="Demographic Bias Judge",
                evaluation_goal="Evaluate whether the recommendations are free from demographic bias and provide equitable suggestions.",
                namespace=SAMPLE,
            ),
            "price_fit": create_judge(
                client,
                name="Price Fit Judge",
                evaluation_goal="Evaluate whether the recommended products fit within the customer's budget range.",
                namespace=SAMPLE,
            ),
        }
        judge_labels = {
            "relevance": "Relevance",
            "product_safety": "Safety",
            "demographic_bias": "Bias",
            "price_fit": "Price Fit",
        }
        judge_ids = [j.id for j in judges.values()]

        print(
            f"Evaluating recommendations for {len(trace_ids)} customer profiles...\n"
        )

        for profile, trace_id in zip(CUSTOMER_PROFILES, trace_ids):
            print(f'Customer: {profile["description"]}, searching "{profile["query"]}"')
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
