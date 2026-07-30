#!/usr/bin/env python3
"""Legal: Contract Review -- LayerLens Python SDK Sample.

Evaluates AI contract review for clause detection accuracy,
risk assessment quality, and confidentiality compliance.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python legal_contracts.py
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
# instrumented ``contract-review`` run over the contracts below (see
# ``samples/data/_generate_fixtures.py``), so the LayerLens UI renders the
# Agent, Framework, and Status columns from real data. The contracts remain
# here as documentation of what was analyzed and to label the evaluation output.
SAMPLE = "legal_contracts"
FIXTURE = recorded_trace_path("industry", "legal_contracts.jsonl")

CONTRACTS: list[dict[str, Any]] = [
    {
        "id": "contract-001",
        "title": "SaaS Agreement (Acme Corp / Widget Inc)",
        "clauses_identified": [
            "term_and_termination",
            "payment_terms",
            "data_protection",
            "liability_limitation",
            "confidentiality",
            "intellectual_property",
            "indemnification",
            "force_majeure",
        ],
        "clauses_expected": [
            "term_and_termination",
            "payment_terms",
            "data_protection",
            "liability_limitation",
            "confidentiality",
            "intellectual_property",
            "indemnification",
            "force_majeure",
        ],
        "risk_flags": [
            {
                "clause": "liability_limitation",
                "risk": "high",
                "note": "Unlimited liability for data breaches",
            },
            {
                "clause": "term_and_termination",
                "risk": "high",
                "note": "Auto-renewal with 180-day notice period",
            },
        ],
        "analysis_output": "Contract review identifies 8 key clauses. Two high-risk items found. Recommend negotiating liability cap and reducing notice period.",
    },
    {
        "id": "contract-002",
        "title": "NDA (Bilateral)",
        "clauses_identified": [
            "definition_of_confidential",
            "obligations",
            "exclusions",
            "term",
            "remedies",
        ],
        "clauses_expected": [
            "definition_of_confidential",
            "obligations",
            "exclusions",
            "term",
            "remedies",
            "return_of_materials",
        ],
        "risk_flags": [
            {
                "clause": "term",
                "risk": "medium",
                "note": "Perpetual NDA with no sunset clause",
            }
        ],
        "analysis_output": "NDA review identifies 5 of 6 expected clauses. Missing return of materials clause. Term is perpetual.",
    },
]

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m", "uncertain": "\033[93m"}
_RESET = "\033[0m"


def main() -> None:
    """Run contract review evaluation."""
    print("=== LayerLens Legal: Contract Review ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded real traces first. Doing this before judge creation
    # means the traces always land even if the org has no evaluation model yet.
    print(f"Uploading {len(CONTRACTS)} recorded contract-review traces...\n")
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
            "clause_detection": create_judge(
                client,
                name="Clause Detection Judge",
                evaluation_goal="Evaluate whether the contract review correctly identifies all key clauses and flags any missing required clauses.",
                namespace=SAMPLE,
            ),
            "risk_assessment": create_judge(
                client,
                name="Risk Assessment Judge",
                evaluation_goal="Evaluate whether the risk flags and risk levels assigned to contract clauses are accurate and complete.",
                namespace=SAMPLE,
            ),
            "confidentiality": create_judge(
                client,
                name="Confidentiality Judge",
                evaluation_goal="Evaluate whether the contract review properly handles confidential information and identifies confidentiality-related issues.",
                namespace=SAMPLE,
            ),
        }
        judge_labels = {
            "clause_detection": "Clause Detection",
            "risk_assessment": "Risk Assessment",
            "confidentiality": "Confidentiality",
        }
        judge_ids = [j.id for j in judges.values()]

        print(f"Reviewing {len(CONTRACTS)} contracts...\n")

        for contract, trace_id in zip(CONTRACTS, trace_ids):
            print(f"Contract: {contract['title']}")
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
                    f"  {label:20s} {color}{verdict.upper()}{_RESET} ({score:.2f}) - {reasoning}"
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
