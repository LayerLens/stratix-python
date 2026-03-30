#!/usr/bin/env python3
"""Document Evaluation -- LayerLens Python SDK Sample.

Evaluates document processing for extraction accuracy, cross-field
consistency, and structural integrity using dedicated judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python document_evaluation.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import upload_trace_dict, poll_evaluation_results, create_judge

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLES: list[dict[str, Any]] = [
    {
        "id": "doc-001",
        "name": "Invoice (complete)",
        "document_type": "invoice",
        "ground_truth": {
            "vendor": "Acme Corp",
            "date": "2026-03-01",
            "total": 1250.00,
            "line_items": [
                {"description": "Widget A", "qty": 10, "price": 50.00},
                {"description": "Widget B", "qty": 5, "price": 150.00},
            ],
        },
        "extracted": {
            "vendor": "Acme Corp",
            "date": "2026-03-01",
            "total": 1250.00,
            "line_items": [
                {"description": "Widget A", "qty": 10, "price": 50.00},
                {"description": "Widget B", "qty": 5, "price": 150.00},
            ],
        },
    },
    {
        "id": "doc-002",
        "name": "Receipt (partial extraction)",
        "document_type": "receipt",
        "ground_truth": {
            "vendor": "Coffee Shop",
            "date": "2026-03-15",
            "total": 12.50,
            "tax": 1.06,
        },
        "extracted": {
            "vendor": "Coffee Shop",
            "date": "2026-03-15",
            "total": 12.50,
        },
    },
]

JUDGE_DEFINITIONS: list[dict[str, str]] = [
    {
        "name": "Document Extraction Judge",
        "evaluation_goal": (
            "Evaluate whether the extracted fields from the document match "
            "the ground truth. Check for missing fields, incorrect values, "
            "and extraction completeness."
        ),
    },
    {
        "name": "Document Consistency Judge",
        "evaluation_goal": (
            "Evaluate whether the extracted document fields are internally "
            "consistent. Check that totals match line items, dates are valid, "
            "and cross-field references are correct."
        ),
    },
    {
        "name": "Document Structure Judge",
        "evaluation_goal": (
            "Evaluate whether the extracted document maintains proper structural "
            "integrity. Check for correct nesting, proper field types, and "
            "valid data formats."
        ),
    },
]

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_PASS_COLOR = "\033[92m"
_FAIL_COLOR = "\033[91m"
_RESET = "\033[0m"


def display_result(label: str, score: float | None, passed: bool | None, reasoning: str) -> None:
    """Display a single evaluation result."""
    score_str = f"{score:.2f}" if score is not None else "N/A"
    if passed is not None:
        color = _PASS_COLOR if passed else _FAIL_COLOR
        status = "PASS" if passed else "FAIL"
    else:
        color = ""
        status = "PEND"
    detail = (reasoning[:60] + "...") if reasoning else ""
    print(f"  {label:14s} {color}{status:6s}{_RESET}  ({score_str}) - {detail}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run document evaluation on all samples."""
    print("=== LayerLens Document Evaluation Sample ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Create judges up front
    print(f"Creating {len(JUDGE_DEFINITIONS)} judges...")
    judges = []
    for jdef in JUDGE_DEFINITIONS:
        judge = create_judge(
            client,
            name=f"{jdef['name']} {int(time.time())}",
            evaluation_goal=jdef["evaluation_goal"],
        )
        judges.append((jdef["name"].replace("Document ", "").replace(" Judge", ""), judge))
        print(f"  Created: {judge.name} (id={judge.id})")
    print()

    if not judges:
        print("ERROR: No judges were created. Cannot proceed.")
        sys.exit(1)

    print(f"Evaluating {len(SAMPLES)} document extractions...\n")
    passed_all = 0

    try:
        for sample in SAMPLES:
            print(f"Sample: {sample['name']}")

            trace_result = upload_trace_dict(
                client,
                input_text=json.dumps(sample["ground_truth"]),
                output_text=json.dumps(sample["extracted"]),
                metadata={
                    "document_type": sample["document_type"],
                    "ground_truth": sample["ground_truth"],
                },
            )
            trace_id = trace_result.trace_ids[0] if trace_result.trace_ids else sample["id"]

            all_passed = True
            for label, judge in judges:
                evaluation = client.trace_evaluations.create(
                    trace_id=trace_id,
                    judge_id=judge.id,
                )
                if not evaluation:
                    display_result(label, None, None, "Failed to create evaluation")
                    all_passed = False
                    continue

                results = poll_evaluation_results(client, evaluation.id)
                if results:
                    r = results[0]
                    display_result(label, r.score, r.passed, r.reasoning or "")
                    if not r.passed:
                        all_passed = False
                else:
                    display_result(label, None, None, "(results pending)")
                    all_passed = False

            if all_passed:
                passed_all += 1
            print()

        print(f"Overall: {passed_all}/{len(SAMPLES)} documents passed all checks")

    finally:
        # Clean up judges
        print("\nCleaning up judges...")
        for _, judge in judges:
            try:
                client.judges.delete(judge.id)
                print(f"  Deleted: {judge.name}")
            except Exception:
                print(f"  WARNING: Could not delete judge {judge.id}")


if __name__ == "__main__":
    main()
