#!/usr/bin/env python3
"""Industry: Retail Catalog Q&A (tool-use) -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL single-agent tool-use workflow. A retail
product-Q&A agent (persona ``product-qa-agent``, backed by OpenAI) answers a
shopper's question by first calling a real ``lookup_product`` tool (OpenAI
``tools=`` / ``tool_calls``) to fetch the catalog record and live inventory for
the item, then answering (price, stock status, key spec) grounded only in the
tool result.

Because the trace was captured under ``@trace`` + ``instrument_openai`` from a
genuine two-step tool loop, it renders a single honest agent node (Agent column
``product-qa-agent``) plus the real ``model.invoke`` / ``tool.call`` /
``tool.result`` / ``cost.record`` events -- no fabrication. The recorded trace
is shipped under samples/data/traces/industry/ (produced by
samples/data/_generate_fixtures.py); this sample uploads it and evaluates the
answer with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python retail_catalog_qa.py
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

SAMPLE = "retail_catalog_qa"
FIXTURE = recorded_trace_path("industry", "retail_catalog_qa.jsonl")

# The shopper question the product-Q&A agent answered. Documents the scenario;
# the recorded tool-use trace was produced by running this through a real OpenAI
# tool loop (the agent called lookup_product, then answered from the record).
QUESTION: dict[str, Any] = {
    "query_id": "QA-77104",
    "sku_expected": "aeron-chair",
    "text": (
        "Do you have the Aeron ergonomic office chair in stock, how much is it, "
        "and does it support up to 300 lbs?"
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded product-Q&A tool-use trace."""
    print("=== LayerLens Industry: Retail Catalog Q&A (tool-use) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded tool-use trace first (renders a single product-qa-agent
    # node with real model.invoke / tool.call / tool.result events). Do this
    # before creating judges so the trace always lands even if the org has no
    # evaluation model yet.
    print("Uploading the recorded product-Q&A trace (lookup_product tool loop)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Query:    {QUESTION['query_id']} (sku {QUESTION['sku_expected']})\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "answer_accuracy": create_judge(
                client,
                name="Answer Accuracy Judge",
                evaluation_goal="Evaluate whether the answer correctly states the product's price, stock status, and the requested spec based on the catalog record.",
                namespace=SAMPLE,
            ),
            "catalog_grounding": create_judge(
                client,
                name="Catalog Grounding Judge",
                evaluation_goal="Evaluate whether the answer is grounded ONLY in the product record returned by the lookup_product tool and does not invent details.",
                namespace=SAMPLE,
            ),
            "helpfulness": create_judge(
                client,
                name="Shopper Helpfulness Judge",
                evaluation_goal="Evaluate whether the answer directly and clearly resolves the shopper's question with a concrete next step.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the product answer:")
        for label, judge in judges.items():
            evaluation = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge.id)
            if evaluation is None:
                print(f"  {label:22s} -- evaluation creation failed")
                continue
            results = poll_evaluation_results(client, evaluation.id)
            if results:
                r = results[0]
                verdict = "pass" if r.passed else "fail"
                color = _VERDICT_COLORS.get(verdict, "")
                print(f"  {label:22s} {color}{verdict.upper()}{_RESET} ({r.score:.2f})")
            else:
                print(f"  {label:22s} -- timed out waiting for results")
    except RuntimeError as exc:
        print(f"\nNOTE: evaluations skipped -- {exc}")
        print("  The trace is uploaded; add a project/public model to enable judges.")
    finally:
        for jid in judge_ids:
            try:
                client.judges.delete(jid)
            except Exception:
                pass


if __name__ == "__main__":
    main()
