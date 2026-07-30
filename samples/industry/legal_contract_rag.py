#!/usr/bin/env python3
"""Legal: Contract-Analysis RAG -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL LlamaIndex retrieval-augmented (RAG) contract
review. A small in-memory ``VectorStoreIndex`` of the clauses of a SaaS master
agreement is queried by a real ``as_query_engine().query(...)`` review question;
the LlamaIndex adapter records the genuine retrieval (``tool.call``/``tool.result``
over the real indexed clauses) plus the synthesis ``model.invoke`` and priced
``cost.record``. Because the query declares no agent identity, the recorded
trace renders honestly as a single RAG query (framework ``llamaindex``, real
model + status) -- no fabricated agent.

The trace was recorded from a real instrumented query run (see
samples/data/_generate_fixtures.py) and is shipped under
samples/data/traces/industry/. This sample uploads it and evaluates the
review answer with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python legal_contract_rag.py
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

SAMPLE = "legal_contract_rag"
FIXTURE = recorded_trace_path("industry", "legal_contract_rag.jsonl")

# The contract under review and the reviewer's question. Documents the scenario;
# the recorded RAG trace was produced by indexing these clauses in a real
# LlamaIndex VectorStoreIndex and running the question through a real query
# engine (retrieval + synthesis).
CONTRACT: dict[str, Any] = {
    "matter_id": "MAT-70241",
    "document": "SaaS Master Agreement (Acme Corp / Widget Inc)",
    "clauses_indexed": [
        "term_and_termination",
        "limitation_of_liability",
        "payment_terms",
        "data_protection",
        "indemnification",
        "confidentiality",
    ],
    "review_question": (
        "What are the biggest legal risks in the liability and termination clauses, "
        "and what should we negotiate?"
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded contract-analysis RAG trace."""
    print("=== LayerLens Legal: Contract-Analysis RAG ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded RAG trace first (real retrieval + synthesis). Do this
    # before creating judges so the trace always lands even if the org has no
    # evaluation model yet.
    print("Uploading the recorded contract-analysis RAG trace (real retrieval + synthesis)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Matter:   {CONTRACT['matter_id']} -- {CONTRACT['document']}")
    print(f"  Question: {CONTRACT['review_question']}\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "risk_identification": create_judge(
                client,
                name="Contract Risk Identification Judge",
                evaluation_goal="Evaluate whether the review correctly identifies the biggest legal risks in the liability and termination clauses (e.g. unlimited data-breach liability, long auto-renewal lock-in).",
                namespace=SAMPLE,
            ),
            "retrieval_grounding": create_judge(
                client,
                name="Retrieval Grounding Judge",
                evaluation_goal="Evaluate whether the answer is grounded in the retrieved contract clauses and does not invent terms that are not in the contract.",
                namespace=SAMPLE,
            ),
            "negotiation_actionability": create_judge(
                client,
                name="Negotiation Actionability Judge",
                evaluation_goal="Evaluate whether the review gives concrete, actionable negotiation recommendations for the flagged clauses.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the contract-review answer:")
        for label, judge in judges.items():
            evaluation = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge.id)
            if evaluation is None:
                print(f"  {label:27s} -- evaluation creation failed")
                continue
            results = poll_evaluation_results(client, evaluation.id)
            if results:
                r = results[0]
                verdict = "pass" if r.passed else "fail"
                color = _VERDICT_COLORS.get(verdict, "")
                print(f"  {label:27s} {color}{verdict.upper()}{_RESET} ({r.score:.2f})")
            else:
                print(f"  {label:27s} -- timed out waiting for results")
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
