#!/usr/bin/env python3
"""Legal: Research Quality -- LayerLens Python SDK Sample.

Evaluates AI legal research for citation accuracy, jurisdictional
correctness, and reasoning quality.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python legal_research.py
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
# instrumented ``legal-research-assistant`` run over the queries below (see
# ``samples/data/_generate_fixtures.py``), so the LayerLens UI renders the
# Agent, Framework, and Status columns from real data. The queries remain
# here as documentation of what was analyzed and to label the evaluation output.
SAMPLE = "legal_research"
FIXTURE = recorded_trace_path("industry", "legal_research.jsonl")

RESEARCH_QUERIES: list[dict[str, Any]] = [
    {
        "id": "research-001",
        "query": "What are the requirements for enforceability of non-compete agreements in California?",
        "response": "Under California Business and Professions Code Section 16600, non-compete agreements are generally void and unenforceable in California. The only recognized exceptions are in the context of sale of a business (Section 16601), dissolution of a partnership (Section 16602), or dissolution of an LLC (Section 16602.5).",
        "citations": [
            "Cal. Bus. & Prof. Code 16600",
            "Cal. Bus. & Prof. Code 16601",
            "Edwards v. Arthur Andersen LLP (2008) 44 Cal.4th 937",
        ],
    },
    {
        "id": "research-002",
        "query": "What is the standard for piercing the corporate veil in Delaware?",
        "response": "Delaware courts apply a two-prong test: (1) the corporate entity is merely an alter ego of its owner, and (2) the corporate form was used to perpetrate fraud or injustice.",
        "citations": [
            "Mabon, Nugent & Co. v. Texas Am. Energy Corp., 1990 Del. LEXIS 312"
        ],
    },
]

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m", "uncertain": "\033[93m"}
_RESET = "\033[0m"


def main() -> None:
    """Run legal research evaluation."""
    print("=== LayerLens Legal: Research Quality ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded real traces first. Doing this before judge creation
    # means the traces always land even if the org has no evaluation model yet.
    print(f"Uploading {len(RESEARCH_QUERIES)} recorded legal-research-assistant traces...\n")
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
            "citation_accuracy": create_judge(
                client,
                name="Citation Accuracy Judge",
                evaluation_goal="Evaluate whether the legal citations are accurate, properly formatted, and support the stated legal conclusions.",
                namespace=SAMPLE,
            ),
            "jurisdictional_correctness": create_judge(
                client,
                name="Jurisdictional Correctness Judge",
                evaluation_goal="Evaluate whether the legal analysis correctly identifies and applies the relevant jurisdiction's laws and precedents.",
                namespace=SAMPLE,
            ),
            "reasoning_quality": create_judge(
                client,
                name="Reasoning Quality Judge",
                evaluation_goal="Evaluate whether the legal reasoning is logically sound, well-structured, and correctly applies legal principles.",
                namespace=SAMPLE,
            ),
        }
        judge_labels = {
            "citation_accuracy": "Citations",
            "jurisdictional_correctness": "Jurisdiction",
            "reasoning_quality": "Reasoning",
        }
        judge_ids = [j.id for j in judges.values()]

        for query, trace_id in zip(RESEARCH_QUERIES, trace_ids):
            print(f"Query: {query['query'][:60]}...")
            print(f"  Citations: {len(query['citations'])} referenced")

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
                    f"  {label:14s} {color}{verdict.upper()}{_RESET} ({score:.2f}) - {reasoning}"
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
