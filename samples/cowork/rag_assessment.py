#!/usr/bin/env python3
"""Co-Work: RAG Quality Assessment -- LayerLens Python SDK Sample.

Demonstrates a Co-Work Channel where a RAG Runner agent executes
queries against a knowledge base and a Quality Judge agent evaluates
retrieval quality, answer groundedness, and completeness.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python rag_assessment.py
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
# instrumented ``rag-qa-agent`` run answering the queries below against the
# knowledge base (see ``samples/data/_generate_fixtures.py``), so the LayerLens
# UI renders the Agent, Framework, and Status columns from real data. The
# knowledge base and queries remain here as documentation of what was retrieved
# and to label the evaluation output; the real grounded answers live in the
# fixture.
SAMPLE = "rag_assessment"
FIXTURE = recorded_trace_path("cowork", "rag_assessment.jsonl")

# ---------------------------------------------------------------------------
# Knowledge base and queries (labels for the recorded traces above)
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE: list[dict[str, Any]] = [
    {
        "id": "doc_001",
        "title": "Refund Policy",
        "content": "Full refunds are available within 30 days of purchase. After 30 days, store credit is issued.",
    },
    {
        "id": "doc_002",
        "title": "Pricing Plans",
        "content": "We offer Free ($0), Pro ($29/mo), and Enterprise (custom) tiers. Annual billing saves 20%.",
    },
    {
        "id": "doc_003",
        "title": "API Rate Limits",
        "content": "Free: 100 req/min. Pro: 1000 req/min. Enterprise: unlimited. Rate limit headers included.",
    },
    {
        "id": "doc_004",
        "title": "Data Retention",
        "content": "Traces are retained for 90 days on Pro, 365 days on Enterprise. Free tier: 7 days.",
    },
]

QUERIES: list[dict[str, Any]] = [
    {
        "id": "q_001",
        "text": "What is your refund policy?",
        "category": "billing",
        "expected_doc_ids": ["doc_001"],
    },
    {
        "id": "q_002",
        "text": "How much does the Pro plan cost?",
        "category": "pricing",
        "expected_doc_ids": ["doc_002"],
    },
    {
        "id": "q_003",
        "text": "What are the API rate limits for enterprise?",
        "category": "technical",
        "expected_doc_ids": ["doc_003"],
    },
]

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m", "uncertain": "\033[93m"}
_RESET = "\033[0m"


def main() -> None:
    """Run the RAG quality assessment Co-Work Channel demo."""
    print("=== LayerLens Co-Work: RAG Quality Assessment ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded real traces first. Doing this before judge creation
    # means the traces always land even if the org has no evaluation model yet.
    print(f"[RAGRunner] Uploading {len(QUERIES)} recorded RAG traces...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no traces uploaded (fixture missing or rejected).")
        sys.exit(1)

    judge_labels = {
        "groundedness": "Grounded",
        "retrieval_quality": "Retrieval",
        "completeness": "Complete",
    }

    # Create judges. If the org has no models available, judge creation raises
    # RuntimeError -- we skip the evaluations (the traces are already uploaded)
    # rather than crash.
    judge_ids: list[str] = []
    try:
        judges = {
            "groundedness": create_judge(
                client,
                name="Groundedness Judge",
                evaluation_goal="Evaluate whether the response is grounded in the retrieved context and does not hallucinate.",
                namespace=SAMPLE,
            ),
            "retrieval_quality": create_judge(
                client,
                name="Retrieval Quality Judge",
                evaluation_goal="Evaluate whether the retrieved documents are relevant and sufficient to answer the query.",
                namespace=SAMPLE,
            ),
            "completeness": create_judge(
                client,
                name="Completeness Judge",
                evaluation_goal="Evaluate whether the response fully and completely addresses the user's question.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        # Phase 1: RAG Runner maps recorded traces to queries
        print("[RAGRunner] Processing queries...\n")
        rag_results: list[dict[str, Any]] = []

        for query, trace_id in zip(QUERIES, trace_ids):
            print(f'[RAGRunner] Query: "{query["text"]}"')

            # Retrieval by ID (no similarity scoring -- scores come from judge evaluation below)
            retrieved_docs = [
                d for d in KNOWLEDGE_BASE if d["id"] in query["expected_doc_ids"]
            ]
            print(f"[RAGRunner] Retrieved {len(retrieved_docs)} document(s)")

            rag_results.append(
                {
                    "query_id": query["id"],
                    "query_text": query["text"],
                    "trace_id": trace_id,
                    "retrieved_docs": retrieved_docs,
                }
            )
            print(f"[RAGRunner] Trace {trace_id} mapped.\n")

        # Phase 2: Quality Judge evaluates
        print("[QualityJudge] Evaluating RAG quality...\n")
        for result in rag_results:
            print(f"[QualityJudge] Evaluating: {result['query_text'][:50]}...")
            for judge_key, judge_obj in judges.items():
                label = judge_labels[judge_key]
                evaluation = client.trace_evaluations.create(
                    trace_id=result["trace_id"],
                    judge_id=judge_obj.id,
                )
                results = poll_evaluation_results(client, evaluation.id)
                score = 0.0
                passed = False
                if results:
                    r = results[0]
                    score = r.score
                    passed = r.passed
                verdict = "pass" if passed else "fail"
                color = _VERDICT_COLORS.get(verdict, "")
                print(f"  {label:12s} {color}{verdict.upper()}{_RESET} ({score:.2f})")
            print()

        print(f"[QualityJudge] All {len(rag_results)} queries evaluated.")

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
