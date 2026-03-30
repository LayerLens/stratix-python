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

import sys
from typing import Any

import os

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import upload_trace_dict, poll_evaluation_results, create_judge

# ---------------------------------------------------------------------------
# Simulated knowledge base and queries
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE: list[dict[str, Any]] = [
    {"id": "doc_001", "title": "Refund Policy", "content": "Full refunds are available within 30 days of purchase. After 30 days, store credit is issued."},
    {"id": "doc_002", "title": "Pricing Plans", "content": "We offer Free ($0), Pro ($29/mo), and Enterprise (custom) tiers. Annual billing saves 20%."},
    {"id": "doc_003", "title": "API Rate Limits", "content": "Free: 100 req/min. Pro: 1000 req/min. Enterprise: unlimited. Rate limit headers included."},
    {"id": "doc_004", "title": "Data Retention", "content": "Traces are retained for 90 days on Pro, 365 days on Enterprise. Free tier: 7 days."},
]

QUERIES: list[dict[str, Any]] = [
    {"id": "q_001", "text": "What is your refund policy?", "category": "billing", "expected_doc_ids": ["doc_001"]},
    {"id": "q_002", "text": "How much does the Pro plan cost?", "category": "pricing", "expected_doc_ids": ["doc_002"]},
    {"id": "q_003", "text": "What are the API rate limits for enterprise?", "category": "technical", "expected_doc_ids": ["doc_003"]},
]

# Simulated RAG answers
SIMULATED_ANSWERS: dict[str, str] = {
    "q_001": "Full refunds are available within 30 days. After that, you receive store credit.",
    "q_002": "The Pro plan costs $29 per month. Annual billing provides a 20% discount.",
    "q_003": "Enterprise tier has unlimited API rate limits. Rate limit headers are included in responses.",
}

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

    # Create judges up front
    judges = {
        "groundedness": create_judge(
            client,
            name="Groundedness Judge",
            evaluation_goal="Evaluate whether the response is grounded in the retrieved context and does not hallucinate.",
        ),
        "retrieval_quality": create_judge(
            client,
            name="Retrieval Quality Judge",
            evaluation_goal="Evaluate whether the retrieved documents are relevant and sufficient to answer the query.",
        ),
        "completeness": create_judge(
            client,
            name="Completeness Judge",
            evaluation_goal="Evaluate whether the response fully and completely addresses the user's question.",
        ),
    }
    judge_labels = {"groundedness": "Grounded", "retrieval_quality": "Retrieval", "completeness": "Complete"}
    judge_ids = [j.id for j in judges.values()]

    try:
        # Phase 1: RAG Runner processes queries
        print("[RAGRunner] Processing queries...\n")
        rag_results: list[dict[str, Any]] = []

        for query in QUERIES:
            answer = SIMULATED_ANSWERS.get(query["id"], "No answer available.")
            print(f'[RAGRunner] Query: "{query["text"]}"')

            # Simple retrieval simulation
            retrieved_docs = [d for d in KNOWLEDGE_BASE if d["id"] in query["expected_doc_ids"]]
            scores_str = ", ".join("0.92" for _ in retrieved_docs)
            print(f"[RAGRunner] Retrieved {len(retrieved_docs)} documents (scores: {scores_str})")

            trace_result = upload_trace_dict(client,
                input_text=query["text"],
                output_text=answer,
                metadata={
                    "query_id": query["id"],
                    "category": query["category"],
                    "retrieved_doc_ids": [d["id"] for d in retrieved_docs],
                    "channel": "co-work-rag-quality",
                },
            )
            trace_id = trace_result.trace_ids[0] if trace_result.trace_ids else f"trc_rag_{query['id']}"

            rag_results.append({
                "query_id": query["id"],
                "query_text": query["text"],
                "trace_id": trace_id,
                "answer": answer,
                "retrieved_docs": retrieved_docs,
            })
            print(f"[RAGRunner] Trace {trace_id} created.\n")

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

    finally:
        for jid in judge_ids:
            try:
                client.judges.delete(jid)
            except Exception:
                pass


if __name__ == "__main__":
    main()
