#!/usr/bin/env python3
"""Industry: Legal Contract-Clause RAG (Haystack, multi-component) -- LayerLens SDK Sample.

Demonstrates evaluating a REAL multi-component Haystack RAG pipeline. A retriever
(BM25 over a contract-clause corpus) -> prompt_builder -> llm (an ``OpenAIGenerator``
backed by OpenAI gpt-4o-mini) answers a contract question about how the
indemnification obligation interacts with the limitation-of-liability cap when the
Vendor's software causes a data breach.

Because the trace was captured under the real ``HaystackAdapter`` from a genuine
``Pipeline.run``, it renders THREE honest Haystack component nodes -- ``retriever``,
``prompt_builder``, and ``llm`` -- as a real component DAG, plus the retriever
``tool.call`` / ``tool.result``, the real ``model.invoke`` / ``cost.record``, and
the pipeline lifecycle. Nothing is fabricated.

HONESTY NOTE: Haystack is a *pipeline* framework, NOT a multi-agent one. It emits
no ``agent.identity`` and no ``agent.handoff`` -- the honest multi-node topology is
a *component* DAG (retriever -> prompt_builder -> llm), not an agent handoff/
delegation graph. So the Agent column renders honest empty-state (no agent is
invented); the graph is the component pipeline and the Framework column is
``haystack``. The token/cost fields are real.

The recorded trace is shipped under samples/data/traces/industry/ (produced by
samples/data/_generate_fixtures.py via samples/data/generators/haystack.py); this
sample uploads it and evaluates the RAG answer with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python legal_haystack_rag.py
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

SAMPLE = "legal_haystack_rag"
FIXTURE = recorded_trace_path("industry", "legal_haystack_rag.jsonl")

# The RAG question the Haystack pipeline answered. Documents the scenario; the
# recorded multi-component trace was produced by running this through a real
# Haystack RAG pipeline (BM25 retriever -> prompt_builder -> OpenAIGenerator).
QUERY: dict[str, Any] = {
    "matter_id": "MSA-2026-0417",
    "question": (
        "If the Vendor's software causes a customer data breach, what does the "
        "indemnification clause require, and does the limitation-of-liability cap "
        "apply to that indemnification obligation?"
    ),
    "components": ["retriever", "prompt_builder", "llm"],
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded multi-component Haystack RAG trace."""
    print("=== LayerLens Industry: Legal Contract-Clause RAG (Haystack, multi-component) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded RAG trace first (renders a Haystack component DAG:
    # retriever -> prompt_builder -> llm, with real tool.call / model.invoke /
    # cost.record events). Do this before creating judges so the trace always
    # lands even if the org has no evaluation model yet.
    print("Uploading the recorded Haystack RAG trace (retriever -> prompt_builder -> llm)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID:   {trace_id}")
    print(f"  Matter:     {QUERY['matter_id']}")
    print(f"  Components: {' -> '.join(QUERY['components'])}\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "retrieval_grounding": create_judge(
                client,
                name="Retrieval Grounding Judge",
                evaluation_goal="Evaluate whether the answer is grounded in the retrieved contract excerpts and cites the relevant section numbers rather than inventing terms.",
                namespace=SAMPLE,
            ),
            "legal_accuracy": create_judge(
                client,
                name="Legal Accuracy Judge",
                evaluation_goal="Evaluate whether the answer correctly explains the Vendor's indemnification obligation for a Vendor-caused breach AND that the indemnification carve-out means the limitation-of-liability cap does not apply to it.",
                namespace=SAMPLE,
            ),
            "completeness": create_judge(
                client,
                name="Answer Completeness Judge",
                evaluation_goal="Evaluate whether the answer addresses BOTH parts of the question: what the indemnification clause requires and how the liability cap interacts with it.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the RAG answer:")
        for label, judge in judges.items():
            evaluation = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge.id)
            if evaluation is None:
                print(f"  {label:20s} -- evaluation creation failed")
                continue
            results = poll_evaluation_results(client, evaluation.id)
            if results:
                r = results[0]
                verdict = "pass" if r.passed else "fail"
                color = _VERDICT_COLORS.get(verdict, "")
                print(f"  {label:20s} {color}{verdict.upper()}{_RESET} ({r.score:.2f})")
            else:
                print(f"  {label:20s} -- timed out waiting for results")
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
