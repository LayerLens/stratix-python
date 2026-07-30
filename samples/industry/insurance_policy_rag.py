#!/usr/bin/env python3
"""Industry: Insurance Policy RAG -- embed->retrieve Loop -- LayerLens Python SDK Sample.

Demonstrates observing a REAL retrieval-augmented (RAG) retrieval loop with
LayerLens. A batch of policyholder questions (windshield crack, parking-lot
dent, vehicle theft, basement water damage) is answered by, for each question,
embedding the query with REAL OpenAI ``text-embedding-3-small`` (the real
``EmbeddingAdapter``) and retrieving the nearest clauses from a real in-process
Chroma index of an insurance policy document (the real ``VectorStoreAdapter``).
Each turn records a genuine ``embedding.create`` (1536-D vector, real token
usage) and a genuine ``retrieval.query`` (real cosine distances) -- nothing is
fabricated.

HONEST EMPTY-STATE / NOT MULTI-AGENT: the embedding + vector-store adapters are
metadata-only, NON-agentic cross-cutting instrumentation. This "multi" lane is a
genuine embed->retrieve retrieval LOOP across two adapters, NOT a multi-agent
graph -- so it renders honestly as an **empty-state** (Agent column = "-") plus
an event waterfall of the real embed/retrieve pairs. LayerLens invents no agent
and no handoff graph (``genuinely_multi_agent`` is marked false in the fixture).
The value is observability of the RAG retrieval layer: per-query embedding cost
signal (tokens/dims) and retrieval quality (match count, distance band, latency).

The trace was recorded from a real instrumented run (see
samples/data/generators/embedding.py) and is shipped under
samples/data/traces/industry/. This sample uploads it and evaluates the
retrieval health with observability judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python insurance_policy_rag.py
"""

from __future__ import annotations

import os
import sys
import json
from typing import Any

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import (
    create_judge,
    poll_evaluation_results,
    recorded_trace_path,
    upload_recorded_trace,
)

SAMPLE = "insurance_policy_rag"
FIXTURE = recorded_trace_path("industry", "insurance_policy_rag.jsonl")

# The RAG retrieval batch. Documents the workload; the recorded trace was
# produced by embedding each question with real OpenAI embeddings and retrieving
# the nearest clauses from a real Chroma index of the policy document.
SCENARIO: dict[str, Any] = {
    "policy_id": "AUTO-2024-8891",
    "knowledge_base": "Personal Auto + Homeowners policy (12 indexed clauses)",
    "questions": [
        "A rock cracked my windshield on the highway. Is the glass covered and do I pay a deductible?",
        "I backed into a pole in a parking lot and dented the bumper. What deductible applies?",
        "My covered car was stolen from my driveway overnight. Which coverage handles theft?",
        "A pipe burst and water flooded my basement. Does my homeowners policy cover that?",
    ],
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def _rag_telemetry(fixture_path: str) -> dict[str, Any]:
    """Read the shipped fixture and summarize the REAL embed->retrieve telemetry
    LayerLens surfaces (per-query embedding dims/tokens + retrieval distances)."""
    with open(fixture_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    events = records[0].get("events", []) if records else []
    emb = [e.get("payload", {}) for e in events if e.get("event_type") == "embedding.create"]
    ret = [e.get("payload", {}) for e in events if e.get("event_type") == "retrieval.query"]
    return {
        "embed_calls": len(emb),
        "retrieval_calls": len(ret),
        "embed_model": next((p.get("model") for p in emb if p.get("model")), None),
        "dims": sorted({p.get("dimensions") for p in emb if p.get("dimensions")}),
        "total_embed_tokens": sum(p.get("total_tokens") or 0 for p in emb),
        "distance_means": [p.get("distance_mean") for p in ret],
    }


def main() -> None:
    """Upload + evaluate the recorded embed->retrieve RAG-loop trace."""
    print("=== LayerLens Industry: Insurance Policy RAG (embed->retrieve loop) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded RAG-loop trace first (renders an honest empty-state
    # waterfall of the real embed/retrieve pairs). Do this before judges so the
    # trace always lands even if the org has no evaluation model yet.
    print("Uploading the recorded policy-RAG trace (real embed->retrieve telemetry)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]

    tel = _rag_telemetry(FIXTURE)
    print(f"  Trace ID:  {trace_id}")
    print(f"  Policy:    {SCENARIO['policy_id']} -- {SCENARIO['knowledge_base']}")
    print(f"  RAG loop:  {tel['embed_calls']} embed->retrieve turns over {len(SCENARIO['questions'])} questions")
    print("  Embedding: model=%s dims=%s  total_tokens=%s"
          % (tel["embed_model"], tel["dims"], tel["total_embed_tokens"]))
    print("  Retrieval: %s queries  cosine-distance means=%s"
          % (tel["retrieval_calls"], tel["distance_means"]))
    print("  Render:    honest empty-state (Agent = '-') + embed/retrieve waterfall "
          "(NOT a multi-agent graph)\n")

    # Best-effort read-back so the sample proves the trace landed and is queryable.
    try:
        fetched = client.traces.get(trace_id)
        if fetched is not None:
            print("  Read-back: trace is stored and queryable in LayerLens.\n")
    except Exception:
        pass

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    # These assess RETRIEVAL OBSERVABILITY/HEALTH from the surfaced telemetry --
    # the honest evaluation surface for a metadata-only retrieval-loop trace.
    judge_ids: list[str] = []
    try:
        judges = {
            "retrieval_health": create_judge(
                client,
                name="RAG Retrieval Health Judge",
                evaluation_goal="Evaluate whether each embed->retrieve turn returned a healthy result set: a non-zero match count and cosine distances tight enough to indicate the retrieved policy clauses are relevant to the question rather than distant noise.",
                namespace=SAMPLE,
            ),
            "embedding_consistency": create_judge(
                client,
                name="Embedding Consistency Judge",
                evaluation_goal="Evaluate whether the embedding calls in the trace are consistent (same model and vector dimensionality across every query) and record real token usage, as expected for a well-formed RAG retrieval loop.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating RAG retrieval observability:")
        for label, judge in judges.items():
            evaluation = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge.id)
            if evaluation is None:
                print(f"  {label:24s} -- evaluation creation failed")
                continue
            results = poll_evaluation_results(client, evaluation.id)
            if results:
                r = results[0]
                verdict = "pass" if r.passed else "fail"
                color = _VERDICT_COLORS.get(verdict, "")
                print(f"  {label:24s} {color}{verdict.upper()}{_RESET} ({r.score:.2f})")
            else:
                print(f"  {label:24s} -- timed out waiting for results")
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
