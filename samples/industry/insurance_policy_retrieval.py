#!/usr/bin/env python3
"""Industry: Insurance Policy-Document Semantic Retrieval -- LayerLens Python SDK Sample.

Demonstrates observing a REAL vector-retrieval workload with LayerLens. A
policyholder question ("a deer dented my car door -- is that covered and what
deductible applies?") is answered by semantically retrieving the most relevant
clauses from an insurance policy knowledge base: a real in-process Chroma
collection whose vectors are REAL OpenAI ``text-embedding-3-small`` embeddings of
the policy document. The query is embedded and the nearest clauses retrieved
through the real ``VectorStoreAdapter``, which records a genuine
``retrieval.query`` event with the provider, match count, and cosine-distance
distribution -- nothing is fabricated.

HONEST EMPTY-STATE RENDER: the vector-store adapter is metadata-only and
NON-agentic -- it emits no ``agent.identity``/``model.invoke``. So the trace
renders honestly as an **empty-state** (Agent column = "-") plus an event
waterfall of the real ``retrieval.query`` telemetry; LayerLens does NOT invent an
agent. This is the correct render for a cross-cutting retrieval workload -- it
gives you observability of the retrieval layer (was the corpus hit, how close
were the matches, how fast) rather than an agent graph.

The trace was recorded from a real instrumented retrieval run (see
samples/data/generators/embedding.py) and is shipped under
samples/data/traces/industry/. This sample uploads it and evaluates the
retrieval health with observability judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python insurance_policy_retrieval.py
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

SAMPLE = "insurance_policy_retrieval"
FIXTURE = recorded_trace_path("industry", "insurance_policy_retrieval.jsonl")

# The retrieval scenario. Documents the workload; the recorded trace was produced
# by embedding this question with real OpenAI embeddings and retrieving the
# nearest clauses from a real Chroma index of the policy document.
SCENARIO: dict[str, Any] = {
    "policy_id": "AUTO-2024-8891",
    "knowledge_base": "Personal Auto + Homeowners policy (12 indexed clauses)",
    "question": (
        "A deer ran into the side of my car on a rural road and dented the door. "
        "Is that damage covered, and what deductible would apply?"
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def _retrieval_telemetry(fixture_path: str) -> dict[str, Any]:
    """Read the shipped fixture and summarize the REAL retrieval telemetry that
    LayerLens surfaces (provider, match count, cosine-distance band, latency)."""
    with open(fixture_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    events = records[0].get("events", []) if records else []
    ret = [e for e in events if e.get("event_type") == "retrieval.query"]
    p = ret[0].get("payload", {}) if ret else {}
    return {
        "provider": p.get("provider"),
        "n_results": p.get("n_results"),
        "result_count": p.get("result_count"),
        "distance_min": p.get("distance_min"),
        "distance_max": p.get("distance_max"),
        "distance_mean": p.get("distance_mean"),
        "latency_ms": p.get("latency_ms"),
    }


def main() -> None:
    """Upload + evaluate the recorded policy-retrieval trace."""
    print("=== LayerLens Industry: Insurance Policy-Document Semantic Retrieval ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded retrieval trace first (renders an honest empty-state +
    # retrieval.query waterfall). Do this before judges so the trace always lands
    # even if the org has no evaluation model yet.
    print("Uploading the recorded policy-retrieval trace (real Chroma retrieval telemetry)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]

    tel = _retrieval_telemetry(FIXTURE)
    print(f"  Trace ID:  {trace_id}")
    print(f"  Policy:    {SCENARIO['policy_id']} -- {SCENARIO['knowledge_base']}")
    print(f"  Question:  {SCENARIO['question']}")
    print("  Retrieval: provider=%s matches=%s/%s  cosine-distance[min/mean/max]=%s/%s/%s  latency=%sms"
          % (tel["provider"], tel["result_count"], tel["n_results"],
             tel["distance_min"], tel["distance_mean"], tel["distance_max"], tel["latency_ms"]))
    print("  Render:    honest empty-state (Agent = '-') + retrieval.query waterfall "
          "(non-agentic retrieval workload)\n")

    # Best-effort read-back so the sample proves the trace landed and is queryable.
    try:
        fetched = client.traces.get(trace_id)
        if fetched is not None:
            print("  Read-back: trace is stored and queryable in LayerLens.\n")
    except Exception:
        pass

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    # These assess RETRIEVAL OBSERVABILITY/HEALTH from the surfaced telemetry --
    # the honest evaluation surface for a metadata-only retrieval trace.
    judge_ids: list[str] = []
    try:
        judges = {
            "retrieval_health": create_judge(
                client,
                name="Retrieval Health Judge",
                evaluation_goal="Evaluate whether the vector retrieval returned a healthy result set for the policy question: a non-zero match count and cosine distances tight enough to indicate the retrieved clauses are relevant rather than distant noise.",
                namespace=SAMPLE,
            ),
            "retrieval_latency": create_judge(
                client,
                name="Retrieval Latency Judge",
                evaluation_goal="Evaluate whether the retrieval latency recorded in the trace is within an acceptable operational range for an interactive policy-lookup workload.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating retrieval observability:")
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
