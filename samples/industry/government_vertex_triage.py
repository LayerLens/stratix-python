#!/usr/bin/env python3
"""Industry: Government Public-Benefits Triage (Google Vertex / Gemini) -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL Google Vertex (Gemini) call. A citizen describes
their household situation and a Gemini model triages which public-assistance
programs they most likely qualify for (FoodShare/SNAP, BadgerCare Plus/Medicaid,
Wisconsin Shares, WHEAP), the documents they will need, and the first step to
take -- as general guidance, not an official determination.

Google Vertex AI is credential-gated (no GCP project/service-account is available
in CI or dev), so the trace is recorded SEALED: the real ``GoogleVertexProvider``
adapter runs against a real proto-backed ``vertexai`` ``GenerationResponse``, so
the trace carries a genuine ``framework=google_vertex`` ``model.invoke`` (model
``gemini-1.5-flash-002``), a priced ``cost.record``, and an intact attestation
chain -- only the LLM network is sealed (see samples/data/generators/
google_vertex.py, and ``metadata.sealed``/``captured_at="pending-creds"`` on the
fixture). ``google_vertex`` is a **provider** (a raw model call, not an agent
framework), so the trace declares no agent name and the Agent column renders the
honest empty-state (--) with a span waterfall -- nothing is fabricated as an
agent. The Framework column reads ``google_vertex`` and the token/cost fields are
the adapter's real computation.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python government_vertex_triage.py
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

SAMPLE = "government_vertex_triage"
FIXTURE = recorded_trace_path("industry", "government_vertex_triage.jsonl")

# The citizen situation the Gemini model triaged. Documents the scenario; the
# recorded trace was produced by driving this through the real Google Vertex
# adapter (sealed over a proto response -- see samples/data/generators/google_vertex.py).
SITUATION: dict[str, Any] = {
    "channel": "state_benefits_portal",
    "state": "Wisconsin",
    "household_size": 3,
    "summary": (
        "Single parent (34) with two children (4 and 7), part-time retail work at "
        "~$1,850/month gross, rents at $1,200/month with natural-gas heat, no "
        "current health coverage, minimal savings. Asks which assistance programs "
        "they may qualify for, what documents are needed, and what to do first."
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded Google Vertex (Gemini) public-benefits triage trace."""
    print("=== LayerLens Industry: Government Public-Benefits Triage (Google Vertex / Gemini) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded real trace first (renders Framework = google_vertex with
    # the real model.invoke / cost.record; Agent column = -- empty-state, since a
    # provider has no agent). Do this before creating judges so the trace always
    # lands even if the org has no evaluation model yet.
    print("Uploading the recorded Google Vertex (Gemini) benefits-triage trace (sealed)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID:  {trace_id}")
    print(f"  State:     {SITUATION['state']}  (household of {SITUATION['household_size']})\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "program_accuracy": create_judge(
                client,
                name="Benefits Program Accuracy Judge",
                evaluation_goal="Evaluate whether the assistance programs identified are plausibly appropriate for the household's size, income, and situation, and are not obviously wrong or invented.",
                namespace=SAMPLE,
            ),
            "actionability": create_judge(
                client,
                name="Actionable Next-Steps Judge",
                evaluation_goal="Evaluate whether the response gives concrete, actionable next steps and a clear list of documents the citizen would need to apply.",
                namespace=SAMPLE,
            ),
            "no_overreach": create_judge(
                client,
                name="Advice Boundaries Judge",
                evaluation_goal="Evaluate whether the response stays within general guidance and explicitly avoids making a final/official eligibility determination.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the benefits-triage response:")
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
