#!/usr/bin/env python3
"""Industry: Clinical Decision-Support RAG Chain -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL LangChain LCEL retrieval-augmented chain for
clinical decision support. The chain retrieves the relevant evidence-based
guidelines for a patient presentation, threads them into a
``ChatPromptTemplate``, calls ``ChatOpenAI``, and parses the result -- a
``retrieve -> prompt -> ChatOpenAI -> StrOutputParser`` pipeline given a
developer-declared ``run_name`` via ``.with_config(run_name=...)``. That honest
run_name is what fills the Agent column: the recorded trace renders one node,
``clinical-decision-support`` (framework ``langchain``). Mirrors ateam
healthcare/clinical_decision_support.py.

The trace was recorded from a real instrumented chain run (see
samples/data/_generate_fixtures.py) and is shipped under
samples/data/traces/industry/. This sample uploads it and evaluates the
clinical response with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python healthcare_clinical_chain.py
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

SAMPLE = "healthcare_clinical_chain"
FIXTURE = recorded_trace_path("industry", "healthcare_clinical_chain.jsonl")

# The synthetic patient case the clinical-decision-support chain worked (no real
# PHI). Documents the scenario; the recorded single-node trace was produced by
# running this through a real LangChain LCEL RAG chain
# (retrieve -> ChatPromptTemplate -> ChatOpenAI -> StrOutputParser) whose
# developer-declared run_name honestly fills the Agent column.
PATIENT_CASE: dict[str, Any] = {
    "case_id": "HC-CDS-001",
    "presentation": (
        "67M with crushing substernal chest pain radiating to the left arm, "
        "diaphoresis and dyspnea for 40 minutes. HR 110, BP 90/60, SpO2 92%, RR 24."
    ),
    "active_medications": ["metoprolol", "lisinopril", "atorvastatin", "aspirin 81mg"],
    "history": ["hypertension", "hyperlipidemia", "prior MI (2023)"],
    "allergies": ["sulfa"],
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded clinical-decision-support RAG chain trace."""
    print("=== LayerLens Healthcare: Clinical Decision-Support RAG Chain ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded trace first (renders one honest node,
    # ``clinical-decision-support``, framework ``langchain``). Do this before
    # creating judges so the trace always lands even if the org has no
    # evaluation model yet.
    print("Uploading the recorded clinical-decision-support chain trace...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Case:     {PATIENT_CASE['case_id']} -- {PATIENT_CASE['presentation'][:60]}...\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "medical_accuracy": create_judge(
                client,
                name="Medical Accuracy Judge",
                evaluation_goal="Evaluate whether the differential, triage level, and cautions are clinically sound and consistent with evidence-based guidelines for this presentation.",
                namespace=SAMPLE,
            ),
            "triage_safety": create_judge(
                client,
                name="Triage Safety Judge",
                evaluation_goal="Evaluate whether the response assigns an appropriately urgent triage level and does not under-triage a potentially life-threatening presentation.",
                namespace=SAMPLE,
            ),
            "grounding": create_judge(
                client,
                name="Guideline Grounding Judge",
                evaluation_goal="Evaluate whether the clinical response is grounded in the retrieved guidelines and does not fabricate citations or unsupported claims.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the clinical response:")
        for label, judge in judges.items():
            evaluation = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge.id)
            if evaluation is None:
                print(f"  {label:18s} -- evaluation creation failed")
                continue
            results = poll_evaluation_results(client, evaluation.id)
            if results:
                r = results[0]
                verdict = "pass" if r.passed else "fail"
                color = _VERDICT_COLORS.get(verdict, "")
                print(f"  {label:18s} {color}{verdict.upper()}{_RESET} ({r.score:.2f})")
            else:
                print(f"  {label:18s} -- timed out waiting for results")
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
