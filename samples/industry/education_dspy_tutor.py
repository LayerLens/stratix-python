#!/usr/bin/env python3
"""Industry: Education Course Tutoring (DSPy RAG) -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL retrieval-augmented tutoring workflow built with
the ``dspy`` framework. A ``TutorRAG`` (a real ``dspy.Module``) answers an enrolled
student's question about Bessel's correction over the STAT-101 Week-3 lecture
notes: it calls a real ``dspy.Tool`` (``search_course_material``) to retrieve the
relevant excerpts, then a real ``dspy.ChainOfThought`` grounds the tutoring answer
in ONLY those excerpts and cites the note ids it used.

Because the trace was captured through the real ``DSPyAdapter`` (registered on
dspy's first-party ``dspy.settings.callbacks`` bus), it renders one honest agent
node -- Agent column = ``TutorRAG`` (the developer-declared class; dspy's own
``ChainOfThought``/``Predict`` primitives are deliberately NOT surfaced as agents),
Framework column = ``dspy``, Status = ok -- plus the real nested module boundary
(TutorRAG -> ChainOfThought -> Predict), the real ``tool.call`` for the retrieval,
and the real ``model.invoke`` / ``cost.record`` of the LM turn. No fabrication.

The recorded trace is shipped under samples/data/traces/industry/ (produced by
samples/data/generators/dspy.py against a local ollama ``llama3:8b``); this sample
uploads it and evaluates the tutoring answer with domain judges. Because the model
that ran is local, its ``cost.record`` honestly carries real token counts and no
dollar cost -- a local run has no provider tariff.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python education_dspy_tutor.py
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

SAMPLE = "education_dspy_tutor"
FIXTURE = recorded_trace_path("industry", "education_dspy_tutor.jsonl")

# The tutoring session the TutorRAG program handled. Documents the scenario; the
# recorded trace was produced by running exactly this through a real dspy.Module
# that retrieved the course notes, then answered against the returned excerpts.
SESSION: dict[str, Any] = {
    "course_id": "STAT-101",
    "unit": "Week 3 -- Measures of spread",
    "student_question": (
        "Why do we divide by n-1 instead of n when we compute the sample variance, "
        "and what does 'degrees of freedom' have to do with it?"
    ),
    "expected_notes": ["STAT101-W3-N2", "STAT101-W3-N3"],
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded dspy TutorRAG course-tutoring trace."""
    print("=== LayerLens Industry: Education Course Tutoring (DSPy RAG) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        print("  Set LAYERLENS_STRATIX_API_KEY to run this sample.")
        sys.exit(1)

    # Upload the recorded tutoring trace first (renders a single TutorRAG node with
    # the real tool.call retrieval + model.invoke). Do this before creating judges
    # so the trace always lands even if the org has no evaluation model yet.
    print("Uploading the recorded TutorRAG trace (search_course_material -> answer)...\n")
    try:
        trace_ids = upload_recorded_trace(client, FIXTURE)
    except Exception as exc:
        print(f"ERROR: failed to upload the recorded trace: {exc}")
        sys.exit(1)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Course:   {SESSION['course_id']} ({SESSION['unit']})")
    print(f"  Question: {SESSION['student_question']}\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "answer_correctness": create_judge(
                client,
                name="Tutoring Answer Correctness Judge",
                evaluation_goal="Evaluate whether the tutor's explanation of why the sample variance divides by n-1 (Bessel's correction, unbiasedness, degrees of freedom) is factually correct for an introductory statistics course.",
                namespace=SAMPLE,
            ),
            "material_grounding": create_judge(
                client,
                name="Course Material Grounding Judge",
                evaluation_goal="Evaluate whether the tutor grounded its answer in the lecture-note excerpts the search_course_material tool actually returned and cited the note ids it used, rather than answering from outside knowledge or inventing citations.",
                namespace=SAMPLE,
            ),
            "pedagogical_quality": create_judge(
                client,
                name="Pedagogical Quality Judge",
                evaluation_goal="Evaluate whether the response teaches the student the underlying intuition in clear plain language appropriate for an intro-statistics undergraduate, rather than merely restating the formula.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the tutoring answer:")
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
