#!/usr/bin/env python3
"""Industry: Legal Contract-Clause Q&A (Haystack) -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL Haystack pipeline run. A single-component Haystack
pipeline (one honest ``llm`` node, an ``OpenAIGenerator`` backed by OpenAI
gpt-4o-mini) reads a specific contract clause (a limitation-of-liability section)
and answers a plain-language question about it, grounded only in the clause text.

Because the trace was captured under the real ``HaystackAdapter`` (which swaps
Haystack's global tracer) from a genuine ``Pipeline.run``, it renders the honest
Haystack component node (``llm``) plus the real ``model.invoke`` / ``cost.record``
and pipeline ``agent.input`` / ``agent.output`` events -- no fabrication. Haystack
is a pipeline framework, not a multi-agent one: it emits no ``agent.identity``, so
the Agent column renders honest empty-state (the graph shows the component node),
while the Framework column shows ``haystack`` and the token/cost fields are real.

The recorded trace is shipped under samples/data/traces/industry/ (produced by
samples/data/_generate_fixtures.py via samples/data/generators/haystack.py); this
sample uploads it and evaluates the clause interpretation with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python legal_haystack_clause_qa.py
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

SAMPLE = "legal_haystack_clause_qa"
FIXTURE = recorded_trace_path("industry", "legal_haystack_clause_qa.jsonl")

# The clause + question the Haystack pipeline answered. Documents the scenario;
# the recorded trace was produced by running this through a real Haystack
# single-component pipeline (OpenAIGenerator over OpenAI gpt-4o-mini).
CASE: dict[str, Any] = {
    "matter_id": "MSA-2026-0417",
    "clause": "Section 9 - Limitation of Liability",
    "question": (
        "Does this limitation-of-liability clause cap the Vendor's liability for a "
        "Vendor-caused data breach, and are there carve-outs that let the cap be exceeded?"
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded Haystack contract-clause-Q&A trace."""
    print("=== LayerLens Industry: Legal Contract-Clause Q&A (Haystack) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded pipeline trace first (renders one honest Haystack
    # component node (``llm``) with real model.invoke / cost.record events). Do
    # this before creating judges so the trace always lands even if the org has
    # no evaluation model yet.
    print("Uploading the recorded Haystack clause-Q&A trace (single llm component)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Matter:   {CASE['matter_id']} ({CASE['clause']})\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "clause_accuracy": create_judge(
                client,
                name="Clause Interpretation Judge",
                evaluation_goal="Evaluate whether the answer correctly interprets the limitation-of-liability clause: that it caps liability and that the indemnification/confidentiality carve-outs let the cap be exceeded.",
                namespace=SAMPLE,
            ),
            "grounding": create_judge(
                client,
                name="Clause Grounding Judge",
                evaluation_goal="Evaluate whether the answer is grounded ONLY in the provided clause text and does not invent terms not present in the clause.",
                namespace=SAMPLE,
            ),
            "clarity": create_judge(
                client,
                name="Plain-Language Clarity Judge",
                evaluation_goal="Evaluate whether the answer explains the clause in clear, plain language a non-lawyer could act on.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the clause interpretation:")
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
