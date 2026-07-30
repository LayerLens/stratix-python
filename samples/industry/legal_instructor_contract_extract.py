#!/usr/bin/env python3
"""Industry: Legal Contract Abstraction (Instructor typed extraction) -- LayerLens SDK Sample.

Demonstrates evaluating a REAL single-agent Instructor run. A legal-operations
contract-abstraction assistant (persona ``contract-metadata-extractor``, a real
``instructor.from_openai(OpenAI())`` patched client backed by gpt-4o-mini) reads
a Master Services Agreement and returns a **validated** Pydantic
``ContractMetadata`` object -- abstracting the contracting parties, the effective
date and initial term, the governing law and exclusive venue, and the renewal
mechanics (auto-renewal, renewal term, non-renewal notice window). This is the
day-one task of every legal-ops abstraction pass, and it showcases Instructor's
structured-output strength: a schema-valid object off a real provider tool call.

Because the trace was captured under the real ``InstructorAdapter`` from a
genuine ``chat.completions.create(response_model=ContractMetadata)``, it renders
a single honest agent node (Agent column ``contract-metadata-extractor``,
Framework ``instructor``, Status ``ok``) plus the real ``model.invoke``
(real model / real ``response_model`` / real token counts) and the priced
``cost.record`` -- no fabrication.

HONESTY NOTE on the Agent column: Instructor declares NO agent identity of its
own -- it is a structured-output layer over a provider client, not an agent
framework. The adapter's only honest identity source is a name the CALLER
declares, so the recorded run passed ``agent_name="contract-metadata-extractor"``
explicitly. Nothing was synthesized from the framework label; an unnamed
Instructor client honestly renders "--" instead.

The recorded trace is shipped under samples/data/traces/industry/ (produced by
samples/data/generators/instructor.py); this sample uploads it and evaluates the
abstraction with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python legal_instructor_contract_extract.py
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

SAMPLE = "legal_instructor_contract_extract"
FIXTURE = recorded_trace_path("industry", "legal_instructor_contract_extract.jsonl")

# The agreement the abstraction agent processed, and the terms it is expected to
# find. Documents the scenario; the recorded trace was produced by running the
# full MSA text through a real Instructor typed-extraction call
# (response_model=ContractMetadata). Every field below is STATED in the
# agreement, so a hallucinated value is a real, visible failure.
MATTER: dict[str, Any] = {
    "matter_id": "MSA-2026-0314",
    "counterparty": "Meridian Health Partners, LLC",
    "document": "Master Services Agreement (Northwind Analytics, Inc. / Meridian Health Partners, LLC)",
    "abstracted_fields": [
        "parties",
        "effective_date",
        "initial_term_months",
        "governing_law",
        "exclusive_venue",
        "renewal (auto_renews / renewal_term_months / non_renewal_notice_days)",
    ],
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded Instructor contract-abstraction trace."""
    print("=== LayerLens Industry: Legal Contract Abstraction (Instructor typed extraction) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded trace first (renders a single contract-metadata-extractor
    # node with real model.invoke / typed output / priced cost.record events). Do
    # this before creating judges so the trace always lands even if the org has no
    # evaluation model yet.
    print("Uploading the recorded contract-abstraction trace (Instructor typed output)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID:     {trace_id}")
    print(f"  Matter:       {MATTER['matter_id']} ({MATTER['counterparty']})")
    print(f"  Document:     {MATTER['document']}")
    print(f"  Abstracting:  {', '.join(MATTER['abstracted_fields'])}\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "abstraction_accuracy": create_judge(
                client,
                # Judge names are namespaced to "<name> (legal_instructor_contract_extract)"
                # and the API caps a judge name at 64 chars, so the name has a
                # 28-char budget here. "Contract Abstraction Accuracy Judge" (35)
                # busted it with a 400 at run time; this says the same thing in 26.
                name="Abstraction Accuracy Judge",
                evaluation_goal="Evaluate whether the abstracted contract metadata (parties and their roles, effective date, initial term, governing law, exclusive venue, renewal terms) matches what the Master Services Agreement actually states.",
                namespace=SAMPLE,
            ),
            "renewal_terms": create_judge(
                client,
                name="Renewal Terms Judge",
                evaluation_goal="Evaluate whether the renewal mechanics were captured correctly: that the agreement auto-renews, the length of each renewal term, and the advance written notice required to prevent renewal.",
                namespace=SAMPLE,
            ),
            "no_hallucination": create_judge(
                client,
                name="No-Hallucination Judge",
                evaluation_goal="Evaluate whether the agent abstracted ONLY terms the agreement actually states and did not invent, infer, or assume any clause, date, or party.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the contract abstraction:")
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
