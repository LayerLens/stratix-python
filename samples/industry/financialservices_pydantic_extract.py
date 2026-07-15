#!/usr/bin/env python3
"""Industry: Financial-Services Loan Intake (PydanticAI typed extraction) -- LayerLens SDK Sample.

Demonstrates evaluating a REAL single-agent PydanticAI run. A consumer-lending
intake agent (persona ``loan-intake-extractor``, a real ``pydantic_ai.Agent``
backed by OpenAI gpt-4o-mini) reads a free-text loan-application message and
returns a **typed** ``LoanApplication`` Pydantic object -- extracting the
applicant name, loan amount, purpose, income, employment status, and requested
term. This showcases PydanticAI's structured-output strength.

Because the trace was captured under the real ``PydanticAIAdapter`` from a
genuine ``run_sync`` (non-streaming), it renders a single honest agent node
(Agent column ``loan-intake-extractor``, Framework ``pydantic-ai``) plus the
real ``model.invoke`` / typed ``agent.output`` / priced ``cost.record`` events --
no fabrication. The recorded trace is shipped under
samples/data/traces/industry/ (produced by
samples/data/generators/pydantic_ai.py); this sample uploads it and evaluates
the extraction with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python financialservices_pydantic_extract.py
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

SAMPLE = "financialservices_pydantic_extract"
FIXTURE = recorded_trace_path("industry", "financialservices_pydantic_extract.jsonl")

# The free-text application the extraction agent processed. Documents the
# scenario; the recorded trace was produced by running this through a real
# PydanticAI typed-extraction Agent (output_type=LoanApplication).
APPLICATION: dict[str, Any] = {
    "application_id": "LN-20418",
    "channel": "web_form_freetext",
    "message": (
        "Hi, my name is Marcus Whitfield. I'd like to apply for a personal loan of "
        "$28,500 to consolidate three high-interest credit-card balances. I've worked "
        "full-time as a registered nurse for six years and earn about $94,000 a year. "
        "I'd like a 48-month term."
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded PydanticAI typed-extraction trace."""
    print("=== LayerLens Industry: Financial-Services Loan Intake (PydanticAI typed extraction) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded trace first (renders a single loan-intake-extractor node
    # with real model.invoke / typed agent.output / cost.record events). Do this
    # before creating judges so the trace always lands even if the org has no
    # evaluation model yet.
    print("Uploading the recorded loan-intake extraction trace (PydanticAI typed output)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID:    {trace_id}")
    print(f"  Application: {APPLICATION['application_id']} ({APPLICATION['channel']})\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "extraction_accuracy": create_judge(
                client,
                name="Extraction Accuracy Judge",
                evaluation_goal="Evaluate whether the extracted loan-application fields (name, amount, purpose, income, employment, term) match the facts stated in the applicant's free-text message.",
                namespace=SAMPLE,
            ),
            "no_hallucination": create_judge(
                client,
                name="No-Hallucination Judge",
                evaluation_goal="Evaluate whether the agent extracted ONLY values the applicant actually stated and did not invent or assume any field.",
                namespace=SAMPLE,
            ),
            "schema_completeness": create_judge(
                client,
                name="Schema Completeness Judge",
                evaluation_goal="Evaluate whether every required field of the structured LoanApplication schema was populated.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the typed extraction:")
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
