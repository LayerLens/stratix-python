#!/usr/bin/env python3
"""Industry: Multi-Agent Underwriting Team -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL multi-agent workflow. An underwriting supervisor
delegates a loan application to specialist sub-agents -- credit-analyst,
risk-assessor, compliance-checker -- and a final decision step, orchestrated as a
LangGraph StateGraph with agent-to-agent handoffs. Each specialist runs on a
different instrumented model provider, so the recorded trace renders as a
multi-node agent graph (5 nodes + handoff edges) whose nodes carry real
model calls across OpenAI, Anthropic, and Ollama.

The trace was recorded from a real run (see samples/data/_generate_fixtures.py)
and is shipped under samples/data/traces/industry/. This sample uploads it and
evaluates the team's output with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python underwriting_team.py
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

SAMPLE = "underwriting_team"
FIXTURE = recorded_trace_path("industry", "underwriting_team.jsonl")

# The loan application the underwriting team assessed. Documents the scenario;
# the recorded multi-agent trace was produced by running this through the real
# LangGraph team (supervisor -> credit-analyst -> risk-assessor ->
# compliance-checker -> decision).
APPLICATION: dict[str, Any] = {
    "applicant_id": "APP-4471",
    "loan_type": "conventional_mortgage",
    "amount": 420000,
    "applicant": {
        "fico": 724,
        "annual_income": 138000,
        "dti_ratio": 0.31,
        "employment_years": 6,
        "down_payment_pct": 20,
    },
    "property": {"type": "single_family", "appraised_value": 525000, "location": "Austin, TX"},
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded multi-agent underwriting-team trace."""
    print("=== LayerLens Industry: Multi-Agent Underwriting Team ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded multi-agent trace first (renders as a multi-node agent
    # graph). Do this before creating judges so the trace always lands even if
    # the org has no evaluation model yet.
    print("Uploading the recorded underwriting-team trace (5-node agent graph)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Application: {APPLICATION['loan_type']} ${APPLICATION['amount']:,} "
          f"(FICO {APPLICATION['applicant']['fico']})\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "risk_accuracy": create_judge(
                client,
                name="Underwriting Risk Judge",
                evaluation_goal="Evaluate whether the team's credit and risk assessment is accurate and well-justified for the application.",
                namespace=SAMPLE,
            ),
            "fair_lending": create_judge(
                client,
                name="Fair Lending Judge",
                evaluation_goal="Evaluate whether the underwriting relies only on permissible factors and complies with fair-lending regulations (ECOA).",
                namespace=SAMPLE,
            ),
            "decision_soundness": create_judge(
                client,
                name="Decision Soundness Judge",
                evaluation_goal="Evaluate whether the final approve/decline decision is sound and consistent with the specialists' assessments.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the team's decision:")
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
