#!/usr/bin/env python3
"""Industry: Credit Underwriting (PydanticAI tool-use loop) -- LayerLens SDK Sample.

Demonstrates evaluating a REAL single-agent PydanticAI **tool-use loop**. A
consumer-lending underwriter (persona ``credit-underwriting-assistant``, a real
``pydantic_ai.Agent`` backed by OpenAI gpt-4o-mini) works a loan application by
calling three real ``@agent.tool_plain`` functions in sequence --
``fetch_credit_score``, ``get_debt_obligations``, and
``lookup_underwriting_policy`` -- then issues an APPROVE / REFER / DECLINE
recommendation grounded in the gathered facts (FICO, DTI, LTV vs policy limits).

HONEST FRAMING: this is a SINGLE-agent tool-use loop, not a multi-agent handoff
graph. PydanticAI has no handoff hook -- cross-agent delegation in PydanticAI
runs each sub-agent as a *separate* trace, so one trace can never carry
``agent.handoff`` or more than one agent node. The trace therefore renders ONE
honest agent node (Agent column ``credit-underwriting-assistant``, Framework
``pydantic-ai``) with the real ``model.invoke`` / ``tool.call`` / ``tool.result``
/ ``cost.record`` events of the loop -- nothing is fabricated. The recorded
trace is shipped under samples/data/traces/industry/ (produced by
samples/data/generators/pydantic_ai.py); this sample uploads it and evaluates
the underwriting decision with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python financialservices_pydantic_underwriting.py
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

SAMPLE = "financialservices_pydantic_underwriting"
FIXTURE = recorded_trace_path("industry", "financialservices_pydantic_underwriting.jsonl")

# The application the underwriting agent processed. Documents the scenario; the
# recorded trace was produced by running this through a real PydanticAI agent
# that called fetch_credit_score, get_debt_obligations, and
# lookup_underwriting_policy (a genuine single-agent tool-use loop).
APPLICATION: dict[str, Any] = {
    "applicant_id": "APP-70412",
    "loan_type": "conventional_mortgage",
    "loan_amount_usd": 420000,
    "property_appraised_usd": 525000,
    "annual_income_usd": 138000,
    "tools_called": ["fetch_credit_score", "get_debt_obligations", "lookup_underwriting_policy"],
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded PydanticAI credit-underwriting tool-use-loop trace."""
    print("=== LayerLens Industry: Credit Underwriting (PydanticAI tool-use loop) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded trace first (renders a single credit-underwriting-
    # assistant node with real model.invoke / tool.call / tool.result / cost.record
    # events -- a genuine tool-use loop). Do this before creating judges so the
    # trace always lands even if the org has no evaluation model yet.
    print("Uploading the recorded credit-underwriting trace (single-agent tool-use loop)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID:  {trace_id}")
    print(f"  Applicant: {APPLICATION['applicant_id']} ({APPLICATION['loan_type']}, "
          f"${APPLICATION['loan_amount_usd']:,})")
    print(f"  Tools:     {', '.join(APPLICATION['tools_called'])}\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "tool_grounding": create_judge(
                client,
                name="Tool Grounding Judge",
                evaluation_goal="Evaluate whether the underwriter called the credit-score, debt, and policy tools and grounded its recommendation in the returned data rather than guessing.",
                namespace=SAMPLE,
            ),
            "policy_compliance": create_judge(
                client,
                name="Policy Compliance Judge",
                evaluation_goal="Evaluate whether the APPROVE/REFER/DECLINE recommendation correctly applies the underwriting policy thresholds (min FICO, max DTI, max LTV) to the applicant's data.",
                namespace=SAMPLE,
            ),
            "decision_soundness": create_judge(
                client,
                name="Decision Soundness Judge",
                evaluation_goal="Evaluate whether the final recommendation is well-reasoned and consistent with the computed debt-to-income and loan-to-value ratios.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the underwriting decision:")
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
