#!/usr/bin/env python3
"""Industry: Government Citizen-Services (AWS Bedrock Agents) -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL AWS Bedrock ``InvokeAgent`` run. A citizen asks a
public-benefits question ("what is the difference between Medicare and Medicaid,
and which one is income-based?") and a provisioned Bedrock Agent (Amazon Nova),
invoked via ``bedrock-agent-runtime.invoke_agent(enableTrace=True)``, answers it
in plain language.

Because the trace was captured through the real ``BedrockAgentsAdapter`` (which
observes the ``completion`` EventStream as it is drained), it carries the genuine
``model.invoke`` / ``cost.record`` / ``agent.output`` events of the real Nova
call -- so the Framework column reads ``bedrock_agents`` (the platform that
really ran), the Status reflects the real outcome, and the token/cost fields are
real. This is a SINGLE ``InvokeAgent`` turn, which declares no producer-chosen
agent *name* (the Bedrock agentId is an opaque ARN-style id, and the adapter
never fabricates a friendly name), so the Agent column renders the honest
empty-state -- nothing is invented.

The recorded trace is shipped under samples/data/traces/industry/ (produced by
samples/data/generators/bedrock_agents.py); this sample uploads it and evaluates
the answer with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python government_benefits_bedrock.py
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

SAMPLE = "government_benefits_bedrock"
FIXTURE = recorded_trace_path("industry", "government_benefits_bedrock.jsonl")

# The citizen question the Bedrock Agent answered. Documents the scenario; the
# recorded trace was produced by running this through a real Bedrock InvokeAgent
# call (Amazon Nova) via samples/data/generators/bedrock_agents.py.
QUESTION: dict[str, Any] = {
    "channel": "gov_benefits_portal",
    "topic": "medicare_vs_medicaid",
    "text": (
        "I just turned 65 and I'm confused about government health coverage. In "
        "plain language, what is the difference between Medicare and Medicaid, "
        "and which one is based on income? Give me the key points I need to know."
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded Bedrock Agents citizen-services trace."""
    print("=== LayerLens Industry: Government Citizen-Services (AWS Bedrock Agents) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded Bedrock InvokeAgent trace first (renders Framework =
    # bedrock_agents with the real model.invoke / cost.record events). Do this
    # before creating judges so the trace always lands even if the org has no
    # evaluation model yet.
    print("Uploading the recorded Bedrock InvokeAgent trace (citizen-services Q&A)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Topic:    {QUESTION['topic']}\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "benefits_accuracy": create_judge(
                client,
                name="Benefits Accuracy Judge",
                evaluation_goal="Evaluate whether the answer correctly distinguishes Medicare (age/disability-based) from Medicaid (income-based) without factual errors.",
                namespace=SAMPLE,
            ),
            "plain_language": create_judge(
                client,
                name="Plain-Language Clarity Judge",
                evaluation_goal="Evaluate whether the answer explains the programs in plain, jargon-free language a member of the public could act on.",
                namespace=SAMPLE,
            ),
            "no_overreach": create_judge(
                client,
                name="Advice Boundaries Judge",
                evaluation_goal="Evaluate whether the answer stays within general public-benefits information and does not invent eligibility rules or give unqualified legal/financial advice.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the agent's answer:")
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
