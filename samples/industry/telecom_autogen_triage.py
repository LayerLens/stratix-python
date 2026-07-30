#!/usr/bin/env python3
"""Industry: Telecom Support Agent (AutoGen tool-use) -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL single-agent tool-use workflow built with AutoGen
(autogen-agentchat). A telecom customer-support agent (persona
``telecom_support_agent``, an ``AssistantAgent`` backed by OpenAI) reviews a
billing question, calls a real ``lookup_account`` function tool to fetch the
customer's plan and recent charges, then answers -- identifying the duplicate
charge, the correct amount owed, and a concrete next step -- grounded in the tool
result.

Because the trace was captured from a genuine AutoGen run (the agent driven
inside autogen's runtime so it is assigned a real ``AgentId``), it renders a
single honest agent node (Agent column ``telecom_support_agent``) plus the real
``model.invoke`` / ``tool.call`` / ``cost.record`` events of the turn -- no
fabrication. The recorded trace is shipped under samples/data/traces/industry/
(produced by samples/data/generators/autogen.py); this sample uploads it and
evaluates the support answer with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python telecom_autogen_triage.py
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

SAMPLE = "telecom_autogen_triage"
FIXTURE = recorded_trace_path("industry", "telecom_autogen_triage.jsonl")

# The billing question the agent handled. Documents the scenario; the recorded
# tool-use trace was produced by running this through a real AutoGen agent (it
# called lookup_account, then answered against the returned account record).
QUESTION: dict[str, Any] = {
    "account_id": "ACCT-55231",
    "channel": "chat",
    "issue": "billing_dispute",
    "text": (
        "I think I was double-charged $79.99 on my last bill. Can you check my "
        "recent charges and tell me what I actually owe and how you'll fix it?"
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded AutoGen telecom-support tool-use trace."""
    print("=== LayerLens Industry: Telecom Support Agent (AutoGen tool-use) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded tool-use trace first (renders a single
    # telecom_support_agent node with real model.invoke / tool.call / cost.record
    # events). Do this before creating judges so the trace always lands even if
    # the org has no evaluation model yet.
    print("Uploading the recorded telecom-support trace (lookup_account tool loop)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Account:  {QUESTION['account_id']} ({QUESTION['issue']})\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "billing_accuracy": create_judge(
                client,
                name="Billing Accuracy Judge",
                evaluation_goal="Evaluate whether the agent correctly identified the duplicate $79.99 charge and stated the correct amount actually owed.",
                namespace=SAMPLE,
            ),
            "tool_grounding": create_judge(
                client,
                name="Tool Grounding Judge",
                evaluation_goal="Evaluate whether the agent grounded its answer in the account record returned by the lookup_account tool rather than inventing charges.",
                namespace=SAMPLE,
            ),
            "resolution_clarity": create_judge(
                client,
                name="Resolution Clarity Judge",
                evaluation_goal="Evaluate whether the response is clear, empathetic, and gives the customer a concrete next step to resolve the dispute.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the support answer:")
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
