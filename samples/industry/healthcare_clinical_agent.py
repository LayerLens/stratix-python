#!/usr/bin/env python3
"""Industry: Clinical Decision-Support Tool-Calling Agent -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL LangChain tool-calling agent for clinical
decision support. A ``create_tool_calling_agent`` + ``AgentExecutor`` works a
patient presentation by calling two real tools -- ``guideline_lookup`` (evidence-
based guideline retrieval) and ``drug_interaction_check`` (medication-interaction
screening) -- before giving a triage read. The AgentExecutor is given a
developer-declared ``run_name`` via ``.with_config(run_name=...)`` -- the honest
way to fill the Agent column (the ``AgentExecutor`` class default renders blank).
The recorded trace renders the agent node ``clinical-decision-support-agent``
(framework ``langchain``) with its real ``tool.call`` / ``tool.result`` and
``model.invoke`` events.

The trace was recorded from a real instrumented agent run (see
samples/data/_generate_fixtures.py) and is shipped under
samples/data/traces/industry/. This sample uploads it and evaluates the agent's
tool use and clinical reasoning with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python healthcare_clinical_agent.py
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

SAMPLE = "healthcare_clinical_agent"
FIXTURE = recorded_trace_path("industry", "healthcare_clinical_agent.jsonl")

# The synthetic patient case the clinical-decision-support agent worked (no real
# PHI), plus the tools it had available. Documents the scenario; the recorded
# tool-use trace was produced by running this through a real LangChain
# create_tool_calling_agent + AgentExecutor, which actually called both tools
# before answering.
CASE: dict[str, Any] = {
    "case_id": "HC-CDS-002",
    "presentation": (
        "67M with crushing substernal chest pain radiating to the left arm, "
        "diaphoresis and dyspnea for 40 minutes. HR 110, BP 90/60, SpO2 92%, RR 24."
    ),
    "active_medications": ["aspirin", "heparin", "metoprolol", "lisinopril"],
    "tools": ["guideline_lookup", "drug_interaction_check"],
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded clinical-decision-support tool-calling agent trace."""
    print("=== LayerLens Healthcare: Clinical Decision-Support Tool-Calling Agent ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded trace first (renders the agent node
    # ``clinical-decision-support-agent`` with its tool calls, framework
    # ``langchain``). Do this before creating judges so the trace always lands
    # even if the org has no evaluation model yet.
    print("Uploading the recorded clinical-decision-support agent trace (with tool calls)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Case:     {CASE['case_id']} ({', '.join(CASE['tools'])})\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "tool_use": create_judge(
                client,
                name="Tool Use Correctness Judge",
                evaluation_goal="Evaluate whether the agent called the appropriate tools (guideline lookup and drug-interaction check) with sensible arguments before giving its clinical assessment.",
                namespace=SAMPLE,
            ),
            "medication_safety": create_judge(
                client,
                name="Medication Safety Judge",
                evaluation_goal="Evaluate whether the agent correctly surfaced medication-interaction risks (e.g. additive bleeding risk) in its final recommendation.",
                namespace=SAMPLE,
            ),
            "triage_accuracy": create_judge(
                client,
                name="Triage Accuracy Judge",
                evaluation_goal="Evaluate whether the agent assigned a clinically appropriate, sufficiently urgent triage level for this presentation.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the agent's tool use and clinical reasoning:")
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
