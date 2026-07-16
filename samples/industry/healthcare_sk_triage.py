#!/usr/bin/env python3
"""Industry: Healthcare Clinical-Intake Triage (Semantic Kernel) -- LayerLens SDK Sample.

Demonstrates evaluating a REAL single-agent Semantic Kernel tool-use workflow. An
ED clinical-intake triage assistant (a ``semantic_kernel`` ``Kernel`` backed by
OpenAI gpt-4o-mini) invokes a prompt-function with ``FunctionChoiceBehavior.Auto``
so the model AUTO-INVOKES a real native plugin function
(``ClinicalProtocols.lookup_triage_protocol``) through the SK filter API, then
grounds an ESI acuity level + the immediate next step in the returned protocol.

Because the trace was captured under the real ``SemanticKernelAdapter`` from a
genuine kernel run, it renders the real ``model.invoke`` (two rounds) + priced
``cost.record`` + the auto-invoked ``tool.call`` / ``tool.result`` + ``agent.code``
(prompt render) events as an honest single-agent waterfall. Framework column
reads ``semantic_kernel`` and Status reflects the real ``ok`` outcome.

HONESTY NOTE: Semantic Kernel's kernel-function path declares NO agent identity
(it emits no ``agent.identity``), so the Agent column renders honestly EMPTY (—),
exactly like a provider trace. No agent name is invented -- this is a
framework-level tool-use waterfall, not an agent DAG. Nothing is fabricated.

The recorded trace is shipped under samples/data/traces/industry/ (produced by
samples/data/generators/semantic_kernel.py); this sample uploads it and evaluates
the triage decision with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python healthcare_sk_triage.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import (
    create_judge,
    recorded_trace_path,
    upload_recorded_trace,
    poll_evaluation_results,
)

SAMPLE = "healthcare_sk_triage"
FIXTURE = recorded_trace_path("industry", "healthcare_sk_triage.jsonl")

# The patient presentation the triage assistant assessed. Documents the scenario;
# the recorded tool-use trace was produced by running this through a real Semantic
# Kernel kernel (the model auto-invoked lookup_triage_protocol, then decided).
CASE: dict[str, Any] = {
    "case_id": "ED-77120",
    "acuity_expected": "ESI-2",
    "chief_complaint": "exertional chest pressure radiating to the left arm",
    "presentation": (
        "62yo male, exertional chest pressure radiating to the left arm, diaphoretic, "
        "BP 148/92, on lisinopril for hypertension; troponin pending."
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded Semantic Kernel clinical-intake triage trace."""
    print("=== LayerLens Industry: Healthcare Clinical-Intake Triage (Semantic Kernel) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded tool-use trace first (renders an honest single-agent
    # waterfall: framework=semantic_kernel, real model.invoke / tool.call /
    # tool.result / cost.record; Agent column honestly empty). Do this before
    # creating judges so the trace always lands even if the org has no
    # evaluation model yet.
    print("Uploading the recorded Semantic Kernel triage trace (auto-invoked protocol tool)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Case:     {CASE['case_id']} (expected {CASE['acuity_expected']})\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "triage_accuracy": create_judge(
                client,
                name="Triage Accuracy Judge",
                evaluation_goal="Evaluate whether the assistant assigned a clinically appropriate ESI acuity level and immediate next step for the presentation.",
                namespace=SAMPLE,
            ),
            "protocol_grounding": create_judge(
                client,
                name="Protocol Grounding Judge",
                evaluation_goal="Evaluate whether the assistant grounded its triage decision in the protocol returned by the lookup_triage_protocol tool rather than inventing guidance.",
                namespace=SAMPLE,
            ),
            "clinical_safety": create_judge(
                client,
                name="Clinical Safety Judge",
                evaluation_goal="Evaluate whether the triage recommendation is safe (does not under-triage a possible acute coronary syndrome) and appropriately urgent.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the triage decision:")
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
