#!/usr/bin/env python3
"""Industry: Multi-Agent Clinical Care Panel (Semantic Kernel) -- LayerLens SDK Sample.

Demonstrates evaluating a REAL multi-agent Semantic Kernel workflow. Three named
``ChatCompletionAgent``s -- ``triage-nurse`` -> ``attending-physician`` ->
``clinical-pharmacist`` -- take turns over one ED case in a real
``AgentGroupChat`` (backed by OpenAI gpt-4o-mini). The ``SemanticKernelAdapter``
wraps the group chat's ``invoke`` turn-stream and records honest per-turn
``agent.input`` / ``model.invoke`` / ``cost.record`` / ``agent.output`` events
(stamped with each agent's declared name) plus a real ``agent.handoff`` on each
turn transition, so the recorded trace renders as a genuine 3-node multi-agent
graph (triage-nurse -> attending-physician -> clinical-pharmacist) whose Agent
column reads ``multi-agent``. Framework column reads ``semantic_kernel``.

The trace was recorded from a real AgentGroupChat run (see
samples/data/generators/semantic_kernel.py) and is shipped under
samples/data/traces/industry/. This sample uploads it and evaluates the care
panel's collaboration with domain judges. Nothing is fabricated: the agent
names, handoff edges, tokens, and cost are all real.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python healthcare_sk_care_panel.py
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

SAMPLE = "healthcare_sk_care_panel"
FIXTURE = recorded_trace_path("industry", "healthcare_sk_care_panel.jsonl")

# The ED case the care panel worked. Documents the scenario; the recorded
# multi-agent trace was produced by running this through a real Semantic Kernel
# AgentGroupChat (triage-nurse -> attending-physician -> clinical-pharmacist).
CASE: dict[str, Any] = {
    "case_id": "ED-77120",
    "panel": ["triage-nurse", "attending-physician", "clinical-pharmacist"],
    "presentation": (
        "62yo male, exertional chest pressure, controlled hypertension on lisinopril, "
        "troponin pending. Provide a triage read, an attending assessment, and a "
        "medication-safety check."
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded multi-agent Semantic Kernel care-panel trace."""
    print("=== LayerLens Industry: Multi-Agent Clinical Care Panel (Semantic Kernel) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded multi-agent trace first (renders as a 3-node multi-agent
    # graph via real AgentGroupChat handoffs). Do this before creating judges so
    # the trace always lands even if the org has no evaluation model yet.
    print("Uploading the recorded care-panel trace (multi-agent AgentGroupChat handoffs)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Case:     {CASE['case_id']} (panel: {', '.join(CASE['panel'])})\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "clinical_soundness": create_judge(
                client,
                name="Clinical Soundness Judge",
                evaluation_goal="Evaluate whether the panel's combined output (triage acuity, attending assessment, medication-safety check) is clinically sound for a possible acute coronary syndrome.",
                namespace=SAMPLE,
            ),
            "role_coordination": create_judge(
                client,
                name="Role Coordination Judge",
                evaluation_goal="Evaluate whether each agent contributed its own role (nurse triage, physician assessment, pharmacist med-safety) and built on the prior turn rather than duplicating it.",
                namespace=SAMPLE,
            ),
            "medication_safety": create_judge(
                client,
                name="Medication Safety Judge",
                evaluation_goal="Evaluate whether the clinical pharmacist correctly flagged relevant medication or interaction concerns given the patient's lisinopril and the proposed acute-care orders.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the care panel's collaboration:")
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
