#!/usr/bin/env python3
"""Industry: Multi-Agent Manufacturing Defect-Triage Swarm (AWS Strands) -- LayerLens SDK Sample.

Demonstrates evaluating a REAL multi-agent AWS Strands ``Swarm`` with genuine
handoffs. A manufacturing defect report is triaged by three specialists: an
``intake-coordinator`` hands off (Strands' built-in ``handoff_to_agent`` tool)
to a ``defect-analyst``, who -- because the defect warrants a process change --
hands off to a ``corrective-action-engineer``. The Strands adapter records the
real ``agent.handoff`` edges and every node's ``model.invoke``, so the recorded
trace renders as a multi-agent graph (intake-coordinator -> defect-analyst ->
corrective-action-engineer) whose Agent column reads ``multi-agent``.

The whole swarm was captured as ONE trace from a real Strands run (see
samples/data/generators/strands.py) and is shipped under
samples/data/traces/industry/. This sample uploads it and evaluates the crew's
root-cause analysis and corrective action with domain judges. Nothing is
fabricated: the Framework column shows ``strands``, the agent nodes are the real
producer-declared identities, and the handoff edges are genuine.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python manufacturing_strands_defect_swarm.py
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

SAMPLE = "manufacturing_strands_defect_swarm"
FIXTURE = recorded_trace_path("industry", "manufacturing_strands_defect_swarm.jsonl")

# The defect report the swarm triaged. Documents the scenario; the recorded
# multi-agent trace was produced by running this through a real Strands Swarm
# (intake-coordinator -> defect-analyst -> corrective-action-engineer via
# handoff_to_agent).
DEFECT_REPORT: dict[str, Any] = {
    "report_id": "DR-3391",
    "line": "assembly line 4",
    "part": "TX-90 gearbox housing",
    "material": "die-cast A380 aluminum",
    "defect": "hairline cracks at the mounting boss (6% of today's run)",
    "observation": "cracks appear only after the press-fit bearing insertion station",
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded multi-agent strands defect-triage-swarm trace."""
    print("=== LayerLens Industry: Multi-Agent Manufacturing Defect Swarm (AWS Strands) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded multi-agent trace first (renders as a multi-agent graph
    # via real handoff_to_agent edges). Do this before creating judges so the
    # trace always lands even if the org has no evaluation model yet.
    print("Uploading the recorded defect-triage-swarm trace (multi-agent handoff graph)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Report:   {DEFECT_REPORT['report_id']} ({DEFECT_REPORT['part']})\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "root_cause_soundness": create_judge(
                client,
                name="Root Cause Soundness Judge",
                evaluation_goal="Evaluate whether the defect-analyst identified a plausible, physically-grounded root cause for the mounting-boss cracks given the press-fit insertion clue.",
                namespace=SAMPLE,
            ),
            "handoff_appropriateness": create_judge(
                client,
                name="Handoff Appropriateness Judge",
                evaluation_goal="Evaluate whether each handoff routed the work to the right specialist (intake -> analyst -> corrective-action) rather than skipping or misrouting steps.",
                namespace=SAMPLE,
            ),
            "corrective_action_quality": create_judge(
                client,
                name="Corrective Action Quality Judge",
                evaluation_goal="Evaluate whether the proposed corrective/preventive action concretely removes the identified root cause so the defect cannot recur.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the swarm's defect triage:")
        for label, judge in judges.items():
            evaluation = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge.id)
            if evaluation is None:
                print(f"  {label:26s} -- evaluation creation failed")
                continue
            results = poll_evaluation_results(client, evaluation.id)
            if results:
                r = results[0]
                verdict = "pass" if r.passed else "fail"
                color = _VERDICT_COLORS.get(verdict, "")
                print(f"  {label:26s} {color}{verdict.upper()}{_RESET} ({r.score:.2f})")
            else:
                print(f"  {label:26s} -- timed out waiting for results")
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
