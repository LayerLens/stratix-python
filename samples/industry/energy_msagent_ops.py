#!/usr/bin/env python3
"""Industry: Multi-Agent Energy Grid-Ops Adjudication -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL multi-agent Microsoft Agent Framework
``AgentGroupChat``. Three named specialists -- ``grid-load-forecaster`` ->
``dispatch-optimizer`` -> ``reliability-auditor`` (each a
``semantic_kernel.agents.ChatCompletionAgent`` backed by OpenAI gpt-4o-mini) --
adjudicate a grid contingency (an N-1 unit trip during a heat wave) in
sequential round-robin turns. The ``MSAgentFrameworkAdapter`` records the real
per-turn ``agent.handoff`` transitions, so the recorded trace renders as a
multi-agent graph (Grid Load Forecaster -> Dispatch Optimizer -> Reliability
Auditor) whose Agent column reads ``multi-agent``. Framework column =
``ms_agent_framework``.

The trace was recorded from a real group-chat run (see
samples/data/generators/ms_agent_framework.py) and is shipped under
samples/data/traces/industry/. This sample uploads it and evaluates the
adjudication with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python energy_msagent_ops.py
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

SAMPLE = "energy_msagent_ops"
FIXTURE = recorded_trace_path("industry", "energy_msagent_ops.jsonl")

# The grid contingency the ops panel adjudicated. Documents the scenario; the
# recorded multi-agent trace was produced by running this through a real MS Agent
# Framework AgentGroupChat (forecaster -> dispatch optimizer -> reliability auditor).
CONTINGENCY: dict[str, Any] = {
    "event_id": "GRID-N1-2291",
    "condition": "summer peak heat wave",
    "forecast_peak_gw": 42.0,
    "available_capacity_gw": 44.0,
    "trip": "1.2 GW nuclear unit tripped offline (N-1 event)",
    "resources": "two 0.4 GW peakers (10-min start); 0.3 GW interchange import offered",
    "transmission": "western corridor at 92% of stability limit",
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded multi-agent grid-ops adjudication trace."""
    print("=== LayerLens Industry: Multi-Agent Energy Grid-Ops Adjudication ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded multi-agent trace first (renders as a multi-agent graph
    # via real turn-transition handoffs). Do this before creating judges so the
    # trace always lands even if the org has no evaluation model yet.
    print("Uploading the recorded grid-ops adjudication trace (multi-agent group chat)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Event:    {CONTINGENCY['event_id']} ({CONTINGENCY['trip']})\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "dispatch_soundness": create_judge(
                client,
                name="Dispatch Soundness Judge",
                evaluation_goal="Evaluate whether the recommended dispatch (committing peakers and/or imports) restores adequate operating reserves at least cost after the N-1 contingency.",
                namespace=SAMPLE,
            ),
            "reliability_compliance": create_judge(
                client,
                name="Reliability Compliance Judge",
                evaluation_goal="Evaluate whether the adjudication confirms N-1 contingency coverage and transmission-limit compliance consistent with NERC reliability standards, flagging any residual risk.",
                namespace=SAMPLE,
            ),
            "adjudication_coherence": create_judge(
                client,
                name="Adjudication Coherence Judge",
                evaluation_goal="Evaluate whether the three specialists' turns (forecast -> dispatch -> reliability audit) form a coherent, non-contradictory grid-operations decision.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the grid-ops adjudication:")
        for label, judge in judges.items():
            evaluation = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge.id)
            if evaluation is None:
                print(f"  {label:24s} -- evaluation creation failed")
                continue
            results = poll_evaluation_results(client, evaluation.id)
            if results:
                r = results[0]
                verdict = "pass" if r.passed else "fail"
                color = _VERDICT_COLORS.get(verdict, "")
                print(f"  {label:24s} {color}{verdict.upper()}{_RESET} ({r.score:.2f})")
            else:
                print(f"  {label:24s} -- timed out waiting for results")
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
