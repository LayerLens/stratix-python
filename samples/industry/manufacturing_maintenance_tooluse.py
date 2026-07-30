#!/usr/bin/env python3
"""Industry: Manufacturing Maintenance with Tool Use (Azure OpenAI) -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL tool-using predictive-maintenance agent on Azure
OpenAI. The ``predictive-maintenance-agent`` reads a hydraulic-press sensor
snapshot, calls a ``get_maintenance_history`` tool to review the equipment's
past repairs, then returns a failure-risk assessment grounded in both the live
sensors and the repair history.

The recorded trace was captured from a real, instrumented ``openai.AzureOpenAI``
run driven through LayerLens's Azure OpenAI adapter (see
samples/data/_generate_fixtures.py). The adapter records the real ``tool.call``
the model made, the ``tool.result`` the local tool genuinely returned, and both
model turns. Azure OpenAI is credential-gated, so the fixture is recorded
SEALED: the real adapter and the real Azure SDK run over an
``httpx.MockTransport``, so the trace carries genuine framework=azure_openai
``model.invoke`` events, priced ``cost.record`` events, the tool.call/tool.result
loop, a synthesized ``agent.identity`` and an intact attestation chain -- only
the LLM network is mocked. The LayerLens UI renders the Agent, Framework
(azure_openai), and Status columns from real data.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python manufacturing_maintenance_tooluse.py
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

SAMPLE = "manufacturing_maintenance_tooluse"
FIXTURE = recorded_trace_path("industry", "manufacturing_maintenance_tooluse.jsonl")

# The machine snapshot the tool-using agent assessed. Documents the scenario;
# the recorded trace was produced by running this through a real Azure OpenAI
# tool-use loop (sealed over a MockTransport -- see data/_generate_fixtures.py):
# the model called get_maintenance_history before advising.
MACHINE: dict[str, Any] = {
    "equipment_id": "MF-EQ-021",
    "equipment_type": "Hydraulic Press (Schuler MSD 630)",
    "operating_hours": 22800,
    "sensor_snapshot": {
        "vibration_mm_s": 4.2,
        "temperature_c": 63.0,
        "pressure_bar": 5.4,
        "current_amps": 61.0,
    },
    "trend": "pressure drifting down toward the 5.0 bar minimum; return-line variability rising",
    "tool_used": "get_maintenance_history",
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded Azure OpenAI tool-use maintenance trace."""
    print("=== LayerLens Manufacturing: Maintenance with Tool Use (Azure OpenAI) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded real trace first (renders framework=azure_openai with
    # the tool.call/tool.result loop). Do this before creating judges so the
    # trace always lands even if the org has no evaluation model yet.
    print("Uploading the recorded tool-use maintenance trace (Azure OpenAI, sealed)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID:  {trace_id}")
    print(f"  Equipment: {MACHINE['equipment_id']} ({MACHINE['equipment_type']})")
    print(f"  Tool used: {MACHINE['tool_used']}\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "tool_use_soundness": create_judge(
                client,
                name="Tool Use Soundness Judge",
                evaluation_goal="Evaluate whether calling get_maintenance_history was appropriate and whether the returned repair history was actually used in the assessment.",
                namespace=SAMPLE,
            ),
            "prediction_quality": create_judge(
                client,
                name="Failure Prediction Judge",
                evaluation_goal="Evaluate whether the failure-risk assessment and remaining-useful-life estimate are justified by the sensor data and the maintenance history.",
                namespace=SAMPLE,
            ),
            "recommendation_quality": create_judge(
                client,
                name="Maintenance Recommendation Judge",
                evaluation_goal="Evaluate whether the maintenance recommendation is concrete, actionable, and appropriately prioritized for the assessed risk.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the tool-use maintenance assessment:")
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
