#!/usr/bin/env python3
"""Industry: Manufacturing Predictive Maintenance (Azure OpenAI) -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL predictive-maintenance agent that runs on Azure
OpenAI. A ``predictive-maintenance-agent`` reads a live sensor snapshot from a
CNC spindle (vibration, temperature, current, acoustic emission) showing a
rising bearing-defect signature and returns a failure-risk assessment, a
remaining-useful-life estimate, and a maintenance recommendation.

The recorded trace was captured from a real, instrumented ``openai.AzureOpenAI``
run driven through LayerLens's Azure OpenAI adapter (see
samples/data/_generate_fixtures.py). Azure OpenAI is credential-gated, so the
fixture is recorded SEALED: the real adapter and the real Azure SDK run over an
``httpx.MockTransport``, so the trace carries a genuine framework=azure_openai
``model.invoke``, a priced ``cost.record``, a synthesized ``agent.identity`` and
an intact attestation chain -- only the LLM network is mocked. The LayerLens UI
therefore renders the Agent, Framework (azure_openai), and Status columns from
real data.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python manufacturing_predictive_maintenance.py
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

SAMPLE = "manufacturing_predictive_maintenance"
FIXTURE = recorded_trace_path("industry", "manufacturing_predictive_maintenance.jsonl")

# The machine sensor snapshot the agent assessed. Documents the scenario; the
# recorded trace was produced by running this through a real Azure OpenAI call
# (sealed over a MockTransport -- see samples/data/_generate_fixtures.py).
MACHINE: dict[str, Any] = {
    "equipment_id": "MF-EQ-014",
    "equipment_type": "CNC Milling Machine (DMG Mori DMU 50 5-Axis)",
    "operating_hours": 12450,
    "sensor_snapshot": {
        "vibration_mm_s": 6.8,
        "temperature_c": 79.4,
        "current_amps": 58.0,
        "acoustic_db": 88.0,
        "rpm": 1750,
    },
    "trend": "vibration up 41% over 72h; 1x + 2x + BPFI harmonics emerging",
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded Azure OpenAI predictive-maintenance trace."""
    print("=== LayerLens Manufacturing: Predictive Maintenance (Azure OpenAI) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded real trace first (renders framework=azure_openai + the
    # predictive-maintenance-agent). Do this before creating judges so the trace
    # always lands even if the org has no evaluation model yet.
    print("Uploading the recorded predictive-maintenance trace (Azure OpenAI, sealed)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID:  {trace_id}")
    print(f"  Equipment: {MACHINE['equipment_id']} ({MACHINE['equipment_type']})\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "prediction_quality": create_judge(
                client,
                name="Failure Prediction Judge",
                evaluation_goal="Evaluate whether the failure-risk assessment and remaining-useful-life estimate are justified by the sensor data and trend.",
                namespace=SAMPLE,
            ),
            "safety_threshold": create_judge(
                client,
                name="Safety Threshold Judge",
                evaluation_goal="Evaluate whether the recommendation respects the equipment's safety thresholds and does not advise running unsafe machinery to failure.",
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

        print("Evaluating the predictive-maintenance assessment:")
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
