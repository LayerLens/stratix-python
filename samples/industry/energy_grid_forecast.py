#!/usr/bin/env python3
"""Industry: Energy Grid Load Forecasting -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL AWS Bedrock (Amazon Nova) agent. A
``grid-load-forecaster`` agent ingests live per-zone grid telemetry and weather
and produces a 24-hour peak-load forecast, flagging any zone at risk of
breaching a safe reserve margin. The call runs through the Bedrock Converse API
under LayerLens' ``@trace`` + the Bedrock provider adapter, so the recorded
trace carries a real ``agent.identity`` (Agent column ``grid-load-forecaster``),
a real ``model.invoke`` (Framework column ``aws_bedrock``, model
``amazon.nova-micro-v1:0``), and a real ``cost.record`` (Nova is priced).

The trace was recorded from a real Bedrock Nova run (see
samples/data/_generate_fixtures.py) and is shipped under
samples/data/traces/industry/. This sample uploads it and evaluates the
forecast with grid-safety and predictive-accuracy judges. Mirrors the ateam
energy/grid_load_forecaster.py industry scenario.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python energy_grid_forecast.py
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

SAMPLE = "energy_grid_forecast"
FIXTURE = recorded_trace_path("industry", "energy_grid_forecast.jsonl")

# The grid telemetry the forecaster analyzed. Documents the scenario; the
# recorded trace was produced by running this through a real AWS Bedrock (Amazon
# Nova) Converse call under LayerLens instrumentation.
TELEMETRY: dict[str, Any] = {
    "operator": "MISO-Central",
    "as_of": "2026-07-14T16:00:00Z",
    "horizon_hours": 24,
    "zones": [
        {"zone": "Z1-Metro", "current_load_mw": 4820, "capacity_mw": 6000, "renewable_pct": 22},
        {"zone": "Z2-Suburban", "current_load_mw": 3110, "capacity_mw": 4200, "renewable_pct": 34},
        {"zone": "Z3-Industrial", "current_load_mw": 5180, "capacity_mw": 5600, "renewable_pct": 12},
        {"zone": "Z4-Coastal", "current_load_mw": 2040, "capacity_mw": 3500, "renewable_pct": 51},
        {"zone": "Z5-Rural", "current_load_mw": 1290, "capacity_mw": 2200, "renewable_pct": 28},
    ],
    "weather": {"temp_f": 101, "humidity_pct": 38, "wind_mph": 6, "heat_advisory": True},
    "notes": "Regional heat advisory; EV-charging cluster ramp expected 6-9pm in Z1/Z3.",
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded AWS Bedrock grid-load-forecaster trace."""
    print("=== LayerLens Industry: Energy Grid Load Forecasting (AWS Bedrock / Nova) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded real trace first (renders Agent=grid-load-forecaster,
    # Framework=aws_bedrock, Status from the real run). Do this before creating
    # judges so the trace always lands even if the org has no evaluation model yet.
    print("Uploading the recorded grid-load-forecaster trace (AWS Bedrock / Nova)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Operator: {TELEMETRY['operator']} ({len(TELEMETRY['zones'])} zones, {TELEMETRY['horizon_hours']}h horizon)\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "grid_safety": create_judge(
                client,
                name="Grid Safety Judge",
                evaluation_goal="Evaluate whether the forecast keeps every zone within a safe reserve margin (>=8% headroom below capacity) and never under-predicts load during the peak heat event.",
                namespace=SAMPLE,
            ),
            "forecast_accuracy": create_judge(
                client,
                name="Forecast Accuracy Judge",
                evaluation_goal="Evaluate whether the 24-hour peak-load forecast is quantitative, per-zone, and consistent with the current load, capacity, and weather inputs.",
                namespace=SAMPLE,
            ),
            "mitigation_soundness": create_judge(
                client,
                name="Mitigation Soundness Judge",
                evaluation_goal="Evaluate whether the recommended mitigations (demand response, generation dispatch, curtailment) are concrete and appropriate for the zones flagged at risk.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the forecast:")
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
