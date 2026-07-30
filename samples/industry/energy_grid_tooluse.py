#!/usr/bin/env python3
"""Industry: Energy Grid Forecasting with Tool Use -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL AWS Bedrock (Amazon Nova) agent that uses the
Bedrock Converse *tool-use* loop. The ``grid-load-forecaster`` agent does not
have the live load in its prompt -- it must call a ``get_sensor_reading`` tool
(a real SCADA sensor lookup) to fetch the current load for a zone, receive the
tool result, and only then produce the next-hour forecast and reserve-margin
risk read. Running through LayerLens' ``@trace`` + the Bedrock provider adapter,
the recorded trace carries the real ``model.invoke`` events (Framework column
``aws_bedrock``, model ``amazon.nova-micro-v1:0``), the real ``tool.call`` event
(``get_sensor_reading``), and the real ``cost.record`` events (Nova is priced),
all under one ``grid-load-forecaster`` agent node.

The trace was recorded from a real Bedrock Nova Converse tool-use run (see
samples/data/_generate_fixtures.py) and is shipped under
samples/data/traces/industry/. This sample uploads it and evaluates whether the
forecast was correctly grounded in the tool's live reading.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python energy_grid_tooluse.py
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

SAMPLE = "energy_grid_tooluse"
FIXTURE = recorded_trace_path("industry", "energy_grid_tooluse.jsonl")

# The scenario the forecaster resolved via a Bedrock Converse tool call.
# Documents what was analyzed; the recorded trace was produced by running this
# through a real AWS Bedrock (Amazon Nova) Converse tool-use loop under
# LayerLens instrumentation. The agent called ``get_sensor_reading`` to read
# Z3-Industrial's live load (5480 MW) before forecasting.
SCENARIO: dict[str, Any] = {
    "zone": "Z3-Industrial",
    "capacity_mw": 5600,
    "safe_reserve_margin_pct": 8,
    "tool": "get_sensor_reading",
    "ask": "What is Z3-Industrial's load right now, and what's the next-hour forecast and risk?",
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded AWS Bedrock Converse tool-use forecaster trace."""
    print("=== LayerLens Industry: Grid Forecasting with Tool Use (AWS Bedrock / Nova) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded real trace first (renders Agent=grid-load-forecaster,
    # Framework=aws_bedrock, with a real tool.call in the timeline). Do this
    # before creating judges so the trace always lands even if the org has no
    # evaluation model yet.
    print("Uploading the recorded tool-use grid-forecaster trace (AWS Bedrock / Nova)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Zone:     {SCENARIO['zone']} (via {SCENARIO['tool']} tool call)\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "tool_grounding": create_judge(
                client,
                name="Tool Grounding Judge",
                evaluation_goal="Evaluate whether the forecast is grounded in the value returned by the get_sensor_reading tool rather than a guessed or hallucinated load.",
                namespace=SAMPLE,
            ),
            "forecast_correctness": create_judge(
                client,
                name="Forecast Correctness Judge",
                evaluation_goal="Evaluate whether the next-hour load forecast and the reserve-margin risk assessment are quantitative and consistent with the zone's live load and capacity.",
                namespace=SAMPLE,
            ),
            "tool_use_discipline": create_judge(
                client,
                name="Tool Use Discipline Judge",
                evaluation_goal="Evaluate whether the agent correctly recognized it lacked the live load and called the sensor tool before forecasting, instead of answering without the reading.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the tool-grounded forecast:")
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
