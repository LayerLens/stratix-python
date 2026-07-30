#!/usr/bin/env python3
"""Industry: Energy Grid-Load Forecasting (single agent) -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL single-agent Microsoft Agent Framework run. A
power-grid load-forecasting agent (persona ``grid-load-forecaster``, a
``semantic_kernel.agents.ChatCompletionAgent`` backed by OpenAI gpt-4o-mini)
reviews an ISO control-room grid snapshot on a summer peak day, forecasts the
evening peak demand, assesses the reserve margin against the operating-reserve
requirement, and recommends a specific proactive operator action.

Because the trace was captured under the real ``MSAgentFrameworkAdapter`` (which
wraps the agent's ``invoke`` turn-stream), it renders a single honest agent node
(Agent column ``Grid Load Forecaster``) plus the real ``model.invoke`` /
``cost.record`` events of the run -- no fabrication. Framework column =
``ms_agent_framework``. The recorded trace is shipped under
samples/data/traces/industry/ (produced by
samples/data/generators/ms_agent_framework.py); this sample uploads it and
evaluates the forecast with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python energy_msagent_forecast.py
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

SAMPLE = "energy_msagent_forecast"
FIXTURE = recorded_trace_path("industry", "energy_msagent_forecast.jsonl")

# The grid snapshot the forecaster assessed. Documents the scenario; the recorded
# trace was produced by running this through a real MS Agent Framework
# ChatCompletionAgent (grid-load-forecaster) backed by OpenAI.
SNAPSHOT: dict[str, Any] = {
    "iso": "control-room summer peak day (heat advisory)",
    "current_demand_gw": 38.4,
    "forecast_peak_gw": 42.1,
    "available_capacity_gw": 44.0,
    "operating_reserve_requirement_gw": 3.0,
    "notes": "One 600 MW combined-cycle unit on forced derate; wind falling after 19:00.",
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded grid-load-forecaster trace."""
    print("=== LayerLens Industry: Energy Grid-Load Forecasting (single agent) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded single-agent trace first (renders a single Grid Load
    # Forecaster node with real model.invoke / cost.record events). Do this before
    # creating judges so the trace always lands even if the org has no evaluation
    # model yet.
    print("Uploading the recorded grid-load-forecaster trace (single-agent, ms_agent_framework)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Snapshot: peak {SNAPSHOT['forecast_peak_gw']} GW vs {SNAPSHOT['available_capacity_gw']} GW available\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "forecast_soundness": create_judge(
                client,
                name="Forecast Soundness Judge",
                evaluation_goal="Evaluate whether the load forecast and reserve-margin assessment are internally consistent and correctly compare available capacity against the operating-reserve requirement.",
                namespace=SAMPLE,
            ),
            "operational_action": create_judge(
                client,
                name="Operational Action Judge",
                evaluation_goal="Evaluate whether the recommended operator action (demand response, committing a peaker, or importing power) is specific, appropriate, and proactive for the forecasted grid conditions.",
                namespace=SAMPLE,
            ),
            "grid_reliability": create_judge(
                client,
                name="Grid Reliability Judge",
                evaluation_goal="Evaluate whether the recommendation maintains adequate operating reserves and grid reliability without proposing unnecessary or unsafe measures.",
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
