#!/usr/bin/env python3
"""Industry: Travel Itinerary Planner -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL single-agent LangGraph workflow that carries
genuine LLM token/cost data. An ``itinerary-planner`` agent -- a one-node
LangGraph ``StateGraph`` whose node makes a real instrumented OpenAI
(gpt-4o-mini) call -- plans a multi-city trip from a trip request (cities, days,
budget, interests, constraints). The LangGraphCallbackHandler records the node
identity (Framework column reads ``langgraph``) and the OpenAI provider adapter
records the real ``model.invoke`` + ``cost.record`` (real prompt/completion
tokens and cost), so the recorded trace renders a single-agent graph (Agent
column ``itinerary-planner``) with real token and cost data.

The trace was recorded from a real run (see samples/data/_generate_fixtures.py)
and is shipped under samples/data/traces/industry/. This sample uploads it and
evaluates the planner's itinerary with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python travel_itinerary.py
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

SAMPLE = "travel_itinerary"
FIXTURE = recorded_trace_path("industry", "travel_itinerary.jsonl")

# The trip request the itinerary-planner agent planned. Documents the scenario;
# the recorded single-agent trace was produced by running this through the real
# one-node LangGraph planner (an 'itinerary-planner' node calling OpenAI
# gpt-4o-mini), so the trace carries real model/token/cost data.
TRIP_REQUEST: dict[str, Any] = {
    "trip_id": "TRIP-20418",
    "traveler": "solo, comfortable walker, moderate budget",
    "cities": ["Lisbon", "Barcelona"],
    "total_days": 7,
    "budget_usd": 2200,
    "interests": ["food", "architecture", "coastal walks"],
    "constraints": ["no red-eye flights", "one relaxed rest day"],
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded single-agent travel-itinerary trace."""
    print("=== LayerLens Industry: Travel Itinerary Planner ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded trace first (renders as a single-agent LangGraph node
    # with real token/cost data). Do this before creating judges so the trace
    # always lands even if the org has no evaluation model yet.
    print("Uploading the recorded travel-itinerary trace (single-agent LangGraph node)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Trip:     {' + '.join(TRIP_REQUEST['cities'])}, "
          f"{TRIP_REQUEST['total_days']} days, ${TRIP_REQUEST['budget_usd']:,} budget\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "itinerary_feasibility": create_judge(
                client,
                name="Itinerary Feasibility Judge",
                evaluation_goal="Evaluate whether the proposed itinerary is feasible: the days add up to the trip length, the cities are sensibly sequenced, and the daily plan has reasonable timing and pacing.",
                namespace=SAMPLE,
            ),
            "budget_adherence": create_judge(
                client,
                name="Budget Adherence Judge",
                evaluation_goal="Evaluate whether the itinerary respects the traveler's stated budget, with plausible cost estimates that keep the total within the budget.",
                namespace=SAMPLE,
            ),
            "preference_alignment": create_judge(
                client,
                name="Traveler Preference Judge",
                evaluation_goal="Evaluate whether the itinerary reflects the traveler's stated interests and constraints (food, architecture, coastal walks, no red-eye flights, one rest day).",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the planner's itinerary:")
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
