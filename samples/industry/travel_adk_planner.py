#!/usr/bin/env python3
"""Industry: Travel Trip Planner (Google ADK / Gemini, multi-agent) -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL hierarchical multi-agent workflow built on the
Google Agent Development Kit (ADK). A ``trip_coordinator`` agent owns two
``sub_agents`` -- a ``flight_specialist`` and a ``hotel_specialist`` -- and
delegates to them using ADK's ``transfer_to_agent`` mechanism: it hands off to
the flight specialist (which calls a real ``search_flights`` tool and reports
back), then to the hotel specialist (which calls a real ``search_hotels`` tool
and reports back), and finally summarizes the full itinerary.

Because the trace was captured from a genuine live Gemini run, the LayerLens graph
engine derives a real MULTI-AGENT graph from the ``agent.handoff`` edges
(coordinator <-> flight_specialist, coordinator <-> hotel_specialist), so the
Agent column renders ``multi-agent`` and the Framework column ``google_adk``. It
carries the real per-agent ``model.invoke`` / ``cost.record`` events (priced from
the real Gemini token counts) and the two specialists' ``tool.call`` /
``tool.result`` searches -- nothing is fabricated. The recorded trace is shipped
under samples/data/traces/industry/ (produced by samples/data/generators/
google_adk.py); this sample uploads it and evaluates the delegation + itinerary
with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python travel_adk_planner.py
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

SAMPLE = "travel_adk_planner"
FIXTURE = recorded_trace_path("industry", "travel_adk_planner.jsonl")

# The trip request the coordinator planned. Documents the scenario; the recorded
# trace was produced by driving this through a real Google ADK hierarchical team
# (trip_coordinator delegating to flight_specialist + hotel_specialist over Gemini).
REQUEST: dict[str, Any] = {
    "channel": "trip_planning_assistant",
    "origin": "Seattle (SEA)",
    "destination": "Denver (DEN)",
    "depart_date": "Nov 12",
    "return_date": "Nov 15",
    "hotel_budget_per_night_usd": 220,
    "request": (
        "Plan a round trip from Seattle (SEA) to Denver (DEN), departing Nov 12 and "
        "returning Nov 15, and book a hotel in Denver for those nights under $220/night."
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded Google ADK multi-agent trip-planner trace."""
    print("=== LayerLens Industry: Travel Trip Planner (Google ADK / Gemini, multi-agent) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded multi-agent trace first (renders Agent = multi-agent with
    # the coordinator + flight/hotel specialists and their handoff edges, Framework
    # = google_adk). Do this before creating judges so the trace always lands even
    # if the org has no evaluation model yet.
    print("Uploading the recorded trip-planner trace (coordinator -> flight/hotel delegation)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID:  {trace_id}")
    print(f"  Trip:      {REQUEST['origin']} -> {REQUEST['destination']}  ({REQUEST['depart_date']}-{REQUEST['return_date']})\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "delegation": create_judge(
                client,
                name="Delegation Correctness Judge",
                evaluation_goal="Evaluate whether the coordinator correctly delegated the flight task to the flight specialist and the hotel task to the hotel specialist, and each specialist used its search tool.",
                namespace=SAMPLE,
            ),
            "itinerary_completeness": create_judge(
                client,
                name="Itinerary Completeness Judge",
                evaluation_goal="Evaluate whether the final summary includes both the recommended flight and a hotel within the stated nightly budget, with a coherent itinerary.",
                namespace=SAMPLE,
            ),
            "budget_compliance": create_judge(
                client,
                name="Budget Compliance Judge",
                evaluation_goal="Evaluate whether the recommended hotel respects the traveler's per-night budget and the flight/hotel choices are internally consistent with the request.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the planned itinerary:")
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
