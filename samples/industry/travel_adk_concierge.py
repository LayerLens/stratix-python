#!/usr/bin/env python3
"""Industry: Travel Destination Concierge (Google ADK / Gemini, tool-use) -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL single-agent tool-use workflow built on the Google
Agent Development Kit (ADK). A ``travel_concierge`` agent (backed by Gemini via
Google ADK) answers a traveler's destination question by FIRST calling a real
``lookup_destination_guide`` tool to fetch a curated guide (best season, top
attractions, local tips, rough daily budget), THEN giving a concise day-by-day
priority plan grounded in that guide.

Google ADK is a **framework** adapter (a plugin on the ADK ``Runner``). Because
the trace was captured from a genuine live Gemini run, it renders a single honest
agent node (Agent column ``travel_concierge``), Framework ``google_adk``, and a
real two-step tool loop -- ``model.invoke`` / ``cost.record`` (priced from the
real Gemini token counts) plus ``tool.call`` / ``tool.result`` for the guide
lookup. Nothing is fabricated. The recorded trace is shipped under
samples/data/traces/industry/ (produced by samples/data/generators/google_adk.py);
this sample uploads it and evaluates the concierge's plan with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python travel_adk_concierge.py
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

SAMPLE = "travel_adk_concierge"
FIXTURE = recorded_trace_path("industry", "travel_adk_concierge.jsonl")

# The traveler question the concierge answered. Documents the scenario; the
# recorded trace was produced by driving this through a real Google ADK + Gemini
# tool loop (the agent called lookup_destination_guide, then planned against it).
REQUEST: dict[str, Any] = {
    "channel": "trip_planning_assistant",
    "destination": "Kyoto, Japan",
    "trip_length_days": 3,
    "season": "spring",
    "question": (
        "I'm visiting Kyoto, Japan for 3 days this spring. What should I prioritize, "
        "and roughly what daily budget should I plan for?"
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded Google ADK travel-concierge tool-use trace."""
    print("=== LayerLens Industry: Travel Destination Concierge (Google ADK / Gemini) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded tool-use trace first (renders a single travel_concierge
    # node with real model.invoke / tool.call / tool.result events). Do this
    # before creating judges so the trace always lands even if the org has no
    # evaluation model yet.
    print("Uploading the recorded travel-concierge trace (lookup_destination_guide tool loop)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID:     {trace_id}")
    print(f"  Destination:  {REQUEST['destination']}  ({REQUEST['trip_length_days']} days, {REQUEST['season']})\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "guide_grounding": create_judge(
                client,
                name="Guide Grounding Judge",
                evaluation_goal="Evaluate whether the concierge grounded its plan in the guide returned by the lookup_destination_guide tool (attractions, season, budget) rather than inventing details.",
                namespace=SAMPLE,
            ),
            "plan_quality": create_judge(
                client,
                name="Itinerary Plan Quality Judge",
                evaluation_goal="Evaluate whether the response gives a clear, sensible day-by-day priority plan appropriate to the trip length and season.",
                namespace=SAMPLE,
            ),
            "budget_realism": create_judge(
                client,
                name="Budget Realism Judge",
                evaluation_goal="Evaluate whether the daily budget guidance is concrete and plausibly matches the destination's cost level as described in the guide.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the concierge's plan:")
        for label, judge in judges.items():
            evaluation = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge.id)
            if evaluation is None:
                print(f"  {label:18s} -- evaluation creation failed")
                continue
            results = poll_evaluation_results(client, evaluation.id)
            if results:
                r = results[0]
                verdict = "pass" if r.passed else "fail"
                color = _VERDICT_COLORS.get(verdict, "")
                print(f"  {label:18s} {color}{verdict.upper()}{_RESET} ({r.score:.2f})")
            else:
                print(f"  {label:18s} -- timed out waiting for results")
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
