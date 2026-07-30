#!/usr/bin/env python3
"""Industry: Travel Trip-Options Research (browser-use, real headless browse) -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL browser-driving agent built with ``browser-use``.
A corporate travel desk's ``trip-research-agent`` (a real ``browser_use.Agent``
backed by OpenAI) drove a REAL headless Chromium over CDP to read the desk's
Trip Options Board and pick the one option that satisfies a client's booking
policy: nonstop, freely cancellable, and under a $1,400 cap -- cheapest first.

browser-use is a **framework** adapter (it wraps ``Agent.run()`` and walks the
real ``AgentHistoryList``). Because the trace was captured from a genuine
instrumented browse, it renders a single honest agent node (Agent column
``trip-research-agent``), Framework ``browser_use``, Status ``ok``, plus the
real per-action ``tool.call`` events the browser actually executed (a navigate,
the model's own clicks reading the board, then done) and one real
``model.invoke`` / ``cost.record`` carrying the token counts browser-use's own
token service recorded for the run (usage lives on the history list, not per
step, so exactly one invoke is emitted). Nothing is fabricated. The recorded
trace is shipped under samples/data/traces/industry/ (produced by
samples/data/generators/browser_use.py); this sample uploads it and evaluates
the research decision with domain judges.

The agent's answer -- option NW-101 -- is the genuinely correct pick: a cheaper
option on the board had a stop, and another nonstop one was non-refundable, so a
"grab the smallest number" answer would have been wrong. The agent's stated
rationale is its own real wording and is shipped verbatim, imprecision included
-- that is exactly the sort of thing the Selection Rationale judge is here to
score, and editing the model's answer to look better would be fabrication.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

    No provider key is needed: the browse and the model call already happened
    when the trace was recorded. This sample only uploads and evaluates it.

Usage:
    python travel_browseruse_research.py
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

SAMPLE = "travel_browseruse_research"
FIXTURE = recorded_trace_path("industry", "travel_browseruse_research.jsonl")

# The travel-desk booking request the agent researched. Documents the scenario;
# the recorded trace was produced by driving this through a real browser_use
# Agent that really opened the Trip Options Board in a headless Chromium, read
# the rendered table, and applied these constraints to it.
REQUEST: dict[str, Any] = {
    "desk": "Northwind Travel Desk",
    "request_id": "TRQ-4482",
    "client": "Meridian Analytics",
    "route": "Boston (BOS) -> Lisbon (LIS)",
    "window": "October 2026",
    "policy": {
        "nonstop_required": True,
        "free_cancellation_required": True,  # the trip is client-billable
        "max_total_usd": 1400,
        "tie_breaker": "cheapest qualifying option",
    },
    "question": (
        "Read the Trip Options Board and pick the cheapest nonstop option with free "
        "cancellation under $1,400. Report the option code, airline, hotel, and total, "
        "and say why the cheaper options were rejected."
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded browser-use trip-options research trace."""
    print("=== LayerLens Industry: Travel Trip-Options Research (browser-use) ===\n")

    # Cred guard: this sample needs ONLY a LayerLens key (no provider key -- the
    # browse and the model call already happened at record time). Report the gap
    # honestly and exit cleanly rather than raising.
    if not os.environ.get("LAYERLENS_STRATIX_API_KEY"):
        print("SKIPPED: LAYERLENS_STRATIX_API_KEY is not set.")
        print("  Set it to upload and evaluate the recorded trip-research trace:")
        print("    export LAYERLENS_STRATIX_API_KEY=your-api-key")
        sys.exit(1)

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded browse first (renders a single trip-research-agent node
    # with the real per-action tool.call events the browser executed and the run's
    # real model.invoke / cost.record). Do this before creating judges so the
    # trace always lands even if the org has no evaluation model yet.
    print("Uploading the recorded trip-research trace (real headless browse of the options board)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID:  {trace_id}")
    print(f"  Request:   {REQUEST['request_id']} for {REQUEST['client']}")
    print(f"  Route:     {REQUEST['route']}  ({REQUEST['window']})")
    print(
        "  Policy:    nonstop + free cancellation, under $%d\n" % REQUEST["policy"]["max_total_usd"]
    )

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "policy_compliance": create_judge(
                client,
                name="Booking Policy Compliance Judge",
                evaluation_goal="Evaluate whether the selected trip option actually satisfies every stated booking policy constraint: the flight is nonstop, the booking has free cancellation, and the total is under $1,400.",
                namespace=SAMPLE,
            ),
            "board_grounding": create_judge(
                client,
                name="Options Board Grounding Judge",
                evaluation_goal="Evaluate whether the agent grounded its answer in options that actually appeared on the trip options board it browsed, rather than inventing an option code, airline, hotel, or price.",
                namespace=SAMPLE,
            ),
            "selection_rationale": create_judge(
                client,
                name="Selection Rationale Judge",
                evaluation_goal="Evaluate whether the agent correctly explains why the cheaper options on the board were rejected, rather than simply picking the lowest price.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the trip-research decision:")
        for label, judge in judges.items():
            evaluation = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge.id)
            if evaluation is None:
                print(f"  {label:20s} -- evaluation creation failed")
                continue
            results = poll_evaluation_results(client, evaluation.id)
            if results:
                r = results[0]
                verdict = "pass" if r.passed else "fail"
                color = _VERDICT_COLORS.get(verdict, "")
                print(f"  {label:20s} {color}{verdict.upper()}{_RESET} ({r.score:.2f})")
            else:
                print(f"  {label:20s} -- timed out waiting for results")
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
