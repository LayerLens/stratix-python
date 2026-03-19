"""Travel Itinerary Planning Demo

Demonstrates LayerLens evaluation for AI itinerary planning using
LangGraph-style multi-step planning for multi-city trips. Evaluates
with itinerary feasibility and booking accuracy judges.

Requires:
    LAYERLENS_STRATIX_API_KEY  - LayerLens platform API key
    OPENAI_API_KEY             - OpenAI API key
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

from openai import OpenAI

from layerlens import Stratix
from layerlens.instrument import STRATIX, emit_input, emit_output, emit_model_invoke

try:
    from langgraph.graph import StateGraph

    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    print("Note: LangGraph not installed. Using direct OpenAI calls.")
    print("  pip install langgraph")


JUDGES = [
    {
        "name": "Itinerary Feasibility Judge",
        "evaluation_goal": (
            "Evaluate whether the AI-generated itinerary is physically feasible "
            "and logistically sound. The response MUST: (1) ensure connection "
            "times between flights meet minimum connection time (MCT) for each "
            "airport (domestic 60 min, international 90 min minimum), (2) not "
            "schedule activities during transit time, (3) account for time zone "
            "changes in the daily schedule, (4) not exceed 14 hours of scheduled "
            "activities per day. Per DOT consumer protection rules (14 CFR 399), "
            "all pricing must include taxes and fees. Score FAIL if any segment "
            "is physically impossible, connection times are insufficient, or "
            "pricing excludes mandatory fees."
        ),
    },
    {
        "name": "Booking Accuracy Judge",
        "evaluation_goal": (
            "Assess whether hotel and flight recommendations contain accurate, "
            "verifiable details. The AI must: (1) recommend real hotels and "
            "airlines that operate on the specified routes, (2) provide realistic "
            "price ranges for the travel dates, (3) not fabricate flight numbers "
            "or hotel amenities, (4) disclose cancellation policies when "
            "recommending bookings. Score FAIL if any recommended property, "
            "airline route, or price is clearly fabricated."
        ),
    },
    {
        "name": "Accessibility and Safety Judge",
        "evaluation_goal": (
            "Verify that the itinerary accounts for traveler safety and "
            "accessibility needs. The output must: (1) include State Department "
            "travel advisory levels for international destinations, (2) note "
            "visa requirements for the traveler's nationality, (3) flag health "
            "advisories or vaccination requirements, (4) provide emergency "
            "contact information for each destination. Score FAIL if safety-"
            "critical travel advisories are omitted for high-risk destinations."
        ),
    },
]

TRAVEL_SCENARIOS = [
    "Plan a 10-day trip: NYC -> Tokyo -> Bangkok -> NYC. Budget $5,000. Prefer direct flights where possible.",
    "Family vacation: 2 adults, 2 children (ages 5, 8). London -> Paris -> Barcelona, 7 days, budget $8,000.",
    "Business trip: San Francisco -> Singapore -> Dubai -> London -> San Francisco. 12 days, need lounge access.",
]


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return val


def build_stratix() -> STRATIX:
    return STRATIX(
        policy_ref="stratix-policy-travel-v1@1.0.0",
        agent_id="itinerary_planning_agent",
        framework="langgraph" if HAS_LANGGRAPH else "openai",
    )


def run_scenario(prompt: str, stratix: STRATIX, oai: OpenAI, model: str) -> str:
    ctx = stratix.start_trial()
    print(f"  [trace]  trial={ctx.trace_id[:12]}...")

    with stratix.context():
        emit_input(prompt, role="human")

        system_msg = (
            "You are a travel itinerary planning AI. Create detailed day-by-day "
            "itineraries with flights, hotels, and activities. Include realistic "
            "pricing with taxes and fees. Account for time zones and minimum "
            "connection times. Include travel advisories, visa requirements, "
            "and health notices for international destinations. Provide "
            "cancellation policy information."
        )

        t0 = time.perf_counter()
        resp = oai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        answer = resp.choices[0].message.content or ""
        usage = resp.usage
        emit_model_invoke(
            provider="openai",
            name=model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            latency_ms=latency_ms,
        )
        emit_output(answer)

    stratix.end_trial()
    return answer


def create_judges_and_evaluate(client: Stratix, stratix: STRATIX):
    events = stratix.get_events()
    if not events:
        print("[warn]  No events captured; skipping evaluation.")
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for ev in events:
            f.write(json.dumps(ev.to_dict()) + "\n")
        trace_path = f.name

    try:
        print("[upload] Uploading trace...")
        resp = client.traces.upload(trace_path)
        if not resp or not resp.trace_ids:
            print("[upload] No trace IDs returned.", file=sys.stderr)
            return
        trace_id = resp.trace_ids[0]
        print(f"[upload] trace_id={trace_id}")
    finally:
        os.unlink(trace_path)

    for jcfg in JUDGES:
        print(f"[judge]  Creating: {jcfg['name']}")
        judge = client.judges.create(name=jcfg["name"], evaluation_goal=jcfg["evaluation_goal"])
        if not judge:
            print(f"[judge]  Failed to create {jcfg['name']}", file=sys.stderr)
            continue
        print(f"[eval]   Running {jcfg['name']} against trace...")
        te = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge.id)
        if te:
            print(f"[eval]   Evaluation submitted: id={te.id}  status={te.status}")
        else:
            print(f"[eval]   Failed to submit evaluation.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Travel Itinerary Planning - LayerLens Evaluation Demo"
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--scenario", type=int, default=0, help="Scenario index 0-2 (default: 0)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip judge creation and evaluation")
    args = parser.parse_args()

    openai_key = _require_env("OPENAI_API_KEY")
    oai = OpenAI(api_key=openai_key)

    stratix = build_stratix()
    prompt = TRAVEL_SCENARIOS[args.scenario % len(TRAVEL_SCENARIOS)]

    print(f"[scenario] {prompt}")
    answer = run_scenario(prompt, stratix, oai, args.model)
    print(f"[answer]   {answer[:120]}...")

    if not args.skip_eval:
        ll_key = _require_env("LAYERLENS_STRATIX_API_KEY")
        client = Stratix(api_key=ll_key)
        create_judges_and_evaluate(client, stratix)

    print("[done]")


if __name__ == "__main__":
    main()
