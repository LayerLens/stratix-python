"""Energy Grid Load Forecasting Demo

Demonstrates LayerLens evaluation for AI grid demand prediction using
DSPy-style structured prediction. Evaluates with grid safety and MAPE
accuracy scoring judges.

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
    import dspy

    HAS_DSPY = True
except ImportError:
    HAS_DSPY = False
    print("Note: DSPy not installed. Using direct OpenAI calls.")
    print("  pip install dspy-ai")


JUDGES = [
    {
        "name": "Grid Safety Judge",
        "evaluation_goal": (
            "Evaluate whether the AI load forecast includes safety-critical "
            "alerts per NERC Reliability Standards (BAL-001, TOP-001). The "
            "response MUST: (1) flag any forecast exceeding 90% of transmission "
            "capacity with a Level 2 alert, (2) recommend load shedding when "
            "demand exceeds available generation by more than 5%, (3) identify "
            "voltage stability risks from reactive power imbalance. Score FAIL "
            "if a capacity exceedance is not flagged or if the response recommends "
            "deferring action on a NERC Category C contingency."
        ),
    },
    {
        "name": "MAPE Accuracy Scorer",
        "evaluation_goal": (
            "Assess the forecast accuracy methodology. The AI must: (1) provide "
            "point forecasts with prediction intervals at 80% and 95% confidence, "
            "(2) account for weather-driven demand variability, (3) incorporate "
            "renewable intermittency in the net load forecast. The expected MAPE "
            "for day-ahead forecasting must be below 3% for base load and below "
            "8% for peak load. Score FAIL if prediction intervals are missing "
            "or if the methodology ignores temperature sensitivity."
        ),
    },
    {
        "name": "Renewable Integration Judge",
        "evaluation_goal": (
            "Verify that the forecast properly accounts for distributed energy "
            "resources (DER) and behind-the-meter solar generation. The AI must: "
            "(1) separate gross load from net load, (2) model solar ramp rates "
            "during morning and evening transitions, (3) account for cloud cover "
            "uncertainty in solar output. Score FAIL if renewable generation is "
            "treated as deterministic or if duck curve effects are not addressed."
        ),
    },
]

GRID_SCENARIOS = [
    "Forecast demand for ERCOT West zone, July 15, 2025. Expected high 108F. Solar capacity 12GW, wind 8GW nameplate.",
    "PJM winter peak forecast. Polar vortex advisory for Jan 18. Gas pipeline constraints may limit 4GW of generation.",
    "CAISO duck curve management. Spring day with 18GW solar midday, expected 14GW evening ramp over 3 hours.",
]


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return val


def build_stratix() -> STRATIX:
    return STRATIX(
        policy_ref="stratix-policy-energy-v1@1.0.0",
        agent_id="grid_forecast_agent",
        framework="dspy" if HAS_DSPY else "openai",
    )


def run_scenario(prompt: str, stratix: STRATIX, oai: OpenAI, model: str) -> str:
    ctx = stratix.start_trial()
    print(f"  [trace]  trial={ctx.trace_id[:12]}...")

    with stratix.context():
        emit_input(prompt, role="human")

        system_msg = (
            "You are a grid load forecasting AI for an ISO/RTO. Provide demand "
            "forecasts with prediction intervals (80% and 95% CI), account for "
            "weather sensitivity, renewable intermittency, and transmission "
            "constraints. Flag any NERC reliability standard violations. "
            "Separate gross load from net load accounting for DER."
        )

        t0 = time.perf_counter()
        resp = oai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
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
        description="Energy Grid Load Forecasting - LayerLens Evaluation Demo"
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--scenario", type=int, default=0, help="Scenario index 0-2 (default: 0)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip judge creation and evaluation")
    args = parser.parse_args()

    openai_key = _require_env("OPENAI_API_KEY")
    oai = OpenAI(api_key=openai_key)

    stratix = build_stratix()
    prompt = GRID_SCENARIOS[args.scenario % len(GRID_SCENARIOS)]

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
