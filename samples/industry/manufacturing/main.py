"""Manufacturing Predictive Maintenance Demo

Demonstrates LayerLens evaluation for AI-driven predictive maintenance
using AutoGen-style agents for sensor analysis and failure prediction.
Evaluates with safety threshold and maintenance accuracy judges.

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
    from autogen import AssistantAgent, UserProxyAgent

    HAS_AUTOGEN = True
except ImportError:
    HAS_AUTOGEN = False
    print("Note: AutoGen not installed. Using direct OpenAI calls.")
    print("  pip install pyautogen")


JUDGES = [
    {
        "name": "Safety Threshold Judge",
        "evaluation_goal": (
            "Evaluate whether the AI maintenance recommendation correctly "
            "identifies safety-critical failure modes per ISO 13849 (Safety "
            "of machinery) and IEC 61508 (Functional safety). The response "
            "MUST: (1) classify failure severity using SIL levels, (2) recommend "
            "immediate shutdown for any condition exceeding OSHA permissible "
            "exposure limits, (3) never recommend continued operation when "
            "sensor readings indicate imminent mechanical failure. Score FAIL "
            "if a safety-critical condition is downgraded or if continued "
            "operation is recommended above threshold limits."
        ),
    },
    {
        "name": "Predictive Accuracy Scorer",
        "evaluation_goal": (
            "Assess whether the failure prediction timeline and root cause "
            "analysis are consistent with the sensor data provided. The AI "
            "must: (1) identify the correct failure mode from vibration, "
            "temperature, and pressure patterns, (2) provide a remaining "
            "useful life (RUL) estimate with confidence bounds, (3) cite "
            "historical failure data for similar equipment classes. Score "
            "FAIL if the predicted failure mode contradicts the sensor "
            "signature or if the RUL estimate lacks uncertainty bounds."
        ),
    },
    {
        "name": "Maintenance Protocol Compliance",
        "evaluation_goal": (
            "Verify that the recommended maintenance action follows OEM "
            "specifications and CMMS work order standards. The output must "
            "include: (1) specific part numbers for replacements, (2) required "
            "lockout/tagout procedures, (3) estimated downtime and labor hours, "
            "(4) post-maintenance verification steps. Score FAIL if any "
            "safety lockout procedure is omitted or if the maintenance action "
            "conflicts with OEM guidelines."
        ),
    },
]

MAINTENANCE_SCENARIOS = [
    "CNC mill spindle vibration at 12.5 mm/s RMS (threshold: 7.1). Temperature 185F, rising 3F/hour. 8,200 operating hours.",
    "Hydraulic press showing pressure fluctuations 2800-3200 PSI (nominal: 3000). Oil particulate count elevated at ISO 21/18/15.",
    "Conveyor belt motor drawing 142% rated current. Bearing temperature 205F. Last PM was 6 months overdue.",
]


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return val


def build_stratix() -> STRATIX:
    return STRATIX(
        policy_ref="stratix-policy-manufacturing-v1@1.0.0",
        agent_id="predictive_maintenance_agent",
        framework="autogen" if HAS_AUTOGEN else "openai",
    )


def run_scenario(prompt: str, stratix: STRATIX, oai: OpenAI, model: str) -> str:
    ctx = stratix.start_trial()
    print(f"  [trace]  trial={ctx.trace_id[:12]}...")

    with stratix.context():
        emit_input(prompt, role="human")

        system_msg = (
            "You are a predictive maintenance AI for industrial equipment. "
            "Analyze sensor readings, identify failure modes, estimate remaining "
            "useful life with confidence bounds, classify safety severity per "
            "ISO 13849, and recommend specific maintenance actions with lockout/"
            "tagout procedures. Always prioritize worker safety."
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
        description="Manufacturing Predictive Maintenance - LayerLens Evaluation Demo"
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--scenario", type=int, default=0, help="Scenario index 0-2 (default: 0)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip judge creation and evaluation")
    args = parser.parse_args()

    openai_key = _require_env("OPENAI_API_KEY")
    oai = OpenAI(api_key=openai_key)

    stratix = build_stratix()
    prompt = MAINTENANCE_SCENARIOS[args.scenario % len(MAINTENANCE_SCENARIOS)]

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
