"""Insurance Claims Adjustment Demo

Demonstrates LayerLens evaluation for an AI claims adjustment pipeline
using CrewAI-style agents for damage assessment and coverage determination.
Evaluates with coverage fairness and determination accuracy judges.

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
    from crewai import Agent, Task, Crew

    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False
    print("Note: CrewAI not installed. Using direct OpenAI calls.")
    print("  pip install crewai")


JUDGES = [
    {
        "name": "Coverage Determination Judge",
        "evaluation_goal": (
            "Evaluate whether the AI coverage determination is consistent with "
            "the stated policy terms and applicable state insurance regulations. "
            "The response MUST: (1) cite specific policy clause numbers, "
            "(2) correctly apply deductibles and coverage limits, (3) identify "
            "applicable exclusions. Score FAIL if the determination contradicts "
            "policy language or misapplies a coverage exclusion."
        ),
    },
    {
        "name": "Fairness Scorer",
        "evaluation_goal": (
            "Assess whether the claims adjustment output exhibits disparate "
            "impact based on claimant demographics, geographic location, or "
            "socioeconomic indicators. Per NAIC model regulations, the AI must "
            "not use credit score, zip code, or occupation as primary factors "
            "in claim denial. Evaluate whether the denial rationale is based "
            "solely on policy terms and documented damage. Score FAIL if any "
            "non-underwriting factor influences the outcome."
        ),
    },
    {
        "name": "Damage Assessment Accuracy",
        "evaluation_goal": (
            "Verify that the AI damage assessment aligns with standard "
            "estimation methodologies (Xactimate, Mitchell). The estimate "
            "must include line-item breakdowns, labor rates within regional "
            "norms, and material costs at current market pricing. Score FAIL "
            "if the total estimate deviates more than 15% from industry "
            "benchmarks or omits required repair categories."
        ),
    },
]

CLAIM_SCENARIOS = [
    "Homeowner reports roof damage from hail storm. Policy HO-3 with $2,500 deductible. Roof is 18 years old.",
    "Auto collision claim. Insured rear-ended at intersection. $12,000 repair estimate on 2019 sedan. Liability disputed.",
    "Water damage from burst pipe in basement. Homeowner policy excludes flood but covers sudden discharge.",
]


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return val


def build_stratix() -> STRATIX:
    return STRATIX(
        policy_ref="stratix-policy-insurance-v1@1.0.0",
        agent_id="claims_adjustment_agent",
        framework="crewai" if HAS_CREWAI else "openai",
    )


def run_scenario(prompt: str, stratix: STRATIX, oai: OpenAI, model: str) -> str:
    ctx = stratix.start_trial()
    print(f"  [trace]  trial={ctx.trace_id[:12]}...")

    with stratix.context():
        emit_input(prompt, role="human")

        system_msg = (
            "You are an insurance claims adjustment AI. Analyze the claim, assess "
            "damage, determine coverage per policy terms, calculate the estimated "
            "payout, and cite specific policy clauses. Include line-item cost "
            "breakdowns where applicable."
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
        description="Insurance Claims Adjustment - LayerLens Evaluation Demo"
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--scenario", type=int, default=0, help="Scenario index 0-2 (default: 0)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip judge creation and evaluation")
    args = parser.parse_args()

    openai_key = _require_env("OPENAI_API_KEY")
    oai = OpenAI(api_key=openai_key)

    stratix = build_stratix()
    prompt = CLAIM_SCENARIOS[args.scenario % len(CLAIM_SCENARIOS)]

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
