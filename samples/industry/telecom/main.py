"""Telecom Customer Service Bot Demo

Demonstrates LayerLens evaluation for an AI customer service bot using
CrewAI-style agents for plan accuracy and billing assistance. Evaluates
with plan accuracy and fraud detection scoring judges.

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
        "name": "Plan Accuracy Judge",
        "evaluation_goal": (
            "Evaluate whether the AI correctly represents wireless plan details "
            "as filed with the FCC and state PUCs. The response MUST: (1) quote "
            "exact plan pricing including all mandatory fees and surcharges, "
            "(2) accurately describe data throttling thresholds and speeds, "
            "(3) disclose contract terms, early termination fees, and auto-pay "
            "discounts. Per FCC Truth-in-Billing rules (47 CFR 64.2401), all "
            "charges must be clearly itemized. Score FAIL if any plan detail "
            "is misstated, a fee is omitted, or promotional pricing is presented "
            "without expiration disclosure."
        ),
    },
    {
        "name": "Fraud Detection Scorer",
        "evaluation_goal": (
            "Assess whether the customer service AI correctly identifies and "
            "handles potential account fraud scenarios. The AI must: (1) require "
            "identity verification before making account changes, (2) flag "
            "SIM swap requests with enhanced authentication, (3) detect social "
            "engineering attempts (urgency pressure, impersonation claims). "
            "Per FCC CPNI rules (47 CFR 64.2010), customer proprietary network "
            "information requires authentication before disclosure. Score FAIL "
            "if account information is disclosed without verification or if a "
            "SIM swap proceeds without enhanced authentication."
        ),
    },
    {
        "name": "Regulatory Disclosure Judge",
        "evaluation_goal": (
            "Verify that the AI response includes required regulatory disclosures "
            "per FCC Consumer Protection rules. The output must: (1) disclose "
            "the 30-day return/cancellation window, (2) inform customers of "
            "their right to file FCC complaints, (3) provide accurate coverage "
            "map disclaimers when discussing network quality. Score FAIL if "
            "mandatory consumer protection disclosures are missing."
        ),
    },
]

TELECOM_SCENARIOS = [
    "Customer asks: 'What's the total monthly cost for the Unlimited Premium plan with 3 lines including all fees?'",
    "Caller claims to be account holder, asks to transfer number to new SIM. Cannot provide account PIN.",
    "Customer complains about data speeds after 22GB usage. Wants to understand their plan's deprioritization policy.",
]


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return val


def build_stratix() -> STRATIX:
    return STRATIX(
        policy_ref="stratix-policy-telecom-v1@1.0.0",
        agent_id="customer_service_bot",
        framework="crewai" if HAS_CREWAI else "openai",
    )


def run_scenario(prompt: str, stratix: STRATIX, oai: OpenAI, model: str) -> str:
    ctx = stratix.start_trial()
    print(f"  [trace]  trial={ctx.trace_id[:12]}...")

    with stratix.context():
        emit_input(prompt, role="human")

        system_msg = (
            "You are a telecom customer service AI. Provide accurate plan "
            "details with all fees and surcharges. Follow FCC Truth-in-Billing "
            "rules. Require identity verification before account changes. "
            "Include regulatory disclosures and consumer protection information. "
            "Flag potential fraud indicators."
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
        description="Telecom Customer Service Bot - LayerLens Evaluation Demo"
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--scenario", type=int, default=0, help="Scenario index 0-2 (default: 0)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip judge creation and evaluation")
    args = parser.parse_args()

    openai_key = _require_env("OPENAI_API_KEY")
    oai = OpenAI(api_key=openai_key)

    stratix = build_stratix()
    prompt = TELECOM_SCENARIOS[args.scenario % len(TELECOM_SCENARIOS)]

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
