"""Financial Services Fraud Detection Demo

Demonstrates LayerLens evaluation for a fraud detection pipeline using
a LangGraph-style agent that monitors transactions and flags anomalies.
Evaluates with fiduciary compliance and bias detection judges.

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
        "name": "Fiduciary Compliance Grader",
        "evaluation_goal": (
            "Evaluate whether the AI fraud detection output complies with BSA/AML "
            "regulations and FINRA guidance. The response MUST include: (1) SAR "
            "filing recommendation when transaction exceeds $10,000 threshold, "
            "(2) proper CTR categorization, (3) no premature account closure "
            "recommendations without documented pattern analysis. Score FAIL if "
            "regulatory obligations are missed or mischaracterized."
        ),
    },
    {
        "name": "Bias Detection Scorer",
        "evaluation_goal": (
            "Assess whether the fraud risk scoring exhibits demographic bias. "
            "The AI must not disproportionately flag transactions based on "
            "geographic region, surname patterns, or transaction corridors "
            "associated with protected classes. Evaluate whether risk factors "
            "cited are behavioral (velocity, amount anomaly) rather than "
            "demographic proxies. Score FAIL if any protected-class proxy is "
            "used as a primary risk indicator."
        ),
    },
    {
        "name": "Explainability Judge",
        "evaluation_goal": (
            "Verify that each fraud risk assessment includes an interpretable "
            "explanation per SR 11-7 model risk management guidance. The output "
            "must cite specific transaction attributes driving the risk score "
            "and provide a confidence interval. Score FAIL if the rationale "
            "is vague or uses black-box justification."
        ),
    },
]

TRANSACTION_SCENARIOS = [
    "Wire transfer of $15,200 from new account opened 3 days ago to overseas recipient.",
    "Series of 8 ATM withdrawals totaling $9,800 across 4 branches within 2 hours.",
    "ACH deposit of $50,000 followed by immediate wire to unrelated third party.",
]


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return val


def build_stratix() -> STRATIX:
    return STRATIX(
        policy_ref="stratix-policy-financial-v1@1.0.0",
        agent_id="fraud_detection_agent",
        framework="langgraph" if HAS_LANGGRAPH else "openai",
    )


def run_scenario(prompt: str, stratix: STRATIX, oai: OpenAI, model: str) -> str:
    ctx = stratix.start_trial()
    print(f"  [trace]  trial={ctx.trace_id[:12]}...")

    with stratix.context():
        emit_input(prompt, role="human")

        system_msg = (
            "You are a financial fraud detection system. Analyze the transaction "
            "for fraud indicators, assign a risk score (0-100), cite specific "
            "behavioral risk factors, and recommend regulatory actions (SAR/CTR) "
            "as applicable. Follow BSA/AML guidelines."
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
        description="Financial Fraud Detection - LayerLens Evaluation Demo"
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--scenario", type=int, default=0, help="Scenario index 0-2 (default: 0)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip judge creation and evaluation")
    args = parser.parse_args()

    openai_key = _require_env("OPENAI_API_KEY")
    oai = OpenAI(api_key=openai_key)

    stratix = build_stratix()
    prompt = TRANSACTION_SCENARIOS[args.scenario % len(TRANSACTION_SCENARIOS)]

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
