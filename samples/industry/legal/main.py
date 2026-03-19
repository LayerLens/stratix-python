"""Legal Contract Review Demo

Demonstrates LayerLens evaluation for AI-assisted contract review using
LlamaIndex-style document retrieval for clause identification and citation
verification. Evaluates with citation accuracy and clause extraction judges.

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
    from llama_index.core import VectorStoreIndex, Document

    HAS_LLAMAINDEX = True
except ImportError:
    HAS_LLAMAINDEX = False
    print("Note: LlamaIndex not installed. Using direct OpenAI calls.")
    print("  pip install llama-index-core llama-index-llms-openai")


JUDGES = [
    {
        "name": "Citation Verification Judge",
        "evaluation_goal": (
            "Verify that every legal citation in the AI response refers to a "
            "real statute, regulation, or case. Cross-reference cited section "
            "numbers against the source contract text. Score FAIL if: (1) any "
            "citation is fabricated or hallucinated, (2) a cited clause number "
            "does not exist in the referenced document, or (3) the cited text "
            "is materially misquoted. Per ABA Model Rule 1.1, AI-assisted "
            "legal work must meet competence standards."
        ),
    },
    {
        "name": "Clause Extraction Accuracy",
        "evaluation_goal": (
            "Evaluate whether the AI correctly identifies and categorizes "
            "contract clauses by type (indemnification, limitation of liability, "
            "termination, force majeure, non-compete, IP assignment). The "
            "extraction must capture the complete clause text without truncation "
            "and assign the correct legal category. Score FAIL if any high-risk "
            "clause (indemnification, liability cap, termination for cause) is "
            "missed or miscategorized."
        ),
    },
    {
        "name": "Risk Assessment Judge",
        "evaluation_goal": (
            "Assess whether the AI contract risk analysis correctly identifies "
            "unfavorable terms, missing standard protections, and one-sided "
            "provisions. The analysis must flag: unlimited liability exposure, "
            "auto-renewal traps, broad IP assignment, and unilateral amendment "
            "rights. Score FAIL if a material risk is omitted or a low-risk "
            "clause is incorrectly flagged as high-risk."
        ),
    },
]

CONTRACT_SCENARIOS = [
    "Review SaaS subscription agreement Section 8 (Limitation of Liability) for uncapped indemnification exposure.",
    "Analyze employment contract non-compete clause: 2-year restriction, 500-mile radius, all competing industries.",
    "Review vendor MSA termination provisions. Contract allows vendor to terminate for convenience with 10-day notice.",
]


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return val


def build_stratix() -> STRATIX:
    return STRATIX(
        policy_ref="stratix-policy-legal-v1@1.0.0",
        agent_id="contract_review_agent",
        framework="llamaindex" if HAS_LLAMAINDEX else "openai",
    )


def run_scenario(prompt: str, stratix: STRATIX, oai: OpenAI, model: str) -> str:
    ctx = stratix.start_trial()
    print(f"  [trace]  trial={ctx.trace_id[:12]}...")

    with stratix.context():
        emit_input(prompt, role="human")

        system_msg = (
            "You are a legal contract review AI. Analyze the contract provision, "
            "identify clause types, assess risk level, cite specific section numbers, "
            "and recommend negotiation points. All citations must reference actual "
            "contract language. Flag any provisions that deviate from market standard."
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
        description="Legal Contract Review - LayerLens Evaluation Demo"
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--scenario", type=int, default=0, help="Scenario index 0-2 (default: 0)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip judge creation and evaluation")
    args = parser.parse_args()

    openai_key = _require_env("OPENAI_API_KEY")
    oai = OpenAI(api_key=openai_key)

    stratix = build_stratix()
    prompt = CONTRACT_SCENARIOS[args.scenario % len(CONTRACT_SCENARIOS)]

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
