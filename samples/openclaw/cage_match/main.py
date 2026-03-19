"""
OpenClaw Cage Match — Live LLM Cage Match

Pits 3 models (Claude, GPT-4o, Gemini) against each other on the same prompt.
A ComparativeJudge scores each response across 4 dimensions: accuracy, clarity,
conciseness, and instruction-following. Results are displayed as a ranked leaderboard.

This demo simulates the OpenClaw Gateway evaluation flow using STRATIX
instrumentation and direct LLM API calls.

Usage:
    python main.py --prompt "Explain quantum entanglement to a 10-year-old"
    python main.py --prompt "Write a Python function to merge two sorted lists" --judge-model gpt-4o
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import openai

from layerlens.instrument import STRATIX, emit_input, emit_output, emit_model_invoke

# ---------------------------------------------------------------------------
# STRATIX initialization
# ---------------------------------------------------------------------------
stratix = STRATIX(
    policy_ref="openclaw-cage-match-v1@1.0.0",
    agent_id="cage_match_evaluator",
    framework="openclaw",
    exporter="otel",
)

# ---------------------------------------------------------------------------
# Judge configuration
# ---------------------------------------------------------------------------
JUDGE_RUBRIC = {
    "dimensions": [
        {"name": "accuracy", "weight": 0.30, "description": "Factual correctness and technical precision"},
        {"name": "clarity", "weight": 0.30, "description": "How easy the response is to understand"},
        {"name": "conciseness", "weight": 0.20, "description": "Economy of language without losing substance"},
        {"name": "instruction_following", "weight": 0.20, "description": "How well the response follows the prompt"},
    ],
    "scale": {"min": 1, "max": 10},
}

JUDGE_SYSTEM_PROMPT = """You are a ComparativeJudge for an LLM evaluation framework.
Score the following response on these dimensions (1-10 scale):
{dimensions}

Respond ONLY with valid JSON:
{{"accuracy": <int>, "clarity": <int>, "conciseness": <int>, "instruction_following": <int>, "rationale": "<brief explanation>"}}"""

# ---------------------------------------------------------------------------
# Contender models
# ---------------------------------------------------------------------------
CONTENDERS = [
    {"id": "claude-sonnet", "provider": "anthropic", "model": "claude-sonnet-4-20250514"},
    {"id": "gpt-4o", "provider": "openai", "model": "gpt-4o"},
    {"id": "gpt-4o-mini", "provider": "openai", "model": "gpt-4o-mini"},
]


def call_anthropic(model: str, prompt: str) -> str:
    """Send a prompt to an Anthropic model and return the response text."""
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def call_openai(model: str, prompt: str) -> str:
    """Send a prompt to an OpenAI model and return the response text."""
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def call_model(contender: dict, prompt: str) -> dict:
    """Dispatch a prompt to the correct provider and return timing + text."""
    t0 = time.time()
    if contender["provider"] == "anthropic":
        text = call_anthropic(contender["model"], prompt)
    else:
        text = call_openai(contender["model"], prompt)
    elapsed = time.time() - t0
    return {"id": contender["id"], "model": contender["model"], "text": text, "latency_s": round(elapsed, 2)}


def judge_response(response_text: str, prompt: str, judge_model: str) -> dict:
    """Use the judge model to score a single response against the rubric."""
    dim_desc = "\n".join(
        f"- {d['name']} (weight {d['weight']}): {d['description']}"
        for d in JUDGE_RUBRIC["dimensions"]
    )
    system = JUDGE_SYSTEM_PROMPT.format(dimensions=dim_desc)
    user_msg = f"Original prompt: {prompt}\n\nResponse to judge:\n{response_text}"

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=judge_model,
        max_tokens=512,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def weighted_score(scores: dict) -> float:
    """Compute weighted aggregate score from dimension scores."""
    total = 0.0
    for dim in JUDGE_RUBRIC["dimensions"]:
        total += scores.get(dim["name"], 0) * dim["weight"]
    return round(total, 2)


def print_leaderboard(results: list[dict]) -> None:
    """Print a formatted leaderboard to stdout."""
    results.sort(key=lambda r: r["weighted"], reverse=True)
    print("\n" + "=" * 72)
    print("  OPENCLAW CAGE MATCH — LEADERBOARD")
    print("=" * 72)
    for rank, r in enumerate(results, 1):
        medal = {1: "[1st]", 2: "[2nd]", 3: "[3rd]"}.get(rank, f"[{rank}th]")
        dims = " | ".join(f"{d['name'][:4]}={r['scores'].get(d['name'], '?')}" for d in JUDGE_RUBRIC["dimensions"])
        print(f"  {medal} {r['id']:<18} SCORE={r['weighted']:<6}  ({dims})  latency={r['latency_s']}s")
    print("-" * 72)
    for r in results:
        print(f"\n  [{r['id']}] rationale: {r['scores'].get('rationale', 'N/A')}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenClaw Cage Match — Live LLM Cage Match")
    parser.add_argument("--prompt", required=True, help="The prompt to send to all contenders")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="Model used for judging (default: gpt-4o-mini)")
    args = parser.parse_args()

    ctx = stratix.start_trial()
    emit_input(args.prompt, role="human")

    print(f"[CageMatch] Prompt: {args.prompt}")
    print(f"[CageMatch] Judge model: {args.judge_model}")
    print(f"[CageMatch] Contenders: {', '.join(c['id'] for c in CONTENDERS)}")
    print("[CageMatch] Calling models...")

    # --- Phase 1: Gather responses (parallel) ---
    responses = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(call_model, c, args.prompt): c for c in CONTENDERS}
        for fut in as_completed(futures):
            result = fut.result()
            emit_model_invoke(provider="openai", name=result["model"])
            responses.append(result)
            print(f"  -> {result['id']} responded in {result['latency_s']}s")

    # --- Phase 2: Judge each response ---
    print("[CageMatch] Judging responses...")
    results = []
    for resp in responses:
        scores = judge_response(resp["text"], args.prompt, args.judge_model)
        ws = weighted_score(scores)
        results.append({**resp, "scores": scores, "weighted": ws})

    # --- Phase 3: Emit and display ---
    winner = max(results, key=lambda r: r["weighted"])
    emit_output(f"Winner: {winner['id']} (score={winner['weighted']})")

    print_leaderboard(results)
    print(f"\n[CageMatch] STRATIX events captured: {len(stratix._event_buffer)}")


if __name__ == "__main__":
    main()
