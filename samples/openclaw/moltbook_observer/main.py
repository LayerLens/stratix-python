"""
OpenClaw Moltbook Observer — Population Quality Audit

Simulates observing 20 autonomous agent outputs and scoring them on 4 dimensions:
reasoning coherence, factual plausibility, task focus, and originality. Displays
histogram-style percentile rankings and flags statistical outliers.

This demo simulates the OpenClaw Moltbook population-audit flow using STRATIX
instrumentation and direct LLM API calls.

Usage:
    python main.py
    python main.py --population-size 10 --model gpt-4o --judge-model gpt-4o-mini
"""

import argparse
import json
import os
import statistics

import openai

from layerlens.instrument import STRATIX, emit_input, emit_output, emit_model_invoke

# ---------------------------------------------------------------------------
# STRATIX initialization
# ---------------------------------------------------------------------------
stratix = STRATIX(
    policy_ref="openclaw-moltbook-observer-v1@1.0.0",
    agent_id="moltbook_observer",
    framework="openclaw",
    exporter="otel",
)

# ---------------------------------------------------------------------------
# Population task prompts (cycled through for the population)
# ---------------------------------------------------------------------------
TASK_POOL = [
    "Explain the trade-offs between microservices and monolithic architectures.",
    "What are the implications of quantum computing for modern cryptography?",
    "Describe three strategies for reducing latency in distributed systems.",
    "Compare eventual consistency and strong consistency in database design.",
    "Explain how transformer attention mechanisms work to a software engineer.",
    "What are the ethical concerns of using LLMs in automated hiring?",
    "Describe the CAP theorem and give a real-world example of each trade-off.",
    "How does garbage collection differ between Go, Java, and Rust?",
    "Explain zero-knowledge proofs and their practical applications.",
    "What are the security risks of server-side request forgery (SSRF)?",
]

# ---------------------------------------------------------------------------
# Judge configuration
# ---------------------------------------------------------------------------
QUALITY_RUBRIC = {
    "dimensions": [
        {"name": "reasoning_coherence", "weight": 0.30, "description": "Logical flow and internal consistency of arguments"},
        {"name": "factual_plausibility", "weight": 0.30, "description": "Claims are factually plausible and technically sound"},
        {"name": "task_focus", "weight": 0.25, "description": "Response stays on topic and addresses the actual question"},
        {"name": "originality", "weight": 0.15, "description": "Provides non-generic insights or novel framing"},
    ],
    "scale": {"min": 1, "max": 10},
}

JUDGE_SYSTEM_PROMPT = """You are a PopulationQualityJudge auditing autonomous agent outputs at scale.
Score this response on these dimensions (1-10):
{dimensions}

Respond ONLY with valid JSON:
{{"reasoning_coherence": <int>, "factual_plausibility": <int>, "task_focus": <int>, "originality": <int>}}"""


def generate_agent_output(task: str, model: str, agent_idx: int) -> str:
    """Simulate an autonomous agent producing output for a task."""
    system = (
        f"You are autonomous agent #{agent_idx}. Answer the following question concisely "
        f"in 2-3 paragraphs. Be substantive and specific."
    )
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model, max_tokens=512,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": task}],
    )
    return resp.choices[0].message.content


def judge_output(task: str, output: str, judge_model: str) -> dict:
    """Score a single agent output against the quality rubric."""
    dim_desc = "\n".join(f"- {d['name']} (weight {d['weight']}): {d['description']}" for d in QUALITY_RUBRIC["dimensions"])
    system = JUDGE_SYSTEM_PROMPT.format(dimensions=dim_desc)
    user_msg = f"Task: {task}\n\nAgent output:\n{output}"
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=judge_model, max_tokens=256,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def weighted_score(scores: dict) -> float:
    """Compute weighted aggregate from dimension scores."""
    total = 0.0
    for dim in QUALITY_RUBRIC["dimensions"]:
        total += scores.get(dim["name"], 0) * dim["weight"]
    return round(total, 2)


def ascii_histogram(values: list[float], bins: int = 5) -> list[str]:
    """Generate a simple ASCII histogram."""
    if not values:
        return []
    mn, mx = min(values), max(values)
    rng = mx - mn if mx > mn else 1
    bin_width = rng / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - mn) / bin_width), bins - 1)
        counts[idx] += 1
    lines = []
    max_count = max(counts) if counts else 1
    for i, c in enumerate(counts):
        lo = round(mn + i * bin_width, 1)
        hi = round(mn + (i + 1) * bin_width, 1)
        bar = "#" * int(c / max_count * 30) if max_count > 0 else ""
        lines.append(f"    {lo:>5}-{hi:<5} | {bar} ({c})")
    return lines


def print_population_report(results: list[dict]) -> None:
    """Print formatted population quality report."""
    scores = [r["weighted"] for r in results]
    mean = statistics.mean(scores)
    stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    p25 = sorted(scores)[len(scores) // 4]
    p50 = statistics.median(scores)
    p75 = sorted(scores)[3 * len(scores) // 4]

    print("\n" + "=" * 72)
    print("  OPENCLAW MOLTBOOK OBSERVER — POPULATION QUALITY REPORT")
    print("=" * 72)
    print(f"  Population size: {len(results)}")
    print(f"  Mean: {mean:.2f} | Stdev: {stdev:.2f} | P25: {p25:.2f} | P50: {p50:.2f} | P75: {p75:.2f}")
    print("\n  Score Distribution:")
    for line in ascii_histogram(scores):
        print(line)

    # Flag outliers (below mean - 1.5*stdev)
    outlier_threshold = mean - 1.5 * stdev if stdev > 0 else mean - 1
    outliers = [r for r in results if r["weighted"] < outlier_threshold]
    print(f"\n  Outlier threshold: < {outlier_threshold:.2f}")
    if outliers:
        print(f"  FLAGGED OUTLIERS ({len(outliers)}):")
        for r in outliers:
            print(f"    [!!] Agent #{r['idx']:>2} score={r['weighted']:<5} task: {r['task'][:50]}...")
    else:
        print("  No outliers detected.")

    print("\n  Per-Agent Scores:")
    for r in sorted(results, key=lambda x: x["weighted"], reverse=True):
        flag = " [OUTLIER]" if r["weighted"] < outlier_threshold else ""
        dims = " | ".join(f"{d['name'][:4]}={r['scores'].get(d['name'], '?')}" for d in QUALITY_RUBRIC["dimensions"])
        print(f"    Agent #{r['idx']:>2}: {r['weighted']:<5} ({dims}){flag}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenClaw Moltbook Observer — Population Quality Audit")
    parser.add_argument("--population-size", type=int, default=20, help="Number of agent outputs to observe (default: 20)")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model for agent outputs (default: gpt-4o-mini)")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="Model for quality judging (default: gpt-4o-mini)")
    args = parser.parse_args()

    ctx = stratix.start_trial()

    print(f"[MoltbookObserver] Population: {args.population_size} agents")
    print(f"[MoltbookObserver] Model: {args.model} | Judge: {args.judge_model}")
    print(f"[MoltbookObserver] Generating and scoring outputs...\n")

    results = []
    for i in range(args.population_size):
        task = TASK_POOL[i % len(TASK_POOL)]
        emit_input(task, role="human")
        print(f"  Agent #{i+1:>2}/{args.population_size}: {task[:50]}...", end=" ", flush=True)

        output = generate_agent_output(task, args.model, i + 1)
        emit_model_invoke(provider="openai", name=args.model)

        scores = judge_output(task, output, args.judge_model)
        ws = weighted_score(scores)
        results.append({"idx": i + 1, "task": task, "output": output, "scores": scores, "weighted": ws})
        print(f"score={ws}")

    emit_output(f"Population audit complete: {len(results)} agents scored")
    print_population_report(results)
    print(f"\n[MoltbookObserver] STRATIX events captured: {len(stratix._event_buffer)}")


if __name__ == "__main__":
    main()
