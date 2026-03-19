"""
OpenClaw Code Gate — Multi-Agent Code Pipeline

Simulates a Coder -> Reviewer -> Tester -> Judge pipeline with an eval quality
gate. A CodeQualityJudge scores the output across 5 dimensions: correctness,
readability, efficiency, test coverage, and security. If the score falls below
a configurable threshold the pipeline retries.

This demo simulates the OpenClaw Sandbox evaluation flow using STRATIX
instrumentation and direct LLM API calls.

Usage:
    python main.py --task "Write a Python function that validates email addresses"
    python main.py --task "Implement a thread-safe LRU cache" --threshold 7.5 --max-retries 2
"""

import argparse
import json
import os

import openai

from layerlens.instrument import STRATIX, emit_input, emit_output, emit_model_invoke, emit_handoff

# ---------------------------------------------------------------------------
# STRATIX initialization
# ---------------------------------------------------------------------------
stratix = STRATIX(
    policy_ref="openclaw-code-gate-v1@1.0.0",
    agent_id="code_pipeline",
    framework="openclaw",
    exporter="otel",
)

# ---------------------------------------------------------------------------
# Quality gate configuration
# ---------------------------------------------------------------------------
QUALITY_RUBRIC = {
    "dimensions": [
        {"name": "correctness", "weight": 0.30, "description": "Does the code produce correct results for expected inputs?"},
        {"name": "readability", "weight": 0.20, "description": "Is the code well-structured, named, and documented?"},
        {"name": "efficiency", "weight": 0.15, "description": "Appropriate algorithmic complexity and resource usage"},
        {"name": "test_coverage", "weight": 0.20, "description": "Do the tests cover edge cases and failure modes?"},
        {"name": "security", "weight": 0.15, "description": "Free from injection, overflow, and other vulnerabilities"},
    ],
    "scale": {"min": 1, "max": 10},
}

AGENT_CONFIGS = {
    "coder": {
        "system": "You are an expert Python developer. Write clean, production-quality code. Return ONLY the code with docstrings, no extra commentary.",
    },
    "reviewer": {
        "system": "You are a senior code reviewer. Review the code for bugs, style issues, and improvements. Provide a revised version incorporating your fixes. Return ONLY the improved code.",
    },
    "tester": {
        "system": "You are a QA engineer. Write comprehensive pytest tests for the given code. Cover happy paths, edge cases, and error handling. Return ONLY the test code.",
    },
    "judge": {
        "system": (
            "You are a CodeQualityJudge. Score the code and tests on these dimensions (1-10):\n"
            "{dimensions}\n\n"
            "Respond ONLY with valid JSON:\n"
            '{{\"correctness\": <int>, \"readability\": <int>, \"efficiency\": <int>, '
            '\"test_coverage\": <int>, \"security\": <int>, \"rationale\": \"<brief>\"}}'
        ),
    },
}


def call_llm(model: str, system: str, user: str) -> str:
    """Generic LLM call via OpenAI API."""
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model, max_tokens=2048,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content


def run_coder(task: str, model: str, feedback: str | None = None) -> str:
    """Coder agent generates code from the task description."""
    prompt = f"Task: {task}"
    if feedback:
        prompt += f"\n\nPrevious attempt feedback:\n{feedback}"
    code = call_llm(model, AGENT_CONFIGS["coder"]["system"], prompt)
    emit_model_invoke(provider="openai", name=model)
    return code


def run_reviewer(code: str, task: str, model: str) -> str:
    """Reviewer agent reviews and improves the code."""
    prompt = f"Original task: {task}\n\nCode to review:\n{code}"
    reviewed = call_llm(model, AGENT_CONFIGS["reviewer"]["system"], prompt)
    emit_model_invoke(provider="openai", name=model)
    return reviewed


def run_tester(code: str, task: str, model: str) -> str:
    """Tester agent writes tests for the code."""
    prompt = f"Task: {task}\n\nCode to test:\n{code}"
    tests = call_llm(model, AGENT_CONFIGS["tester"]["system"], prompt)
    emit_model_invoke(provider="openai", name=model)
    return tests


def run_judge(code: str, tests: str, task: str, model: str) -> dict:
    """Judge scores the code and tests against the quality rubric."""
    dim_desc = "\n".join(f"- {d['name']} (weight {d['weight']}): {d['description']}" for d in QUALITY_RUBRIC["dimensions"])
    system = AGENT_CONFIGS["judge"]["system"].format(dimensions=dim_desc)
    prompt = f"Task: {task}\n\nCode:\n{code}\n\nTests:\n{tests}"
    raw = call_llm(model, system, prompt)
    if raw.strip().startswith("```"):
        raw = raw.strip().split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def weighted_score(scores: dict) -> float:
    """Compute weighted aggregate from dimension scores."""
    total = 0.0
    for dim in QUALITY_RUBRIC["dimensions"]:
        total += scores.get(dim["name"], 0) * dim["weight"]
    return round(total, 2)


def print_pipeline_result(attempt: int, scores: dict, ws: float, threshold: float, passed: bool) -> None:
    """Print results for a single pipeline attempt."""
    status = "PASS" if passed else "FAIL"
    print(f"\n  --- Attempt {attempt} [{status}] (weighted={ws}, threshold={threshold}) ---")
    for dim in QUALITY_RUBRIC["dimensions"]:
        val = scores.get(dim["name"], "?")
        bar = "#" * int(val) if isinstance(val, (int, float)) else ""
        print(f"    {dim['name']:<16} {val:<4} {bar}")
    print(f"    rationale: {scores.get('rationale', 'N/A')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenClaw Code Gate — Multi-Agent Code Pipeline")
    parser.add_argument("--task", required=True, help="Coding task description")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model for all agents (default: gpt-4o-mini)")
    parser.add_argument("--threshold", type=float, default=7.0, help="Quality gate threshold (default: 7.0)")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retry attempts (default: 2)")
    args = parser.parse_args()

    ctx = stratix.start_trial()
    emit_input(args.task, role="human")

    print(f"[CodeGate] Task: {args.task}")
    print(f"[CodeGate] Model: {args.model} | Threshold: {args.threshold} | Max retries: {args.max_retries}")

    feedback = None
    for attempt in range(1, args.max_retries + 2):
        print(f"\n[CodeGate] Pipeline attempt {attempt}...")

        print("  [Coder] Generating code...")
        emit_handoff(source_agent="orchestrator", target_agent="coder")
        code = run_coder(args.task, args.model, feedback)

        print("  [Reviewer] Reviewing code...")
        emit_handoff(source_agent="coder", target_agent="reviewer")
        reviewed_code = run_reviewer(code, args.task, args.model)

        print("  [Tester] Writing tests...")
        emit_handoff(source_agent="reviewer", target_agent="tester")
        tests = run_tester(reviewed_code, args.task, args.model)

        print("  [Judge] Scoring quality...")
        emit_handoff(source_agent="tester", target_agent="judge")
        scores = run_judge(reviewed_code, tests, args.task, args.model)
        ws = weighted_score(scores)
        passed = ws >= args.threshold

        print_pipeline_result(attempt, scores, ws, args.threshold, passed)

        if passed:
            emit_output(f"Pipeline PASSED on attempt {attempt} with score {ws}")
            print(f"\n[CodeGate] QUALITY GATE: PASS (score {ws} >= {args.threshold})")
            break
        else:
            feedback = scores.get("rationale", "Improve code quality across all dimensions.")
            if attempt <= args.max_retries:
                print(f"  Retrying with feedback: {feedback}")
    else:
        emit_output(f"Pipeline FAILED after {args.max_retries + 1} attempts, best score {ws}")
        print(f"\n[CodeGate] QUALITY GATE: FAIL after {args.max_retries + 1} attempts")

    print(f"[CodeGate] STRATIX events captured: {len(stratix._event_buffer)}")


if __name__ == "__main__":
    main()
