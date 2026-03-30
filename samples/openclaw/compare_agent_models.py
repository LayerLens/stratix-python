#!/usr/bin/env python3
"""
Compare Agent Models -- LayerLens + OpenClaw
=============================================
Compares different LLM backends for an OpenClaw agent by executing the
same tasks on each model, uploading all executions as traces, evaluating
with consistent judges, and printing a comparison table.

Workflow:
  1. Create OpenClaw agents with different model backends.
  2. Execute the same set of tasks on each agent.
  3. Upload all executions as LayerLens traces with model metadata.
  4. Evaluate all traces with the same judges.
  5. Print a comparison table showing which model performed best.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package openclaw
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python compare_agent_models.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import poll_evaluation_results, upload_trace_dict, create_judge

# ---------------------------------------------------------------------------
# Models to compare
# ---------------------------------------------------------------------------

MODELS = [
    "claude-sonnet-4-20250514",
    "gpt-5.3",
    "gemini-2.5-pro",
]

# ---------------------------------------------------------------------------
# Tasks to execute on each model
# ---------------------------------------------------------------------------

TASKS = [
    "Explain the difference between TCP and UDP in plain English.",
    "Write a Python function that checks if a string is a valid email address.",
    "Summarize the key principles of the Agile Manifesto.",
    "What are the pros and cons of microservices vs monolithic architecture?",
]

# ---------------------------------------------------------------------------
# Simulated outputs per model
# ---------------------------------------------------------------------------

SIMULATED_OUTPUTS: dict[str, list[dict[str, Any]]] = {
    "claude-sonnet-4-20250514": [
        {
            "result": (
                "TCP (Transmission Control Protocol) is like sending a registered letter -- "
                "it guarantees delivery, keeps things in order, and confirms receipt. "
                "UDP (User Datagram Protocol) is like shouting across a room -- faster, "
                "but no guarantee the message arrives or arrives in order. Use TCP for "
                "web pages and email; use UDP for video streaming and gaming where speed "
                "matters more than perfection."
            ),
            "duration_ms": 2100,
        },
        {
            "result": (
                "```python\nimport re\n\ndef is_valid_email(email: str) -> bool:\n"
                '    pattern = r\'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$\'\n'
                "    return bool(re.match(pattern, email))\n```\n\n"
                "This validates the general structure. For production use, consider "
                "the `email-validator` library which handles edge cases per RFC 5322."
            ),
            "duration_ms": 1800,
        },
        {
            "result": (
                "The Agile Manifesto has four core values:\n"
                "1. Individuals and interactions over processes and tools\n"
                "2. Working software over comprehensive documentation\n"
                "3. Customer collaboration over contract negotiation\n"
                "4. Responding to change over following a plan\n\n"
                "The right side still matters, but the left side matters more."
            ),
            "duration_ms": 1500,
        },
        {
            "result": (
                "Microservices pros: independent deployment, tech diversity, team autonomy, "
                "fault isolation. Cons: network complexity, distributed debugging, data "
                "consistency challenges, operational overhead.\n\n"
                "Monolith pros: simpler development, easier testing, straightforward "
                "deployment, lower latency. Cons: scaling limitations, tight coupling, "
                "slower CI/CD at scale.\n\nStart monolithic; extract services when needed."
            ),
            "duration_ms": 2400,
        },
    ],
    "gpt-5.3": [
        {
            "result": (
                "Think of TCP as a phone call -- you establish a connection, talk back "
                "and forth, and know if the other person heard you. UDP is like a postcard "
                "-- you send it and hope for the best. TCP is reliable but slower; UDP is "
                "fast but unreliable. Web browsing uses TCP; live video uses UDP."
            ),
            "duration_ms": 2800,
        },
        {
            "result": (
                "```python\ndef is_valid_email(email):\n"
                "    import re\n"
                '    return bool(re.fullmatch(r"[^@\\s]+@[^@\\s]+\\.[^@\\s]+", email))\n```\n\n'
                "This checks for basic email format. Note: true email validation "
                "requires sending a confirmation email."
            ),
            "duration_ms": 2200,
        },
        {
            "result": (
                "The Agile Manifesto prioritizes: people over processes, working "
                "software over docs, customer collaboration over contracts, and "
                "adaptability over rigid plans. It's about delivering value "
                "incrementally through iterative development cycles."
            ),
            "duration_ms": 1900,
        },
        {
            "result": (
                "Microservices: great for large teams, independent scaling, polyglot "
                "tech stacks. But they add complexity in networking, monitoring, and "
                "data management.\n\nMonoliths: simpler to build and deploy initially, "
                "but become harder to maintain as they grow. Best for small teams and "
                "early-stage products."
            ),
            "duration_ms": 2600,
        },
    ],
    "gemini-2.5-pro": [
        {
            "result": (
                "TCP and UDP are both internet protocols. TCP is connection-oriented "
                "and reliable -- it ensures all data packets arrive in order. UDP is "
                "connectionless and faster but doesn't guarantee delivery. TCP = "
                "downloading files. UDP = video calls."
            ),
            "duration_ms": 1900,
        },
        {
            "result": (
                "```python\nimport re\n\ndef validate_email(email: str) -> bool:\n"
                '    regex = r"^[\\w.+-]+@[\\w-]+\\.[\\w.]+$"\n'
                "    return re.match(regex, email) is not None\n\n"
                "# Examples:\n"
                '# validate_email("user@example.com")  -> True\n'
                '# validate_email("invalid@")          -> False\n```'
            ),
            "duration_ms": 2000,
        },
        {
            "result": (
                "Agile Manifesto summary: Value individuals over processes, working "
                "software over documentation, customer collaboration over contracts, "
                "and responding to change over following plans. These four values guide "
                "iterative, people-centric software development."
            ),
            "duration_ms": 1600,
        },
        {
            "result": (
                "Microservices advantages: scalability, resilience, flexibility. "
                "Disadvantages: complexity, latency, testing difficulty.\n\n"
                "Monolith advantages: simplicity, performance, easy debugging. "
                "Disadvantages: scaling bottlenecks, deployment coupling.\n\n"
                "Choose based on team size and project maturity."
            ),
            "duration_ms": 2100,
        },
    ],
}


def _execute_tasks_for_model(model: str) -> list[dict[str, Any]]:
    """Execute tasks for a specific model via OpenClaw, with fallback."""
    try:
        from openclaw import OpenClawClient  # type: ignore[import-untyped]

        oc_client = OpenClawClient()
        agent = oc_client.agents.create(
            name=f"compare-{model}",
            model=model,
            description=f"Comparison agent using {model}.",
        )
        results = []
        for task in TASKS:
            start = time.monotonic()
            result = agent.execute(task)
            duration_ms = round((time.monotonic() - start) * 1000)
            results.append({
                "task": task,
                "result": str(result),
                "duration_ms": duration_ms,
            })
        return results
    except ImportError:
        return [
            {"task": TASKS[i], **SIMULATED_OUTPUTS[model][i]}
            for i in range(len(TASKS))
        ]
    except Exception:
        return [
            {"task": TASKS[i], **SIMULATED_OUTPUTS[model][i]}
            for i in range(len(TASKS))
        ]


# Judge definitions
JUDGE_DEFINITIONS = [
    (
        "Accuracy",
        "Evaluate whether the response is factually correct with no errors.",
    ),
    (
        "Clarity",
        "Evaluate whether the response is clearly written, well-structured, "
        "and easy to understand for the target audience.",
    ),
    (
        "Completeness",
        "Evaluate whether the response thoroughly addresses the question "
        "without omitting important aspects.",
    ),
]


def _ensure_judges(client: Stratix) -> list[tuple[str, str]]:
    """Create or find judges. Returns list of (judge_id, label)."""
    judge_pairs: list[tuple[str, str]] = []
    existing_resp = client.judges.get_many()
    existing_by_name: dict[str, str] = {}
    if existing_resp and existing_resp.judges:
        for j in existing_resp.judges:
            existing_by_name[j.name.lower()] = j.id

    for name, goal in JUDGE_DEFINITIONS:
        existing_id = existing_by_name.get(name.lower())
        if existing_id:
            judge_pairs.append((existing_id, name))
        else:
            judge = create_judge(client, name=name, evaluation_goal=goal)
            if judge:
                judge_pairs.append((judge.id, judge.name))
            else:
                print(f"  WARNING: Failed to create judge '{name}'")
    return judge_pairs


def main() -> None:
    """Run the model comparison demo."""
    print("=== LayerLens + OpenClaw: Compare Agent Models ===\n")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Tasks:  {len(TASKS)}\n")

    # --- 1. Initialize LayerLens ---
    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # --- 2. Execute tasks for each model ---
    # model_traces[model] = list of trace_ids
    model_traces: dict[str, list[str]] = {}

    openclaw_available = True
    try:
        from openclaw import OpenClawClient  # type: ignore[import-untyped]
    except ImportError:
        openclaw_available = False
        print("(openclaw not installed -- using simulated execution data)")
        print("  Install with: pip install openclaw\n")

    for model in MODELS:
        print(f"Running tasks with {model}...")
        executions = _execute_tasks_for_model(model)
        trace_ids: list[str] = []
        for ex in executions:
            trace_result = upload_trace_dict(
                client,
                input_text=ex["task"],
                output_text=ex["result"],
                metadata={
                    "source": "openclaw",
                    "model": model,
                    "duration_ms": ex["duration_ms"],
                },
            )
            if not trace_result or not trace_result.trace_ids:
                print(f"  WARNING: Trace upload returned no IDs")
                continue
            trace_ids.append(trace_result.trace_ids[0])
        model_traces[model] = trace_ids
        print(f"  Uploaded {len(trace_ids)} trace(s)")

    # --- 3. Create judges ---
    print()
    judge_pairs = _ensure_judges(client)
    print(f"Judges: {', '.join(label for _, label in judge_pairs)}\n")

    # --- 4. Evaluate all traces ---
    # scores[model][judge_label] = list of scores
    scores: dict[str, dict[str, list[float]]] = {
        model: {label: [] for _, label in judge_pairs}
        for model in MODELS
    }
    pass_counts: dict[str, dict[str, int]] = {
        model: {label: 0 for _, label in judge_pairs}
        for model in MODELS
    }

    for model in MODELS:
        print(f"Evaluating {model}...")
        for trace_id in model_traces[model]:
            for judge_id, label in judge_pairs:
                evaluation = client.trace_evaluations.create(
                    trace_id=trace_id, judge_id=judge_id,
                )
                results = poll_evaluation_results(client, evaluation.id)
                if results:
                    r = results[0]
                    scores[model][label].append(r.score if r.score is not None else 0.0)
                    if r.passed:
                        pass_counts[model][label] += 1

    # --- 5. Print comparison table ---
    print("\n" + "=" * 76)
    print("MODEL COMPARISON RESULTS")
    print("=" * 76)

    # Header
    header = f"\n{'Model':<30}"
    for _, label in judge_pairs:
        header += f" {label:>12}"
    header += f" {'Avg Score':>12}"
    print(header)
    print("-" * (30 + 13 * (len(judge_pairs) + 1)))

    # Rows
    model_averages: dict[str, float] = {}
    for model in MODELS:
        row = f"{model:<30}"
        all_scores: list[float] = []
        for _, label in judge_pairs:
            s = scores[model][label]
            avg = sum(s) / len(s) if s else 0.0
            total = len(s)
            passed = pass_counts[model][label]
            row += f" {avg:>5.2f} {passed}/{total:<4}"
            all_scores.extend(s)
        overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
        model_averages[model] = overall
        row += f" {overall:>12.2f}"
        print(row)

    # Winner
    if model_averages:
        best_model = max(model_averages, key=model_averages.get)  # type: ignore[arg-type]
        print(f"\nBest overall: \033[92m{best_model}\033[0m "
              f"(avg score: {model_averages[best_model]:.2f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
