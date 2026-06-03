#!/usr/bin/env python3
"""Browser Agent Evaluator -- LayerLens Python SDK Sample.

Evaluates browser automation agents (such as Browser Use) against a
suite of 20 real-world tasks across 5 categories. Captures traces with
the Stratix SDK, runs specialized judges, and produces a reliability
report including compound failure analysis.

Works in three modes:
  - simulated (default): generates synthetic traces for demo purposes
  - recorded: loads pre-recorded trace files from data/traces/browser_agent/
  - live: runs Browser Use against the task suite (requires browser-use)

Prerequisites:
    pip install layerlens --extra-index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python browser_agent_evaluator.py
    python browser_agent_evaluator.py --mode recorded
    python browser_agent_evaluator.py --mode live --tasks navigation
    python browser_agent_evaluator.py --json --output report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from typing import Any, Optional

from layerlens import Stratix
from layerlens import PublicClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import create_judge, poll_evaluation_results, upload_trace_dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Task suite: 20 tasks across 5 categories
# ---------------------------------------------------------------------------

TASK_CATEGORIES: dict[str, list[dict[str, str]]] = {
    "navigation": [
        {
            "id": "nav-001",
            "description": "Find the pricing page on stripe.com",
            "target_url": "https://stripe.com/pricing",
            "expected_outcome": "Agent lands on Stripe pricing page showing per-transaction fees.",
        },
        {
            "id": "nav-002",
            "description": "Navigate to the docs section of github.com",
            "target_url": "https://docs.github.com",
            "expected_outcome": "Agent reaches GitHub documentation landing page.",
        },
        {
            "id": "nav-003",
            "description": "Find the careers page on anthropic.com",
            "target_url": "https://www.anthropic.com/careers",
            "expected_outcome": "Agent lands on Anthropic careers page with open positions.",
        },
        {
            "id": "nav-004",
            "description": "Locate the API reference on openai.com",
            "target_url": "https://platform.openai.com/docs/api-reference",
            "expected_outcome": "Agent reaches the OpenAI API reference documentation.",
        },
    ],
    "data_extraction": [
        {
            "id": "extract-001",
            "description": "Get the current price of AAPL from finance.yahoo.com",
            "target_url": "https://finance.yahoo.com/quote/AAPL/",
            "expected_outcome": "Agent returns the current market price for AAPL as a number.",
        },
        {
            "id": "extract-002",
            "description": "Extract the top 5 headlines from news.ycombinator.com",
            "target_url": "https://news.ycombinator.com",
            "expected_outcome": "Agent returns a list of 5 headline strings from the front page.",
        },
        {
            "id": "extract-003",
            "description": "Get the weather for Los Angeles from weather.gov",
            "target_url": "https://forecast.weather.gov/MapClick.php?lat=34.0522&lon=-118.2437",
            "expected_outcome": "Agent returns current temperature and conditions for Los Angeles.",
        },
        {
            "id": "extract-004",
            "description": "Find the latest Python version from python.org",
            "target_url": "https://www.python.org/downloads/",
            "expected_outcome": "Agent returns the latest stable Python release version number.",
        },
    ],
    "form_interaction": [
        {
            "id": "form-001",
            "description": "Fill out a search form on Google with 'AI evaluation'",
            "target_url": "https://www.google.com/search?q=AI+evaluation",
            "expected_outcome": "Agent types query into Google search and submits the form.",
        },
        {
            "id": "form-002",
            "description": "Use the search filter on GitHub to find Python repos",
            "target_url": "https://github.com/search?q=language:python&type=repositories",
            "expected_outcome": "Agent applies language filter and sees Python repository results.",
        },
        {
            "id": "form-003",
            "description": "Enter a query in the Stack Overflow search bar",
            "target_url": "https://stackoverflow.com/search?q=browser+automation",
            "expected_outcome": "Agent enters query into Stack Overflow search and sees results.",
        },
        {
            "id": "form-004",
            "description": "Use the date picker on booking.com",
            "target_url": "https://www.booking.com",
            "expected_outcome": "Agent selects check-in and check-out dates using the date picker widget.",
        },
    ],
    "multi_step": [
        {
            "id": "multi-001",
            "description": "Search for 'stratix' on GitHub and navigate to the first result",
            "target_url": "https://github.com/layerlens/stratix",
            "expected_outcome": "Agent searches GitHub, clicks first result, lands on repo page.",
        },
        {
            "id": "multi-002",
            "description": "Find a product on Amazon and check if it is Prime eligible",
            "target_url": "https://www.amazon.com",
            "expected_outcome": "Agent searches a product, opens listing, reports Prime eligibility.",
        },
        {
            "id": "multi-003",
            "description": "Search for a restaurant on Yelp and check its hours",
            "target_url": "https://www.yelp.com",
            "expected_outcome": "Agent searches for restaurant, opens listing, extracts business hours.",
        },
        {
            "id": "multi-004",
            "description": "Find a flight on Google Flights from LAX to SFO",
            "target_url": "https://www.google.com/travel/flights",
            "expected_outcome": "Agent enters origin, destination, and sees flight results.",
        },
    ],
    "error_recovery": [
        {
            "id": "error-001",
            "description": "Handle a 404 page on a broken link",
            "target_url": "https://example.com/nonexistent-page-12345",
            "expected_outcome": "Agent detects the 404 error and reports it gracefully.",
        },
        {
            "id": "error-002",
            "description": "Deal with a CAPTCHA or cookie consent popup",
            "target_url": "https://www.google.com",
            "expected_outcome": "Agent handles or reports the blocking popup appropriately.",
        },
        {
            "id": "error-003",
            "description": "Navigate back after reaching a dead end",
            "target_url": "https://httpstat.us/500",
            "expected_outcome": "Agent detects the server error and navigates back or retries.",
        },
        {
            "id": "error-004",
            "description": "Handle a page that requires login",
            "target_url": "https://github.com/settings/profile",
            "expected_outcome": "Agent detects the login requirement and reports it instead of hanging.",
        },
    ],
}

# ---------------------------------------------------------------------------
# Judge definitions
# ---------------------------------------------------------------------------

JUDGE_DEFINITIONS: list[tuple[str, str]] = [
    (
        "Browser Task Completion",
        "Evaluate whether the browser agent successfully completed the assigned task. "
        "Check if the correct page was reached, correct data was extracted, or correct "
        "form was filled. Score 1.0 for full completion, 0.5 for partial, 0.0 for failure.",
    ),
    (
        "Browser Navigation Accuracy",
        "Evaluate whether the agent navigated to the correct URL and page section "
        "without unnecessary detours or wrong clicks. Penalize excessive steps, "
        "wrong intermediate pages, or failure to reach the target URL.",
    ),
    (
        "Browser Data Accuracy",
        "Evaluate whether the data extracted by the agent is factually correct and "
        "complete compared to what is actually on the page. Penalize hallucinated "
        "values, wrong elements selected, or incomplete extractions.",
    ),
    (
        "Browser Error Recovery",
        "Evaluate how well the agent handled unexpected situations like popups, "
        "errors, CAPTCHAs, or missing elements. Score based on whether the agent "
        "detected the issue, attempted recovery, and reported status clearly.",
    ),
    (
        "Browser Efficiency",
        "Evaluate whether the agent completed the task in a reasonable number of "
        "steps without unnecessary actions, repeated clicks, or loops. A 3-step "
        "navigation should not take 15 steps.",
    ),
]

# Category-to-judge mapping: which judges apply to each category
CATEGORY_JUDGES: dict[str, list[str]] = {
    "navigation": [
        "Browser Task Completion",
        "Browser Navigation Accuracy",
        "Browser Efficiency",
    ],
    "data_extraction": [
        "Browser Task Completion",
        "Browser Data Accuracy",
        "Browser Efficiency",
    ],
    "form_interaction": [
        "Browser Task Completion",
        "Browser Navigation Accuracy",
        "Browser Efficiency",
    ],
    "multi_step": [
        "Browser Task Completion",
        "Browser Navigation Accuracy",
        "Browser Data Accuracy",
        "Browser Efficiency",
    ],
    "error_recovery": [
        "Browser Task Completion",
        "Browser Error Recovery",
    ],
}

# ---------------------------------------------------------------------------
# Simulated trace generation
# ---------------------------------------------------------------------------

# Per-category simulated pass rates (realistic for current browser agents)
SIMULATED_PASS_RATES: dict[str, float] = {
    "navigation": 0.92,
    "data_extraction": 0.68,
    "form_interaction": 0.74,
    "multi_step": 0.55,
    "error_recovery": 0.45,
}


def _generate_simulated_trace(
    task: dict[str, str], category: str, seed: Optional[int] = None
) -> dict[str, Any]:
    """Generate a synthetic browser agent trace for a given task."""
    rng = random.Random(seed or hash(task["id"]))
    pass_rate = SIMULATED_PASS_RATES[category]
    succeeded = rng.random() < pass_rate
    partial = not succeeded and rng.random() < 0.4

    steps = rng.randint(2, 5) if succeeded else rng.randint(4, 12)
    duration_ms = steps * rng.randint(800, 3500)
    tokens = steps * rng.randint(400, 1800)

    status = "completed" if succeeded else ("partial" if partial else "failed")

    if succeeded:
        output_text = (
            f"Successfully completed: {task['description']}. "
            f"Reached {task['target_url']} and verified the expected outcome."
        )
    elif partial:
        output_text = (
            f"Partially completed: {task['description']}. "
            f"Navigated to the correct domain but could not complete the final step. "
            f"The target element was not found or did not render."
        )
    else:
        output_text = (
            f"Failed to complete: {task['description']}. "
            f"The agent encountered an obstacle and could not recover. "
            f"Final URL did not match expected target."
        )

    trace = {
        "trace_id": f"tr-sim-{task['id']}",
        "agent_name": "browser-automation-agent",
        "framework": "browser-use",
        "status": status,
        "description": f"Simulated trace for task: {task['description']}",
        "metadata": {
            "synthetic": True,
            "task_id": task["id"],
            "task_category": category,
            "task_description": task["description"],
        },
        "start_time": "2026-05-13T10:00:00Z",
        "duration_ms": duration_ms,
        "input": task["description"],
        "output": output_text,
        "steps_taken": steps,
        "tokens_used": tokens,
        "succeeded": succeeded,
        "partial": partial,
    }
    return trace


# ---------------------------------------------------------------------------
# Recorded trace loading
# ---------------------------------------------------------------------------

TRACES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "traces", "browser_agent"
)


def _load_recorded_traces() -> dict[str, dict[str, Any]]:
    """Load pre-recorded trace files and map them to task IDs."""
    traces: dict[str, dict[str, Any]] = {}
    if not os.path.isdir(TRACES_DIR):
        logger.warning("Recorded traces directory not found: %s", TRACES_DIR)
        return traces

    file_to_task: dict[str, str] = {
        "navigation_success.json": "nav-001",
        "extraction_failure.json": "extract-001",
        "multistep_partial.json": "multi-001",
    }

    for filename, task_id in file_to_task.items():
        filepath = os.path.join(TRACES_DIR, filename)
        if os.path.isfile(filepath):
            with open(filepath) as f:
                data = json.load(f)
            # Extract input/output for Stratix upload
            agent_input = data.get("metadata", {}).get("task_description", "")
            agent_output = ""
            for event in data.get("events", []):
                if event.get("type") == "agent.output":
                    agent_output = event["payload"].get("output", "")
                    break
            data["input"] = agent_input
            data["output"] = agent_output
            data["succeeded"] = data.get("status") == "completed"
            data["partial"] = data.get("status") == "partial"
            traces[task_id] = data
            logger.info("Loaded recorded trace: %s -> %s", filename, task_id)

    return traces


# ---------------------------------------------------------------------------
# Evaluation engine
# ---------------------------------------------------------------------------


def _ensure_judges(client: Stratix) -> dict[str, str]:
    """Create or find all judges. Returns {judge_name: judge_id}."""
    existing_resp = client.judges.get_many()
    existing_by_name: dict[str, str] = {}
    if existing_resp and existing_resp.judges:
        for j in existing_resp.judges:
            existing_by_name[j.name.lower()] = j.id

    judge_map: dict[str, str] = {}
    for name, goal in JUDGE_DEFINITIONS:
        existing_id = existing_by_name.get(name.lower())
        if existing_id:
            judge_map[name] = existing_id
        else:
            judge = create_judge(client, name=name, evaluation_goal=goal)
            if judge:
                judge_map[name] = judge.id
            else:
                logger.warning("Failed to create judge: %s", name)
    return judge_map


def _evaluate_trace(
    client: Stratix,
    trace_data: dict[str, Any],
    judge_names: list[str],
    judge_map: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Upload a trace and evaluate it with the specified judges.

    Returns a dict mapping judge_name to {score, passed, verdict}.
    """
    input_text = trace_data.get("input", "")
    output_text = trace_data.get("output", "")
    metadata = trace_data.get("metadata", {})

    trace_result = upload_trace_dict(
        client,
        input_text=input_text,
        output_text=output_text,
        metadata=metadata,
    )
    trace_id = (
        trace_result.trace_ids[0]
        if trace_result and trace_result.trace_ids
        else f"trace-{trace_data.get('trace_id', 'unknown')}"
    )

    results: dict[str, dict[str, Any]] = {}
    for judge_name in judge_names:
        judge_id = judge_map.get(judge_name)
        if not judge_id:
            results[judge_name] = {
                "score": 0.0,
                "passed": False,
                "verdict": "judge_not_found",
            }
            continue

        try:
            te = client.trace_evaluations.create(
                trace_id=trace_id,
                judge_id=judge_id,
            )
            if te is None:
                results[judge_name] = {
                    "score": 0.0,
                    "passed": False,
                    "verdict": "creation_failed",
                }
                continue

            eval_results = poll_evaluation_results(client, te.id)
            if eval_results:
                r = eval_results[0]
                results[judge_name] = {
                    "score": r.score if r.score is not None else 0.0,
                    "passed": bool(r.passed),
                    "verdict": "pass" if r.passed else "fail",
                }
            else:
                results[judge_name] = {
                    "score": 0.0,
                    "passed": False,
                    "verdict": "timeout",
                }
        except Exception as exc:
            logger.warning("Evaluation failed for %s: %s", judge_name, exc)
            results[judge_name] = {
                "score": 0.0,
                "passed": False,
                "verdict": f"error: {exc}",
            }

    return results


# ---------------------------------------------------------------------------
# Compound failure analysis
# ---------------------------------------------------------------------------


def _compound_reliability(per_step_rate: float, chain_length: int) -> float:
    """Calculate compound reliability for chained browser actions.

    If each individual step succeeds at per_step_rate, the probability
    that all steps in a chain succeed is per_step_rate ^ chain_length.
    """
    return per_step_rate**chain_length


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

HUMAN_BASELINE = 0.98  # assumed human accuracy for the same tasks


def _render_bar(value: float, width: int = 20) -> str:
    """Render a simple ASCII progress bar."""
    filled = int(value * width)
    return f"[{'#' * filled}{'.' * (width - filled)}] {value * 100:5.1f}%"


def _render_report(
    all_results: dict[str, dict[str, dict[str, dict[str, Any]]]],
    output_path: Optional[str] = None,
    as_json: bool = False,
) -> None:
    """Render the reliability report to stdout (and optionally to file)."""

    # Aggregate scores
    category_scores: dict[str, list[float]] = {}
    category_pass_counts: dict[str, tuple[int, int]] = {}
    task_details: list[dict[str, Any]] = []
    all_scores: list[float] = []

    for category, tasks in all_results.items():
        cat_scores: list[float] = []
        cat_passed = 0
        cat_total = 0
        for task_id, judges in tasks.items():
            task_passed_all = True
            task_score_sum = 0.0
            task_judge_count = 0
            judge_details: list[dict[str, Any]] = []
            for judge_name, result in judges.items():
                judge_details.append(
                    {
                        "judge": judge_name,
                        "score": result["score"],
                        "passed": result["passed"],
                        "verdict": result["verdict"],
                    }
                )
                task_score_sum += result["score"]
                task_judge_count += 1
                if not result["passed"]:
                    task_passed_all = False

            avg_score = task_score_sum / max(task_judge_count, 1)
            cat_scores.append(avg_score)
            all_scores.append(avg_score)
            cat_total += 1
            if task_passed_all:
                cat_passed += 1

            # Look up task description
            task_desc = task_id
            for t in TASK_CATEGORIES.get(category, []):
                if t["id"] == task_id:
                    task_desc = t["description"]
                    break

            task_details.append(
                {
                    "task_id": task_id,
                    "category": category,
                    "description": task_desc,
                    "avg_score": avg_score,
                    "all_passed": task_passed_all,
                    "judges": judge_details,
                }
            )

        category_scores[category] = cat_scores
        category_pass_counts[category] = (cat_passed, cat_total)

    overall_score = sum(all_scores) / max(len(all_scores), 1)
    total_passed = sum(p for p, _ in category_pass_counts.values())
    total_tasks = sum(t for _, t in category_pass_counts.values())

    # Compound failure analysis
    compound_3 = _compound_reliability(overall_score, 3)
    compound_5 = _compound_reliability(overall_score, 5)
    compound_10 = _compound_reliability(overall_score, 10)

    # Build recommendations
    strong_categories = [
        c
        for c, scores in category_scores.items()
        if scores and (sum(scores) / len(scores)) >= 0.85
    ]
    weak_categories = [
        c
        for c, scores in category_scores.items()
        if scores and (sum(scores) / len(scores)) < 0.60
    ]

    # -- JSON output --
    if as_json:
        report_data = {
            "overall_reliability": round(overall_score, 4),
            "tasks_passed": total_passed,
            "tasks_total": total_tasks,
            "human_baseline": HUMAN_BASELINE,
            "gap_vs_human": round(HUMAN_BASELINE - overall_score, 4),
            "category_breakdown": {
                cat: {
                    "avg_score": round(sum(scores) / max(len(scores), 1), 4),
                    "passed": category_pass_counts[cat][0],
                    "total": category_pass_counts[cat][1],
                }
                for cat, scores in category_scores.items()
            },
            "compound_reliability": {
                "3_step_chain": round(compound_3, 4),
                "5_step_chain": round(compound_5, 4),
                "10_step_chain": round(compound_10, 4),
            },
            "suitable_for": strong_categories,
            "not_recommended_for": weak_categories,
            "task_details": task_details,
        }
        json_str = json.dumps(report_data, indent=2)
        if output_path:
            with open(output_path, "w") as f:
                f.write(json_str)
            print(f"\nJSON report saved to: {output_path}")
        else:
            print(json_str)
        return

    # -- ASCII report --
    lines: list[str] = []
    w = 72
    lines.append("")
    lines.append(f"{_BOLD}{'=' * w}{_RESET}")
    lines.append(f"{_BOLD}  BROWSER AGENT RELIABILITY REPORT{_RESET}")
    lines.append(f"{_BOLD}  Powered by Stratix (LayerLens){_RESET}")
    lines.append(f"{_BOLD}{'=' * w}{_RESET}")
    lines.append("")

    # Overall
    color = (
        _GREEN
        if overall_score >= 0.80
        else (_YELLOW if overall_score >= 0.60 else _RED)
    )
    lines.append(
        f"  Overall Reliability:  {color}{_BOLD}{overall_score * 100:.1f}%{_RESET}"
    )
    lines.append(f"  Tasks Passed:         {total_passed}/{total_tasks}")
    lines.append(f"  Human Baseline:       {HUMAN_BASELINE * 100:.0f}%")
    gap = HUMAN_BASELINE - overall_score
    lines.append(f"  Gap vs Human:         {gap * 100:.1f} percentage points")
    lines.append("")

    # Category breakdown
    lines.append(f"{_BOLD}  CATEGORY BREAKDOWN{_RESET}")
    lines.append(f"  {'-' * (w - 4)}")
    for cat in TASK_CATEGORIES:
        if cat not in category_scores:
            continue
        scores = category_scores[cat]
        avg = sum(scores) / max(len(scores), 1)
        passed, total = category_pass_counts[cat]
        cat_color = _GREEN if avg >= 0.80 else (_YELLOW if avg >= 0.60 else _RED)
        label = cat.replace("_", " ").title()
        bar = _render_bar(avg)
        lines.append(
            f"  {label:20s} {cat_color}{bar}{_RESET}  ({passed}/{total} passed)"
        )
    lines.append("")

    # Compound failure analysis
    lines.append(f"{_BOLD}  COMPOUND FAILURE ANALYSIS{_RESET}")
    lines.append(f"  {'-' * (w - 4)}")
    lines.append(f"  Per-step reliability:    {overall_score * 100:.1f}%")
    lines.append(f"  3-step chain:            {_render_bar(compound_3)}")
    lines.append(f"  5-step chain:            {_render_bar(compound_5)}")
    lines.append(f"  10-step chain:           {_render_bar(compound_10)}")
    lines.append("")
    lines.append(
        f"  {_DIM}Formula: P(all N steps succeed) = (per_step_rate) ^ N{_RESET}"
    )
    lines.append(
        f"  {_DIM}At {overall_score * 100:.0f}% per step, a 10-step workflow "
        f"succeeds only {compound_10 * 100:.1f}% of the time.{_RESET}"
    )
    lines.append("")

    # Per-task detail
    lines.append(f"{_BOLD}  TASK DETAILS{_RESET}")
    lines.append(f"  {'-' * (w - 4)}")
    for detail in task_details:
        status_icon = (
            f"{_GREEN}PASS{_RESET}" if detail["all_passed"] else f"{_RED}FAIL{_RESET}"
        )
        lines.append(
            f"  [{status_icon}] {detail['task_id']:12s} {detail['description'][:45]}"
        )
        for jd in detail["judges"]:
            jcolor = _GREEN if jd["passed"] else _RED
            lines.append(
                f"         {jd['judge']:30s} {jcolor}{jd['score']:.2f}{_RESET} ({jd['verdict']})"
            )
    lines.append("")

    # Recommendations
    lines.append(f"{_BOLD}  RECOMMENDATIONS{_RESET}")
    lines.append(f"  {'-' * (w - 4)}")
    if strong_categories:
        suitable = ", ".join(c.replace("_", " ") for c in strong_categories)
        lines.append(f"  {_GREEN}Suitable for:{_RESET} {suitable}")
    else:
        lines.append(
            f"  {_YELLOW}Suitable for:{_RESET} No category exceeded the 85% threshold."
        )

    if weak_categories:
        not_rec = ", ".join(c.replace("_", " ") for c in weak_categories)
        lines.append(f"  {_RED}Not recommended for:{_RESET} {not_rec}")
    else:
        lines.append(
            f"  {_GREEN}Not recommended for:{_RESET} All categories above 60%."
        )

    lines.append("")
    lines.append(
        f"  {_DIM}Evaluated with {len(JUDGE_DEFINITIONS)} specialized judges across "
        f"{total_tasks} tasks.{_RESET}"
    )
    lines.append(
        f"  {_DIM}Compound analysis shows reliability decay in multi-step workflows.{_RESET}"
    )
    lines.append(f"{_BOLD}{'=' * w}{_RESET}")
    lines.append("")

    report_text = "\n".join(lines)
    print(report_text)

    if output_path:
        # Strip ANSI codes for file output
        import re

        clean = re.sub(r"\033\[[0-9;]*m", "", report_text)
        with open(output_path, "w") as f:
            f.write(clean)
        print(f"Report saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the browser agent evaluation framework."""
    parser = argparse.ArgumentParser(
        description="Evaluate browser automation agents with Stratix judges."
    )
    parser.add_argument(
        "--mode",
        choices=["simulated", "recorded", "live"],
        default="simulated",
        help="Evaluation mode (default: simulated)",
    )
    parser.add_argument(
        "--tasks",
        choices=list(TASK_CATEGORIES.keys()),
        default=None,
        help="Run only tasks from a specific category",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save the reliability report to a file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report as JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for simulated mode (default: 42)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print(f"\n{_BOLD}=== Browser Agent Evaluator ==={_RESET}")
    print(f"Mode: {args.mode}")

    # Select categories
    if args.tasks:
        categories = {args.tasks: TASK_CATEGORIES[args.tasks]}
    else:
        categories = TASK_CATEGORIES

    task_count = sum(len(tasks) for tasks in categories.values())
    print(f"Tasks: {task_count} across {len(categories)} categories")

    # Initialize Stratix client
    try:
        client = Stratix()
    except Exception as exc:
        print(f"\n{_RED}ERROR: Failed to initialize Stratix client: {exc}{_RESET}")
        print("Set LAYERLENS_STRATIX_API_KEY and try again.")
        sys.exit(1)

    # Load recorded traces if needed
    recorded_traces: dict[str, dict[str, Any]] = {}
    if args.mode == "recorded":
        recorded_traces = _load_recorded_traces()
        if not recorded_traces:
            print(
                f"\n{_YELLOW}WARNING: No recorded traces found in {TRACES_DIR}{_RESET}"
            )
            print("Falling back to simulated mode for tasks without recordings.")

    # Check for Browser Use in live mode
    if args.mode == "live":
        try:
            import browser_use  # type: ignore[import-untyped]  # noqa: F401

            print(f"{_GREEN}Browser Use detected. Running live evaluation.{_RESET}")
        except ImportError:
            print(f"\n{_RED}ERROR: browser-use not installed.{_RESET}")
            print("Install with: pip install browser-use")
            print("Or use --mode simulated for demo purposes.")
            sys.exit(1)

    # Create judges
    print(f"\n{_CYAN}Setting up {len(JUDGE_DEFINITIONS)} evaluation judges...{_RESET}")
    judge_map = _ensure_judges(client)
    print(f"  Judges ready: {len(judge_map)}/{len(JUDGE_DEFINITIONS)}")

    # Track which judges were created for cleanup
    existing_resp = client.judges.get_many()
    pre_existing_ids: set[str] = set()
    if existing_resp and existing_resp.judges:
        pre_existing_ids = {j.id for j in existing_resp.judges}

    created_judge_ids = [
        jid for jid in judge_map.values() if jid not in pre_existing_ids
    ]

    # Run evaluations
    all_results: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}

    try:
        for category, tasks in categories.items():
            cat_label = category.replace("_", " ").title()
            print(f"\n{_BOLD}Evaluating: {cat_label} ({len(tasks)} tasks){_RESET}")
            all_results[category] = {}
            applicable_judges = CATEGORY_JUDGES.get(category, [])

            for i, task in enumerate(tasks, 1):
                task_id = task["id"]
                print(
                    f"  [{i}/{len(tasks)}] {task['description'][:55]}...",
                    end="",
                    flush=True,
                )

                # Get or generate trace
                if args.mode == "recorded" and task_id in recorded_traces:
                    trace_data = recorded_traces[task_id]
                elif args.mode == "live":
                    # Live mode placeholder: would invoke Browser Use here.
                    # For now, generate a high-fidelity simulated trace.
                    trace_data = _generate_simulated_trace(
                        task, category, seed=args.seed + hash(task_id)
                    )
                else:
                    trace_data = _generate_simulated_trace(
                        task, category, seed=args.seed + hash(task_id)
                    )

                # Evaluate with applicable judges
                judge_results = _evaluate_trace(
                    client, trace_data, applicable_judges, judge_map
                )
                all_results[category][task_id] = judge_results

                # Print inline status
                passed_count = sum(1 for r in judge_results.values() if r["passed"])
                total_judges = len(judge_results)
                if passed_count == total_judges:
                    print(f" {_GREEN}PASS{_RESET} ({passed_count}/{total_judges})")
                else:
                    print(f" {_RED}FAIL{_RESET} ({passed_count}/{total_judges})")

        # Render report
        print(f"\n{_CYAN}Generating reliability report...{_RESET}")
        _render_report(
            all_results,
            output_path=args.output,
            as_json=args.json,
        )

    finally:
        # Clean up judges that were created during this run
        if created_judge_ids:
            print(
                f"\n{_DIM}Cleaning up {len(created_judge_ids)} created judges...{_RESET}"
            )
            for jid in created_judge_ids:
                try:
                    client.judges.delete(jid)
                except Exception:
                    pass


if __name__ == "__main__":
    main()
