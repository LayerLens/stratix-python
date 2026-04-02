#!/usr/bin/env python3
"""
Trace Agent Execution -- LayerLens + OpenClaw
==============================================
Traces an OpenClaw agent's execution with LayerLens, then evaluates
the result for quality using an AI judge.

Workflow:
  1. Create an OpenClaw agent (or connect to an existing one).
  2. Execute a task via the agent.
  3. Upload the execution as a LayerLens trace with metadata.
  4. Create a judge and evaluate the trace for quality.
  5. Print results.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package openclaw
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python trace_agent_execution.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import create_judge, upload_trace_dict, poll_evaluation_results

# ---------------------------------------------------------------------------
# Simulated OpenClaw execution (used when openclaw is not installed)
# ---------------------------------------------------------------------------

SIMULATED_EXECUTION: dict[str, Any] = {
    "agent_name": "research-assistant",
    "model": "claude-sonnet-4-20250514",
    "task": "Find the top 3 trending Python libraries this week on GitHub and summarize what each one does.",
    "result": (
        "Here are the top 3 trending Python libraries on GitHub this week:\n\n"
        "1. **uv** (astral-sh/uv) -- An extremely fast Python package installer and "
        "resolver written in Rust. Drop-in replacement for pip that is 10-100x faster.\n\n"
        "2. **marimo** (marimo-team/marimo) -- A reactive notebook for Python that "
        "replaces Jupyter with reproducible, git-friendly, and deployable notebooks.\n\n"
        "3. **crawl4ai** (unclecode/crawl4ai) -- An open-source LLM-friendly web "
        "crawler that extracts structured data optimized for AI/RAG pipelines."
    ),
    "duration_ms": 4200,
    "skills_used": ["browser_search", "web_scrape", "summarize"],
}


def _execute_openclaw_task(task: str) -> dict[str, Any]:
    """Execute a task via OpenClaw, falling back to simulated data."""
    try:
        from openclaw import OpenClawClient  # type: ignore[import-untyped]

        print("(Connecting to OpenClaw agent...)\n")
        oc_client = OpenClawClient()
        agent = oc_client.agents.create(
            name="research-assistant",
            model="claude-sonnet-4-20250514",
            description="Research assistant that finds and summarizes information.",
        )
        start = time.monotonic()
        result = agent.execute(task)
        duration_ms = round((time.monotonic() - start) * 1000)
        return {
            "agent_name": "research-assistant",
            "model": "claude-sonnet-4-20250514",
            "task": task,
            "result": str(result),
            "duration_ms": duration_ms,
            "skills_used": getattr(result, "skills_used", []),
        }
    except ImportError:
        print("(openclaw not installed -- using simulated execution data)")
        print("  Install with: pip install openclaw\n")
        return SIMULATED_EXECUTION
    except Exception as exc:
        print(f"(OpenClaw connection failed: {exc} -- using simulated data)\n")
        return SIMULATED_EXECUTION


def main() -> None:
    """Run the trace agent execution demo."""
    print("=== LayerLens + OpenClaw: Trace Agent Execution ===\n")

    # --- 1. Execute a task via OpenClaw ---
    task = "Find the top 3 trending Python libraries this week on GitHub and summarize what each one does."
    execution = _execute_openclaw_task(task)

    print(f"Agent:    {execution['agent_name']}")
    print(f"Model:    {execution['model']}")
    print(f"Duration: {execution['duration_ms']}ms")
    print(f"Task:     {execution['task'][:80]}...")
    print(f"Result:   {execution['result'][:120]}...\n")

    # --- 2. Initialize LayerLens client ---
    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # --- 3. Upload execution as a trace ---
    trace_result = upload_trace_dict(
        client,
        input_text=execution["task"],
        output_text=execution["result"],
        metadata={
            "source": "openclaw",
            "agent_name": execution["agent_name"],
            "model": execution["model"],
            "duration_ms": execution["duration_ms"],
            "skills_used": execution.get("skills_used", []),
        },
    )
    if not trace_result or not trace_result.trace_ids:
        print("WARNING: Trace upload returned no IDs")
        return
    trace_id = trace_result.trace_ids[0]
    print(f"Uploaded trace: {trace_id}")

    # --- 4. Create a judge and evaluate ---
    judge = create_judge(
        client,
        name="OpenClaw Quality Judge",
        evaluation_goal=(
            "Evaluate whether the agent's response is accurate, complete, "
            "well-structured, and directly addresses the user's task. "
            "Check that claims are plausible and the output is actionable."
        ),
    )
    print(f"Created judge:  {judge.name} (ID: {judge.id})")

    try:
        evaluation = client.trace_evaluations.create(
            trace_id=trace_id,
            judge_id=judge.id,
        )
        print(f"Evaluation:     {evaluation.id}\n")

        # --- 5. Poll for results ---
        print("Waiting for evaluation results...")
        results = poll_evaluation_results(client, evaluation.id)
        if results:
            r = results[0]
            verdict = "PASS" if r.passed else "FAIL"
            color = "\033[92m" if r.passed else "\033[91m"
            reset = "\033[0m"
            print(f"\n  Verdict:   {color}{verdict}{reset}")
            print(f"  Score:     {r.score}")
            print(f"  Reasoning: {r.reasoning}")
        else:
            print("  No results yet (evaluation may still be processing)")
    finally:
        try:
            client.judges.delete(judge.id)
        except Exception:
            pass

    print("\nDone.")


if __name__ == "__main__":
    main()
