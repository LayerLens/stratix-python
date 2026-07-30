#!/usr/bin/env python3
"""Cowork: Multi-Agent Research Report Team -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL multi-agent workflow. A planner briefs a
writer, which drafts a report and iterates with a critic in a revise loop
(writer -> critic -> writer -> critic) before the final brief is produced,
orchestrated as a LangGraph StateGraph with agent-to-agent handoffs. Each
role runs on a different instrumented model provider, so the recorded trace
renders as a multi-node agent graph whose nodes carry real model calls
across OpenAI, Anthropic, and Ollama.

The trace was recorded from a real run (see samples/data/_generate_fixtures.py)
and is shipped under samples/data/traces/cowork/. This sample uploads it and
evaluates the team's brief with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python research_report_team.py
"""

from __future__ import annotations

import os
import sys

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import (
    create_judge,
    poll_evaluation_results,
    recorded_trace_path,
    upload_recorded_trace,
)

SAMPLE = "research_report_team"
FIXTURE = recorded_trace_path("cowork", "research_report_team.jsonl")

# The topic the research team briefed on. Documents the scenario; the
# recorded multi-agent trace was produced by running this through the real
# LangGraph team (planner -> writer -> critic -> writer -> critic).
TOPIC = "the benefits of automated code review"

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded multi-agent research-report-team trace."""
    print("=== LayerLens Cowork: Multi-Agent Research Report Team ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded multi-agent trace first (renders as a multi-node agent
    # graph). Do this before creating judges so the trace always lands even if
    # the org has no evaluation model yet.
    print("Uploading the recorded research-report-team trace (planner/writer/critic revise loop)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Topic: {TOPIC}\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "groundedness": create_judge(
                client,
                name="Groundedness Judge",
                evaluation_goal="Evaluate whether the brief is accurate and well-supported.",
                namespace=SAMPLE,
            ),
            "coherence": create_judge(
                client,
                name="Coherence Judge",
                evaluation_goal="Evaluate whether the brief is clear, well-structured, and coherent.",
                namespace=SAMPLE,
            ),
            "revision_quality": create_judge(
                client,
                name="Revision Quality Judge",
                evaluation_goal="Evaluate whether the critic's feedback meaningfully improved the draft.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the team's brief:")
        for label, judge in judges.items():
            evaluation = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge.id)
            if evaluation is None:
                print(f"  {label:20s} -- evaluation creation failed")
                continue
            results = poll_evaluation_results(client, evaluation.id)
            if results:
                r = results[0]
                verdict = "pass" if r.passed else "fail"
                color = _VERDICT_COLORS.get(verdict, "")
                print(f"  {label:20s} {color}{verdict.upper()}{_RESET} ({r.score:.2f})")
            else:
                print(f"  {label:20s} -- timed out waiting for results")
    except RuntimeError as exc:
        print(f"\nNOTE: evaluations skipped -- {exc}")
        print("  The trace is uploaded; add a project/public model to enable judges.")
    finally:
        for jid in judge_ids:
            try:
                client.judges.delete(jid)
            except Exception:
                pass


if __name__ == "__main__":
    main()
