#!/usr/bin/env python3
"""Industry: Multi-Agent Newsroom Research Team (SmolAgents) -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL multi-agent HuggingFace ``smolagents`` workflow
with genuine managed-agent delegation. A ``newsroom_editor`` manager delegates a
story assignment to two named managed sub-agents -- ``research_agent`` (which
runs a real ``search_news`` / ``read_article`` tool loop to gather the facts) and
``story_writer`` (which writes the brief) -- via smolagents' built-in
managed-agent delegation. The ``SmolAgentsAdapter`` recursively instruments the
sub-agents, so the recorded trace carries a real ``agent.handoff`` per delegation
(newsroom_editor -> research_agent, newsroom_editor -> story_writer) and three
distinct honest agent identities. It renders as a multi-agent graph whose Agent
column reads ``multi-agent``.

The trace was recorded from a real managed-agent run (see
samples/data/generators/smolagents.py) and is shipped under
samples/data/traces/industry/. Nothing is fabricated: the Framework column shows
``smolagents``, and the nodes/handoff edges are the real named sub-agents and
delegations the framework emitted. This sample uploads the trace and evaluates
the team's output with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python media_smolagents_research_team.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import (
    create_judge,
    poll_evaluation_results,
    recorded_trace_path,
    upload_recorded_trace,
)

SAMPLE = "media_smolagents_research_team"
FIXTURE = recorded_trace_path("industry", "media_smolagents_research_team.jsonl")

# The story assignment the newsroom team worked. Documents the scenario; the
# recorded multi-agent trace was produced by running this through a real
# smolagents manager (newsroom_editor) delegating to research_agent + story_writer.
ASSIGNMENT: dict[str, Any] = {
    "assignment_id": "NEWS-40088",
    "desk": "metro",
    "topic": "city transit expansion",
    "team": ["newsroom_editor", "research_agent", "story_writer"],
    "brief": (
        "Produce a short, sourced news brief on the city transit-expansion story: "
        "research_agent gathers the facts (funding amount, service year, who "
        "benefits) and story_writer writes the final 2-3 sentence brief."
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded multi-agent newsroom-research-team trace."""
    print("=== LayerLens Industry: Multi-Agent Newsroom Research Team (SmolAgents) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded multi-agent trace first (renders as a multi-agent graph
    # via real managed-agent handoffs). Do this before creating judges so the
    # trace always lands even if the org has no evaluation model yet.
    print("Uploading the recorded newsroom-research-team trace (managed-agent delegation graph)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Assignment: {ASSIGNMENT['assignment_id']} (team: {', '.join(ASSIGNMENT['team'])})\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "delegation_soundness": create_judge(
                client,
                name="Newsroom Delegation Judge",
                evaluation_goal="Evaluate whether the editor delegated research to the research agent and the write-up to the story writer (each part handled by the right specialist).",
                namespace=SAMPLE,
            ),
            "factual_grounding": create_judge(
                client,
                name="Team Factual Grounding Judge",
                evaluation_goal="Evaluate whether the final brief is grounded in the facts the research agent actually gathered from the wire rather than invented details.",
                namespace=SAMPLE,
            ),
            "brief_quality": create_judge(
                client,
                name="Story Brief Quality Judge",
                evaluation_goal="Evaluate whether the final brief is a tight, accurate, publishable 2-3 sentence news brief citing the key figures.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the team's story brief:")
        for label, judge in judges.items():
            evaluation = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge.id)
            if evaluation is None:
                print(f"  {label:22s} -- evaluation creation failed")
                continue
            results = poll_evaluation_results(client, evaluation.id)
            if results:
                r = results[0]
                verdict = "pass" if r.passed else "fail"
                color = _VERDICT_COLORS.get(verdict, "")
                print(f"  {label:22s} {color}{verdict.upper()}{_RESET} ({r.score:.2f})")
            else:
                print(f"  {label:22s} -- timed out waiting for results")
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
