#!/usr/bin/env python3
"""Industry: Media Newsroom Research (SmolAgents tool-use) -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL single-agent tool-use workflow built with
HuggingFace ``smolagents``. A newsroom research assistant (persona
``newsroom_research_agent``, a ``ToolCallingAgent`` backed by OpenAI gpt-4o-mini)
runs a genuine two-tool research loop -- it calls ``search_news`` to find the
relevant wire headlines, then ``read_article`` to read the most relevant
article -- and finally drafts a short, sourced story brief.

Because the trace was captured under the real ``SmolAgentsAdapter`` from a
genuine multi-step tool loop, it renders a single honest agent node (Agent
column ``newsroom_research_agent``) plus the real ``model.invoke`` /
``cost.record`` / ``tool.call`` / ``tool.result`` events -- no fabrication. The
Framework column shows ``smolagents`` (the framework that actually ran) and the
token/cost figures are real. The recorded trace is shipped under
samples/data/traces/industry/ (produced by samples/data/generators/smolagents.py
via samples/data/_generate_fixtures.py); this sample uploads it and evaluates
the story brief with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python media_smolagents_newsroom.py
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

SAMPLE = "media_smolagents_newsroom"
FIXTURE = recorded_trace_path("industry", "media_smolagents_newsroom.jsonl")

# The research assignment the newsroom agent worked. Documents the scenario; the
# recorded tool-use trace was produced by running this through a real smolagents
# ToolCallingAgent (the agent called search_news, then read_article, then wrote
# the brief).
ASSIGNMENT: dict[str, Any] = {
    "assignment_id": "NEWS-40021",
    "desk": "metro",
    "topic": "city transit expansion",
    "brief": (
        "Research the city transit-expansion story on the wire and write a "
        "factual 2-3 sentence news brief citing the key figures (funding amount "
        "and service year)."
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded newsroom-research tool-use trace."""
    print("=== LayerLens Industry: Media Newsroom Research (SmolAgents tool-use) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded tool-use trace first (renders a single newsroom_research_
    # agent node with real model.invoke / tool.call / tool.result events). Do this
    # before creating judges so the trace always lands even if the org has no
    # evaluation model yet.
    print("Uploading the recorded newsroom-research trace (search_news + read_article loop)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Assignment: {ASSIGNMENT['assignment_id']} ({ASSIGNMENT['topic']})\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "factual_grounding": create_judge(
                client,
                name="Factual Grounding Judge",
                evaluation_goal="Evaluate whether the story brief is grounded in the articles the agent actually read via the tools (search_news/read_article) rather than inventing facts.",
                namespace=SAMPLE,
            ),
            "key_figures": create_judge(
                client,
                name="Key Figures Judge",
                evaluation_goal="Evaluate whether the brief correctly cites the key figures from the source (the funding amount and the target service year).",
                namespace=SAMPLE,
            ),
            "brevity": create_judge(
                client,
                name="Newsroom Brevity Judge",
                evaluation_goal="Evaluate whether the brief is a tight, publishable 2-3 sentence summary suitable for a news wire.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the story brief:")
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
