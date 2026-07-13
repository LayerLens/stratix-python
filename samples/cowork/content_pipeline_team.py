#!/usr/bin/env python3
"""Cowork: Multi-Agent Content Pipeline Team -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL multi-agent group chat. A content strategist, a
copywriter, and an editor collaborate as a real AutoGen `RoundRobinGroupChat`
(not simulated) to produce a short launch announcement -- each agent reads the
running conversation and contributes its turn. Because the trace was recorded
from an actual AutoGen run, it carries genuine per-agent `agent.input`/
`model.invoke`/`cost.record` events and renders as a 3-agent trace
(content_strategist -> copywriter -> editor) with real model calls.

The trace was recorded from a real run (see samples/data/_generate_fixtures.py)
and is shipped under samples/data/traces/cowork/. This sample uploads it and
evaluates the collaboration with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python content_pipeline_team.py
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

SAMPLE = "content_pipeline_team"
FIXTURE = recorded_trace_path("cowork", "content_pipeline_team.jsonl")

# The brief the content team collaborated on. Documents the scenario; the
# recorded multi-agent trace was produced by running this through the real
# AutoGen group chat (content_strategist -> copywriter -> editor).
BRIEF = "Write a short launch announcement for a privacy-first note-taking app called Quill."

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded multi-agent content-pipeline-team trace."""
    print("=== LayerLens Cowork: Multi-Agent Content Pipeline Team ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded multi-agent trace first (renders as a multi-node agent
    # graph). Do this before creating judges so the trace always lands even if
    # the org has no evaluation model yet.
    print("Uploading the recorded content-pipeline-team trace (strategist/copywriter/editor group chat)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Brief: {BRIEF}\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "on_brief": create_judge(
                client,
                name="On-Brief Judge",
                evaluation_goal="Evaluate whether the final copy stays on the requested brief and message.",
                namespace=SAMPLE,
            ),
            "clarity": create_judge(
                client,
                name="Copy Clarity Judge",
                evaluation_goal="Evaluate whether the final announcement is clear, concise, and well written.",
                namespace=SAMPLE,
            ),
            "collaboration": create_judge(
                client,
                name="Collaboration Judge",
                evaluation_goal="Evaluate whether each agent (strategist, copywriter, editor) contributed its role.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the team's output:")
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
