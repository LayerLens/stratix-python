#!/usr/bin/env python3
"""Industry: Multi-Agent Support Triage Team -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL multi-agent workflow. A triage-router reads an
incoming customer-support ticket and routes it to the right specialist --
billing, technical, or account -- which in turn hands off to a resolver that
closes the ticket, orchestrated as a LangGraph StateGraph with
agent-to-agent handoffs. For this ticket the router selects the
technical-specialist, so the recorded trace renders a 3-node agent graph
(triage-router -> technical-specialist -> resolver) whose nodes carry real
model calls across OpenAI and Ollama.

The trace was recorded from a real run (see samples/data/_generate_fixtures.py)
and is shipped under samples/data/traces/industry/. This sample uploads it and
evaluates the team's resolution with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python support_triage_team.py
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

SAMPLE = "support_triage_team"
FIXTURE = recorded_trace_path("industry", "support_triage_team.jsonl")

# The support ticket the triage team handled. Documents the scenario; the
# recorded multi-agent trace was produced by running this through the real
# LangGraph team (triage-router -> technical-specialist -> resolver).
TICKET = (
    "My API calls started returning 401 after I rotated my key this morning; "
    "billing looks fine."
)

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded multi-agent support-triage-team trace."""
    print("=== LayerLens Industry: Multi-Agent Support Triage Team ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded multi-agent trace first (renders as a multi-node agent
    # graph). Do this before creating judges so the trace always lands even if
    # the org has no evaluation model yet.
    print("Uploading the recorded support-triage-team trace (router/specialist/resolver graph)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Ticket: {TICKET}\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "routing_accuracy": create_judge(
                client,
                name="Routing Accuracy Judge",
                evaluation_goal="Evaluate whether the ticket was routed to the correct specialist.",
                namespace=SAMPLE,
            ),
            "resolution_quality": create_judge(
                client,
                name="Resolution Quality Judge",
                evaluation_goal="Evaluate whether the resolution correctly and completely addresses the issue.",
                namespace=SAMPLE,
            ),
            "tone": create_judge(
                client,
                name="Tone Judge",
                evaluation_goal="Evaluate whether the response is professional and empathetic.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the team's resolution:")
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
