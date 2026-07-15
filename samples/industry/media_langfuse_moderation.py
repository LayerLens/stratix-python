#!/usr/bin/env python3
"""Industry: Media Content Moderation — Langfuse trace migration (single) —
LayerLens Python SDK Sample.

Demonstrates migrating an EXISTING observability trace from Langfuse into
LayerLens and evaluating it. A media platform had been logging its
content-moderation decisions to Langfuse; the LayerLens ``LangfuseAdapter``
imports one such trace (a single ``generation`` observation — the moderation
LLM call) and re-emits it as flat LayerLens events.

Because the trace was captured by importing a REAL Langfuse trace (the migrated
``generation`` carries a genuine ``gpt-4o-mini`` moderation call's model, token
usage, and decision text; the cost is Langfuse's own calculated figure), it
renders honestly: the Framework column shows ``langfuse`` (the tool the trace
was migrated FROM), the single node is named after the real Langfuse trace
(``content-moderation``), and the ``model.invoke`` / ``cost.record`` events are
real. A migrated observability trace is single-node / non-agentic, so it does
NOT render a multi-agent graph — that is correct, not a gap.

The recorded trace is shipped under samples/data/traces/industry/ (produced by
samples/data/generators/langfuse.py); this sample uploads it and evaluates the
moderation decision with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python media_langfuse_moderation.py
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

SAMPLE = "media_langfuse_moderation"
FIXTURE = recorded_trace_path("industry", "media_langfuse_moderation.jsonl")

# The post the moderation decision (now migrated from Langfuse) reviewed.
# Documents the scenario; the recorded trace was produced by importing a real
# Langfuse trace whose generation was a genuine gpt-4o-mini moderation call.
POST: dict[str, Any] = {
    "post_id": "POST-70418",
    "surface": "social_feed",
    "category": "spam_scam",
    "text": (
        "Just DM me the word CRYPTO and I'll 10x your money in 48 hours, "
        "guaranteed! Limited spots — first 50 people only, don't miss out. "
        "#CryptoKing #GetRichQuick"
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the migrated Langfuse content-moderation trace."""
    print("=== LayerLens Industry: Media Content Moderation (Langfuse migration) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the migrated trace first (renders a single content-moderation node
    # with a real, framework=langfuse model.invoke / cost.record). Do this before
    # creating judges so the trace always lands even if the org has no evaluation
    # model yet.
    print("Migrating the recorded Langfuse content-moderation trace into LayerLens...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Post:     {POST['post_id']} ({POST['category']})\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "moderation_accuracy": create_judge(
                client,
                name="Moderation Accuracy Judge",
                evaluation_goal="Evaluate whether the moderation decision (ALLOW/FLAG/REMOVE) correctly applies platform policy to the post's content.",
                namespace=SAMPLE,
            ),
            "migration_fidelity": create_judge(
                client,
                name="Migration Fidelity Judge",
                evaluation_goal="Evaluate whether the migrated trace preserves the moderation decision and its justification faithfully enough to audit.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the migrated moderation decision:")
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
        print("  The trace is migrated; add a project/public model to enable judges.")
    finally:
        for jid in judge_ids:
            try:
                client.judges.delete(jid)
            except Exception:
                pass


if __name__ == "__main__":
    main()
