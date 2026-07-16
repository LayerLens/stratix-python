#!/usr/bin/env python3
"""Industry: Media Content Moderation Review Pipeline — Langfuse trace migration
(multi-observation) — LayerLens Python SDK Sample.

Demonstrates migrating a richer EXISTING observability trace from Langfuse into
LayerLens. A media platform's content-moderation review pipeline was logged to
Langfuse as a multi-observation tree, and the LayerLens ``LangfuseAdapter``
imports the whole tree into flat LayerLens events:

* a ``generation`` -> ``model.invoke`` + ``cost.record`` — the moderation LLM
  call (a real ``gpt-4o-mini`` decision: model, token usage, decision text);
* a ``span``       -> ``tool.call`` — the ``policy_lookup`` step;
* an ``event``     -> ``agent.state.change`` — an auto-escalation to human review;
* a ``score``      -> ``evaluation.result`` — a real LLM-as-judge
  ``policy_adherence`` rating (a genuine ``gpt-4o-mini`` judge's numeric score),
  the langfuse-distinctive grading signal that the migration preserves.

Because every observation was imported from a REAL Langfuse trace, the migrated
trace renders honestly: Framework = ``langfuse`` (the tool migrated FROM), a
single node named after the real Langfuse trace (``content-moderation-review``),
a real ``model.invoke`` / ``cost.record``, the tool + state-change events, and
the real judge score. A migrated observability trace is single-node /
non-agentic, so it renders a single node plus its observation waterfall — NOT a
multi-agent graph (correct, not a gap).

The recorded trace is shipped under samples/data/traces/industry/ (produced by
samples/data/generators/langfuse.py); this sample uploads it and evaluates the
migrated review with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python media_langfuse_moderation_pipeline.py
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

SAMPLE = "media_langfuse_moderation_pipeline"
FIXTURE = recorded_trace_path("industry", "media_langfuse_moderation_pipeline.jsonl")

# The post the migrated review pipeline processed. Documents the scenario; the
# recorded trace was produced by importing a real multi-observation Langfuse
# trace (generation + span + event + a real LLM-as-judge score).
POST: dict[str, Any] = {
    "post_id": "POST-70552",
    "surface": "social_feed",
    "category": "health_safety_misinformation",
    "text": (
        "URGENT: every batch of NutriStart baby formula is being recalled for "
        "containing toxic heavy metals — throw yours out NOW and share to save "
        "lives! The authorities are covering it up so spread the word before "
        "this post gets deleted."
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the migrated multi-observation Langfuse review-pipeline trace."""
    print("=== LayerLens Industry: Content Moderation Review Pipeline (Langfuse migration) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the migrated multi-observation trace first (renders a single
    # content-moderation-review node with real model.invoke / cost.record plus
    # the tool.call / agent.state.change / evaluation.result events). Do this
    # before creating judges so the trace lands even without an evaluation model.
    print("Migrating the recorded Langfuse review-pipeline trace into LayerLens...\n")
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
                evaluation_goal="Evaluate whether the moderation decision correctly applies platform policy to the misinformation post.",
                namespace=SAMPLE,
            ),
            "escalation_appropriateness": create_judge(
                client,
                name="Escalation Appropriateness Judge",
                evaluation_goal="Evaluate whether escalating this post to human review was an appropriate handling of an unverified safety-recall claim.",
                namespace=SAMPLE,
            ),
            "grading_signal_preserved": create_judge(
                client,
                name="Grading Signal Judge",
                evaluation_goal="Evaluate whether the migrated trace preserved the LLM-as-judge policy-adherence score alongside the moderation decision.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the migrated review pipeline:")
        for label, judge in judges.items():
            evaluation = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge.id)
            if evaluation is None:
                print(f"  {label:28s} -- evaluation creation failed")
                continue
            results = poll_evaluation_results(client, evaluation.id)
            if results:
                r = results[0]
                verdict = "pass" if r.passed else "fail"
                color = _VERDICT_COLORS.get(verdict, "")
                print(f"  {label:28s} {color}{verdict.upper()}{_RESET} ({r.score:.2f})")
            else:
                print(f"  {label:28s} -- timed out waiting for results")
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
