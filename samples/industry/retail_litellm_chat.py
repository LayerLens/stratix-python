#!/usr/bin/env python3
"""Industry: Retail customer-support chat via LiteLLM -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL provider-proxy trace. LiteLLM is a multi-provider
**proxy**, not an agent framework, so it has NO agent -- the trace renders an
honest EMPTY-STATE: the Agent column is ``—`` (LayerLens does not invent an
agent), the Framework column is ``litellm``, and the run shows as a span
waterfall with the real ``model.invoke`` / ``cost.record`` events. Here a single
retail customer-support question is answered through ``litellm.completion`` on
its default route (OpenAI ``gpt-4o-mini``).

Because the trace was captured under ``instrument_litellm`` from a genuine
completion, the token counts and the ``cost.record`` (attributed to the
underlying ``openai`` provider) are real -- nothing is fabricated, and there is
deliberately no Agent because a proxy has none. The recorded trace is shipped
under samples/data/traces/industry/ (produced by
samples/data/_generate_fixtures.py); this sample uploads it and evaluates the
support answer with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python retail_litellm_chat.py
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

SAMPLE = "retail_litellm_chat"
FIXTURE = recorded_trace_path("industry", "retail_litellm_chat.jsonl")

# The shopper question the support gateway answered. Documents the scenario; the
# recorded trace was produced by running this through a real litellm.completion
# call (default OpenAI route) under instrument_litellm.
QUESTION: dict[str, Any] = {
    "query_id": "NW-SUP-4471",
    "route": "openai/gpt-4o-mini (litellm default)",
    "text": (
        "I ordered a rain jacket 10 days ago but it arrived a size too small. "
        "What are my options for exchanging it, and will I have to pay return "
        "shipping?"
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded LiteLLM retail-support chat trace (empty-state)."""
    print("=== LayerLens Industry: Retail Support Chat via LiteLLM ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded trace first (renders empty-state: Agent ``—``, Framework
    # ``litellm``, real model.invoke / cost.record). Do this before creating judges
    # so the trace always lands even if the org has no evaluation model yet.
    print("Uploading the recorded LiteLLM support trace (default OpenAI route)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Query:    {QUESTION['query_id']}  (route {QUESTION['route']})")
    print("  Agent:    —  (LiteLLM is a provider proxy; no agent, honest empty-state)")
    print("  Framework: litellm\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "policy_accuracy": create_judge(
                client,
                name="Return Policy Accuracy Judge",
                evaluation_goal="Evaluate whether the answer correctly explains the shopper's exchange/return options and who pays return shipping, consistent with a standard retail policy.",
                namespace=SAMPLE,
            ),
            "helpfulness": create_judge(
                client,
                name="Support Helpfulness Judge",
                evaluation_goal="Evaluate whether the answer directly resolves the shopper's question with empathy and a concrete next step.",
                namespace=SAMPLE,
            ),
            "conciseness": create_judge(
                client,
                name="Conciseness Judge",
                evaluation_goal="Evaluate whether the answer is clear and concise (roughly under 120 words) without omitting the key policy detail.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the support answer:")
        for label, judge in judges.items():
            evaluation = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge.id)
            if evaluation is None:
                print(f"  {label:18s} -- evaluation creation failed")
                continue
            results = poll_evaluation_results(client, evaluation.id)
            if results:
                r = results[0]
                verdict = "pass" if r.passed else "fail"
                color = _VERDICT_COLORS.get(verdict, "")
                print(f"  {label:18s} {color}{verdict.upper()}{_RESET} ({r.score:.2f})")
            else:
                print(f"  {label:18s} -- timed out waiting for results")
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
