#!/usr/bin/env python3
"""Industry: Retail multi-provider routing gateway (LiteLLM) -- LayerLens Sample.

Demonstrates evaluating LiteLLM's headline feature: a **multi-provider routing
gateway**. A cost/capability router sends three real retail customer-support
turns to THREE different providers -- a cheap FAQ to OpenAI ``gpt-4o-mini``,
nuanced policy reasoning to Anthropic ``claude-haiku-4-5``, and a quick factual
lookup to Google ``gemini-2.5-flash`` -- all through the SAME
``litellm.completion`` seam. Each turn emits its own ``cost.record`` whose
``provider`` is the UNDERLYING provider that actually served it, so the trace
carries a genuine PER-PROVIDER cost breakdown.

This is a routing LOOP, not a multi-agent graph: LiteLLM is a proxy with no
agent, so the trace renders an honest EMPTY-STATE (Agent column ``—`` -- no DAG,
LayerLens does not invent an agent) + a span waterfall. The value it proves is
per-provider cost attribution across a mixed-provider workload, not agent
topology. The token/cost fields are real; nothing is fabricated.

The recorded trace is shipped under samples/data/traces/industry/ (produced by
samples/data/_generate_fixtures.py); this sample uploads it and evaluates the
answers with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python retail_litellm_gateway.py
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

SAMPLE = "retail_litellm_gateway"
FIXTURE = recorded_trace_path("industry", "retail_litellm_gateway.jsonl")

# The three support turns the gateway routed, each to a different provider by a
# cost/capability router. Documents the scenario; the recorded trace was produced
# by running these through real litellm.completion calls under instrument_litellm.
ROUTED_TURNS: list[dict[str, Any]] = [
    {"tier": "faq", "provider": "openai/gpt-4o-mini",
     "text": "What are your standard shipping options and how long does each take?"},
    {"tier": "reasoning", "provider": "anthropic/claude-haiku-4-5",
     "text": "A gifted tent arrived cracked but I reported it 45 days later -- what can you do?"},
    {"tier": "factual", "provider": "gemini/gemini-2.5-flash",
     "text": "How do I track order #NW-58217 and can I still change the delivery address?"},
]

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded LiteLLM multi-provider gateway trace (empty-state)."""
    print("=== LayerLens Industry: Retail Multi-Provider Gateway (LiteLLM) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload first (renders empty-state: Agent ``—``, Framework ``litellm``, with a
    # per-provider cost.record for each routed turn). Do this before creating judges
    # so the trace always lands even if the org has no evaluation model yet.
    print("Uploading the recorded LiteLLM gateway trace (openai + anthropic + google)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print("  Routed turns (per-provider cost.record):")
    for t in ROUTED_TURNS:
        print(f"    - [{t['tier']:9s}] {t['provider']}")
    print("  Agent:    —  (LiteLLM is a provider proxy; no agent, honest empty-state)")
    print("  Framework: litellm\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "answer_quality": create_judge(
                client,
                name="Gateway Answer Quality Judge",
                evaluation_goal="Evaluate whether the routed support answers correctly and helpfully resolve each shopper's question (shipping, warranty/return, order tracking).",
                namespace=SAMPLE,
            ),
            "routing_fit": create_judge(
                client,
                name="Routing Appropriateness Judge",
                evaluation_goal="Evaluate whether each turn's answer matches the complexity it was routed for -- a concise FAQ reply, nuanced reasoning for the policy edge case, and a direct factual lookup answer.",
                namespace=SAMPLE,
            ),
            "policy_consistency": create_judge(
                client,
                name="Policy Consistency Judge",
                evaluation_goal="Evaluate whether the answers apply retail policies (shipping, returns, warranty) accurately and do not contradict one another across the routed turns.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the gateway answers:")
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
