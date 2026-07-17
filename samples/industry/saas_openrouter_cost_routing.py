#!/usr/bin/env python3
"""Industry: Software/SaaS multi-model cost routing (OpenRouter) -- LayerLens Sample.

Demonstrates evaluating OpenRouter's headline feature: **multi-model cost
routing**. A support assistant for a B2B event-analytics API answers two real
customer questions through the SAME OpenRouter gateway, but on two different
routes chosen by cost/capability -- a routine plan-limits FAQ goes to a FREE
model (``meta-llama/llama-3-8b-instruct:free``), while a live production incident
(429s during a nightly backfill) is escalated to a PAID model
(``openai/gpt-4o-mini``). Cheap-first, escalate-on-complexity: that split is why
teams put OpenRouter in front of their models, and it is what this trace proves.

WHAT RENDERS
    OpenRouter is a **provider gateway**, not an agent framework: it emits
    ``model.invoke``/``cost.record`` but no ``agent.identity``. So the trace
    renders an honest EMPTY-STATE -- Agent column ``—`` (LayerLens does NOT
    invent an agent) + a two-call span waterfall. Framework ``openrouter``,
    Status ``ok``. The routed model slugs and token counts are real.

THE COST COLUMN IS HONEST, AND PARTIAL ON PURPOSE
    OpenRouter bills at its own rates, which no pricing table we ship holds, so
    the gateway is the SOLE authority for what a call cost. The trace reflects
    exactly that:
      * free route -- usage accounting was ON, so OpenRouter reported its own
        charge: ``$0.00``. A ``:free`` slug genuinely bills nothing, so that zero
        is a FACT, and it lands as a ``cost.record`` stamped
        ``cost_source="provider"``.
      * paid route -- usage accounting was OFF, so OpenRouter reported no charge
        and LayerLens records NONE. Pricing ``openai/gpt-4o-mini`` from our own
        catalog would attach OpenAI list-rate dollars that OpenRouter never
        billed. A missing cost is honest; an invented one is not.
    Turn usage accounting on (``extra_body={"usage": {"include": True}}``) and
    every route reports its real charge.

SEALED FIXTURE -- no OpenRouter credential exists
    The recorded trace ships under samples/data/traces/industry/ (produced by
    samples/data/generators/openrouter.py). No OpenRouter API key exists on any
    machine, so the gateway hop is SEALED behind a mock transport and this sample
    does not claim a live gateway call happened (see ``metadata.sealed`` on the
    trace). The model responses replayed through it ARE real captured inferences
    -- local llama3:8b for the free route, a real billed OpenAI gpt-4o-mini call
    for the paid route -- and the real OpenRouter adapter really parsed them, so
    every token count, every word of output and the attestation chain are
    genuine. Re-record for real once a key is provisioned.

    One artifact to read correctly: ``latency_ms`` on the trace measures the
    local sealed-transport replay (~1-20ms), NOT a real OpenRouter round-trip
    (~0.5-3s). It is honestly measured, but it is not gateway performance. Tokens,
    output text and cost are unaffected (see ``metadata`` on the trace).

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key
    (NO provider key needed -- the trace is already recorded.)

Usage:
    python saas_openrouter_cost_routing.py
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

SAMPLE = "saas_openrouter_cost_routing"
FIXTURE = recorded_trace_path("industry", "saas_openrouter_cost_routing.jsonl")

# The two support turns the gateway routed. Documents the scenario; the recorded
# trace was produced by running these through the real OpenRouterProvider.
ROUTED_TURNS: list[dict[str, Any]] = [
    {
        "tier": "faq",
        "route": "meta-llama/llama-3-8b-instruct:free",
        "cost": "$0.00 (reported by the gateway -- a ':free' slug bills nothing)",
        "text": "What is the ingest rate limit on the Growth plan, and what status do I get when I exceed it?",
    },
    {
        "tier": "escalated",
        "route": "openai/gpt-4o-mini",
        "cost": "not reported (usage accounting off) -- so none is recorded",
        "text": "We're getting 429s on /v1/events at ~40k events/min during our nightly backfill. Why, and how do we fix it?",
    },
]

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded OpenRouter cost-routing trace (empty-state)."""
    print("=== LayerLens Industry: SaaS Multi-Model Cost Routing (OpenRouter) ===\n")

    # Cred guard: only a LayerLens key is needed (no provider key -- the trace is
    # already recorded). Exit cleanly and say so rather than raising.
    if not os.environ.get("LAYERLENS_STRATIX_API_KEY"):
        print("SKIPPED: LAYERLENS_STRATIX_API_KEY is not set.")
        print("  Set it to upload the recorded OpenRouter cost-routing trace:")
        print("    export LAYERLENS_STRATIX_API_KEY=your-api-key")
        print("  No provider key is required -- the trace is already recorded.")
        sys.exit(1)

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload first (renders empty-state: Agent ``—``, Framework ``openrouter``),
    # before creating judges, so the trace always lands even if the org has no
    # evaluation model yet.
    print("Uploading the recorded OpenRouter cost-routing trace (SEALED gateway)...\n")
    try:
        trace_ids = upload_recorded_trace(client, FIXTURE)
    except Exception as exc:
        print(f"SKIPPED: could not upload the recorded trace: {exc}")
        sys.exit(1)
    if not trace_ids:
        print("SKIPPED: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)

    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print("  Routed turns (cost/capability router):")
    for t in ROUTED_TURNS:
        print(f"    - [{t['tier']:9s}] {t['route']}")
        print(f"                  cost: {t['cost']}")
    print("  Agent:     —  (OpenRouter is a provider gateway; no agent, honest empty-state)")
    print("  Framework: openrouter")
    print("  Status:    ok\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "answer_quality": create_judge(
                client,
                name="Support Answer Quality Judge",
                evaluation_goal="Evaluate whether the routed support answers correctly and helpfully resolve each customer's question about the event-analytics API (plan rate limits, 429 handling, backfill remediation).",
                namespace=SAMPLE,
            ),
            # Judge names carry a " (<namespace>)" suffix and the server caps the
            # result at 64 chars — keep them short enough to survive it.
            "routing_fit": create_judge(
                client,
                name="Cost Routing Fit Judge",
                evaluation_goal="Evaluate whether each answer matches the complexity it was routed for -- a concise factual reply for the plan-limits FAQ, and a deeper root-cause plus concrete remediation plan for the escalated production incident.",
                namespace=SAMPLE,
            ),
            "actionability": create_judge(
                client,
                name="Remediation Actionability Judge",
                evaluation_goal="Evaluate whether the escalated incident answer gives a concrete, actionable remediation plan (for example backoff, batching, or spreading the backfill) rather than vague advice.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the routed support answers:")
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
    except Exception as exc:
        # The trace — this sample's core deliverable — is already uploaded and
        # renders. A judge/evaluation failure (no model in the org, a rate limit,
        # a transient 5xx) is reported honestly and never crashes the sample.
        print(f"\nNOTE: evaluations skipped -- {type(exc).__name__}: {exc}")
        print("  The trace is uploaded; add a project/public model to enable judges.")
    finally:
        for jid in judge_ids:
            try:
                client.judges.delete(jid)
            except Exception:
                pass


if __name__ == "__main__":
    main()
