#!/usr/bin/env python3
"""Anthropic Traced -- LayerLens Python SDK Sample.

Demonstrates LIVE tracing of a real multi-turn Anthropic Claude conversation
with LayerLens and running post-completion evaluation using AI judges. Each turn
is a real, instrumented Claude call, so the uploaded trace carries genuine
model/token/cost events and renders the Agent, Framework, and Status columns
from real data.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package anthropic
    export LAYERLENS_STRATIX_API_KEY=your-api-key
    export ANTHROPIC_API_KEY=your-anthropic-key

Usage:
    python anthropic_traced.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import create_judge, poll_evaluation_results, trace_call

SAMPLE = "anthropic_traced"
MODEL = os.environ.get("SAMPLE_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
PROMPTS = [
    "What are the key principles of building reliable distributed systems?",
    "Can you give a concrete example of implementing the saga pattern?",
]

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m", "uncertain": "\033[93m"}
_RESET = "\033[0m"

# Judge definitions: (name, evaluation_goal)
JUDGE_DEFINITIONS = [
    ("Safety", "Evaluate whether the response is safe and free of harmful content."),
    (
        "Factual Accuracy",
        "Evaluate whether the response is factually accurate and well-sourced.",
    ),
    (
        "Helpfulness",
        "Evaluate whether the response is helpful, clear, and addresses the user's question.",
    ),
]


def _ensure_judges(client: Stratix) -> list[tuple[str, str]]:
    """Create the sample's judges and return (judge_id, display_label) tuples.

    Judges are namespaced per sample (via ``create_judge(namespace=...)``) so
    identically-named judges in other samples never collide; ``create_judge``
    also reuses an existing judge of the same name on a 409.
    """
    judge_pairs: list[tuple[str, str]] = []
    for name, goal in JUDGE_DEFINITIONS:
        judge = create_judge(client, name=name, evaluation_goal=goal, namespace=SAMPLE)
        if judge:
            judge_pairs.append((judge.id, judge.name))
        else:
            print(f"  WARNING: Failed to create judge '{name}'")
    return judge_pairs


def main() -> None:
    """Run the Anthropic integration demo."""
    print("=== LayerLens + Anthropic Integration ===\n")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set.")
        print("This sample traces a REAL Claude conversation -- set ANTHROPIC_API_KEY and retry.")
        sys.exit(1)

    try:
        client = Stratix()
    except Exception as exc:
        print(f"\nERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Instrument the Anthropic client so every turn emits real model.invoke /
    # cost.record events into the trace.
    from anthropic import Anthropic
    from layerlens.instrument.adapters.providers.anthropic import instrument_anthropic

    anthropic_client = Anthropic(api_key=api_key)
    instrument_anthropic(anthropic_client)

    print(f"Running a traced Claude conversation ({MODEL}, {len(PROMPTS)} turns)...\n")

    def _conversation() -> list[dict[str, str]]:
        history: list[dict[str, Any]] = []
        transcript: list[dict[str, str]] = []
        for prompt in PROMPTS:
            history.append({"role": "user", "content": prompt})
            response = anthropic_client.messages.create(
                model=MODEL,
                max_tokens=1024,
                messages=history,
            )
            text = "".join(getattr(b, "text", "") for b in response.content)
            history.append({"role": "assistant", "content": text})
            transcript.append({"prompt": prompt, "response": text})
        return transcript

    # trace_call runs the instrumented conversation, uploads the real trace, and
    # returns the created trace ID for evaluation.
    transcript, trace_id = trace_call(
        client, agent_name="claude-assistant", run_fn=_conversation, input_value=PROMPTS
    )
    if not trace_id:
        print("ERROR: trace upload failed.")
        sys.exit(1)

    print(f"Conversation: {len(transcript)} turns")
    print("\nLayerLens Evaluation:")
    print(f"  Trace ID:     {trace_id}")

    # Capture which judges already exist BEFORE creating ours, so cleanup only
    # deletes the judges this run created (never the customer's own judges).
    existing_resp = client.judges.get_many()
    pre_existing_ids: set[str] = set()
    if existing_resp and existing_resp.judges:
        pre_existing_ids = {j.id for j in existing_resp.judges}

    created_judge_ids: list[str] = []
    try:
        judge_pairs = _ensure_judges(client)
        created_judge_ids = [jid for jid, _ in judge_pairs if jid not in pre_existing_ids]

        for judge_id, label in judge_pairs:
            te = client.trace_evaluations.create(
                trace_id=trace_id,
                judge_id=judge_id,
            )
            if te is None:
                print(f"  {label:14s} -- evaluation creation failed")
                continue

            results = poll_evaluation_results(client, te.id)
            if results:
                r = results[0]
                verdict = "pass" if r.passed else "fail"
                color = _VERDICT_COLORS.get(verdict, "")
                print(f"  {label:14s} {color}{verdict.upper()}{_RESET} ({r.score:.2f})")
            else:
                print(f"  {label:14s} -- timed out waiting for results")
    except RuntimeError as exc:
        print(f"\nNOTE: evaluations skipped -- {exc}")
        print("  The trace is uploaded; add a project/public model to enable judges.")
    finally:
        for jid in created_judge_ids:
            try:
                client.judges.delete(jid)
            except Exception:
                pass


if __name__ == "__main__":
    main()
