#!/usr/bin/env python3
"""OpenAI Traced -- LayerLens Python SDK Sample.

Demonstrates LIVE tracing of a real OpenAI API call with LayerLens and running
post-completion evaluation using AI judges. The OpenAI call is instrumented, so
the uploaded trace carries genuine model/token/cost events and renders the
Agent, Framework, and Status columns from real data.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package openai
    export LAYERLENS_STRATIX_API_KEY=your-api-key
    export OPENAI_API_KEY=your-openai-key

Usage:
    python openai_traced.py
"""

from __future__ import annotations

import os
import sys

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import create_judge, poll_evaluation_results, trace_call

SAMPLE = "openai_traced"
MODEL = os.environ.get("SAMPLE_OPENAI_MODEL", "gpt-4o-mini")
PROMPT = "Explain the CAP theorem in distributed systems."

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
    """Run the OpenAI integration demo."""
    print("=== LayerLens + OpenAI Integration ===\n")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.")
        print("This sample traces a REAL OpenAI call -- set OPENAI_API_KEY and retry.")
        sys.exit(1)

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Instrument the OpenAI client so the call below emits real model.invoke /
    # cost.record events into the trace.
    from openai import OpenAI
    from layerlens.instrument.adapters.providers.openai import instrument_openai

    openai_client = OpenAI(api_key=api_key)
    instrument_openai(openai_client)

    print(f"Running a traced OpenAI completion ({MODEL})...\n")

    def _call() -> str:
        completion = openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT}],
        )
        return completion.choices[0].message.content or ""

    # trace_call runs the instrumented completion, uploads the real trace, and
    # returns the created trace ID for evaluation.
    response_text, trace_id = trace_call(
        client, agent_name="openai-assistant", run_fn=_call, input_value=PROMPT
    )
    if not trace_id:
        print("ERROR: trace upload failed.")
        sys.exit(1)

    print(f'Prompt:   "{PROMPT}"')
    print(f"Response: {response_text[:80]}{'...' if len(response_text) > 80 else ''}")
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
