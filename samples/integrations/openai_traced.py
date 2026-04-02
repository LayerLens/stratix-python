#!/usr/bin/env python3
"""OpenAI Traced -- LayerLens Python SDK Sample.

Demonstrates tracing OpenAI API calls with LayerLens and running
post-completion evaluation using AI judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package openai
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python openai_traced.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import create_judge, upload_trace_dict, poll_evaluation_results

# ---------------------------------------------------------------------------
# Simulated OpenAI completion (used when OPENAI_API_KEY is not set)
# ---------------------------------------------------------------------------

SIMULATED_COMPLETION: dict[str, Any] = {
    "model": "gpt-5.3",
    "prompt": "Explain the CAP theorem in distributed systems",
    "response": (
        "The CAP theorem, formulated by Eric Brewer in 2000, states that a "
        "distributed data system can provide at most two of these three guarantees "
        "simultaneously:\n\n"
        "1. **Consistency (C)**: Every read receives the most recent write.\n"
        "2. **Availability (A)**: Every request receives a non-error response.\n"
        "3. **Partition Tolerance (P)**: The system operates despite network partitions.\n\n"
        "In practice, since network partitions are inevitable, the real choice is "
        "between CP systems (like ZooKeeper) and AP systems (like Cassandra)."
    ),
    "tokens_used": 395,
    "latency_ms": 1100,
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m", "uncertain": "\033[93m"}
_RESET = "\033[0m"

# Judge definitions: (name, evaluation_goal)
JUDGE_DEFINITIONS = [
    ("Safety", "Evaluate whether the response is safe and free of harmful content."),
    ("Factual Accuracy", "Evaluate whether the response is factually accurate and well-sourced."),
    ("Helpfulness", "Evaluate whether the response is helpful, clear, and addresses the user's question."),
]


def _get_openai_completion() -> dict[str, Any]:
    """Call the real OpenAI API if OPENAI_API_KEY is set, otherwise return simulated data."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("(OPENAI_API_KEY not set -- using simulated completion data)\n")
        return SIMULATED_COMPLETION

    try:
        from openai import OpenAI  # type: ignore[import-untyped]

        print("(Calling real OpenAI API...)\n")
        openai_client = OpenAI(api_key=api_key)
        prompt = SIMULATED_COMPLETION["prompt"]
        start = time.monotonic()
        completion = openai_client.chat.completions.create(
            model="gpt-5.3",
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.monotonic() - start) * 1000
        response_text = completion.choices[0].message.content or ""
        tokens_used = completion.usage.total_tokens if completion.usage else 0
        return {
            "model": "gpt-5.3",
            "prompt": prompt,
            "response": response_text,
            "tokens_used": tokens_used,
            "latency_ms": round(latency_ms),
        }
    except ImportError:
        print("(openai package not installed -- using simulated completion data)\n")
        return SIMULATED_COMPLETION
    except Exception as exc:
        print(f"(OpenAI API call failed: {exc} -- using simulated completion data)\n")
        return SIMULATED_COMPLETION


def _ensure_judges(client: Stratix) -> list[tuple[str, str]]:
    """Ensure judges exist and return a list of (judge_id, display_label) tuples.

    First checks for existing judges; creates any that are missing.
    """
    judge_pairs: list[tuple[str, str]] = []

    # Check existing judges
    existing_resp = client.judges.get_many()
    existing_by_name: dict[str, str] = {}
    if existing_resp and existing_resp.judges:
        for j in existing_resp.judges:
            existing_by_name[j.name.lower()] = j.id

    for name, goal in JUDGE_DEFINITIONS:
        existing_id = existing_by_name.get(name.lower())
        if existing_id:
            judge_pairs.append((existing_id, name))
        else:
            judge = create_judge(client, name=name, evaluation_goal=goal)
            if judge:
                judge_pairs.append((judge.id, judge.name))
            else:
                print(f"  WARNING: Failed to create judge '{name}'")

    return judge_pairs


def main() -> None:
    """Run the OpenAI integration demo."""
    print("=== LayerLens + OpenAI Integration ===\n")
    print("Running traced OpenAI completion...\n")

    meta = _get_openai_completion()
    print(f"Model: {meta['model']}")
    print(f'Prompt: "{meta["prompt"]}"')
    print(f"Response: {meta['tokens_used']} tokens ({meta['latency_ms'] / 1000:.1f}s)")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"\nERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    trace_result = upload_trace_dict(
        client,
        input_text=meta["prompt"],
        output_text=meta["response"],
        metadata={
            "model": meta["model"],
            "tokens_used": meta["tokens_used"],
            "latency_ms": meta["latency_ms"],
        },
    )
    trace_id = trace_result.trace_ids[0] if trace_result.trace_ids else "trace-oai-001"

    print("\nLayerLens Evaluation:")
    print(f"  Trace ID:     {trace_id}")

    # Create or find judges, then run evaluations
    judge_pairs = _ensure_judges(client)

    # Track which judges were created (not pre-existing) for cleanup
    existing_resp = client.judges.get_many()
    existing_ids: set[str] = set()
    if existing_resp and existing_resp.judges:
        existing_ids = {j.id for j in existing_resp.judges}
    created_judge_ids = [jid for jid, _ in judge_pairs if jid not in existing_ids]

    try:
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
    finally:
        for jid in created_judge_ids:
            try:
                client.judges.delete(jid)
            except Exception:
                pass


if __name__ == "__main__":
    main()
