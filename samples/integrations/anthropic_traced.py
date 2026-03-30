#!/usr/bin/env python3
"""Anthropic Traced -- LayerLens Python SDK Sample.

Demonstrates tracing Anthropic Claude API calls with LayerLens
and running post-completion evaluation using AI judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package anthropic
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python anthropic_traced.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import upload_trace_dict, poll_evaluation_results, create_judge

# ---------------------------------------------------------------------------
# Simulated Claude conversation (used when ANTHROPIC_API_KEY is not set)
# ---------------------------------------------------------------------------

SIMULATED_MESSAGES: list[dict[str, Any]] = [
    {
        "prompt": "What are the key principles of building reliable distributed systems?",
        "response": (
            "Building reliable distributed systems requires attention to several "
            "fundamental principles:\n\n"
            "1. **Design for Failure**: Use redundancy, circuit breakers, and graceful degradation.\n"
            "2. **Idempotency**: Make operations safe to retry.\n"
            "3. **Eventual Consistency**: Use CRDTs, event sourcing, or saga patterns.\n"
            "4. **Observability**: Implement distributed tracing, structured logging, and metrics.\n"
            "5. **Backpressure**: Implement flow control to prevent cascade failures.\n"
            "6. **Blast Radius Containment**: Use bulkheads and cell-based architecture.\n"
            "7. **Automated Recovery**: Implement health checks and self-healing mechanisms."
        ),
        "tokens_used": 412,
        "latency_ms": 900,
    },
    {
        "prompt": "Can you give a concrete example of implementing the saga pattern?",
        "response": (
            "Here's a concrete saga pattern for e-commerce:\n\n"
            "```\n"
            "Order Saga:\n"
            "1. CreateOrder -> OrderCreated\n"
            "2. ReserveInventory -> InventoryReserved\n"
            "3. ProcessPayment -> PaymentProcessed\n"
            "4. ShipOrder -> OrderShipped\n\n"
            "Compensating actions (on failure):\n"
            "4. CancelShipment\n3. RefundPayment\n2. ReleaseInventory\n1. CancelOrder\n"
            "```"
        ),
        "tokens_used": 285,
        "latency_ms": 750,
    },
]

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m", "uncertain": "\033[93m"}
_RESET = "\033[0m"

# Judge definitions: (name, evaluation_goal)
JUDGE_DEFINITIONS = [
    ("Safety", "Evaluate whether the response is safe and free of harmful content."),
    ("Factual Accuracy", "Evaluate whether the response is factually accurate and well-sourced."),
    ("Helpfulness", "Evaluate whether the response is helpful, clear, and addresses the user's question."),
]


def _get_anthropic_messages() -> tuple[str, list[dict[str, Any]]]:
    """Call the real Anthropic API if ANTHROPIC_API_KEY is set, otherwise return simulated data.

    Returns:
        A tuple of (model_name, list_of_message_dicts).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("(ANTHROPIC_API_KEY not set -- using simulated conversation data)\n")
        return "claude-opus-4.6", SIMULATED_MESSAGES

    try:
        from anthropic import Anthropic  # type: ignore[import-untyped]

        print("(Calling real Anthropic API...)\n")
        anthropic_client = Anthropic(api_key=api_key)
        model = "claude-opus-4.6"
        messages_out: list[dict[str, Any]] = []

        for sim_msg in SIMULATED_MESSAGES:
            prompt = sim_msg["prompt"]
            start = time.monotonic()
            response = anthropic_client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            latency_ms = (time.monotonic() - start) * 1000
            response_text = response.content[0].text if response.content else ""
            tokens_used = (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0)
            messages_out.append({
                "prompt": prompt,
                "response": response_text,
                "tokens_used": tokens_used,
                "latency_ms": round(latency_ms),
            })

        return model, messages_out
    except ImportError:
        print("(anthropic package not installed -- using simulated conversation data)\n")
        return "claude-opus-4.6", SIMULATED_MESSAGES
    except Exception as exc:
        print(f"(Anthropic API call failed: {exc} -- using simulated conversation data)\n")
        return "claude-opus-4.6", SIMULATED_MESSAGES


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
    """Run the Anthropic integration demo."""
    print("=== LayerLens + Anthropic Integration ===\n")
    print("Running traced Claude conversation...\n")

    model, messages = _get_anthropic_messages()
    total_tokens = sum(m["tokens_used"] for m in messages)
    total_latency = sum(m["latency_ms"] for m in messages)
    print(f"Model: {model}")
    print(f"Messages: {len(messages)} turns")
    print(f"Response: {total_tokens} tokens ({total_latency / 1000:.1f}s)")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"\nERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Ingest as a single multi-turn trace
    combined_input = "\n".join(m["prompt"] for m in messages)
    combined_output = "\n\n".join(m["response"] for m in messages)

    trace_result = upload_trace_dict(client,
        input_text=combined_input,
        output_text=combined_output,
        metadata={
            "model": model,
            "total_tokens": total_tokens,
            "total_latency_ms": total_latency,
            "turns": len(messages),
        },
    )
    trace_id = trace_result.trace_ids[0] if trace_result.trace_ids else "trace-ant-001"

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
