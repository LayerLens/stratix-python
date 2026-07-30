#!/usr/bin/env python3
"""Industry: Financial-services A2A multi-hop dispute delegation -- LayerLens SDK Sample.

Demonstrates observing a genuine **multi-hop Agent-to-Agent (A2A) delegation
graph**. A ``dispute-orchestrator`` delegates a customer refund dispute to a
``billing-specialist`` over A2A, which in turn delegates the ledger posting to a
``ledger-adjuster`` -- two real ``a2a.delegation`` edges forming a 3-node
delegation DAG:

    dispute-orchestrator -> billing-specialist -> ledger-adjuster

A2A is an **LLM-free protocol** surface, so the trace carries NO model/token/cost
data; it renders the delegation topology instead. Each hop records:

* an ``a2a.agent.discovered`` event with the delegatee's card skills + signature
  PROVENANCE (present + keyed-HMAC fingerprint -- never the raw JWS),
* an ``a2a.delegation`` edge (from_agent -> to_agent) that the graph engine
  derives into the multi-hop agent DAG (exactly like ``agent.handoff``),
* the ``a2a.task.created`` / ``a2a.task.updated`` lifecycle with the real
  terminal ``TaskState`` (both hops end ``completed``).

The recorded trace was captured under the real ``A2AProtocolAdapter`` from genuine
``a2a-sdk`` v1.1.0 model objects (real signed ``AgentCard``s + real ``Task``s at
``TaskState.completed``). Nothing is fabricated: the Framework column shows
``a2a`` (the protocol that ran), the Agent column renders the multi-agent
delegation graph, and the Status reflects the real terminal states. It is shipped
under samples/data/traces/industry/ (produced by
samples/data/_generate_fixtures.py); this sample uploads it and evaluates whether
the multi-hop delegation chain routed each sub-task to the right specialist.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python financial_a2a_dispute_delegation.py
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

SAMPLE = "financial_a2a_dispute_delegation"
FIXTURE = recorded_trace_path("industry", "financial_a2a_dispute_delegation.jsonl")

# The multi-hop delegation the orchestrator ran over A2A. Documents the scenario;
# the recorded trace was produced by driving both hops through the real
# A2AProtocolAdapter (discover card + send_task, twice).
DELEGATION: dict[str, Any] = {
    "dispute_id": "DSP-20260712-4471",
    "account": "ACME-4471",
    "invoice": "INV-20260712",
    "amount_usd": 79.99,
    "reason": "duplicate charge (billed twice on 2026-07-12)",
    "chain": ["dispute-orchestrator", "billing-specialist", "ledger-adjuster"],
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Upload and evaluate the recorded multi-hop A2A delegation trace."""
    print("=== LayerLens Industry: Financial-services A2A multi-hop dispute delegation ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded multi-hop delegation trace first (renders the 3-node
    # orchestrator -> billing -> ledger DAG + per-hop card provenance + completed
    # statuses). Do this before judges so the trace always lands.
    print("Uploading the recorded multi-hop A2A delegation trace "
          "(orchestrator -> billing-specialist -> ledger-adjuster)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID:   {trace_id}")
    print(f"  Dispute:    {DELEGATION['dispute_id']} (${DELEGATION['amount_usd']} on {DELEGATION['invoice']})")
    print(f"  Delegation: {' -> '.join(DELEGATION['chain'])} (both hops completed)\n")

    # Judges evaluate the multi-hop delegation itself (A2A is LLM-free -- the
    # delegation topology + skill descriptions are the signal).
    judge_ids: list[str] = []
    try:
        judges = {
            "delegation_chain": create_judge(
                client,
                name="A2A Delegation Chain Judge",
                evaluation_goal="Evaluate whether the multi-hop delegation routed each sub-task to the correct specialist: the refund dispute to a billing specialist, and the resulting ledger credit posting onward to a ledger-adjustment agent.",
                namespace=SAMPLE,
            ),
            "card_provenance": create_judge(
                client,
                name="A2A Card Provenance Judge",
                evaluation_goal="Evaluate whether each delegatee agent's capability card was discovered (with a present signature) before a task was delegated to it, so delegation targets were verified rather than assumed.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the multi-hop delegation:")
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
