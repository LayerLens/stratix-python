#!/usr/bin/env python3
"""Industry: Retail RAG Customer Support (OpenInference ingestion) -- LayerLens SDK Sample.

Demonstrates evaluating a trace that LayerLens **ingested** rather than produced.
OpenInference is the Arize set of OpenTelemetry semantic conventions +
auto-instrumentation libraries; the ``openinference`` adapter patches nothing and
calls no model -- it consumes the spans an OpenInference-instrumented app already
emits (live via ``adapter.span_processor()``, or offline via ``ingest_spans``).
That means ANY OpenInference-instrumented app gets LayerLens coverage with no
per-framework work.

THE RECORDED RUN (all real -- nothing fabricated)
------------------------------------------------
A footwear e-commerce support assistant answers a real warranty question: a
customer's boots split at the seam ~4 months after delivery, so the 30-day return
window has expired but the 12-month manufacturing-defect warranty still applies
and makes return shipping free. The answer is only correct if retrieval surfaces
the right policy -- a genuine RAG task, not a toy.

The shipped trace was captured from ONE real instrumented run (see
``samples/data/generators/openinference.py``) in which:

* the LLM span was emitted by the REAL ``openinference-instrumentation-openai``
  auto-instrumentor around a REAL OpenAI ``gpt-4o-mini`` call -- so ``model.invoke``
  carries the resolved dated model id and the real 372/80/452 token counts, and
  ``cost.record`` is really DERIVED from those counts (never a fabricated 0.0);
* the AGENT / TOOL / RETRIEVER spans came from the real OpenInference ``OITracer``
  wrapping the run's real order-lookup and real policy-retrieval steps.

So the trace renders the real span topology the run actually had --
``retail_support_agent`` over ``order_lookup`` (tool.call), ``policy_retriever``
(retrieval.query, 3 real documents), and ``ChatCompletion`` (model.invoke +
cost.record) -- with the real source OTel trace/span ids preserved, so it still
correlates back to the producer's own telemetry.

HONESTY NOTE -- the Agent column is deliberately EMPTY
-----------------------------------------------------
OpenInference is an INGESTION surface, not an agent framework. An OpenInference
AGENT span declares its identity only as a span NAME, and LayerLens's
``_identity.py`` forbids a span name as an Agent-column source (a span name is not
an attested agent identity). So the adapter keeps the name in ``agent_id`` and
never writes ``agent_name``: the Agent column renders an honest ``—`` rather than
an invented agent. The Framework column is ``openinference``, the Status comes
from the real span statuses (the real LLM span reported OK), and the graph is the
real ingested span tree. Nothing is fabricated to fill a column.

This sample uploads that recorded trace with ONLY a LayerLens API key -- no
provider key is needed or used, because no model is called here.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python retail_openinference_support.py
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

SAMPLE = "retail_openinference_support"
FIXTURE = recorded_trace_path("industry", "retail_openinference_support.jsonl")

# The support case the instrumented RAG assistant handled. Documents the scenario;
# the recorded trace was produced by running exactly this through a real
# OpenInference-instrumented run whose spans the adapter then ingested.
CASE: dict[str, Any] = {
    "order_id": "SO-884213",
    "item": "Summit Trail Waterproof Hiking Boot, M9",
    "delivered_on": "2026-03-19",
    "customer_message": (
        "I ordered the Summit Trail boots back in March — about four months ago — "
        "and the seam along the left heel has completely split open. They were not "
        "final sale. Can I still get a refund, and do I have to pay for the return "
        "shipping?"
    ),
    # The spans the adapter ingested, in the order the real run produced them.
    "spans": ["order_lookup", "policy_retriever", "ChatCompletion", "retail_support_agent"],
    "controlling_policies": ["POL-WAR-04", "POL-RET-03"],
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded OpenInference-ingested retail-support RAG trace."""
    print("=== LayerLens Industry: Retail RAG Customer Support (OpenInference ingestion) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        print("  Set LAYERLENS_STRATIX_API_KEY to run this sample.")
        sys.exit(1)

    # Upload the recorded ingested trace first (renders the real span tree with
    # the real tool.call / retrieval.query / model.invoke / cost.record). Do this
    # before creating judges so the trace always lands even if the org has no
    # evaluation model yet.
    print("Uploading the recorded OpenInference trace (order_lookup -> policy_retriever -> ChatCompletion)...\n")
    try:
        trace_ids = upload_recorded_trace(client, FIXTURE)
    except Exception as exc:
        print(f"ERROR: failed to upload the recorded trace: {exc}")
        sys.exit(1)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID:  {trace_id}")
    print(f"  Order:     {CASE['order_id']}  ({CASE['item']}, delivered {CASE['delivered_on']})")
    print(f"  Spans:     {' -> '.join(CASE['spans'][:3])}")
    print("  Agent:     —  (OpenInference is an ingestion surface; a span name is not")
    print("                an attested agent identity, so no agent is invented)")
    print("  Framework: openinference")
    print("  Status:    from the real ingested span statuses (LLM span reported OK)\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "policy_grounding": create_judge(
                client,
                name="Policy Grounding Judge",
                evaluation_goal="Evaluate whether the support answer is grounded in the store policy excerpts the retriever actually returned and cites the policy ids it relied on, rather than inventing policy terms.",
                namespace=SAMPLE,
            ),
            "warranty_correctness": create_judge(
                client,
                name="Warranty Correctness Judge",
                evaluation_goal="Evaluate whether the answer correctly applies the 12-month manufacturing-defect warranty to a split seam reported about four months after delivery -- concluding a refund IS available even though the 30-day return window has expired, and that return shipping is free because the item failed under warranty.",
                namespace=SAMPLE,
            ),
            "completeness": create_judge(
                client,
                name="Answer Completeness Judge",
                evaluation_goal="Evaluate whether the answer addresses BOTH parts of the customer's question: whether a refund is available, and who pays for the return shipping.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the support answer:")
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
        print("  The trace is uploaded; add a project/public model to enable judges.")
    finally:
        for jid in judge_ids:
            try:
                client.judges.delete(jid)
            except Exception:
                pass


if __name__ == "__main__":
    main()
