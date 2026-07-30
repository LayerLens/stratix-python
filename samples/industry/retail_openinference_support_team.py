"""LayerLens industry sample — Retail multi-agent support triage (OpenInference).

Uploads a RECORDED, real multi-agent OpenInference-ingested trace and evaluates
it with only a ``LAYERLENS_STRATIX_API_KEY`` — no provider key, no browser, no
network to any model. The trace was produced once by
``samples/data/generators/openinference.py::generate_openinference_multi``:

    support-triage-supervisor          (a real OpenInference AGENT span)
      ├─ warranty-specialist           (real AGENT span: real retrieval + real LLM)
      └─ returns-specialist            (real AGENT span: real retrieval + real LLM)

Each specialist did its own real retrieval over the real store-policy corpus and
made its own real ``gpt-4o-mini`` call; the supervisor synthesized their findings
in a third real call. The openinference adapter INGESTED those real OTel spans —
it patches nothing and makes no call of its own.

WHAT RENDERS: unlike the single-agent lane, this is a genuine multi-agent DAG.
The three AGENT spans carry three distinct declared identities, so the Agent
column shows ``multi-agent`` with the supervisor and both specialists as nodes
(the LLM/retriever span names and the model id are operations, never agents).
"""

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

SAMPLE = "retail_openinference_support_team"
FIXTURE = recorded_trace_path("industry", "retail_openinference_support_team.jsonl")

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
    # The three real AGENT spans the adapter ingested — the DAG nodes.
    "agents": ["support-triage-supervisor", "warranty-specialist", "returns-specialist"],
    "controlling_policies": ["POL-WAR-04", "POL-RET-03"],
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded multi-agent OpenInference-ingested support trace."""
    print("=== LayerLens Industry: Retail Multi-Agent Support Triage (OpenInference ingestion) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        print("  Set LAYERLENS_STRATIX_API_KEY to run this sample.")
        sys.exit(1)

    print(
        "Uploading the recorded multi-agent OpenInference trace "
        "(triage supervisor -> warranty + returns specialists)...\n"
    )
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
    print(f"  Agents:    {', '.join(CASE['agents'])}")
    print("  Agent:     multi-agent  (three real AGENT spans -> a multi-agent DAG;")
    print("                the LLM/retriever span names are operations, not agents)")
    print("  Framework: openinference")
    print("  Status:    from the real ingested span statuses\n")

    judge_ids: list[str] = []
    try:
        judges = {
            "delegation_soundness": create_judge(
                client,
                name="Delegation Soundness Judge",
                evaluation_goal=(
                    "Evaluate whether the triage supervisor's final answer faithfully combines the "
                    "warranty specialist's and returns specialist's findings — the manufacturing-defect "
                    "warranty applying to the split seam, and the expired 30-day return window — rather "
                    "than contradicting either specialist or inventing a conclusion neither reached."
                ),
                namespace=SAMPLE,
            ),
            "warranty_correctness": create_judge(
                client,
                name="Warranty Correctness Judge",
                evaluation_goal=(
                    "Evaluate whether the final answer correctly concludes a refund IS available under the "
                    "12-month manufacturing-defect warranty despite the expired 30-day return window, and "
                    "that return shipping is free because the item failed under warranty."
                ),
                namespace=SAMPLE,
            ),
            "completeness": create_judge(
                client,
                name="Answer Completeness Judge",
                evaluation_goal=(
                    "Evaluate whether the final answer addresses BOTH parts of the customer's question: "
                    "whether a refund is available, and who pays for the return shipping."
                ),
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the supervisor's synthesized answer:")
        for label, judge in judges.items():
            evaluation = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge.id)
            if evaluation is None:
                print(f"  {label:24s} -- evaluation creation failed")
                continue
            results = poll_evaluation_results(client, evaluation.id)
            if results:
                r = results[0]
                verdict = "pass" if r.passed else "fail"
                color = _VERDICT_COLORS.get(verdict, "")
                print(f"  {label:24s} {color}{verdict.upper()}{_RESET} ({r.score:.2f})")
            else:
                print(f"  {label:24s} -- timed out waiting for results")
    except RuntimeError as exc:
        print(f"\nNOTE: evaluations skipped -- {exc}")
        print("  The trace is uploaded; add a project/public model to enable judges.")
    finally:
        for jid in judge_ids:
            try:
                client.judges.delete(jid)
            except Exception:
                pass

    print("\nDone.")


if __name__ == "__main__":
    main()
