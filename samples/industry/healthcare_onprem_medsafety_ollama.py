#!/usr/bin/env python3
"""Industry: On-Prem Medication-Safety Tool-Use Loop (Ollama) -- LayerLens SDK Sample.

Demonstrates evaluating a REAL two-turn tool-use loop on a local LLM. An
on-premise medication-safety assistant -- backed by a local Ollama ``llama3:8b``
model inside the hospital network -- reconciles a newly proposed prescription
(amiodarone) against a synthetic patient's active medication list:

  1. (model.invoke) the assistant names the drug pairs that must be verified,
  2. (tool.result) a REAL deterministic ``check_interactions`` tool queries a
     genuine drug-interaction reference and returns the significant findings,
  3. (model.invoke) the assistant grounds its final medication-safety assessment
     on the verified findings (overall risk verdict + management + monitoring).

Ollama is a **provider** adapter, not an agent framework: the trace carries the
two real ``model.invoke`` events (framework=ollama, model=llama3:8b, real token
counts) + the ``tool.result`` (real reference data, not model-fabricated) +
``cost.record`` events (``cost_usd = None`` -- a local model incurs no API cost),
but declares NO agent. So the trace renders the HONEST empty-state: the Agent
column is ``—`` (this is a provider tool-use LOOP, NOT a multi-agent graph) with
the real OTel span waterfall (``trace.root`` -> model.invoke -> tool.result ->
model.invoke). The recorded trace is shipped under samples/data/traces/industry/
(produced by samples/data/generators/ollama.py); this sample uploads it and
evaluates the medication-safety assessment with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python healthcare_onprem_medsafety_ollama.py
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

SAMPLE = "healthcare_onprem_medsafety_ollama"
FIXTURE = recorded_trace_path("industry", "healthcare_onprem_medsafety_ollama.jsonl")

# The synthetic medication-reconciliation scenario the assistant worked through.
# Documents the scenario; the recorded trace is what the real local llama3:8b
# tool-use loop produced (the check_interactions tool really ran).
SCENARIO: dict[str, Any] = {
    "case_id": "MEDSAFE-4402",
    "active_medications": ["warfarin", "lisinopril", "atorvastatin"],
    "proposed_new_medication": "amiodarone (new-onset atrial fibrillation)",
    "expected_flags": [
        "warfarin + amiodarone (major — raises INR/bleeding risk)",
        "atorvastatin + amiodarone (moderate — myopathy/rhabdomyolysis risk)",
    ],
    "deployment": "on-premise Ollama (llama3:8b) — no patient data leaves the hospital network",
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded on-prem medication-safety tool-use trace."""
    print("=== LayerLens Industry: On-Prem Medication-Safety Tool-Use Loop (Ollama) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded provider tool-use trace first (renders the honest
    # empty-state: Agent ``—``, a provider LOOP not a DAG, with the real
    # model.invoke / tool.result / cost.record waterfall). Do this before creating
    # judges so the trace always lands even if the org has no evaluation model yet.
    print("Uploading the recorded on-prem medication-safety trace (llama3:8b tool loop)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Case:     {SCENARIO['case_id']}")
    print(f"  New drug: {SCENARIO['proposed_new_medication']}")
    print(f"  Runs on:  {SCENARIO['deployment']}\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "interaction_detection": create_judge(
                client,
                name="Interaction Detection Judge",
                evaluation_goal="Evaluate whether the final assessment flags the clinically significant warfarin+amiodarone and atorvastatin+amiodarone interactions from the verified findings.",
                namespace=SAMPLE,
            ),
            "tool_grounding": create_judge(
                client,
                name="Tool Grounding Judge",
                evaluation_goal="Evaluate whether the assessment is grounded in the check_interactions tool's returned findings rather than the model inventing or omitting interactions.",
                namespace=SAMPLE,
            ),
            "actionable_safety": create_judge(
                client,
                name="Actionable Safety Judge",
                evaluation_goal="Evaluate whether the assessment gives a clear risk verdict with concrete, safe management and monitoring guidance (e.g. warfarin dose reduction + INR monitoring).",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the medication-safety assessment:")
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


if __name__ == "__main__":
    main()
