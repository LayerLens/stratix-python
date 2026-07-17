#!/usr/bin/env python3
"""Industry: Insurance FNOL Intake (Mirascope typed extraction) -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL structured first-notice-of-loss (FNOL) intake built
with the ``mirascope`` framework. An auto-insurance ``fnol_intake_agent`` -- a plain
Python function decorated with ``@llm.call(..., format=FirstNoticeOfLoss)`` -- reads
the free-text intake notes a call-centre rep typed while a policyholder reported a
collision, and returns the structured FNOL record a claims system can actually open
a claim from: policy number, claimant, loss date/type/location, insured vehicle,
damage, injuries, police report, and a triage severity.

Because the trace was captured through the real ``MirascopeAdapter`` (which wraps
mirascope v2's ``Call`` classes), it renders a single honest agent node (Agent
column = ``fnol_intake_agent``, resolved from the decorated function's own name)
plus the real ``model.invoke`` / ``cost.record`` / ``tool.call`` / ``tool.result``
events of the typed extraction turn -- no fabrication. The recorded run used a real
local ``ollama/llama3:8b``, so ``model.invoke`` carries the genuine model id,
provider and token counts; there is no ``cost_usd`` because a local model has no
billed cost, and inventing one would be a lie.

The recorded trace is shipped under samples/data/traces/industry/ (produced by
samples/data/generators/mirascope.py); this sample uploads it and evaluates the
extraction with domain judges. No provider key is needed -- only a LayerLens key.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python insurance_mirascope_fnol_intake.py
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

SAMPLE = "insurance_mirascope_fnol_intake"
FIXTURE = recorded_trace_path("industry", "insurance_mirascope_fnol_intake.jsonl")

# The loss the intake agent processed. Documents the scenario; the recorded trace
# was produced by running these verbatim intake notes through a real mirascope
# ``@llm.call`` with a typed ``FirstNoticeOfLoss`` output spec.
LOSS: dict[str, Any] = {
    "policy_number": "AUTO-TX-4482910",
    "claimant": "Denise Okonkwo",
    "loss_type": "rear-end collision",
    "reported_via": "inbound call to the claims line",
    "description": (
        "Policyholder stopped at a light on Lamar Blvd, Austin was rear-ended by a "
        "pickup; rear end crushed, whiplash reported, APD report filed."
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded mirascope FNOL typed-extraction trace."""
    print("=== LayerLens Industry: Insurance FNOL Intake (Mirascope typed extraction) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        print("  Set LAYERLENS_STRATIX_API_KEY to run this sample.")
        sys.exit(1)

    # Upload the recorded extraction trace first (renders a single fnol_intake_agent
    # node with real model.invoke / tool.call / tool.result events). Do this before
    # creating judges so the trace always lands even if the org has no evaluation
    # model yet.
    print("Uploading the recorded FNOL intake trace (typed extraction turn)...\n")
    try:
        trace_ids = upload_recorded_trace(client, FIXTURE)
    except Exception as exc:
        print(f"ERROR: could not upload the recorded trace: {exc}")
        sys.exit(1)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Loss:     {LOSS['loss_type']} on policy {LOSS['policy_number']} ({LOSS['claimant']})\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "extraction_accuracy": create_judge(
                client,
                name="FNOL Extraction Accuracy Judge",
                evaluation_goal="Evaluate whether the extracted FNOL fields (policy number, claimant, loss date, location, vehicle, police report number) match the facts stated in the policyholder's intake notes.",
                namespace=SAMPLE,
            ),
            "no_fabrication": create_judge(
                client,
                name="FNOL No-Fabrication Judge",
                evaluation_goal="Evaluate whether every extracted field is grounded in a fact the caller actually stated, penalising any invented policy number, report number, date, or injury the notes do not support.",
                namespace=SAMPLE,
            ),
            "severity_triage": create_judge(
                client,
                name="FNOL Severity Triage Judge",
                evaluation_goal="Evaluate whether the assigned triage severity correctly reflects the reported injury and vehicle damage, since a reported injury should escalate the claim.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the FNOL extraction:")
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
