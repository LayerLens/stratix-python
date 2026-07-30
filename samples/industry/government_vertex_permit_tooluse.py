#!/usr/bin/env python3
"""Industry: Government Building-Permit Determination (Google Vertex / Gemini, tool-use) -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL Google Vertex (Gemini) **function-call loop**. A
resident describes a home-addition project; the Gemini model first calls a
``lookup_permit_requirements`` tool to fetch the jurisdiction's current permit
rules, then returns whether a permit is required, which permit(s), the review
timeline, and what to submit.

Google Vertex AI is credential-gated (no GCP project/service-account is available
in CI or dev), so the trace is recorded SEALED: the real ``GoogleVertexProvider``
adapter runs against real proto-backed ``vertexai`` ``GenerationResponse`` objects
(a first turn carrying a genuine ``function_call`` part, a second carrying the
final answer). The adapter surfaces the model's request as a real ``tool.call``,
the local tool genuinely runs and is recorded as a ``tool.result``, and both
turns emit a genuine ``framework=google_vertex`` ``model.invoke`` + priced
``cost.record`` (model ``gemini-1.5-flash-002``) with an intact attestation chain
-- only the LLM network is sealed (see samples/data/generators/google_vertex.py,
and ``metadata.sealed``/``captured_at="pending-creds"`` on the fixture).

``google_vertex`` is a **provider** (a raw model call, not an agent framework),
so the trace declares no agent name and the Agent column renders the honest
empty-state (--) with a span waterfall (tool.call + tool.result are shown in the
timeline) -- nothing is fabricated as an agent.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python government_vertex_permit_tooluse.py
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

SAMPLE = "government_vertex_permit_tooluse"
FIXTURE = recorded_trace_path("industry", "government_vertex_permit_tooluse.jsonl")

# The resident's project the Gemini permit assistant assessed. Documents the
# scenario; the recorded trace was produced by driving this through the real
# Google Vertex adapter as a function-call loop (sealed over proto responses --
# see samples/data/generators/google_vertex.py).
PROJECT: dict[str, Any] = {
    "channel": "city_permits_portal",
    "jurisdiction": "City of Madison, WI",
    "project_type": "residential_addition",
    "summary": (
        "A 240 sq ft single-story addition (new bedroom) on the back of a "
        "single-family home, with a new concrete foundation, tying into the "
        "existing electrical and HVAC. Asks whether a permit is required, which "
        "permits, the review timeline, and what to submit."
    ),
    "tool": "lookup_permit_requirements",
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded Google Vertex (Gemini) permit-determination tool-use trace."""
    print("=== LayerLens Industry: Government Building-Permit Determination (Google Vertex / Gemini, tool-use) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded real trace first (renders Framework = google_vertex with
    # two real model.invoke / cost.record events + a tool.call/tool.result loop;
    # Agent column = -- empty-state, since a provider has no agent). Do this before
    # creating judges so the trace always lands even if the org has no model yet.
    print("Uploading the recorded Google Vertex (Gemini) permit-determination trace (sealed, tool-use)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID:      {trace_id}")
    print(f"  Jurisdiction:  {PROJECT['jurisdiction']}")
    print(f"  Project type:  {PROJECT['project_type']}  (tool: {PROJECT['tool']})\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "determination": create_judge(
                client,
                name="Permit Determination Judge",
                evaluation_goal="Evaluate whether the response correctly determines that a permit is required for a foundation-bearing residential addition and identifies the relevant permit types.",
                namespace=SAMPLE,
            ),
            "grounded_in_tool": create_judge(
                client,
                name="Tool-Grounded Answer Judge",
                evaluation_goal="Evaluate whether the answer is grounded in the looked-up permit requirements (review timeline and submittal list) rather than invented.",
                namespace=SAMPLE,
            ),
            "actionability": create_judge(
                client,
                name="Actionable Submittals Judge",
                evaluation_goal="Evaluate whether the response gives the resident concrete, actionable next steps and a clear list of what to submit.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the permit-determination response:")
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
