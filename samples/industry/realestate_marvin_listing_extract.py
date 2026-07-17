#!/usr/bin/env python3
"""Industry: Real-Estate MLS Listing Intake (Marvin structured extraction) -- LayerLens SDK Sample.

Demonstrates evaluating a REAL Marvin 3.x structured-extraction run. A listing
coordinator pastes an agent's freeform property write-up into the MLS, and a
``listing-extraction-agent`` (a real ``marvin.Agent`` backed by OpenAI
gpt-4o-mini) normalizes it with two real Marvin primitives:

* ``marvin.cast``    -> a typed ``PropertyListing`` record (address, property
  type, beds, baths, square feet, list price, year built, garage, lot size, HOA)
* ``marvin.extract`` -> the marketable feature/amenity list the same prose
  supports

Because the trace was captured through the real ``MarvinAdapter`` (which patches
Marvin's module-level primitives), it renders the agent node honestly: the Agent
column is ``listing-extraction-agent`` (the name the developer declared on the
``marvin.Agent``), the Framework column is ``marvin``, the Status column is the
real run outcome, and each primitive contributes a real ``tool.call`` +
``model.invoke`` carrying the genuinely extracted values. Nothing is fabricated.

HONESTY NOTE -- NO TOKENS AND NO ``cost.record``, ON PURPOSE. Marvin surfaces no
usage on its primitives, so there are no token counts at this layer and the
pricing hook has nothing real to price. The adapter therefore omits them rather
than inventing a figure: this trace shows real latency and a real model
(``gpt-4o-mini``, read off ``marvin.Agent.model``) with the token/cost fields
genuinely absent. That absence is the truth about Marvin's instrumentation
surface, not a gap in the recording -- the underlying provider response really
did carry usage, and the recorded-corpus gate
(``tests/instrument/adapters/frameworks/test_marvin_recorded.py``) pins exactly
that: real tokens in the body, none invented at Marvin's layer.

HONESTY NOTE -- FRAGMENTED SPAN TREE. Marvin's primitives are ambient module
functions, so the adapter opens a fresh run scope per call and never emits an
event on that scope's own root span. Inside the enclosing ``@trace`` the two
primitive subtrees therefore hang off spans the SDK emitted no event for, and the
frontend roots them under a synthesized parent and flags the trace ``fragmented``.
The events themselves are complete and real; only the parent linkage is missing.

The recorded trace is shipped under samples/data/traces/industry/ (produced by
samples/data/generators/marvin.py); this sample uploads it and evaluates the
extraction with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

    No provider key is needed -- the listing run was recorded once; this sample
    only uploads and evaluates it.

Usage:
    python realestate_marvin_listing_extract.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import (
    create_judge,
    recorded_trace_path,
    upload_recorded_trace,
    poll_evaluation_results,
)

SAMPLE = "realestate_marvin_listing_extract"
FIXTURE = recorded_trace_path("industry", "realestate_marvin_listing_extract.jsonl")

# The listing the extraction agent normalized. Documents the scenario; the
# recorded trace was produced by running this freeform write-up through a real
# marvin.Agent via marvin.cast (typed record) + marvin.extract (feature list).
LISTING: dict[str, Any] = {
    "listing_id": "MLS-4471-OAKRIDGE",
    "source": "agent freeform write-up (MLS intake)",
    "primitives": ["marvin.cast -> PropertyListing", "marvin.extract -> features"],
    "description": (
        "Welcome to 1428 Oakridge Lane, a beautifully maintained 1997 craftsman-style "
        "single-family home tucked into the sought-after Oakridge Park neighborhood of "
        "Round Rock. Offered at $749,000, this 2,340 square foot residence gives you four "
        "generous bedrooms and two and a half bathrooms, including a main-floor primary "
        "suite with a spa-inspired walk-in shower and dual vanities. The chef's kitchen was "
        "fully renovated in 2023 with quartz countertops, a gas range, and a walk-in pantry, "
        "and it opens onto a light-filled great room anchored by a wood-burning fireplace. "
        "Enjoy hardwood floors throughout the main level, a dedicated home office, and a "
        "finished bonus room over the attached two-car garage. Outside, the 0.28 acre lot is "
        "fully fenced and backs to a greenbelt, with a covered patio, mature oaks, and an "
        "in-ground sprinkler system. Recent updates include a 2022 roof and a new 16-SEER "
        "HVAC system. Zoned to the highly rated Oakridge Elementary, and just minutes from "
        "the tollway. HOA dues are $45/month and cover the neighborhood pool and trails."
    ),
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded Marvin MLS listing-extraction trace."""
    print("=== LayerLens Industry: Real-Estate MLS Listing Intake (Marvin structured extraction) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        print("  Set LAYERLENS_STRATIX_API_KEY to run this sample.")
        print("  No provider key is required -- the run was recorded once.")
        sys.exit(1)

    # Upload the recorded trace first (renders a single listing-extraction-agent
    # node with the real cast/extract tool.call + model.invoke events). Do this
    # before creating judges so the trace always lands even if the org has no
    # evaluation model yet.
    print("Uploading the recorded MLS listing-extraction trace (marvin.cast + marvin.extract)...\n")
    try:
        trace_ids = upload_recorded_trace(client, FIXTURE)
    except Exception as exc:
        print(f"SKIP: could not upload the recorded trace: {exc}")
        return
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID:   {trace_id}")
    print(f"  Listing:    {LISTING['listing_id']} ({LISTING['source']})")
    print(f"  Primitives: {', '.join(LISTING['primitives'])}")
    print("  NOTE: this trace carries no tokens and no cost.record -- Marvin surfaces no")
    print("        usage on its primitives, so the adapter omits them instead of")
    print("        inventing a figure. The model (gpt-4o-mini) and latency are real.\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    judge_ids: list[str] = []
    try:
        judges = {
            "field_accuracy": create_judge(
                client,
                name="Listing Field Accuracy Judge",
                evaluation_goal="Evaluate whether the extracted MLS fields (address, property type, bedrooms, bathrooms, square feet, list price, year built, garage spaces, lot size, HOA dues) match the facts stated in the freeform listing description.",
                namespace=SAMPLE,
            ),
            "no_invented_fields": create_judge(
                client,
                name="No-Invented-Fields Judge",
                evaluation_goal="Evaluate whether the agent extracted ONLY values the listing description actually states and did not infer, round, or invent any field the prose does not support.",
                namespace=SAMPLE,
            ),
            "feature_grounding": create_judge(
                client,
                name="Feature Grounding Judge",
                evaluation_goal="Evaluate whether every property feature or amenity in the extracted feature list is genuinely supported by the listing description, with no feature added that the text does not mention.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the listing extraction:")
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
