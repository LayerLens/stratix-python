#!/usr/bin/env python3
"""Industry: Retail AG-UI Cart Session (protocol, state round-trip) -- LayerLens Python SDK Sample.

Demonstrates evaluating a fuller REAL AG-UI (CopilotKit) cart-management session
that exercises ALL THREE ``agui.*`` telemetry families in one flow. AG-UI is the
agent<->frontend SSE transport; the LayerLens ``AGUIProtocolAdapter`` observes the
stream and reconstructs:

* ``agui.state`` -- a ``STATE_SNAPSHOT`` of the cart followed by several
  ``STATE_DELTA`` JSON-Patch rounds (add a line item, replace a quantity, remove
  an item), with the adapter's chained before/after SHA-256 state hashes (each
  delta's ``before_hash`` == the prior event's ``after_hash``);
* ``agui.tool_call`` -- a multi-fragment ``add_to_cart`` call whose split
  ``TOOL_CALL_ARGS`` deltas the adapter accumulates and parses to a JSON object;
* ``agui.message`` -- the assistant's streamed confirmations.

Like every AG-UI session this is single-agent / agent-empty (a UI transport, not
an agent framework): there is no ``agent.identity`` and no agent graph, so the
trace renders as an HONEST EMPTY-STATE -- Agent column ``—``, Framework ``agui``,
an OTel-style waterfall of the real protocol events rooted at a single
``trace.root`` span. "Multi" here means a multi-FAMILY protocol session, NOT
multi-agent; nothing invents an agent.

The trace was recorded from a real ``AGUIProtocolAdapter.wrap_stream`` run over a
genuine CopilotKit SSE session (see samples/data/_generate_fixtures.py +
samples/data/generators/agui.py) and is shipped under
samples/data/traces/industry/. This sample uploads it, confirms it persisted +
attested, shows the honest empty-state + the state-hash chain, and evaluates the
cart interaction with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python retail_agui_cart.py
"""

from __future__ import annotations

import json
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

SAMPLE = "retail_agui_cart"
FIXTURE = recorded_trace_path("industry", "retail_agui_cart.jsonl")

# The cart interaction the AG-UI session captured. Documents the scenario; the
# recorded trace was produced by driving a real CopilotKit SSE stream
# (STATE_SNAPSHOT + STATE_DELTA round-trip + a multi-fragment add_to_cart tool
# call + streamed messages) through the real AGUIProtocolAdapter.
SESSION: dict[str, Any] = {
    "thread_id": "th-cart-1",
    "surface": "copilotkit_chat",
    "flow": "add two items, bump a quantity, remove one, summarize",
    "final_subtotal_usd": 298.0,
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def _describe_empty_state(fixture_path: str) -> None:
    """Print the honest empty-state summary + state-hash chain from the fixture.

    Reads the shipped trace's own events (deterministic, offline) to show WHY this
    protocol trace renders as an empty-state waterfall (no ``agent.identity`` /
    handoff, framework ``agui``, rooted at ``trace.root``) and to surface the real
    ``agui.state`` before/after hash chain across the SNAPSHOT -> DELTA rounds.
    """
    with open(fixture_path) as f:
        payload = json.loads(f.readline())
    events = payload.get("events", [])
    counts: dict[str, int] = {}
    for e in events:
        counts[e.get("event_type")] = counts.get(e.get("event_type"), 0) + 1
    frameworks = sorted(
        {(e.get("payload") or {}).get("framework") for e in events} - {None}
    )
    print("  Honest empty-state render (AG-UI is a UI transport, not an agent):")
    print(f"    Agent column:     — (agent.identity events: {counts.get('agent.identity', 0)})")
    print(f"    Framework column: {', '.join(frameworks) or '—'}")
    print(f"    Rooted at:        trace.root ({counts.get('trace.root', 0)} synthesized root span)")
    print(f"    Protocol events:  agui.state={counts.get('agui.state', 0)} "
          f"agui.tool_call={counts.get('agui.tool_call', 0)} "
          f"agui.message={counts.get('agui.message', 0)}\n")

    # Show the real chained state hashes the adapter emitted across the round-trip.
    states = [e.get("payload") or {} for e in events if e.get("event_type") == "agui.state"]
    if states:
        print("  Cart state hash chain (adapter's chained before/after SHA-256):")
        prev = None
        for s in states:
            chained = "chained" if (prev is None or s.get("before_hash") == prev) else "BROKEN"
            print(f"    {s.get('state_event',''):15s} after={str(s.get('after_hash',''))[:23]}… ({chained})")
            prev = s.get("after_hash")
        print()


def main() -> None:
    """Evaluate the recorded AG-UI cart-management session."""
    print("=== LayerLens Industry: Retail AG-UI Cart Session (protocol) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded AG-UI session first (renders as an honest empty-state
    # waterfall of the real protocol events). Do this before creating judges so
    # the trace always lands even if the org has no evaluation model yet.
    print("Uploading the recorded AG-UI cart session (state round-trip SSE protocol trace)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Session:  {SESSION['thread_id']} ({SESSION['flow']})\n")

    # Confirm the trace persisted + attested (GET-by-id), then show the honest
    # empty-state render + state-hash chain derived from the fixture's own events.
    got = client.traces.get(trace_id)
    if got is not None:
        print("  Persisted + attested (get-by-id: FOUND).\n")
    _describe_empty_state(FIXTURE)

    # Judge scoped to this sample (namespace avoids cross-sample name collisions).
    # NOTE ON EMPTY-STATE PROTOCOL TRACES: AG-UI captures protocol telemetry
    # (messages/tool-calls/state), not a scored agent input->output pair, so an
    # agent-output judge has no output to grade — trace_evaluations return no
    # score for these traces. We still create + attempt one evaluation (the SDK
    # judge API works the same way you'd use it on your OWN agent traces), then
    # report the honest outcome; the real value here is the uploaded, rendered,
    # attested protocol trace (+ state-hash chain) above. A bounded poll keeps
    # the sample snappy.
    judge_ids: list[str] = []
    try:
        judge = create_judge(
            client,
            name="Cart Consistency Judge",
            evaluation_goal="Evaluate whether the assistant's cart actions and final summary are internally consistent with the add/replace/remove operations applied to the cart during the session.",
            namespace=SAMPLE,
        )
        judge_ids = [judge.id]

        print("Attempting a trace evaluation of the cart interaction:")
        evaluation = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge.id)
        results = (
            poll_evaluation_results(client, evaluation.id, max_attempts=12)
            if evaluation is not None
            else None
        )
        if results:
            r = results[0]
            verdict = "pass" if r.passed else "fail"
            color = _VERDICT_COLORS.get(verdict, "")
            print(f"  cart_consistency         {color}{verdict.upper()}{_RESET} ({r.score:.2f})")
        else:
            print("  cart_consistency         -- no score (empty-state protocol trace has no")
            print("                              agent output to grade; use judges on your own")
            print("                              agent traces — the observability above is the value).")
    except RuntimeError as exc:
        print(f"\nNOTE: evaluation skipped -- {exc}")
        print("  The trace is uploaded + rendered; add a project/public model to enable judges.")
    finally:
        for jid in judge_ids:
            try:
                client.judges.delete(jid)
            except Exception:
                pass


if __name__ == "__main__":
    main()
