#!/usr/bin/env python3
"""Industry: Retail AG-UI Shopping Assistant (protocol) -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL AG-UI (CopilotKit) shopping-assistant session. AG-UI
is the agent<->frontend SSE transport CopilotKit runtimes speak; the LayerLens
``AGUIProtocolAdapter`` sits as middleware around that stream and reconstructs
telemetry from it -- a streamed assistant message, a multi-fragment
``product_lookup`` tool call (the adapter accumulates the split ``TOOL_CALL_ARGS``
deltas and parses the JSON), the tool result, and the assistant's streamed
recommendation.

AG-UI is a UI transport, NOT an agent framework: the session carries no
``agent.identity`` and no agent-to-agent topology, so the recorded trace renders
as an HONEST EMPTY-STATE -- the Agent column reads ``—`` and the trace shows an
OTel-style waterfall of the real protocol events (``agui.message`` /
``agui.tool_call`` / lifecycle ``protocol.stream.event``s), rooted at a single
content-free ``trace.root`` span. That empty-state is the CORRECT rendering for a
UI transport; nothing invents an agent. The Framework column reads ``agui``.

The trace was recorded from a real ``AGUIProtocolAdapter.wrap_stream`` run over a
genuine CopilotKit SSE session (see samples/data/_generate_fixtures.py +
samples/data/generators/agui.py) and is shipped under
samples/data/traces/industry/. This sample uploads it, confirms it persisted +
attested, shows the honest empty-state, and evaluates the streamed shopping
recommendation with a domain judge.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python retail_agui_shopping.py
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

SAMPLE = "retail_agui_shopping"
FIXTURE = recorded_trace_path("industry", "retail_agui_shopping.jsonl")

# The shopper interaction the AG-UI session captured. Documents the scenario; the
# recorded trace was produced by driving a real CopilotKit SSE stream (assistant
# message + product_lookup tool call + streamed recommendation) through the real
# AGUIProtocolAdapter.
SESSION: dict[str, Any] = {
    "thread_id": "th-shop-1",
    "surface": "copilotkit_chat",
    "shopper_query": "Find me wireless headphones under $200.",
    "tool": "product_lookup",
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def _describe_empty_state(fixture_path: str) -> None:
    """Print the honest empty-state summary from the recorded fixture itself.

    Reads the shipped trace's own events (deterministic, offline) to show WHY this
    protocol trace renders as an empty-state waterfall rather than an agent DAG:
    no ``agent.identity`` / handoff, framework ``agui``, rooted at ``trace.root``,
    with the reconstructed ``agui.*`` protocol families.
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
    print(f"    Protocol events:  agui.message={counts.get('agui.message', 0)} "
          f"agui.tool_call={counts.get('agui.tool_call', 0)} "
          f"protocol.stream.event={counts.get('protocol.stream.event', 0)}\n")


def main() -> None:
    """Evaluate the recorded AG-UI shopping-assistant session."""
    print("=== LayerLens Industry: Retail AG-UI Shopping Assistant (protocol) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded AG-UI session first (renders as an honest empty-state
    # waterfall of the real protocol events). Do this before creating judges so
    # the trace always lands even if the org has no evaluation model yet.
    print("Uploading the recorded AG-UI shopping-assistant session (SSE protocol trace)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID: {trace_id}")
    print(f"  Session:  {SESSION['thread_id']} ({SESSION['shopper_query']})\n")

    # Confirm the trace persisted + attested (GET-by-id), then show the honest
    # empty-state render derived from the fixture's own events.
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
    # attested protocol trace above. A bounded poll keeps the sample snappy.
    judge_ids: list[str] = []
    try:
        judge = create_judge(
            client,
            name="Shopping Recommendation Judge",
            evaluation_goal="Evaluate whether the assistant's streamed product recommendation is helpful, relevant to the shopper's request (wireless headphones under $200), and consistent with the product_lookup tool result.",
            namespace=SAMPLE,
        )
        judge_ids = [judge.id]

        print("Attempting a trace evaluation of the streamed recommendation:")
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
            print(f"  recommendation_quality   {color}{verdict.upper()}{_RESET} ({r.score:.2f})")
        else:
            print("  recommendation_quality   -- no score (empty-state protocol trace has no")
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
