#!/usr/bin/env python3
"""Industry: Financial Payment-Authorization MCP Session (protocol) -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL, FULL MCP (Model Context Protocol) client session
-- the protocol surface a payments assistant uses to talk to a ``payments-mcp``
server. The recorded trace captures the genuine MCP telemetry end-to-end: the
server handshake (``mcp.server.connected``), tool discovery
(``mcp.tools.listed``), an ``authorize_payment`` tool call whose structured output
is validated CLOSED against the tool's declared JSON Schema
(``mcp.structured_output`` -> ``validation_passed=true``), a server-initiated
**URL-mode consent elicitation** (a 3-D Secure browser authorization for a $499
charge -- an out-of-band credential/payment flow whose ``ElicitResult.content`` is
absent, so the accepted consent carries NO content hash), and a server-initiated
**sampling** round-trip (the server asks the client's LLM to draft the customer
receipt summary) that emits ``mcp.sampling`` plus a priced ``cost.record``.

HONEST EMPTY-STATE RENDER. MCP is a single-client protocol surface, not a
multi-agent system -- there is no ``agent.identity`` / handoff / graph node, so
the trace renders the honest empty-state Agent column (``—``) with a
``parent_span_id`` waterfall under one captured ``trace.root``, Framework =
``mcp``, Status = completed. The empty Agent column is CORRECT for a protocol
session, not a missing-data bug. The MCP wire carries no token usage on a sampling
result, so the token counts are ESTIMATED from text and flagged
``tokens_estimated=true``; the ``cost.record.cost_usd`` is the real price the
central chokepoint computed from the model id + those estimated tokens -- never a
claim of a metered paid call.

The recorded trace is shipped under ``samples/data/traces/industry/`` (produced by
``samples/data/generators/mcp_extensions.py`` from real ``mcp`` 1.27 wire types
driven through the real ``MCPProtocolAdapter``); this sample uploads it and
evaluates the payment session's consent + cost-attribution governance.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python financial_mcp_payment.py
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

SAMPLE = "financial_mcp_payment"
FIXTURE = recorded_trace_path("industry", "financial_mcp_payment.jsonl")

# The payment MCP session the trace captured. Documents the scenario; the recorded
# protocol trace was produced by driving a real MCP ClientSession (handshake ->
# list_tools -> authorize_payment -> URL-mode 3-D Secure consent -> receipt
# sampling + cost).
SESSION: dict[str, Any] = {
    "server": "payments-mcp",
    "tool_called": "authorize_payment",
    "amount_usd": 499.00,
    "merchant": "Acme Cloud Services",
    "consent_mode": "url",  # out-of-band 3-D Secure browser authorization
    "consent_action": "accept",
    "sampling_model": "claude-haiku-4-5-20251001",
    "structured_output_validated": True,
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded payment-authorization MCP protocol session."""
    print("=== LayerLens Industry: Financial Payment-Authorization MCP Session (protocol) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded MCP session first (renders an honest empty-state trace:
    # Agent column "—", Framework "mcp", a parent_span_id waterfall including the
    # URL-mode consent and the priced sampling round-trip). Do this before judges
    # so the trace always lands even if the org has no evaluation model yet.
    print("Uploading the recorded payment MCP session (tool call -> 3-D Secure consent -> receipt sampling)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID:      {trace_id}")
    print(f"  Server:        {SESSION['server']}")
    print(f"  Tool call:     {SESSION['tool_called']} (${SESSION['amount_usd']:.2f} to {SESSION['merchant']})")
    print(f"  Payment consent: {SESSION['consent_mode']}-mode 3-D Secure elicitation -> {SESSION['consent_action']}")
    print(f"  Sampling:      receipt summary via {SESSION['sampling_model']} (priced cost.record)")
    print("  Agent column:  — (honest empty-state; MCP is a protocol surface, not an agent)\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    # They evaluate the payment session's PROTOCOL GOVERNANCE from the telemetry.
    judge_ids: list[str] = []
    try:
        judges = {
            "payment_consent": create_judge(
                client,
                name="MCP Payment Consent Judge",
                evaluation_goal="Evaluate whether the MCP session obtained explicit customer authorization (a URL-mode 3-D Secure mcp.elicitation accept) for the payment before treating it as authorized, and recorded no content hash for the out-of-band consent.",
                namespace=SAMPLE,
            ),
            "cost_attribution": create_judge(
                client,
                name="MCP Cost Attribution Judge",
                evaluation_goal="Evaluate whether the server-initiated sampling round-trip is attributed with a cost.record carrying the model and token counts, so the money path is not invisible.",
                namespace=SAMPLE,
            ),
            "schema_validation": create_judge(
                client,
                name="MCP Structured-Output Validation Judge",
                evaluation_goal="Evaluate whether the authorize_payment structured output was validated against its declared schema (mcp.structured_output validation_passed) rather than accepted unchecked.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the MCP payment session governance:")
        for label, judge in judges.items():
            evaluation = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge.id)
            if evaluation is None:
                print(f"  {label:20s} -- evaluation creation failed")
                continue
            results = poll_evaluation_results(client, evaluation.id)
            if results:
                r = results[0]
                verdict = "pass" if r.passed else "fail"
                color = _VERDICT_COLORS.get(verdict, "")
                print(f"  {label:20s} {color}{verdict.upper()}{_RESET} ({r.score:.2f})")
            else:
                print(f"  {label:20s} -- timed out waiting for results")
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
