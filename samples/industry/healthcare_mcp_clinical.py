#!/usr/bin/env python3
"""Industry: Healthcare Clinical MCP Session (protocol) -- LayerLens Python SDK Sample.

Demonstrates evaluating a REAL MCP (Model Context Protocol) client session -- the
protocol surface a clinical decision-support tool uses to talk to a
``clinical-records-mcp`` server. The recorded trace captures the genuine MCP
telemetry: the server handshake (``mcp.server.connected``), tool discovery
(``mcp.tools.listed``), a ``get_patient_record`` tool call whose structured output
is validated CLOSED against the tool's declared JSON Schema
(``mcp.structured_output`` -> ``validation_passed=true``), the async-task
lifecycle, and a server-initiated **form-mode consent elicitation** requesting the
clinician's approval to access PHI (``mcp.elicitation`` request/response, the
clinician accepts, so the accepted form content carries a privacy-preserving
``content_hash`` -- never the raw submission).

HONEST EMPTY-STATE RENDER. MCP is a single-client protocol surface, not a
multi-agent system -- there is no ``agent.identity`` / handoff / graph node, so
the trace renders the honest empty-state Agent column (``—``) with a
``parent_span_id`` waterfall under one captured ``trace.root``, Framework =
``mcp``, Status = completed. That empty Agent column is CORRECT for a protocol
session (there is no agent to name), not a missing-data bug. The MCP adapter is
privacy-preserving by design: the tool-call event records only a result *shape*
and a structured-output *hash*, never the raw patient record.

The recorded trace is shipped under ``samples/data/traces/industry/`` (produced by
``samples/data/generators/mcp_extensions.py`` from real ``mcp`` 1.27 wire types
driven through the real ``MCPProtocolAdapter``); this sample uploads it and
evaluates the session's consent + schema-validation governance with domain judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python healthcare_mcp_clinical.py
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

SAMPLE = "healthcare_mcp_clinical"
FIXTURE = recorded_trace_path("industry", "healthcare_mcp_clinical.jsonl")

# The clinical MCP session the trace captured. Documents the scenario; the
# recorded protocol trace was produced by driving a real MCP ClientSession
# (handshake -> list_tools -> get_patient_record -> form-mode PHI consent).
SESSION: dict[str, Any] = {
    "server": "clinical-records-mcp",
    "tool_called": "get_patient_record",
    "patient_id": "PT-8842",
    "consent_mode": "form",
    "consent_action": "accept",
    "structured_output_validated": True,
}

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    """Evaluate the recorded clinical MCP protocol session."""
    print("=== LayerLens Industry: Healthcare Clinical MCP Session (protocol) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded MCP session first (renders an honest empty-state trace:
    # Agent column "—", Framework "mcp", a parent_span_id waterfall of the
    # protocol events). Do this before creating judges so the trace always lands
    # even if the org has no evaluation model yet.
    print("Uploading the recorded clinical MCP session (handshake -> tool call -> PHI consent)...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no trace uploaded (fixture missing or rejected).")
        sys.exit(1)
    trace_id = trace_ids[0]
    print(f"  Trace ID:      {trace_id}")
    print(f"  Server:        {SESSION['server']}")
    print(f"  Tool call:     {SESSION['tool_called']} (patient {SESSION['patient_id']})")
    print(f"  PHI consent:   {SESSION['consent_mode']}-mode elicitation -> {SESSION['consent_action']}")
    print("  Agent column:  — (honest empty-state; MCP is a protocol surface, not an agent)\n")

    # Judges scoped to this sample (namespace avoids cross-sample name collisions).
    # They evaluate the session's PROTOCOL GOVERNANCE from the captured telemetry.
    judge_ids: list[str] = []
    try:
        judges = {
            "consent_fidelity": create_judge(
                client,
                name="MCP Consent Fidelity Judge",
                evaluation_goal="Evaluate whether the MCP session obtained explicit clinician consent (an mcp.elicitation accept) to access PHI before using the patient record, and recorded the consent decision faithfully.",
                namespace=SAMPLE,
            ),
            "schema_validation": create_judge(
                client,
                name="MCP Structured-Output Validation Judge",
                evaluation_goal="Evaluate whether the tool's structured output was validated against its declared schema (mcp.structured_output validation_passed) rather than accepted unchecked.",
                namespace=SAMPLE,
            ),
            "privacy": create_judge(
                client,
                name="MCP PHI Privacy Judge",
                evaluation_goal="Evaluate whether the captured telemetry avoids exposing raw patient PHI, recording only shapes/hashes of the tool result and consent submission.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in judges.values()]

        print("Evaluating the MCP session governance:")
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
