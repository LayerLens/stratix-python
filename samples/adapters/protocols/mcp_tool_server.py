"""MCP (Model Context Protocol) tool server instrumentation demo.

Demonstrates:
- Tool discovery (tools/list) and resource reads (resources/read)
- Tool invocations with structured JSON output
- Async task tracking for long-running tool executions

If ``mcp`` is not installed the protocol is simulated via STRATIX emit functions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid

from layerlens.instrument import (
    STRATIX,
    emit,
    emit_input,
    emit_output,
    emit_tool_call,
)

# ---------------------------------------------------------------------------
# Try importing the real MCP SDK; fall back to simulation
# ---------------------------------------------------------------------------

try:
    from mcp.types import Tool, Resource, TextContent  # type: ignore[import-untyped]

    HAS_MCP_SDK = True
except ImportError:
    HAS_MCP_SDK = False
    print(
        "[mcp] SDK not installed. Run: pip install mcp\n"
        "      Continuing with simulated protocol flow.\n"
    )


# ---------------------------------------------------------------------------
# Mock MCP server definitions
# ---------------------------------------------------------------------------

MOCK_TOOLS = [
    {"name": "query_database", "description": "Execute a read-only SQL query.",
     "inputSchema": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}},
    {"name": "generate_report", "description": "Generate a PDF report. Long-running async task.",
     "inputSchema": {"type": "object", "properties": {"title": {"type": "string"}, "data_ref": {"type": "string"}},
                     "required": ["title", "data_ref"]}},
]

MOCK_RESOURCES = [
    {"uri": "db://analytics/schema", "name": "Analytics Schema", "mimeType": "application/json"},
]


def _mock_query_result() -> dict:
    return {"columns": ["quarter", "revenue"], "rows": [["Q3", 4_200_000], ["Q4", 4_700_000]], "row_count": 2}


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MCP tool server instrumentation demo")
    parser.add_argument("--policy-ref", default="stratix-policy-cs-v1@1.0.0")
    parser.add_argument("--agent-id", default="analytics_agent")
    parser.add_argument("--query", default="SELECT quarter, revenue, growth_pct FROM earnings ORDER BY quarter")
    args = parser.parse_args()

    stratix = STRATIX(policy_ref=args.policy_ref, agent_id=args.agent_id, framework="mcp_extensions")
    ctx = stratix.start_trial()
    print(f"[STRATIX] Trial started  trace_id={ctx.trace_id}")

    with stratix.context():
        emit_input(args.query, role="human")
        print(f"[mcp] User query: {args.query}")

        # -- 1. Tool discovery (tools/list) --
        t0 = time.perf_counter()
        emit_tool_call(
            name="mcp.tools/list",
            input_data={},
            output_data={"tools": MOCK_TOOLS},
            latency_ms=(time.perf_counter() - t0) * 1000,
            integration="service",
        )
        print(f"[mcp] Discovered {len(MOCK_TOOLS)} tools: {[t['name'] for t in MOCK_TOOLS]}")

        # -- 2. Resource read (resources/read) --
        resource = MOCK_RESOURCES[0]
        schema_content = {"tables": ["earnings", "users", "events"], "version": "3.1"}
        t0 = time.perf_counter()
        emit_tool_call(
            name="mcp.resources/read",
            input_data={"uri": resource["uri"]},
            output_data={"contents": [{"uri": resource["uri"], "text": json.dumps(schema_content)}]},
            latency_ms=(time.perf_counter() - t0) * 1000,
            integration="service",
        )
        print(f"[mcp] Resource read: {resource['name']}  tables={schema_content['tables']}")

        # -- 3. Tool invocation: query_database (structured output) --
        t0 = time.perf_counter()
        query_result = _mock_query_result()
        latency = (time.perf_counter() - t0) * 1000 + 85.0  # simulate DB latency
        emit_tool_call(
            name="query_database",
            input_data={"sql": args.query, "limit": 100},
            output_data={"structuredOutput": query_result, "isError": False},
            latency_ms=latency,
            version="1.2.0",
            integration="service",
        )
        print(f"[mcp] query_database  rows={query_result['row_count']}  latency={latency:.0f}ms")

        # -- 4. Async task: generate_report --
        task_id = str(uuid.uuid4())
        t0 = time.perf_counter()

        # Submit async task
        emit_tool_call(
            name="generate_report",
            input_data={"title": "Q4 Earnings Summary", "data_ref": "result://query/1"},
            output_data={"asyncTaskId": task_id, "status": "accepted"},
            latency_ms=(time.perf_counter() - t0) * 1000,
            version="1.0.0",
            integration="service",
        )
        print(f"[mcp] generate_report submitted  asyncTaskId={task_id[:8]}...")

        # Poll for completion
        for status in ["in_progress", "in_progress", "completed"]:
            time.sleep(0.05)
            elapsed = (time.perf_counter() - t0) * 1000
            result_data = {"asyncTaskId": task_id, "status": status}
            if status == "completed":
                result_data["artifact"] = {"uri": f"reports://{task_id}.pdf", "mimeType": "application/pdf"}
            emit_tool_call(
                name="mcp.tasks/status",
                input_data={"asyncTaskId": task_id},
                output_data=result_data,
                latency_ms=elapsed,
                integration="service",
            )
            print(f"[mcp] Async task poll  status={status}  elapsed={elapsed:.0f}ms")

        # -- 5. Final output --
        output_msg = f"Query returned {query_result['row_count']} rows. Report generated: reports://{task_id[:8]}.pdf"
        emit_output(output_msg)
        print(f"[mcp] Output: {output_msg}")

    # -- Summary --
    summary = stratix.end_trial()
    events = stratix.get_events()
    print(f"\n{'=' * 60}")
    print(f"[STRATIX] Trial ended   status={summary.get('status')}  events={len(events)}")
    print(f"[STRATIX] Trace ID:     {ctx.trace_id}")
    print(f"[STRATIX] SDK present:  {HAS_MCP_SDK}")
    for i, ev in enumerate(events):
        print(f"  [{i}] {ev.event_type:30s}  ts={ev.timestamp}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
