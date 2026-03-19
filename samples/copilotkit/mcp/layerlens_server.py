"""MCP Server — LayerLens API Tools.

Exposes 8 LayerLens platform operations as MCP tools for use with CopilotKit
or any MCP-compatible client.

Tools:
    list_traces     - List and filter traces
    get_trace       - Retrieve a single trace by ID
    run_evaluation  - Submit an evaluation run
    list_judges     - List available judges
    create_judge    - Create a new custom judge
    get_results     - Get evaluation results
    replay_trace    - Trigger a trace replay
    export_data     - Export traces or evaluations

Requires:
    LAYERLENS_STRATIX_API_KEY  - LayerLens platform API key

Run:
    python layerlens_server.py          # stdio transport (default)
    python layerlens_server.py --sse    # SSE transport for remote clients
"""

from __future__ import annotations

import os
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from layerlens import Stratix

# ── Server setup ──────────────────────────────────────────────────────────────

app = Server("layerlens-mcp")
client = Stratix(api_key=os.environ.get("LAYERLENS_STRATIX_API_KEY", ""))


def _ok(data: Any) -> list[TextContent]:
    """Wrap a response as MCP text content."""
    import json
    text = json.dumps(data, indent=2, default=str) if not isinstance(data, str) else data
    return [TextContent(type="text", text=text)]


def _err(msg: str) -> list[TextContent]:
    return [TextContent(type="text", text=f"Error: {msg}")]


# ── Tool definitions ──────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(name="list_traces", description="List and filter traces",
             inputSchema={"type": "object", "properties": {
                 "limit": {"type": "integer", "default": 10},
                 "agent_id": {"type": "string"}, "search": {"type": "string"},
             }}),
        Tool(name="get_trace", description="Retrieve a single trace by ID",
             inputSchema={"type": "object", "properties": {
                 "trace_id": {"type": "string"},
             }, "required": ["trace_id"]}),
        Tool(name="run_evaluation", description="Submit an evaluation run",
             inputSchema={"type": "object", "properties": {
                 "judge_id": {"type": "string"}, "dataset_id": {"type": "string"},
             }, "required": ["judge_id", "dataset_id"]}),
        Tool(name="list_judges", description="List available judges",
             inputSchema={"type": "object", "properties": {
                 "limit": {"type": "integer", "default": 20},
             }}),
        Tool(name="create_judge", description="Create a new custom judge",
             inputSchema={"type": "object", "properties": {
                 "name": {"type": "string"}, "description": {"type": "string"},
                 "criteria": {"type": "array", "items": {"type": "object"}},
                 "model": {"type": "string", "default": "gpt-4o"},
                 "pass_threshold": {"type": "number", "default": 0.7},
             }, "required": ["name", "criteria"]}),
        Tool(name="get_results", description="Get evaluation results",
             inputSchema={"type": "object", "properties": {
                 "evaluation_id": {"type": "string"},
             }, "required": ["evaluation_id"]}),
        Tool(name="replay_trace", description="Trigger a trace replay with optional model override",
             inputSchema={"type": "object", "properties": {
                 "trace_id": {"type": "string"}, "model_override": {"type": "string"},
             }, "required": ["trace_id"]}),
        Tool(name="export_data", description="Export traces or evaluations as CSV/JSON/Parquet",
             inputSchema={"type": "object", "properties": {
                 "export_type": {"type": "string", "enum": ["traces", "evaluations"]},
                 "format": {"type": "string", "enum": ["csv", "json", "parquet"], "default": "json"},
             }, "required": ["export_type"]}),
    ]


# ── Tool handlers ─────────────────────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "list_traces":
            traces = client.traces.list(
                limit=arguments.get("limit", 10),
                agent_id=arguments.get("agent_id"),
                search=arguments.get("search"),
            )
            return _ok([t.to_dict() for t in (traces or [])])

        elif name == "get_trace":
            trace = client.traces.get(arguments["trace_id"])
            return _ok(trace.to_dict()) if trace else _err("Trace not found")

        elif name == "run_evaluation":
            evaluation = client.evaluations.create(
                judge_id=arguments["judge_id"],
                dataset_id=arguments["dataset_id"],
            )
            return _ok(evaluation.to_dict()) if evaluation else _err("Failed to create evaluation")

        elif name == "list_judges":
            judges = client.judges.list(limit=arguments.get("limit", 20))
            return _ok([j.to_dict() for j in (judges or [])])

        elif name == "create_judge":
            judge = client.judges.create(
                name=arguments["name"],
                description=arguments.get("description", ""),
                criteria=arguments["criteria"],
                model=arguments.get("model", "gpt-4o"),
                pass_threshold=arguments.get("pass_threshold", 0.7),
            )
            return _ok(judge.to_dict()) if judge else _err("Failed to create judge")

        elif name == "get_results":
            results = client.evaluations.get_results(arguments["evaluation_id"])
            return _ok(results.to_dict()) if results else _err("No results found")

        elif name == "replay_trace":
            replay = client.replays.create(
                trace_id=arguments["trace_id"],
                model_override=arguments.get("model_override"),
            )
            return _ok(replay.to_dict()) if replay else _err("Failed to trigger replay")

        elif name == "export_data":
            export = client.exports.create(
                export_type=arguments["export_type"],
                format=arguments.get("format", "json"),
            )
            return _ok(export.to_dict()) if export else _err("Export failed")

        else:
            return _err(f"Unknown tool: {name}")

    except Exception as exc:
        return _err(str(exc))


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    import sys
    if "--sse" in sys.argv:
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Route
        import uvicorn

        sse = SseServerTransport("/messages")
        starlette_app = Starlette(routes=[
            Route("/sse", endpoint=sse.handle_sse_request),
            Route("/messages", endpoint=sse.handle_post_message, methods=["POST"]),
        ])
        uvicorn.run(starlette_app, host="0.0.0.0", port=8080)
    else:
        async with stdio_server() as (read, write):
            await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
