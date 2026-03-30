"""LayerLens MCP Server -- powered by the LayerLens Python SDK.

An MCP (Model Context Protocol) server that exposes LayerLens trace
inspection, evaluation, and judge management capabilities as tools that
any MCP-compatible AI assistant can invoke.

Prerequisites
-------------
    pip install layerlens --index-url https://sdk.layerlens.ai/package mcp

Environment
-----------
    LAYERLENS_STRATIX_API_KEY   API key used by the SDK (picked up
                                automatically by ``Stratix()``).

Usage
-----
    python layerlens_server.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from layerlens import Stratix, StratixError, NotFoundError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import create_judge as _create_judge_helper

# ---------------------------------------------------------------------------
# Lazy-initialised SDK client
# ---------------------------------------------------------------------------

_client_lock = threading.Lock()
_client: Optional[Stratix] = None


def _get_client() -> Stratix:
    """Return the module-level Stratix client, creating it on first use."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:  # double-check after acquiring lock
                _client = Stratix()
    return _client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _obj_to_text(obj: Any) -> str:
    """Convert a Pydantic model, dict, or list to a readable string."""
    if obj is None:
        return "(no data)"
    if hasattr(obj, "model_dump"):
        return json.dumps(obj.model_dump(), indent=2, default=str)
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, indent=2, default=str)
    return str(obj)


def _error_text(exc: Exception) -> str:
    """Format an exception into a user-friendly error message."""
    if isinstance(exc, NotFoundError):
        return f"Not found: {exc}"
    if isinstance(exc, StratixError):
        return f"LayerLens API error: {exc}"
    return f"Error: {exc}"


# ---------------------------------------------------------------------------
# MCP server factory
# ---------------------------------------------------------------------------


def create_server() -> Server:
    """Create and return the MCP server with all tool handlers wired up."""

    server = Server("layerlens")

    # ----- tool catalogue ------------------------------------------------

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="list_traces",
                description=(
                    "List traces stored in LayerLens. Returns the most "
                    "recent traces with optional pagination."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of traces to return (default 20, max 500).",
                        },
                    },
                },
            ),
            Tool(
                name="get_trace",
                description="Retrieve the full details of a single trace by its ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "trace_id": {
                            "type": "string",
                            "description": "The trace ID.",
                        },
                    },
                    "required": ["trace_id"],
                },
            ),
            Tool(
                name="run_evaluation",
                description=(
                    "Run a judge evaluation against a trace. Returns the "
                    "created trace-evaluation object (initially in pending state)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "trace_id": {
                            "type": "string",
                            "description": "The trace ID to evaluate.",
                        },
                        "judge_id": {
                            "type": "string",
                            "description": "The judge ID to use for evaluation.",
                        },
                    },
                    "required": ["trace_id", "judge_id"],
                },
            ),
            Tool(
                name="get_evaluation",
                description=(
                    "Get the status and results of a trace evaluation. "
                    "If the evaluation has completed, the results are included."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "evaluation_id": {
                            "type": "string",
                            "description": "The trace-evaluation ID.",
                        },
                    },
                    "required": ["evaluation_id"],
                },
            ),
            Tool(
                name="create_judge",
                description="Create a new evaluation judge with a name and goal.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Display name for the judge.",
                        },
                        "goal": {
                            "type": "string",
                            "description": "The evaluation goal / criteria the judge should assess.",
                        },
                    },
                    "required": ["name", "goal"],
                },
            ),
            Tool(
                name="list_judges",
                description="List all judges configured in the current project.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    # ----- tool dispatcher -----------------------------------------------

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            client = _get_client()

            if name == "list_traces":
                return await _handle_list_traces(client, arguments)

            if name == "get_trace":
                return await _handle_get_trace(client, arguments)

            if name == "run_evaluation":
                return await _handle_run_evaluation(client, arguments)

            if name == "get_evaluation":
                return await _handle_get_evaluation(client, arguments)

            if name == "create_judge":
                return await _handle_create_judge(client, arguments)

            if name == "list_judges":
                return await _handle_list_judges(client, arguments)

            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        except Exception as exc:
            return [TextContent(type="text", text=_error_text(exc))]

    return server


# ---------------------------------------------------------------------------
# Individual tool handlers
#
# Each handler wraps synchronous SDK calls in asyncio.to_thread() so they
# do not block the async MCP event loop.
# ---------------------------------------------------------------------------


async def _handle_list_traces(client: Stratix, arguments: dict) -> list[TextContent]:
    limit = arguments.get("limit", 20)
    resp = await asyncio.to_thread(
        client.traces.get_many, page_size=limit, sort_by="created_at", sort_order="desc"
    )
    if resp is None:
        return [TextContent(type="text", text="No traces found.")]

    lines: list[str] = [f"Traces (showing {resp.count} of {resp.total_count}):"]
    for t in resp.traces:
        eval_info = ""
        if t.evaluations_count:
            eval_info = f" | {t.evaluations_count} evaluation(s)"
        lines.append(f"  - {t.id}  created={t.created_at}  file={t.filename}{eval_info}")
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_get_trace(client: Stratix, arguments: dict) -> list[TextContent]:
    trace_id: str = arguments["trace_id"]
    trace = await asyncio.to_thread(client.traces.get, trace_id)
    if trace is None:
        return [TextContent(type="text", text=f"Trace {trace_id} not found.")]
    return [TextContent(type="text", text=_obj_to_text(trace))]


async def _handle_run_evaluation(client: Stratix, arguments: dict) -> list[TextContent]:
    trace_id: str = arguments["trace_id"]
    judge_id: str = arguments["judge_id"]
    evaluation = await asyncio.to_thread(
        client.trace_evaluations.create, trace_id=trace_id, judge_id=judge_id
    )
    if evaluation is None:
        return [TextContent(type="text", text="Failed to create evaluation.")]
    return [TextContent(
        type="text",
        text=(
            f"Evaluation created.\n"
            f"  ID:      {evaluation.id}\n"
            f"  Status:  {evaluation.status}\n"
            f"  Trace:   {trace_id}\n"
            f"  Judge:   {judge_id}"
        ),
    )]


async def _handle_get_evaluation(client: Stratix, arguments: dict) -> list[TextContent]:
    eid: str = arguments["evaluation_id"]
    evaluation = await asyncio.to_thread(client.trace_evaluations.get, eid)
    if evaluation is None:
        return [TextContent(type="text", text=f"Evaluation {eid} not found.")]

    parts: list[str] = [
        f"Evaluation {eid}:",
        f"  Status:  {evaluation.status}",
    ]

    # If the evaluation finished, fetch and append results.
    if hasattr(evaluation.status, "value") and evaluation.status.value == "success" or str(evaluation.status) == "success":
        results_resp = await asyncio.to_thread(client.trace_evaluations.get_results, id=eid)
        if results_resp and results_resp.results:
            for r in results_resp.results:
                parts.append("")
                parts.append(f"  Result:")
                parts.append(f"    Score:     {r.score}")
                parts.append(f"    Passed:    {r.passed}")
                parts.append(f"    Reasoning: {r.reasoning}")
                parts.append(f"    Latency:   {r.latency_ms} ms")
                parts.append(f"    Cost:      {r.total_cost}")

    return [TextContent(type="text", text="\n".join(parts))]


async def _handle_create_judge(client: Stratix, arguments: dict) -> list[TextContent]:
    name: str = arguments["name"]
    goal: str = arguments["goal"]
    judge = await asyncio.to_thread(_create_judge_helper, client, name=name, evaluation_goal=goal)
    if judge is None:
        return [TextContent(type="text", text="Failed to create judge.")]
    return [TextContent(
        type="text",
        text=(
            f"Judge created.\n"
            f"  ID:      {judge.id}\n"
            f"  Name:    {judge.name}"
        ),
    )]


async def _handle_list_judges(client: Stratix, arguments: dict) -> list[TextContent]:
    resp = await asyncio.to_thread(client.judges.get_many)
    if resp is None or not resp.judges:
        return [TextContent(type="text", text="No judges found.")]

    lines: list[str] = [f"Judges ({len(resp.judges)}):"]
    for j in resp.judges:
        lines.append(f"  - {j.id}  name={j.name!r}")
    return [TextContent(type="text", text="\n".join(lines))]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _run() -> None:
    server = create_server()
    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options)


def main() -> None:
    """Run the MCP server over stdio."""
    asyncio.run(_run())


if __name__ == "__main__":
    main()
