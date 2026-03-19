"""LangGraph CoAgent — Root-Cause Investigator for Failed Traces.

Fetches a trace from LayerLens, analyzes its events to identify failure points,
and suggests fixes. Designed for use as a CopilotKit CoAgent.

Flow: fetch_trace -> analyze_events -> identify_issues -> suggest_fixes

Requires:
    LAYERLENS_STRATIX_API_KEY  - LayerLens platform API key
    OPENAI_API_KEY             - For the LLM powering the analysis
"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from layerlens import Stratix
from layerlens.instrument import STRATIX

# ── State schema ──────────────────────────────────────────────────────────────

class InvestigatorState(TypedDict):
    messages: Annotated[list, add_messages]
    trace_id: str
    trace_data: dict[str, Any]
    events: list[dict[str, Any]]
    issues: list[dict[str, Any]]
    suggestions: list[str]
    status: str


# ── Clients ───────────────────────────────────────────────────────────────────

ll_client = Stratix(api_key=os.environ.get("LAYERLENS_STRATIX_API_KEY", ""))
llm = ChatOpenAI(model="gpt-4o", temperature=0)
stratix = STRATIX(
    policy_ref="copilotkit-investigator-v1@1.0.0",
    agent_id="investigator_coagent",
    framework="langgraph",
)


# ── Node implementations ─────────────────────────────────────────────────────

def fetch_trace(state: InvestigatorState) -> dict:
    """Retrieve the trace and its events from LayerLens."""
    trace_id = state.get("trace_id", "")

    # If no trace_id, try to extract from the last message
    if not trace_id and state.get("messages"):
        last_msg = state["messages"][-1].content
        # Look for trace ID patterns (e.g., "tr_abc123" or UUID)
        import re
        match = re.search(r"(tr_[a-zA-Z0-9]+|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", last_msg)
        if match:
            trace_id = match.group(1)

    if not trace_id:
        return {
            "messages": [AIMessage(content="I need a trace ID to investigate. Please provide one (e.g., `tr_abc123`).")],
            "status": "error",
        }

    with stratix.context():
        trace = ll_client.traces.get(trace_id)

    if not trace:
        return {
            "messages": [AIMessage(content=f"Trace `{trace_id}` not found. Please check the ID and try again.")],
            "status": "error",
        }

    trace_dict = trace.to_dict() if hasattr(trace, "to_dict") else {"id": trace_id}
    events = trace_dict.get("events", [])

    return {
        "trace_id": trace_id,
        "trace_data": trace_dict,
        "events": events,
        "status": "fetched",
        "messages": [AIMessage(content=f"Fetched trace `{trace_id}` with {len(events)} events. Analyzing...")],
    }


def analyze_events(state: InvestigatorState) -> dict:
    """Use the LLM to analyze the event timeline for anomalies."""
    events = state.get("events", [])
    if not events:
        return {"issues": [], "status": "no_events"}

    # Summarize events for the LLM (truncate large payloads)
    event_summaries = []
    for i, ev in enumerate(events[:50]):
        summary = {
            "index": i,
            "type": ev.get("type", "unknown"),
            "timestamp": ev.get("timestamp", ""),
            "duration_ms": ev.get("duration_ms"),
            "status": ev.get("status", ""),
            "error": ev.get("error"),
        }
        if ev.get("metadata"):
            summary["metadata_keys"] = list(ev["metadata"].keys())
        event_summaries.append(summary)

    with stratix.context():
        response = llm.invoke([
            {"role": "system", "content": (
                "You are an AI observability expert. Analyze this trace event timeline and "
                "identify issues: errors, high latency, unexpected patterns, missing events. "
                "Respond as JSON: {\"issues\": [{\"event_index\": N, \"type\": \"...\", "
                "\"severity\": \"high|medium|low\", \"description\": \"...\"}]}"
            )},
            {"role": "user", "content": json.dumps(event_summaries, indent=2)},
        ])

    try:
        analysis = json.loads(response.content)
        issues = analysis.get("issues", [])
    except (json.JSONDecodeError, AttributeError):
        issues = [{"type": "parse_error", "severity": "low", "description": "Could not parse LLM analysis"}]

    return {"issues": issues, "status": "analyzed"}


def identify_issues(state: InvestigatorState) -> dict:
    """Consolidate and prioritize the identified issues."""
    issues = state.get("issues", [])
    if not issues:
        return {
            "messages": [AIMessage(content="No issues found in this trace. The execution appears normal.")],
            "status": "clean",
        }

    # Sort by severity
    severity_order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda x: severity_order.get(x.get("severity", "low"), 3))

    issue_report = "## Issues Found\n\n"
    for i, issue in enumerate(issues, 1):
        sev = issue.get("severity", "unknown").upper()
        desc = issue.get("description", "No description")
        ev_idx = issue.get("event_index", "N/A")
        issue_report += f"{i}. **[{sev}]** (event #{ev_idx}) {desc}\n"

    return {
        "issues": issues,
        "messages": [AIMessage(content=issue_report)],
        "status": "issues_identified",
    }


def suggest_fixes(state: InvestigatorState) -> dict:
    """Generate actionable fix suggestions for the identified issues."""
    issues = state.get("issues", [])
    trace_data = state.get("trace_data", {})

    with stratix.context():
        response = llm.invoke([
            {"role": "system", "content": (
                "You are an AI systems engineer. Given these trace issues, suggest "
                "concrete fixes. Be specific: mention config changes, code patterns, "
                "or operational steps. Format as a numbered list."
            )},
            {"role": "user", "content": (
                f"Agent: {trace_data.get('agent_id', 'unknown')}\n"
                f"Model: {trace_data.get('model', 'unknown')}\n\n"
                f"Issues:\n{json.dumps(issues, indent=2)}"
            )},
        ])

    suggestions = response.content if response.content else "No suggestions generated."

    return {
        "suggestions": [suggestions],
        "messages": [AIMessage(content=f"## Suggested Fixes\n\n{suggestions}")],
        "status": "done",
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_fetch(state: InvestigatorState) -> str:
    if state.get("status") == "error":
        return END
    return "analyze_events"


def route_after_identify(state: InvestigatorState) -> str:
    if state.get("status") == "clean":
        return END
    return "suggest_fixes"


# ── Build graph ───────────────────────────────────────────────────────────────

builder = StateGraph(InvestigatorState)
builder.add_node("fetch_trace", fetch_trace)
builder.add_node("analyze_events", analyze_events)
builder.add_node("identify_issues", identify_issues)
builder.add_node("suggest_fixes", suggest_fixes)

builder.set_entry_point("fetch_trace")
builder.add_conditional_edges("fetch_trace", route_after_fetch)
builder.add_edge("analyze_events", "identify_issues")
builder.add_conditional_edges("identify_issues", route_after_identify)
builder.add_edge("suggest_fixes", END)

graph = builder.compile()
