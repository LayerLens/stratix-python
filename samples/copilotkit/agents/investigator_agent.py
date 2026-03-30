"""
CopilotKit Investigator Agent -- LayerLens SDK edition.

Given a trace ID the agent:
  1. Fetches the full trace via the SDK
  2. Extracts events from the trace data
  3. Runs a battery of analysis helpers (errors, slow spans, token usage, etc.)
  4. Produces a structured report with issues and suggestions

All LayerLens API calls go through the Python SDK (`layerlens.Stratix`)
instead of raw httpx.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph

# CopilotKit helpers
from copilotkit.langchain import copilotkit_emit_message, copilotkit_emit_state

from layerlens import Stratix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level SDK client (lazy)
# ---------------------------------------------------------------------------

_client_lock = threading.Lock()
_client: Optional[Stratix] = None


def _get_client() -> Stratix:
    """Return (and lazily create) a module-level Stratix client."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:  # double-check after acquiring lock
                _client = Stratix()
    return _client


# ---------------------------------------------------------------------------
# Pydantic models for structured analysis output
# ---------------------------------------------------------------------------


class TraceEvent(BaseModel):
    """A single event extracted from a trace's data payload."""

    name: str
    type: str = "unknown"
    timestamp: Optional[str] = None
    duration_ms: Optional[float] = None
    status: Optional[str] = None
    error: Optional[str] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    model: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Issue(BaseModel):
    """A detected issue within the trace."""

    severity: str  # "error", "warning", "info"
    category: str  # e.g. "error", "latency", "cost"
    title: str
    description: str
    event_name: Optional[str] = None


class Suggestion(BaseModel):
    """An actionable suggestion derived from detected issues."""

    title: str
    description: str
    priority: str = "medium"  # "high", "medium", "low"


class InvestigationReport(BaseModel):
    """Full investigation report for a trace."""

    trace_id: str
    filename: str
    created_at: str
    total_events: int
    issues: List[Issue]
    suggestions: List[Suggestion]
    summary: str


# ---------------------------------------------------------------------------
# SDK helper: fetch trace
# ---------------------------------------------------------------------------


def _get_trace(trace_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a trace by ID via the SDK and return its data as a dict."""
    client = _get_client()
    try:
        trace = client.traces.get(trace_id)
        if trace is None:
            return None
        return {
            "id": trace.id,
            "filename": trace.filename,
            "created_at": trace.created_at,
            "data": trace.data,
            "input": trace.input,
        }
    except Exception as exc:
        logger.error("Failed to fetch trace %s: %s", trace_id, exc)
        return None


# ---------------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------------


def _extract_events(trace_data: Dict[str, Any]) -> List[TraceEvent]:
    """Best-effort extraction of events from the trace data dict.

    Trace data can be structured in several ways depending on the ingestion
    source.  We look for common shapes:
      - trace_data["events"]  (list of dicts)
      - trace_data["spans"]   (OpenTelemetry-like)
      - trace_data["steps"]   (agent frameworks)
    If nothing matches we wrap the entire data dict as a single pseudo-event.
    """
    events: List[TraceEvent] = []

    raw_events: list = []
    for key in ("events", "spans", "steps", "messages", "calls"):
        candidate = trace_data.get(key)
        if isinstance(candidate, list) and candidate:
            raw_events = candidate
            break

    if not raw_events:
        # Treat the whole trace data as one event
        raw_events = [trace_data]

    for raw in raw_events:
        if not isinstance(raw, dict):
            continue
        events.append(
            TraceEvent(
                name=raw.get("name", raw.get("role", "unknown")),
                type=raw.get("type", raw.get("kind", "unknown")),
                timestamp=raw.get("timestamp", raw.get("start_time")),
                duration_ms=_safe_float(raw.get("duration_ms", raw.get("duration"))),
                status=raw.get("status", raw.get("status_code")),
                error=raw.get("error", raw.get("exception", {}).get("message") if isinstance(raw.get("exception"), dict) else None),
                tokens_in=_safe_int(raw.get("tokens_in", raw.get("prompt_tokens", raw.get("input_tokens")))),
                tokens_out=_safe_int(raw.get("tokens_out", raw.get("completion_tokens", raw.get("output_tokens")))),
                model=raw.get("model", raw.get("model_id")),
                metadata={
                    k: v
                    for k, v in raw.items()
                    if k
                    not in {
                        "name",
                        "type",
                        "kind",
                        "timestamp",
                        "start_time",
                        "duration_ms",
                        "duration",
                        "status",
                        "status_code",
                        "error",
                        "exception",
                        "tokens_in",
                        "tokens_out",
                        "prompt_tokens",
                        "completion_tokens",
                        "input_tokens",
                        "output_tokens",
                        "model",
                        "model_id",
                        "role",
                    }
                },
            )
        )

    return events


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def _detect_error_events(events: List[TraceEvent]) -> List[Issue]:
    """Flag events that have explicit errors or failure status."""
    issues: List[Issue] = []
    for ev in events:
        if ev.error:
            issues.append(
                Issue(
                    severity="error",
                    category="error",
                    title=f"Error in '{ev.name}'",
                    description=ev.error,
                    event_name=ev.name,
                )
            )
        elif ev.status and ev.status.lower() in ("error", "failure", "failed"):
            issues.append(
                Issue(
                    severity="error",
                    category="error",
                    title=f"Failed event '{ev.name}'",
                    description=f"Event status: {ev.status}",
                    event_name=ev.name,
                )
            )
    return issues


def _detect_slow_events(
    events: List[TraceEvent],
    threshold_ms: float = 5_000,
) -> List[Issue]:
    """Flag events whose duration exceeds a threshold."""
    issues: List[Issue] = []
    for ev in events:
        if ev.duration_ms is not None and ev.duration_ms > threshold_ms:
            issues.append(
                Issue(
                    severity="warning",
                    category="latency",
                    title=f"Slow event '{ev.name}'",
                    description=f"Duration {ev.duration_ms:.0f} ms exceeds {threshold_ms:.0f} ms threshold.",
                    event_name=ev.name,
                )
            )
    return issues


def _detect_high_token_usage(
    events: List[TraceEvent],
    threshold: int = 10_000,
) -> List[Issue]:
    """Flag events with high token consumption."""
    issues: List[Issue] = []
    for ev in events:
        total = (ev.tokens_in or 0) + (ev.tokens_out or 0)
        if total > threshold:
            issues.append(
                Issue(
                    severity="warning",
                    category="cost",
                    title=f"High token usage in '{ev.name}'",
                    description=f"Total tokens: {total} (in={ev.tokens_in}, out={ev.tokens_out}).",
                    event_name=ev.name,
                )
            )
    return issues


def _generate_suggestions(issues: List[Issue]) -> List[Suggestion]:
    """Derive actionable suggestions from the detected issues."""
    suggestions: List[Suggestion] = []
    categories = {i.category for i in issues}

    if "error" in categories:
        error_count = sum(1 for i in issues if i.category == "error")
        suggestions.append(
            Suggestion(
                title="Fix errors in the trace",
                description=f"{error_count} error(s) detected. Review the failing events and add retry logic or input validation.",
                priority="high",
            )
        )

    if "latency" in categories:
        suggestions.append(
            Suggestion(
                title="Optimize slow spans",
                description="Consider caching, parallelism, or a faster model to reduce latency.",
                priority="medium",
            )
        )

    if "cost" in categories:
        suggestions.append(
            Suggestion(
                title="Reduce token usage",
                description="Trim system prompts, summarize context, or switch to a smaller model where quality allows.",
                priority="medium",
            )
        )

    if not suggestions:
        suggestions.append(
            Suggestion(
                title="Trace looks healthy",
                description="No significant issues detected. Consider adding more detailed instrumentation for deeper insights.",
                priority="low",
            )
        )

    return suggestions


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


@dataclass
class InvestigatorState:
    """LangGraph state for the investigator agent."""

    messages: List[BaseMessage] = field(default_factory=list)
    trace_id: Optional[str] = None
    trace_data: Optional[Dict[str, Any]] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    report: Optional[Dict[str, Any]] = None
    step: str = "start"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# LangGraph node functions
# ---------------------------------------------------------------------------


async def fetch_trace_node(state: InvestigatorState) -> Dict[str, Any]:
    """Node: retrieve the trace from LayerLens."""
    trace_id = state.trace_id
    if not trace_id:
        # Try to extract a trace ID from the last human message
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage) and msg.content:
                trace_id = msg.content.strip()
                break

    if not trace_id:
        return {
            "step": "error",
            "error": "No trace ID provided.",
            "messages": state.messages
            + [AIMessage(content="Please provide a trace ID to investigate.")],
        }

    data = await asyncio.to_thread(_get_trace, trace_id)
    if data is None:
        return {
            "step": "error",
            "error": f"Trace '{trace_id}' not found.",
            "messages": state.messages
            + [AIMessage(content=f"Could not find trace `{trace_id}`. Please check the ID.")],
        }

    msg = f"Fetched trace `{trace_id}` ({data.get('filename', 'unknown')}). Analyzing..."
    return {
        "trace_id": trace_id,
        "trace_data": data,
        "step": "analyze",
        "messages": state.messages + [AIMessage(content=msg)],
    }


async def analyze_node(state: InvestigatorState) -> Dict[str, Any]:
    """Node: run all analysis helpers on the extracted events."""
    if state.trace_data is None:
        return {"step": "error", "error": "No trace data to analyze."}

    raw_data = state.trace_data.get("data", {})
    events = _extract_events(raw_data)

    # Run detectors
    issues: List[Issue] = []
    issues.extend(_detect_error_events(events))
    issues.extend(_detect_slow_events(events))
    issues.extend(_detect_high_token_usage(events))

    suggestions = _generate_suggestions(issues)

    # Build summary line
    error_count = sum(1 for i in issues if i.severity == "error")
    warning_count = sum(1 for i in issues if i.severity == "warning")
    summary = f"{len(events)} event(s), {error_count} error(s), {warning_count} warning(s)."

    report = InvestigationReport(
        trace_id=state.trace_id or "",
        filename=state.trace_data.get("filename", "unknown"),
        created_at=state.trace_data.get("created_at", "unknown"),
        total_events=len(events),
        issues=issues,
        suggestions=suggestions,
        summary=summary,
    )

    # Serialise events for state (dataclass-friendly)
    events_dicts = [e.model_dump() for e in events]

    # Build human-readable message
    lines: List[str] = [f"**Investigation Report** for `{report.trace_id}`", ""]
    lines.append(f"File: {report.filename} | Created: {report.created_at}")
    lines.append(f"Events: {report.total_events} | {report.summary}")
    lines.append("")

    if issues:
        lines.append("**Issues:**")
        for issue in issues:
            icon = {"error": "!!!", "warning": "(!)", "info": "(i)"}.get(issue.severity, "   ")
            lines.append(f"  {icon} [{issue.category}] {issue.title}: {issue.description}")
        lines.append("")

    lines.append("**Suggestions:**")
    for s in suggestions:
        lines.append(f"  [{s.priority}] {s.title} -- {s.description}")

    msg = "\n".join(lines)

    return {
        "events": events_dicts,
        "report": report.model_dump(),
        "step": "done",
        "messages": state.messages + [AIMessage(content=msg)],
    }


async def error_node(state: InvestigatorState) -> Dict[str, Any]:
    """Terminal node for error cases."""
    return {"step": "done"}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_step(state: InvestigatorState) -> str:
    step = state.step
    if step == "analyze":
        return "analyze"
    if step == "error":
        return "error"
    return "done"


# ---------------------------------------------------------------------------
# Build the LangGraph StateGraph
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    graph = StateGraph(InvestigatorState)

    graph.add_node("fetch_trace", fetch_trace_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("error", error_node)

    graph.set_entry_point("fetch_trace")

    graph.add_conditional_edges(
        "fetch_trace",
        route_step,
        {
            "analyze": "analyze",
            "error": "error",
            "done": END,
        },
    )
    graph.add_edge("analyze", END)
    graph.add_edge("error", END)

    return graph


# Pre-compiled graph for import
investigator_graph = build_graph().compile()


# ---------------------------------------------------------------------------
# main() for test compatibility
# ---------------------------------------------------------------------------


def main() -> None:
    """Print usage information (for test / CI compatibility)."""
    print("Investigator Agent (LayerLens SDK)")
    print("=" * 40)
    print()
    print("This module exposes a LangGraph + CopilotKit agent that:")
    print("  1. Fetches a trace by ID from LayerLens")
    print("  2. Extracts events from the trace data")
    print("  3. Detects errors, slow spans, and high token usage")
    print("  4. Produces an investigation report with suggestions")
    print()
    print("Import `investigator_graph` and wire it into your CopilotKit server.")
    print()
    print("Required env var: LAYERLENS_STRATIX_API_KEY")


if __name__ == "__main__":
    main()
