"""
CopilotKit Evaluator Agent -- LayerLens SDK edition.

This agent lets an operator:
  1. List available judges
  2. List recent traces
  3. Pick a judge (with human-in-the-loop confirmation via LangGraph interrupt)
  4. Run evaluations for every selected trace against the confirmed judge
  5. Poll / fetch evaluation results

All LayerLens API calls go through the Python SDK (`layerlens.Stratix`)
instead of raw httpx.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from langgraph.graph import END, StateGraph

# CopilotKit helpers -- match the wiring in CopilotKit's official
# ``examples/integrations/langgraph-fastapi`` reference sample so the
# agent speaks the protocol the CopilotKit frontend expects.
from copilotkit import CopilotKitState
from copilotkit.langgraph import copilotkit_interrupt
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from layerlens import Stratix

logger = logging.getLogger(__name__)

MAX_POLL_ATTEMPTS = 30
POLL_INITIAL_INTERVAL = 2.0
POLL_MAX_INTERVAL = 15.0
POLL_BACKOFF_FACTOR = 1.5

# ---------------------------------------------------------------------------
# Module-level SDK client (lazy)
# ---------------------------------------------------------------------------

_client_lock = threading.Lock()
_client: Optional[Stratix] = None


def _get_client() -> Stratix:
    """Return (and lazily create) a module-level Stratix client.

    Reads ``LAYERLENS_STRATIX_API_KEY`` automatically from the environment.
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:  # double-check after acquiring lock
                _client = Stratix()
    return _client


# ---------------------------------------------------------------------------
# Lightweight data-transfer objects used inside the graph state
# ---------------------------------------------------------------------------


# These DTOs are Pydantic models (not plain dataclasses) because LangGraph
# persists ``EvaluatorState`` through the checkpointer whenever the graph
# hits an ``interrupt()``. The default ``JsonPlusSerializer`` serializes
# Pydantic models out of the box; plain dataclasses raise
# ``TypeError: Type is not msgpack serializable`` during checkpointing,
# which breaks human-in-the-loop resume.
class JudgeInfo(BaseModel):
    """Minimal representation of a LayerLens judge for the UI."""

    id: str
    name: str
    goal: str
    created_at: str


class TraceInfo(BaseModel):
    """Minimal representation of a LayerLens trace for the UI."""

    id: str
    filename: str
    created_at: str


class EvaluationInfo(BaseModel):
    """Tracks a single trace-evaluation that has been kicked off."""

    evaluation_id: str
    trace_id: str
    judge_id: str
    status: str
    passed: Optional[bool] = None
    score: Optional[float] = None
    reasoning: Optional[str] = None


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class EvaluatorState(CopilotKitState, total=False):
    """LangGraph state for the evaluator agent.

    Inherits from ``CopilotKitState`` so ``messages`` uses LangGraph's
    ``add_messages`` reducer (node updates APPEND rather than REPLACE) and
    the ``copilotkit`` field carries frontend-injected properties. This
    matches CopilotKit's official ``langgraph-fastapi`` reference sample.
    """

    judges: List[JudgeInfo]
    traces: List[TraceInfo]
    selected_judge: Optional[JudgeInfo]
    confirmed_judge: Optional[JudgeInfo]
    evaluations: List[EvaluationInfo]
    step: str
    error: Optional[str]
    poll_count: int


# ---------------------------------------------------------------------------
# SDK helper wrappers
# ---------------------------------------------------------------------------


def _list_judges() -> List[JudgeInfo]:
    """Fetch all judges via the SDK and map to JudgeInfo."""
    client = _get_client()
    try:
        resp = client.judges.get_many()
        if resp is None:
            return []
        return [
            JudgeInfo(
                id=j.id,
                name=j.name,
                goal=j.evaluation_goal,
                created_at=j.created_at,
            )
            for j in resp.judges
        ]
    except Exception as exc:
        logger.error("Failed to list judges: %s", exc)
        return []


def _list_traces(limit: int = 20) -> List[TraceInfo]:
    """Fetch recent traces via the SDK, sorted newest-first."""
    client = _get_client()
    try:
        resp = client.traces.get_many(
            page_size=limit,
            sort_by="created_at",
            sort_order="desc",
        )
        if resp is None:
            return []
        return [
            TraceInfo(
                id=t.id,
                filename=t.filename,
                created_at=t.created_at,
            )
            for t in resp.traces
        ]
    except Exception as exc:
        logger.error("Failed to list traces: %s", exc)
        return []


def _create_evaluation(trace_id: str, judge_id: str) -> Optional[EvaluationInfo]:
    """Kick off an evaluation for a single trace/judge pair."""
    client = _get_client()
    try:
        te = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge_id)
        if te is None:
            return None
        return EvaluationInfo(
            evaluation_id=te.id,
            trace_id=te.trace_id,
            judge_id=te.judge_id,
            status=te.status.value if hasattr(te.status, "value") else str(te.status),
        )
    except Exception as exc:
        logger.error("Failed to create evaluation: %s", exc)
        return None


def _get_evaluation(evaluation_id: str) -> Optional[EvaluationInfo]:
    """Fetch the current status (and results if finished) for an evaluation."""
    client = _get_client()
    try:
        te = client.trace_evaluations.get(evaluation_id)
        if te is None:
            return None
        info = EvaluationInfo(
            evaluation_id=te.id,
            trace_id=te.trace_id,
            judge_id=te.judge_id,
            status=te.status.value if hasattr(te.status, "value") else str(te.status),
        )

        # If the evaluation finished, pull detailed results
        if info.status == "success":
            results_resp = client.trace_evaluations.get_results(id=evaluation_id)
            if results_resp and results_resp.score is not None:
                info.passed = results_resp.passed
                info.score = results_resp.score
                info.reasoning = results_resp.reasoning

        return info
    except Exception as exc:
        logger.error("Failed to get evaluation %s: %s", evaluation_id, exc)
        return None


# ---------------------------------------------------------------------------
# LangGraph node functions
# ---------------------------------------------------------------------------


async def fetch_judges_node(state: EvaluatorState) -> Dict[str, Any]:
    """Node: fetch the list of available judges."""
    judges = await asyncio.to_thread(_list_judges)
    if not judges:
        return {
            "judges": [],
            "step": "error",
            "error": "No judges found. Create a judge in LayerLens first.",
            # With add_messages reducer, returning a list APPENDS.
            "messages": [AIMessage(content="No judges found. Please create a judge in LayerLens first.")],
        }

    summary = "\n".join(f"  - **{j.name}** (`{j.id}`): {j.goal}" for j in judges)
    return {
        "judges": judges,
        "step": "fetch_traces",
        "messages": [AIMessage(content=f"Found {len(judges)} judge(s):\n{summary}")],
    }


async def fetch_traces_node(state: EvaluatorState) -> Dict[str, Any]:
    """Node: fetch the most recent traces."""
    traces = await asyncio.to_thread(_list_traces, 20)
    if not traces:
        return {
            "traces": [],
            "step": "error",
            "error": "No traces found. Upload traces to LayerLens first.",
            "messages": [AIMessage(content="No traces found. Please upload traces first.")],
        }

    summary = "\n".join(f"  - `{t.id}` ({t.filename}, {t.created_at})" for t in traces[:10])
    return {
        "traces": traces,
        "step": "confirm_judge",
        "messages": [AIMessage(content=f"Found {len(traces)} recent trace(s). Showing first 10:\n{summary}")],
    }


async def confirm_judge_node(state: EvaluatorState) -> Dict[str, Any]:
    """Node: ask the human to confirm which judge to use (HITL)."""
    judges = state.get("judges") or []
    if not judges:
        return {"step": "error", "error": "No judges available."}

    # Default to the first judge; the user can override by id or name.
    default = judges[0]
    prompt = (
        f"Which judge should I use? Default: **{default.name}** (`{default.id}`).\n"
        "Reply with a judge ID or name, or 'ok' to accept the default."
    )

    # ``copilotkit_interrupt`` (NOT the raw ``langgraph.types.interrupt``)
    # emits the prompt as an ``AIMessage`` the CopilotKit chat UI renders.
    # The bare ``interrupt(prompt)`` pattern emits a CUSTOM event the UI
    # ignores, so the user never sees the confirmation question.
    answer, _response = copilotkit_interrupt(message=prompt)

    selected = default
    if answer and answer.strip().lower() != "ok":
        needle = answer.strip().lower()
        for j in judges:
            if needle in (j.id.lower(), j.name.lower()):
                selected = j
                break

    return {
        "selected_judge": selected,
        "confirmed_judge": selected,
        "step": "run_evaluations",
        "messages": [AIMessage(content=f"Using judge **{selected.name}** (`{selected.id}`).")],
    }


async def run_evaluations_node(state: EvaluatorState) -> Dict[str, Any]:
    """Node: kick off evaluations for every trace with the confirmed judge."""
    judge = state.get("confirmed_judge")
    if judge is None:
        return {"step": "error", "error": "No judge confirmed."}

    results: List[EvaluationInfo] = []
    for trace in state.get("traces") or []:
        info = await asyncio.to_thread(_create_evaluation, trace.id, judge.id)
        if info is not None:
            results.append(info)

    if not results:
        return {
            "evaluations": [],
            "step": "error",
            "error": "All evaluation requests failed.",
            "messages": [AIMessage(content="Failed to create any evaluations.")],
        }

    return {
        "evaluations": results,
        "step": "poll_results",
        "messages": [AIMessage(content=f"Started {len(results)} evaluation(s) with judge **{judge.name}**.")],
    }


async def poll_results_node(state: EvaluatorState) -> Dict[str, Any]:
    """Node: poll evaluation results with bounded retries and backoff."""
    evaluations = state.get("evaluations") or []
    updated: List[EvaluationInfo] = []
    for ev in evaluations:
        refreshed = await asyncio.to_thread(_get_evaluation, ev.evaluation_id)
        updated.append(refreshed if refreshed is not None else ev)

    finished = [e for e in updated if e.status in ("success", "failure")]
    pending = [e for e in updated if e.status not in ("success", "failure")]

    lines: List[str] = []
    for e in finished:
        verdict = "PASS" if e.passed else "FAIL" if e.passed is not None else "N/A"
        lines.append(f"  - trace `{e.trace_id}`: {verdict} (score={e.score})")
    for e in pending:
        lines.append(f"  - trace `{e.trace_id}`: {e.status}")

    summary = "\n".join(lines) if lines else "(no evaluations)"

    poll_count = state.get("poll_count", 0)

    if not pending or poll_count >= MAX_POLL_ATTEMPTS:
        if pending:
            summary += (
                f"\n\n(Stopped polling after {MAX_POLL_ATTEMPTS} attempts; {len(pending)} evaluation(s) still pending.)"
            )
        return {
            "evaluations": updated,
            "step": "done",
            "messages": [AIMessage(
                content=f"Evaluation results ({len(finished)} done, {len(pending)} pending):\n{summary}"
            )],
        }

    # Sleep between polls with exponential backoff to avoid hammering the API.
    poll_delay = min(
        POLL_INITIAL_INTERVAL * (POLL_BACKOFF_FACTOR**poll_count),
        POLL_MAX_INTERVAL,
    )
    await asyncio.sleep(poll_delay)

    return {
        "evaluations": updated,
        "step": "poll_results",
        "poll_count": poll_count + 1,
        "messages": [AIMessage(
            content=f"Evaluation results ({len(finished)} done, {len(pending)} pending):\n{summary}"
        )],
    }


async def error_node(state: EvaluatorState) -> Dict[str, Any]:
    """Terminal node when an error has occurred."""
    return {"step": "done"}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


# Routing tokens are distinct from node names to avoid a collision with the
# ``error`` field on ``EvaluatorState``. LangGraph rejects a node whose name
# matches a state key on older versions (``'error' is already being used as
# a state key``); the node name ``handle_error`` sidesteps that while the
# ``error`` routing token remains purely a conditional-edge key.
def route_step(state: EvaluatorState) -> str:
    step = state.get("step")
    if step == "fetch_traces":
        return "fetch_traces"
    if step == "confirm_judge":
        return "confirm_judge"
    if step == "run_evaluations":
        return "run_evaluations"
    if step == "poll_results":
        return "poll_results"
    if step == "error":
        return "error"
    return "done"


# ---------------------------------------------------------------------------
# Build the LangGraph StateGraph
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    graph = StateGraph(EvaluatorState)

    graph.add_node("fetch_judges", fetch_judges_node)
    graph.add_node("fetch_traces", fetch_traces_node)
    graph.add_node("confirm_judge", confirm_judge_node)
    graph.add_node("run_evaluations", run_evaluations_node)
    graph.add_node("poll_results", poll_results_node)
    graph.add_node("handle_error", error_node)

    graph.set_entry_point("fetch_judges")

    graph.add_conditional_edges(
        "fetch_judges",
        route_step,
        {
            "fetch_traces": "fetch_traces",
            "error": "handle_error",
            "done": END,
        },
    )
    graph.add_conditional_edges(
        "fetch_traces",
        route_step,
        {
            "confirm_judge": "confirm_judge",
            "error": "handle_error",
            "done": END,
        },
    )
    graph.add_conditional_edges(
        "confirm_judge",
        route_step,
        {
            "run_evaluations": "run_evaluations",
            "error": "handle_error",
            "done": END,
        },
    )
    graph.add_conditional_edges(
        "run_evaluations",
        route_step,
        {
            "poll_results": "poll_results",
            "error": "handle_error",
            "done": END,
        },
    )
    graph.add_conditional_edges(
        "poll_results",
        route_step,
        {
            "poll_results": "poll_results",
            "done": END,
        },
    )
    graph.add_edge("handle_error", END)

    return graph


# ---------------------------------------------------------------------------
# Checkpointer
# ---------------------------------------------------------------------------
#
# A checkpointer is MANDATORY for this graph because ``confirm_judge_node``
# calls ``interrupt()``. LangGraph uses the checkpointer to persist state at
# the pause boundary so the run can resume from the same thread later.
#
# Without a checkpointer, the resume call (``Command(resume=...)``) produces
# zero events -- the AG-UI stream never emits ``RUN_FINISHED``, and the
# CopilotKit frontend then rejects subsequent messages with:
#
#   "Cannot send 'RUN_STARTED' while a run is still active. The previous
#    run must be finished with 'RUN_FINISHED' before starting a new run."
#
# This sample uses ``InMemorySaver`` so it runs out-of-the-box with no
# external dependencies. In-memory checkpoints are lost on process restart,
# so for any deployment beyond a local demo you MUST swap to a durable
# checkpointer. See ``samples/copilotkit/README.md`` for the full matrix
# (Postgres / SQLite / Redis / LangGraph Platform).
#
# Quick Postgres swap (requires ``pip install langgraph-checkpoint-postgres``):
#
#     from langgraph.checkpoint.postgres import PostgresSaver
#
#     DB_URI = "postgresql://user:pass@host:5432/langgraph?sslmode=require"
#     checkpointer = PostgresSaver.from_conn_string(DB_URI)
#     checkpointer.setup()   # one-time: creates the checkpoint tables
#     evaluator_graph = build_graph().compile(checkpointer=checkpointer)
#
# Register the Pydantic DTOs on the checkpoint serializer's msgpack allowlist.
# Without this, LangGraph 1.x emits a ``Deserializing unregistered type``
# warning every time the graph resumes from a checkpoint, and future releases
# will refuse to deserialize unregistered types outright. The kwarg is
# langgraph>=1.0 only -- on older versions we fall back to the default serde,
# which does not enforce the allowlist.
_DTOS = (
    (__name__, "JudgeInfo"),
    (__name__, "TraceInfo"),
    (__name__, "EvaluationInfo"),
)
try:
    _serde = JsonPlusSerializer(allowed_msgpack_modules=_DTOS)
except TypeError:
    _serde = JsonPlusSerializer()
checkpointer = InMemorySaver(serde=_serde)
evaluator_graph = build_graph().compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# RunIdPreservingAgent -- local workaround for ag-ui-protocol/ag-ui#1582
# ---------------------------------------------------------------------------
#
# Upstream bug: ``ag_ui_langgraph.LangGraphAgent._handle_stream_events``
# (inherited by ``copilotkit.LangGraphAGUIAgent``) overwrites
# ``self.active_run["id"]`` on every LangGraph event with the event's
# internal chain ``run_id``. By the time ``RUN_FINISHED`` is dispatched,
# ``active_run["id"]`` is LangGraph's UUID -- not the ``input.run_id``
# the client supplied. ``@copilotkit/runtime`` tracks active runs by
# the client runId, so the mismatch surfaces in the browser as:
#
#   RUN_ERROR {code: "INCOMPLETE_STREAM",
#              message: "Cannot send 'RUN_STARTED' while a run is still
#                        active. The previous run must be finished with
#                        'RUN_FINISHED' before starting a new run."}
#
# We restore ``input.run_id`` on terminal events so the AG-UI protocol
# state machine can correlate the stream closure. Remove this subclass
# once https://github.com/ag-ui-protocol/ag-ui/issues/1582 ships and
# the fixed ``ag-ui-langgraph`` release is pinned in this sample's
# requirements.


def _build_langgraph_agui_agent(**kwargs):
    """Factory for the CopilotKit agent wrapper with the runId workaround.

    Kept as a factory (rather than a module-level class import) so the
    sample imports cleanly even when ``copilotkit`` / ``ag_ui_langgraph``
    are not installed -- users who only want to inspect the graph don't
    need the full runtime stack.
    """
    from typing import Optional as _Optional

    from ag_ui.core import EventType
    from copilotkit import LangGraphAGUIAgent

    class RunIdPreservingAgent(LangGraphAGUIAgent):
        def __init__(self, **inner_kwargs):
            super().__init__(**inner_kwargs)
            self._client_run_id: _Optional[str] = None

        async def run(self, input):
            self._client_run_id = input.run_id
            async for ev in super().run(input):
                yield ev

        def _dispatch_event(self, event):
            event_type = getattr(event, "type", None)
            if (
                event_type in (EventType.RUN_FINISHED, EventType.RUN_ERROR)
                and self._client_run_id
                and hasattr(event, "run_id")
            ):
                event.run_id = self._client_run_id
            return super()._dispatch_event(event)

    return RunIdPreservingAgent(**kwargs)


# ---------------------------------------------------------------------------
# main() for test compatibility
# ---------------------------------------------------------------------------


def main() -> None:
    """Print usage information (for test / CI compatibility)."""
    print("Evaluator Agent (LayerLens SDK)")
    print("=" * 40)
    print()
    print("This module exposes a LangGraph + CopilotKit agent that:")
    print("  1. Lists LayerLens judges")
    print("  2. Lists recent traces")
    print("  3. Asks the operator to confirm a judge (human-in-the-loop)")
    print("  4. Runs evaluations for each trace")
    print("  5. Polls for results")
    print()
    print("Import `evaluator_graph` and wire it into your CopilotKit server.")
    print()
    print("Required env var: LAYERLENS_STRATIX_API_KEY")


if __name__ == "__main__":
    main()
