"""CopilotKit Evaluator Agent — LayerLens SDK edition.

Uses CopilotKit's current HITL idiom: ``langchain.agents.create_agent`` +
``CopilotKitMiddleware``. The LLM drives the conversation and calls tools;
the human-in-the-loop step is a **frontend-defined tool** (``confirm_judge``)
registered with ``useCopilotAction`` in the browser. ``CopilotKitMiddleware``
bridges the frontend tool into the LLM's toolbelt, so from the graph's
perspective it looks like any other tool call.

This replaces an earlier design that used a custom ``StateGraph`` with
``langgraph.types.interrupt()`` for HITL. That path hit two upstream bugs
in ``ag-ui-langgraph`` (``ag-ui-protocol/ag-ui#1582`` and ``#1584``) and
needed a local workaround subclass. This version sidesteps the
``interrupt()`` pipeline entirely — the LLM + frontend-tool pattern is
what CopilotKit's active showcases use (``hitl_in_chat_agent.py``,
``interrupt_agent.py``) and is the pattern we recommend.

Flow driven by the system prompt:

1. LLM calls ``list_recent_traces`` to see what's available.
2. LLM calls ``list_judges`` to see evaluation criteria.
3. LLM calls ``confirm_judge`` (frontend tool) with the candidates.
   The frontend renders a picker; the user selects one; the picker
   resolves with the chosen judge. The LLM receives the selection as
   the tool's return value.
4. LLM calls ``run_trace_evaluation`` for each trace with the chosen
   judge, then ``get_evaluation_result`` to fetch outcomes.
5. LLM summarises the pass/fail results for the user.

Requires ``OPENAI_API_KEY`` in the environment. Set
``LAYERLENS_STRATIX_API_KEY`` so the tools can call the LayerLens API.
"""

import os
import json
import logging
import operator
import threading
from typing import Any, Dict, Optional, Annotated

from copilotkit import CopilotKitMiddleware
from langgraph.types import Command
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState
from copilotkit.langgraph import copilotkit_emit_state
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools.base import InjectedToolCallId
from langchain.agents.middleware import AgentState
from langgraph.checkpoint.memory import InMemorySaver

from layerlens import Stratix

logger = logging.getLogger(__name__)
# When run under uvicorn the root logger is configured at INFO; make sure
# our messages propagate so tool activity is visible in the harness log.
logger.setLevel(logging.INFO)

# Polling bounds for get_evaluation_result so a stuck job doesn't hang a
# customer's chat indefinitely. The LLM will call get_evaluation_result
# repeatedly; we don't block the tool itself.
DEFAULT_POLL_DELAY_SECONDS = 1.0
MAX_POLL_WAIT_SECONDS = 60.0


# ---------------------------------------------------------------------------
# Lazy, thread-safe LayerLens client
# ---------------------------------------------------------------------------

_client_lock = threading.Lock()
_client: Optional[Stratix] = None


def _get_client() -> Stratix:
    """Return (and lazily create) a module-level ``Stratix`` client.

    Reads ``LAYERLENS_STRATIX_API_KEY`` automatically from the environment.
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = Stratix()
    return _client


# ---------------------------------------------------------------------------
# Agent state schema
#
# Extending ``AgentState`` lets each tool return a ``Command(update={...})``
# that mutates these fields. The frontend's ``useCoAgentStateRender`` then
# renders progressive cards (recent traces, available judges, running
# evaluations, completed results) keyed off the state — the canonical
# CopilotKit pattern from ``coagents-research-canvas``.
#
# ``Annotated[..., operator.add]`` makes a field accumulate across tool
# calls (we use this for ``evaluations`` and ``results`` since the LLM
# fires one ``run_trace_evaluation`` per trace and one
# ``get_evaluation_result`` per evaluation). The non-annotated fields
# replace on update.
# ---------------------------------------------------------------------------


class EvaluatorState(AgentState):
    """Extended state schema surfaced to the frontend via STATE_SNAPSHOT."""

    traces: list[dict[str, Any]]
    judges: list[dict[str, Any]]
    evaluations: Annotated[list[dict[str, Any]], operator.add]
    results: Annotated[list[dict[str, Any]], operator.add]


# ---------------------------------------------------------------------------
# Backend tools — these call the LayerLens SDK
# ---------------------------------------------------------------------------


@tool
async def list_judges(
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
    state: Annotated[dict, InjectedState] = None,
    config: RunnableConfig = None,
) -> Command:
    """List available LayerLens judges (evaluation criteria).

    Updates ``state.judges`` with ``{"id", "name", "goal"}`` dicts so the
    frontend can render the available-judges card.
    """
    client = _get_client()
    resp = client.judges.get_many()
    judges: list[dict[str, Any]] = []
    if resp is not None:
        judges = [{"id": j.id, "name": j.name, "goal": j.evaluation_goal} for j in resp.judges]
    # Push state to the frontend immediately so the canvas updates as
    # this tool completes — without this, ag-ui-langgraph batches state
    # snapshots until the LLM's tool-calling round wraps up, which makes
    # ``MetricCard`` / ``JudgesCard`` lag the actual data.
    if config is not None:
        merged = {**(state or {}), "judges": judges}
        await copilotkit_emit_state(config, merged)
    return Command(
        update={
            "judges": judges,
            "messages": [
                ToolMessage(
                    content=json.dumps(judges),
                    tool_call_id=tool_call_id,
                    name="list_judges",
                )
            ],
        }
    )


@tool
async def list_recent_traces(
    limit: int = 20,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
    state: Annotated[dict, InjectedState] = None,
    config: RunnableConfig = None,
) -> Command:
    """List the most recent LayerLens traces in this project.

    Args:
        limit: Maximum number of traces to return (default 20).

    Updates ``state.traces`` with rich trace dicts (id, filename,
    created_at, model, duration_ms, tokens, evaluations_count) so the
    frontend's ``TraceCard`` can render real per-trace metrics.
    """
    client = _get_client()
    resp = client.traces.get_many(page_size=limit, sort_by="created_at", sort_order="desc")
    traces: list[dict[str, Any]] = []
    if resp is not None:
        for t in resp.traces:
            data = getattr(t, "data", None) or {}
            if hasattr(data, "model_dump"):
                data = data.model_dump()
            traces.append(
                {
                    "id": t.id,
                    "filename": t.filename,
                    "created_at": t.created_at,
                    "model": (data.get("model") if isinstance(data, dict) else None) or "",
                    "duration_ms": (data.get("latency_ms") if isinstance(data, dict) else None) or 0,
                    "tokens": (data.get("tokens") if isinstance(data, dict) else None) or 0,
                    "evaluations_count": getattr(t, "evaluations_count", 0) or 0,
                }
            )
    if config is not None:
        merged = {**(state or {}), "traces": traces}
        await copilotkit_emit_state(config, merged)
    return Command(
        update={
            "traces": traces,
            "messages": [
                ToolMessage(
                    content=json.dumps(traces),
                    tool_call_id=tool_call_id,
                    name="list_recent_traces",
                )
            ],
        }
    )


@tool
async def run_trace_evaluation(
    trace_id: str,
    judge_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
    state: Annotated[dict, InjectedState] = None,
    config: RunnableConfig = None,
) -> Command:
    """Start a LayerLens evaluation for a single trace against a judge.

    Args:
        trace_id: The trace to evaluate.
        judge_id: The judge (evaluation criteria) to use.

    Appends an ``{"evaluation_id", "trace_id", "judge_id", "status"}``
    entry to ``state.evaluations`` so the frontend can render a
    "running" card. Status starts as ``"pending"`` (or whatever the
    backend reports); poll ``get_evaluation_result`` for the verdict.
    """
    client = _get_client()
    ev = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge_id)
    if ev is None:
        ev_data = {
            "evaluation_id": "",
            "trace_id": trace_id,
            "judge_id": judge_id,
            "status": "failed",
            "error": "create returned None",
        }
    else:
        status = ev.status.value if hasattr(ev.status, "value") else str(ev.status)
        ev_data = {
            "evaluation_id": ev.id,
            "trace_id": ev.trace_id,
            "judge_id": ev.judge_id,
            "status": status,
        }
    if config is not None:
        prev_evals = (state or {}).get("evaluations", []) or []
        merged = {**(state or {}), "evaluations": [*prev_evals, ev_data]}
        await copilotkit_emit_state(config, merged)
    return Command(
        update={
            "evaluations": [ev_data],
            "messages": [
                ToolMessage(
                    content=json.dumps(ev_data),
                    tool_call_id=tool_call_id,
                    name="run_trace_evaluation",
                )
            ],
        }
    )


@tool
async def get_evaluation_result(
    evaluation_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
    state: Annotated[dict, InjectedState] = None,
    config: RunnableConfig = None,
) -> Command:
    """Get the result of a previously-started LayerLens evaluation.

    Args:
        evaluation_id: The id returned by ``run_trace_evaluation``.

    Returns ``{"status", "passed", "score", "reasoning"}`` once the
    evaluation completes, or ``{"status": <state>}`` when still running.
    On success, also appends an entry to ``state.results`` so the
    frontend can render an EvaluationCard for it.
    """
    client = _get_client()
    ev = client.trace_evaluations.get(evaluation_id)
    if ev is None:
        result_data: Dict[str, Any] = {
            "evaluation_id": evaluation_id,
            "status": "not_found",
        }
    else:
        status = ev.status.value if hasattr(ev.status, "value") else str(ev.status)
        # ``trace_id`` and ``judge_id`` come straight off the evaluation
        # record so the frontend can resolve them against ``state.judges``
        # (judge name) without a second round-trip.
        base = {
            "evaluation_id": evaluation_id,
            "status": status,
            "trace_id": getattr(ev, "trace_id", "") or "",
            "judge_id": getattr(ev, "judge_id", "") or "",
        }
        if status != "success":
            result_data = base
        else:
            details = client.trace_evaluations.get_results(id=evaluation_id)
            if details is None or details.score is None:
                result_data = base
            else:
                result_data = {
                    **base,
                    "passed": bool(details.passed),
                    "score": float(details.score),
                    "reasoning": details.reasoning,
                }
    update: Dict[str, Any] = {
        "messages": [
            ToolMessage(
                content=json.dumps(result_data),
                tool_call_id=tool_call_id,
                name="get_evaluation_result",
            )
        ],
    }
    # Only append to ``results`` once we have a real verdict — pending
    # polls shouldn't pile cards onto the UI.
    if result_data.get("status") == "success" and "score" in result_data:
        update["results"] = [result_data]
        if config is not None:
            prev_results = (state or {}).get("results", []) or []
            merged = {**(state or {}), "results": [*prev_results, result_data]}
            await copilotkit_emit_state(config, merged)
    return Command(update=update)


# Expose the backend tools for tests and for the server wiring. The
# ``confirm_judge`` HITL tool is defined on the frontend via
# ``useCopilotAction`` and bridged into the LLM's toolbelt by
# ``CopilotKitMiddleware``.
BACKEND_TOOLS = [
    list_judges,
    list_recent_traces,
    run_trace_evaluation,
    get_evaluation_result,
]


# ---------------------------------------------------------------------------
# System prompt — guides the LLM through the evaluation flow
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are LayerLens's evaluation assistant. The frontend renders a rich
canvas showing traces, judges, and evaluation results from agent
state. Your chat messages are NOT the primary UI — keep them short,
factual, and forward-looking.

Flow (call tools in this order, do not skip or reorder):

1. ``list_recent_traces`` with the DEFAULT limit (do not pass any
   ``limit`` argument). The canvas needs the full trace list.
2. ``list_judges``
3. ``confirm_judge`` with NO arguments — the frontend reads candidates
   from state and shows a picker. Wait for the user's selection.
4. From the trace list returned in step 1, take the FIRST FIVE traces
   only and call ``run_trace_evaluation(trace_id, judge_id)`` once for
   each, using the judge_id the user picked. Do not call
   run_trace_evaluation for more than five traces.
5. Real LayerLens evaluations take 30+ seconds to complete, longer
   than a chat turn should block. Do NOT poll. The frontend polls
   the backend's ``/evaluations/{id}`` endpoint and folds completed
   verdicts into the canvas as each one finishes.
   - You MAY call ``get_evaluation_result(evaluation_id)`` ONCE per
     evaluation to capture any that already happen to be done.
     Do not call it more than once for the same id.
6. End with ONE short message. Choose the variant that matches what
   you actually observed (do NOT include zero-counts or stale clauses):

   - If K = 0 (no evaluations are done yet):
     "Started <N> evaluations against <judge>. They will appear in
     the canvas as they finish."

   - If 0 < K < N (some done, some pending):
     "Started <N> evaluations against <judge>. <K> already complete
     (<K-passed> passed, <K-failed> failed). The rest will appear in
     the canvas as they finish."

   - If K == N (all done already):
     "Evaluated <N> traces against <judge>: <K-passed> passed,
     <K-failed> failed."

   K = number of get_evaluation_result calls that returned
   status="success". Do NOT classify pending evaluations as failed,
   and never include "0 already complete" or similar zero-clauses.

Chat narration rules:
- Do NOT echo what the canvas already shows (counts, judge name,
  per-trace verdicts, scores).
- Do NOT repeat steps you have already completed ("now I'll load the
  judges" after you've already listed them).
- One short status sentence per phase is enough; silence is fine.
- NEVER mention internal field names like ``state.traces``.
"""


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------


def _default_model() -> Any:
    """Build the default chat model from environment variables.

    The sample accepts any OpenAI-compatible endpoint, not just OpenAI
    itself. Three environment variables shape the call:

    - ``OPENAI_API_KEY`` (required) — credential for the endpoint.
    - ``OPENAI_BASE_URL`` (optional) — full base URL of an
      OpenAI-compatible server. Leave unset for OpenAI itself.
    - ``OPENAI_MODEL`` (optional) — model name to send. Defaults to
      ``gpt-4o-mini``.

    Anything that speaks OpenAI's HTTP shape works: hosted gateways,
    self-hosted inference (Ollama, LM Studio, vLLM, llama.cpp), or
    private deployments. For non-OpenAI-compatible models (Anthropic,
    Google, etc.) pass the relevant LangChain ``BaseChatModel``
    directly via ``build_graph(model=ChatAnthropic(...))``.
    """
    kwargs: Dict[str, Any] = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    }
    if api_key := os.environ.get("OPENAI_API_KEY"):
        kwargs["api_key"] = api_key
    if base_url := os.environ.get("OPENAI_BASE_URL"):
        kwargs["base_url"] = base_url
    # Parallel tool calls are enabled (the default) so the LLM can fan
    # out ``run_trace_evaluation`` and ``get_evaluation_result`` across
    # many traces in a single turn. The HITL picker race that motivated
    # disabling this is no longer a concern: ``JudgePicker`` subscribes
    # to ``useCoAgent`` directly and re-renders when ``state.judges``
    # populates, so a same-turn emission of ``confirm_judge`` is fine.
    return ChatOpenAI(**kwargs)


def build_graph(model: Optional[Any] = None):
    """Construct the evaluator agent graph.

    Pattern matches CopilotKit's ``examples/integrations/langgraph-fastapi``
    reference and its ``hitl_in_chat_agent.py`` showcase: a ``create_agent``
    with backend tools, ``CopilotKitMiddleware``, and a system prompt that
    references a frontend-defined HITL tool (``confirm_judge``).

    Args:
        model: Override for the underlying chat model. Defaults to
            ``_default_model()`` which honours ``OPENAI_API_KEY``,
            ``OPENAI_BASE_URL``, and ``OPENAI_MODEL`` so any
            OpenAI-compatible endpoint works. Tests inject a fake
            model here.
    """
    if model is None:
        model = _default_model()
    return create_agent(
        model=model,
        tools=BACKEND_TOOLS,
        middleware=[CopilotKitMiddleware()],
        system_prompt=SYSTEM_PROMPT,
        state_schema=EvaluatorState,
        # ag-ui-langgraph's endpoint calls ``graph.aget_state(config)``
        # to look up per-thread state on each request -- which fails
        # with "No checkpointer set" if the graph wasn't compiled with
        # one. Production deployments should swap to a durable saver
        # (Postgres, SQLite, Redis, LangGraph Platform).
        checkpointer=InMemorySaver(),
    )
    # NOTE: recursion_limit is set on ``LangGraphAGUIAgent`` (in
    # ``build_agui_agent`` below) rather than via ``graph.with_config``
    # because ag-ui-langgraph constructs its own RunnableConfig from
    # ``self.config`` per request — ``with_config`` bindings don't
    # propagate through that path.


# Pre-compiled graph for import by the FastAPI server. Customers with
# ``OPENAI_API_KEY`` set will have this initialised at import time; those
# without a key can still inspect ``BACKEND_TOOLS`` and ``SYSTEM_PROMPT``
# without instantiating the model.
try:
    evaluator_graph = build_graph()
except Exception as exc:  # pragma: no cover — informative import-time guard
    logger.warning(
        "Could not build default evaluator_graph (%s). Set OPENAI_API_KEY "
        "or call build_graph(model=<your_model>) explicitly.",
        exc,
    )
    evaluator_graph = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# build_agui_agent — minimal runId workaround for ag-ui-langgraph#1582
# ---------------------------------------------------------------------------
#
# ``ag_ui_langgraph.LangGraphAgent._handle_stream_events`` (inherited by
# ``copilotkit.LangGraphAGUIAgent``) overwrites ``self.active_run["id"]``
# with each LangGraph event's internal chain ``run_id``, so the eventual
# ``RUN_FINISHED`` is emitted with LangGraph's UUID instead of
# ``input.run_id``. Tracked at
# https://github.com/ag-ui-protocol/ag-ui/issues/1582.
#
# The current ``@ag-ui/client@0.0.52`` validator does NOT enforce runId
# continuity (only within-stream start/finish ordering), so this
# mismatch isn't user-visible today. But: older clients did enforce it,
# future strict ones likely will, and a paying customer's logs
# shouldn't carry an obvious protocol violation. So this thin subclass
# restores ``input.run_id`` on terminal events.
#
# This is the ONLY workaround needed for the current evaluator
# architecture. Bug #1584 (duplicate RUN_STARTED in the
# ``has_active_interrupts`` path) is not reachable here because the
# evaluator does not call ``langgraph.types.interrupt()`` -- HITL is
# handled by a frontend tool (``confirm_judge``) instead.
#
# Remove this subclass once #1582 ships and the fixed
# ``ag-ui-langgraph`` release is pinned in ``requirements.lock``.


def build_agui_agent(**kwargs):
    """Build a ``LangGraphAGUIAgent`` with the runId workaround applied.

    Caller-supplied ``config`` is merged onto our defaults — in
    particular ``recursion_limit`` is bumped from LangGraph's default
    (25) to 200 so a 20-trace evaluation run with parallel tool calls
    has plenty of headroom for the chat / tool / chat / poll cycle.
    """
    from typing import Optional as _Optional

    from ag_ui.core import EventType
    from copilotkit import LangGraphAGUIAgent

    default_config = {"recursion_limit": 200}
    merged_config = {**default_config, **(kwargs.pop("config", None) or {})}
    kwargs["config"] = merged_config

    class _RunIdPreserving(LangGraphAGUIAgent):
        def __init__(self, **inner_kwargs):
            super().__init__(**inner_kwargs)
            self._client_run_id: _Optional[str] = None

        async def run(self, input):
            self._client_run_id = input.run_id
            async for event in super().run(input):
                event_type = getattr(event, "type", None)
                if (
                    event_type in (EventType.RUN_FINISHED, EventType.RUN_ERROR)
                    and self._client_run_id
                    and hasattr(event, "run_id")
                ):
                    event.run_id = self._client_run_id
                yield event

    return _RunIdPreserving(**kwargs)


# ---------------------------------------------------------------------------
# main() for test compatibility
# ---------------------------------------------------------------------------


def main() -> None:
    """Print usage information (for test / CI compatibility)."""
    print("Evaluator Agent (LayerLens SDK)")
    print("=" * 40)
    print()
    print("A CopilotKit + LangGraph agent that:")
    print("  1. Fetches recent LayerLens traces.")
    print("  2. Fetches available judges (evaluation criteria).")
    print("  3. Uses a frontend `confirm_judge` tool to ask the user")
    print("     which judge to apply (rendered via useCopilotAction).")
    print("  4. Runs evaluations and summarises the results.")
    print()
    print("Import `evaluator_graph` into your CopilotKit FastAPI server.")
    print()
    print("Required env vars:")
    print("  LAYERLENS_STRATIX_API_KEY  -- LayerLens API key")
    print("  OPENAI_API_KEY             -- credentials for the LLM")
    print()
    print("Optional (for non-OpenAI endpoints):")
    print("  OPENAI_BASE_URL  -- any OpenAI-compatible base URL")
    print("                      (any OpenAI-compatible server)")
    print("  OPENAI_MODEL     -- model name override (default: gpt-4o-mini)")
    print()
    print("For non-OpenAI-compatible models, call build_graph(model=...)")
    print("with any LangChain BaseChatModel (ChatAnthropic, ChatGoogle, ...).")


if __name__ == "__main__":
    main()
