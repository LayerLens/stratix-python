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

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, Dict, List, Optional

from copilotkit import CopilotKitMiddleware
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from layerlens import Stratix

logger = logging.getLogger(__name__)

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
# Backend tools — these call the LayerLens SDK
# ---------------------------------------------------------------------------


@tool
def list_judges() -> List[Dict[str, str]]:
    """List available LayerLens judges (evaluation criteria).

    Returns a list of ``{"id", "name", "goal"}`` dicts. Call this before
    asking the user which judge to use.
    """
    client = _get_client()
    resp = client.judges.get_many()
    if resp is None:
        return []
    return [
        {"id": j.id, "name": j.name, "goal": j.evaluation_goal}
        for j in resp.judges
    ]


@tool
def list_recent_traces(limit: int = 20) -> List[Dict[str, str]]:
    """List the most recent LayerLens traces in this project.

    Args:
        limit: Maximum number of traces to return (default 20).

    Returns a list of ``{"id", "filename", "created_at"}`` dicts sorted
    newest-first.
    """
    client = _get_client()
    resp = client.traces.get_many(
        page_size=limit, sort_by="created_at", sort_order="desc"
    )
    if resp is None:
        return []
    return [
        {
            "id": t.id,
            "filename": t.filename,
            "created_at": t.created_at,
        }
        for t in resp.traces
    ]


@tool
def run_trace_evaluation(trace_id: str, judge_id: str) -> Dict[str, Any]:
    """Start a LayerLens evaluation for a single trace against a judge.

    Args:
        trace_id: The trace to evaluate.
        judge_id: The judge (evaluation criteria) to use.

    Returns ``{"evaluation_id", "trace_id", "judge_id", "status"}``.
    Status starts as ``"pending"``; poll ``get_evaluation_result`` to
    get the final verdict.
    """
    client = _get_client()
    ev = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge_id)
    if ev is None:
        return {
            "evaluation_id": "",
            "trace_id": trace_id,
            "judge_id": judge_id,
            "status": "failed",
            "error": "create returned None",
        }
    status = ev.status.value if hasattr(ev.status, "value") else str(ev.status)
    return {
        "evaluation_id": ev.id,
        "trace_id": ev.trace_id,
        "judge_id": ev.judge_id,
        "status": status,
    }


@tool
def get_evaluation_result(evaluation_id: str) -> Dict[str, Any]:
    """Get the result of a previously-started LayerLens evaluation.

    Args:
        evaluation_id: The id returned by ``run_trace_evaluation``.

    Returns ``{"status", "passed", "score", "reasoning"}`` when the
    evaluation is finished, or ``{"status": "pending"}`` when it's
    still running. The LLM should call this repeatedly (with reasonable
    back-off) until ``status`` is ``"success"`` or ``"failure"``.
    """
    client = _get_client()
    ev = client.trace_evaluations.get(evaluation_id)
    if ev is None:
        return {"status": "not_found"}
    status = ev.status.value if hasattr(ev.status, "value") else str(ev.status)
    if status != "success":
        return {"status": status}
    results = client.trace_evaluations.get_results(id=evaluation_id)
    if results is None or results.score is None:
        return {"status": status}
    return {
        "status": status,
        "passed": bool(results.passed),
        "score": float(results.score),
        "reasoning": results.reasoning,
    }


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
You are LayerLens's evaluation assistant. Your job is to evaluate the
user's agent traces against judges and surface the results.

When the user asks to evaluate their traces, follow this flow:

1. Call ``list_recent_traces`` to see what traces are available. If the
   list is empty, tell the user to upload traces first and stop.
2. Call ``list_judges`` to see the evaluation criteria this project has.
   If the list is empty, tell the user to create a judge first and stop.
3. Call the ``confirm_judge`` tool with the full list of candidates from
   step 2. The frontend renders a picker; wait for the user to choose
   before continuing. The tool returns the chosen judge's id and name.
4. For each trace from step 1, call ``run_trace_evaluation(trace_id,
   judge_id)`` using the id the user picked in step 3.
5. For each evaluation id returned by step 4, call
   ``get_evaluation_result`` until its status is ``"success"`` or
   ``"failure"``. If an evaluation stays ``"pending"`` after several
   polls, move on and note it as pending in your summary.
6. Summarise the results in plain English — number of passes, failures,
   pending — plus any noteworthy reasoning from the judge verdicts.

Keep chat messages concise. Don't paste raw JSON blobs; use the tool
widgets to surface structured data.
"""


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------


def build_graph(model: Optional[Any] = None):
    """Construct the evaluator agent graph.

    Pattern matches CopilotKit's ``examples/integrations/langgraph-fastapi``
    reference and its ``hitl_in_chat_agent.py`` showcase: a ``create_agent``
    with backend tools, ``CopilotKitMiddleware``, and a system prompt that
    references a frontend-defined HITL tool (``confirm_judge``).

    Args:
        model: Override for the underlying chat model. Defaults to
            ``ChatOpenAI(model="gpt-4o-mini")`` which requires
            ``OPENAI_API_KEY``. Tests inject a fake model here.
    """
    if model is None:
        model = ChatOpenAI(model="gpt-4o-mini")
    return create_agent(
        model=model,
        tools=BACKEND_TOOLS,
        middleware=[CopilotKitMiddleware()],
        system_prompt=SYSTEM_PROMPT,
    )


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
    print("  LAYERLENS_STRATIX_API_KEY  — LayerLens API key")
    print("  OPENAI_API_KEY             — LLM for the agent")


if __name__ == "__main__":
    main()
