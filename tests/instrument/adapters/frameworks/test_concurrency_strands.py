"""Interleaved-run isolation for the Strands adapter (LAY-3576 / D1b).

THE INVARIANT: two runs driven through ONE ``StrandsAdapter`` instance whose
lifecycle events interleave must upload two traces with distinct trace_ids, each
holding exactly its own run's events — no cross-contamination, no lost events.

FIXED (D1b): the adapter no longer keeps run state in instance scalars
(``self._collector``, ``self._run_span_id``, ``self._timers``,
``self._model_span_ids``); it routes every invocation through
``_begin_run``/``_end_run`` so the collector + run span id + timers + the
per-run model.invoke span ids live in a per-run ``RunState`` pushed into
ContextVars (isolated per ``asyncio.Task`` / thread — the same mechanism
PydanticAIAdapter uses). The Strands hooks fire *inside* the agent invocation,
i.e. within that run's context, so they resolve their own run's collector.

The interleave is exercised deterministically (no real threads, no RNG) by
running each run's hook-handler calls inside its OWN ``contextvars.Context`` —
exactly how concurrent agent invocations in separate threads/tasks see
independent ContextVar state — across several fixed schedules (sequential,
fully interleaved, reversed, nested). Under the GIL a forced interleaving is a
stricter race detector for this code than naive threading, and a fixed schedule
cannot flap.
"""

from __future__ import annotations

import json
import contextvars
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from .conftest import record_for_schema_lock

strands_mod = pytest.importorskip("strands")
from strands.hooks.events import (  # noqa: E402
    AfterModelCallEvent,
    AfterInvocationEvent,
    BeforeModelCallEvent,
    BeforeInvocationEvent,
)

from layerlens.models import CreateTracesResponse  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.strands import (
    StrandsAdapter,
)  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_traces(mock_client: Any) -> List[Dict[str, Any]]:
    """Accumulate each uploaded trace payload separately (one entry per flush)."""
    traces: List[Dict[str, Any]] = []

    def _capture(path: str) -> CreateTracesResponse:
        with open(path) as f:
            data = json.load(f)
        traces.append(data[0])
        record_for_schema_lock(data[0].get("events", []))
        # Non-empty trace_ids or the upload counts as a REJECT (F-L7-002).
        return CreateTracesResponse(trace_ids=[data[0].get("trace_id") or "mock-trace-id"])

    mock_client.traces.upload.side_effect = _capture
    return traces


def _make_agent(name: str, model_id: str) -> Mock:
    """Mock Strands agent — only the attributes the adapter reads.

    ``agent_invocations`` is an empty list so ``_emit_per_cycle_tokens``
    has no Mock cycles to leak into payloads.
    """
    agent = Mock()
    agent.name = name
    agent.model = Mock()
    agent.model.config = {"model_id": model_id}
    agent.tool_names = []
    agent.system_prompt = None
    agent.event_loop_metrics = Mock()
    agent.event_loop_metrics.agent_invocations = []
    return agent


def _make_result(message: str) -> Mock:
    result = Mock()
    result.stop_reason = "end_turn"
    result.message = message
    return result


def _stop_response(stop_reason: str = "end_turn") -> Any:
    return AfterModelCallEvent.ModelStopResponse(message=Mock(), stop_reason=stop_reason)


# ---------------------------------------------------------------------------
# Isolation invariant
# ---------------------------------------------------------------------------

# Each schedule is a list of (run_key, action). Every schedule must yield two
# fully isolated traces — that is the post-D1b invariant.
_SCHEDULES = {
    "sequential": [("a", "start"), ("a", "model"), ("a", "end"), ("b", "start"), ("b", "model"), ("b", "end")],
    "interleaved": [("a", "start"), ("b", "start"), ("a", "model"), ("b", "model"), ("a", "end"), ("b", "end")],
    "reversed": [("b", "start"), ("a", "start"), ("b", "model"), ("a", "model"), ("b", "end"), ("a", "end")],
    "nested": [("a", "start"), ("b", "start"), ("b", "model"), ("b", "end"), ("a", "model"), ("a", "end")],
}

_MARKERS = {
    "a": {"prompt": "prompt-alpha", "agent": "agent-alpha", "model": "model-alpha", "answer": "answer-alpha"},
    "b": {"prompt": "prompt-beta", "agent": "agent-beta", "model": "model-beta", "answer": "answer-beta"},
}


@pytest.mark.parametrize("schedule_name", list(_SCHEDULES))
def test_interleaved_invocations_produce_two_isolated_traces(mock_client, schedule_name):
    traces = _collect_traces(mock_client)
    # Full capture so the run-content markers the isolation check searches for are
    # actually captured — the default standard() config redacts content.
    adapter = StrandsAdapter(mock_client, capture_config=CaptureConfig.full())
    adapter.connect()
    agents = {k: _make_agent(m["agent"], m["model"]) for k, m in _MARKERS.items()}

    # Each run gets its own Context — the deterministic stand-in for concurrent
    # agent invocations on separate threads/tasks, where ContextVar state is
    # independent. Running each hook in its run's context is what isolates them.
    ctxs = {k: contextvars.copy_context() for k in _MARKERS}

    for run_key, action in _SCHEDULES[schedule_name]:
        ctx, agent, m = ctxs[run_key], agents[run_key], _MARKERS[run_key]
        if action == "start":
            ctx.run(
                adapter._on_before_invocation,
                BeforeInvocationEvent(agent=agent, invocation_state={}, messages=m["prompt"]),
            )
        elif action == "model":
            ctx.run(adapter._on_before_model, BeforeModelCallEvent(agent=agent, invocation_state={}))
            ctx.run(
                adapter._on_after_model,
                AfterModelCallEvent(agent=agent, invocation_state={}, stop_response=_stop_response()),
            )
        elif action == "end":
            ctx.run(
                adapter._on_after_invocation,
                AfterInvocationEvent(agent=agent, invocation_state={}, result=_make_result(m["answer"])),
            )

    adapter.disconnect()

    summaries = [[(e["event_type"], e.get("span_name")) for e in t["events"]] for t in traces]
    assert len(traces) == 2, (
        f"[{schedule_name}] each invocation must flush its own trace: expected 2, got {len(traces)}: {summaries}"
    )
    assert len({t["trace_id"] for t in traces}) == 2, f"[{schedule_name}] runs must not share a trace_id"

    by_prompt: Dict[str, Dict[str, Any]] = {}
    for trace in traces:
        inputs = [e for e in trace["events"] if e["event_type"] == "agent.input"]
        assert len(inputs) == 1, f"[{schedule_name}] each trace holds exactly one run's agent.input"
        by_prompt[inputs[0]["payload"]["input"]] = trace
    assert set(by_prompt) == {m["prompt"] for m in _MARKERS.values()}, f"[{schedule_name}] one trace per prompt"

    for key, m in _MARKERS.items():
        text = json.dumps(by_prompt[m["prompt"]]["events"])
        for marker in (m["prompt"], m["agent"], m["model"], m["answer"]):
            assert marker in text, f"[{schedule_name}] trace for {m['prompt']!r} lost its own marker {marker!r}"
        other = _MARKERS["b" if key == "a" else "a"]
        for marker in (other["prompt"], other["agent"], other["model"], other["answer"]):
            assert marker not in text, f"[{schedule_name}] trace for {m['prompt']!r} contaminated by {marker!r}"
