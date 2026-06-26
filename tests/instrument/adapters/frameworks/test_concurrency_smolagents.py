"""Interleaved-run isolation for the SmolAgents adapter (LAY-3576 / D1b).

THE INVARIANT: two runs driven through ONE ``SmolAgentsAdapter`` instance whose
lifecycle events interleave must upload two traces with distinct trace_ids, each
holding exactly its own run's events — no cross-contamination, no lost events.

FIXED (D1b): the adapter no longer keeps run state in instance scalars; it routes
every run through ``_begin_run``/``_end_run`` so the collector + span ids + step
count + timers live in a per-run ``RunState`` pushed into ContextVars (isolated
per ``asyncio.Task`` / thread — the same mechanism PydanticAIAdapter uses). The
step callbacks fire *inside* ``agent.run()``, i.e. within that run's context, so
they resolve their own run's collector.

The interleave is exercised deterministically (no real threads, no RNG) by
running each run's handler calls inside its OWN ``contextvars.Context`` — exactly
how concurrent ``agent.run()`` calls in separate threads/tasks see independent
ContextVar state — across several fixed schedules (sequential, fully interleaved,
reversed, nested). Under the GIL a forced interleaving is a stricter race
detector for this code than naive threading, and a fixed schedule cannot flap.
"""

from __future__ import annotations

import json
import contextvars
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from .conftest import record_for_schema_lock

smolagents = pytest.importorskip("smolagents")
from smolagents import ToolCall, ActionStep  # noqa: E402
from smolagents.memory import Timing  # noqa: E402
from smolagents.monitoring import TokenUsage  # noqa: E402

from layerlens.instrument.adapters.frameworks.smolagents import SmolAgentsAdapter  # noqa: E402


def _collect_traces(mock_client: Any) -> List[Dict[str, Any]]:
    """Accumulate each uploaded trace payload separately (one entry per flush)."""
    traces: List[Dict[str, Any]] = []

    def _capture(path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        traces.append(data[0])
        record_for_schema_lock(data[0].get("events", []))

    mock_client.traces.upload.side_effect = _capture
    return traces


def _make_mock_agent(name: str, model_id: str) -> MagicMock:
    agent = MagicMock()
    agent.name = name
    agent.model = MagicMock()
    agent.model.model_id = model_id
    agent.tools = None
    agent.managed_agents = None
    return agent


def _make_action_step(tool_name: str, observations: str) -> ActionStep:
    step = ActionStep(step_number=1, timing=Timing(start_time=100.0, end_time=101.0))
    step.tool_calls = [ToolCall(name=tool_name, arguments={"q": tool_name}, id=f"tc-{tool_name}")]
    step.token_usage = TokenUsage(input_tokens=10, output_tokens=5)
    step.model_output = None
    step.observations = observations
    step.error = None
    step.is_final_answer = False
    step.code_action = None
    return step


# Each schedule is a list of (run_key, action). Every schedule must yield two
# fully isolated traces — that is the post-D1b invariant.
_SCHEDULES = {
    "sequential": [("a", "start"), ("a", "step"), ("a", "end"), ("b", "start"), ("b", "step"), ("b", "end")],
    "interleaved": [("a", "start"), ("b", "start"), ("a", "step"), ("b", "step"), ("a", "end"), ("b", "end")],
    "reversed": [("b", "start"), ("a", "start"), ("b", "step"), ("a", "step"), ("b", "end"), ("a", "end")],
    "nested": [("a", "start"), ("b", "start"), ("b", "step"), ("b", "end"), ("a", "step"), ("a", "end")],
}

_MARKERS = {
    "a": {
        "task": "task-alpha",
        "agent": "agent-alpha",
        "tool": "tool-alpha",
        "obs": "obs-alpha",
        "result": "result-alpha",
    },
    "b": {"task": "task-beta", "agent": "agent-beta", "tool": "tool-beta", "obs": "obs-beta", "result": "result-beta"},
}


@pytest.mark.parametrize("schedule_name", list(_SCHEDULES))
def test_interleaved_runs_produce_two_isolated_traces(mock_client, schedule_name):
    traces = _collect_traces(mock_client)
    adapter = SmolAgentsAdapter(mock_client)
    agents = {k: _make_mock_agent(m["agent"], f"model-{k}") for k, m in _MARKERS.items()}
    adapter.connect(target=agents["a"])

    # Each run gets its own Context — the deterministic stand-in for concurrent
    # agent.run() calls on separate threads/tasks, where ContextVar state is
    # independent. Running each step in its run's context is what isolates them.
    ctxs = {k: contextvars.copy_context() for k in _MARKERS}

    for run_key, action in _SCHEDULES[schedule_name]:
        ctx, agent, m = ctxs[run_key], agents[run_key], _MARKERS[run_key]
        if action == "start":
            ctx.run(adapter._on_run_start, agent, m["task"])
        elif action == "step":
            ctx.run(adapter._handle_action_step, _make_action_step(m["tool"], m["obs"]), agent)
        elif action == "end":
            ctx.run(adapter._on_run_end, agent, m["result"], None)

    adapter.disconnect()

    summaries = [[(e["event_type"], e.get("span_name")) for e in t["events"]] for t in traces]
    assert len(traces) == 2, (
        f"[{schedule_name}] each run must flush its own trace: expected 2, got {len(traces)}: {summaries}"
    )
    assert len({t["trace_id"] for t in traces}) == 2, f"[{schedule_name}] runs must not share a trace_id"

    by_task: Dict[str, Dict[str, Any]] = {}
    for trace in traces:
        inputs = [e for e in trace["events"] if e["event_type"] == "agent.input"]
        assert len(inputs) == 1, f"[{schedule_name}] each trace holds exactly one run's agent.input"
        by_task[inputs[0]["payload"]["input"]] = trace
    assert set(by_task) == {m["task"] for m in _MARKERS.values()}, f"[{schedule_name}] one trace per task"

    for key, m in _MARKERS.items():
        text = json.dumps(by_task[m["task"]]["events"])
        for marker in (m["task"], m["agent"], m["tool"], m["obs"], m["result"]):
            assert marker in text, f"[{schedule_name}] trace for {m['task']!r} lost its own marker {marker!r}"
        other = _MARKERS["b" if key == "a" else "a"]
        for marker in (other["task"], other["tool"], other["obs"], other["result"]):
            assert marker not in text, f"[{schedule_name}] trace for {m['task']!r} contaminated by {marker!r}"
