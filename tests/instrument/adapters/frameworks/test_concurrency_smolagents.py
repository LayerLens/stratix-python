"""Interleaved-run isolation guard for the SmolAgents adapter (LAY-3576 / T2).

THE INVARIANT: two runs driven through ONE ``SmolAgentsAdapter`` instance,
with lifecycle events interleaved (start A, start B, mid-run A, mid-run B,
end A, end B), must upload two traces with distinct trace_ids, each holding
exactly its own run's events — no cross-contamination, no lost events.

The adapter keeps run state in instance scalars
(``src/layerlens/instrument/adapters/frameworks/smolagents.py`` —
``self._collector``, ``self._run_span_id``, ``self._current_step_span_id``,
``self._step_count``, ``self._timers``), so run B's start replaces run A's
collector: run A's opening events are dropped unflushed, both runs' mid-run
events land in run B's collector, one merged trace flushes when run A ends,
and run B's closing events vanish (no collector left).

Expected XFAIL (strict): this is the RED guard for the D1b
collector-convergence work (stability report §3.1), which moves the
self-flushing adapters onto the per-run RunState/ContextVar isolation that
PydanticAIAdapter already uses (see test_concurrency.py). When D1b lands,
the strict xfail turns into a strict XPASS failure and the marker must be
removed. The interleave is single-threaded direct handler calls, so the
corruption is deterministic and the xfail cannot flap.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from .conftest import record_for_schema_lock

smolagents = pytest.importorskip("smolagents")
from smolagents import ToolCall, ActionStep  # noqa: E402
from smolagents.memory import Timing  # noqa: E402
from smolagents.monitoring import TokenUsage  # noqa: E402

from layerlens.instrument.adapters.frameworks.smolagents import (
    SmolAgentsAdapter,
)  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers (copied minimal — do not import private helpers from siblings)
# ---------------------------------------------------------------------------


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
    """Mock smolagents agent — only the attributes the adapter reads."""
    agent = MagicMock()
    agent.name = name
    agent.model = MagicMock()
    agent.model.model_id = model_id
    agent.tools = None
    agent.managed_agents = None
    return agent


def _make_action_step(tool_name: str, observations: str) -> ActionStep:
    """Real smolagents ActionStep carrying run-distinctive markers."""
    step = ActionStep(
        step_number=1,
        timing=Timing(start_time=100.0, end_time=101.0),
    )
    step.tool_calls = [ToolCall(name=tool_name, arguments={"q": tool_name}, id=f"tc-{tool_name}")]
    step.token_usage = TokenUsage(input_tokens=10, output_tokens=5)
    step.model_output = None
    step.observations = observations
    step.error = None
    step.is_final_answer = False
    step.code_action = None
    return step


# ---------------------------------------------------------------------------
# Isolation invariant
# ---------------------------------------------------------------------------

XFAIL_REASON = (
    "LAY-3576: SmolAgentsAdapter keeps run state in instance scalars "
    "(smolagents.py — self._collector, self._run_span_id, self._current_step_span_id, "
    "self._step_count, self._timers); interleaved runs on one instance corrupt traces. "
    "RED guard for the D1b collector-convergence work (stability report §3.1) — "
    "NOT fixed in phase 4."
)


@pytest.mark.xfail(strict=True, reason=XFAIL_REASON)
def test_interleaved_runs_produce_two_isolated_traces(mock_client):
    traces = _collect_traces(mock_client)
    adapter = SmolAgentsAdapter(mock_client)
    agent_a = _make_mock_agent(name="agent-alpha", model_id="model-alpha")
    agent_b = _make_mock_agent(name="agent-beta", model_id="model-beta")
    adapter.connect(target=agent_a)

    # Deterministic single-thread interleave: start A, start B, mid A, mid B,
    # end A, end B.  (_handle_action_step is the unwrapped step handler — it
    # raises loudly on a malformed step instead of swallowing the error.)
    adapter._on_run_start(agent_a, "task-alpha")
    adapter._on_run_start(agent_b, "task-beta")
    adapter._handle_action_step(_make_action_step("tool-alpha", "obs-alpha"), agent_a)
    adapter._handle_action_step(_make_action_step("tool-beta", "obs-beta"), agent_b)
    adapter._on_run_end(agent_a, "result-alpha", None)
    adapter._on_run_end(agent_b, "result-beta", None)
    adapter.disconnect()

    summaries = [[(e["event_type"], e.get("span_name")) for e in t["events"]] for t in traces]
    assert traces, "SETUP BUG (not the isolation invariant): no trace was uploaded at all"
    assert len(traces) == 2, (
        f"Each run must flush its own trace: expected 2 uploads, got {len(traces)}. "
        f"Uploaded traces as (event_type, span_name): {summaries}"
    )

    trace_ids = {t["trace_id"] for t in traces}
    assert len(trace_ids) == 2, f"The two runs must not share a trace_id: {trace_ids}"

    markers = {
        "task-alpha": ("agent-alpha", "tool-alpha", "obs-alpha", "result-alpha"),
        "task-beta": ("agent-beta", "tool-beta", "obs-beta", "result-beta"),
    }
    by_task: Dict[str, Dict[str, Any]] = {}
    for trace in traces:
        inputs = [e for e in trace["events"] if e["event_type"] == "agent.input"]
        assert len(inputs) == 1, (
            f"Each trace must hold exactly one run's agent.input, got {[e['payload'].get('input') for e in inputs]}"
        )
        by_task[inputs[0]["payload"]["input"]] = trace
    assert set(by_task) == set(markers), f"Expected one trace per task, got {sorted(by_task)}"

    for task, own_markers in markers.items():
        text = json.dumps(by_task[task]["events"])
        for marker in (task, *own_markers):
            assert marker in text, f"Trace for {task!r} lost its own event marker {marker!r}"
        (other_task,) = [t for t in markers if t != task]
        for marker in (other_task, *markers[other_task]):
            assert marker not in text, f"Trace for {task!r} contaminated by other run's marker {marker!r}"
