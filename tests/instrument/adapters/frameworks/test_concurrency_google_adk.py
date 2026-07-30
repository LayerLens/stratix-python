"""Interleaved-run isolation for the Google ADK adapter (LAY-3576 / D1b).

THE INVARIANT: two runs driven through ONE ``GoogleADKAdapter`` instance whose
lifecycle events interleave must upload two traces with distinct trace_ids, each
holding exactly its own run's events — no cross-contamination, no lost events.

FIXED (D1b): the adapter no longer keeps run state in instance scalars; it routes
every run through ``_begin_run``/``_end_run`` so the collector + run/agent span
ids + current-agent marker + config-seen set + timers live in a per-run
``RunState`` pushed into ContextVars (isolated per ``asyncio.Task`` / thread —
the same mechanism PydanticAIAdapter uses). The ADK Runner invokes the plugin
callbacks *inside* each run's task, i.e. within that run's context, so the
handlers resolve their own run's collector.

The interleave is exercised deterministically (no real threads, no RNG) by
running each run's handler calls inside its OWN ``contextvars.Context`` — exactly
how concurrent runs in separate threads/tasks see independent ContextVar state —
across several fixed schedules (sequential, fully interleaved, reversed, nested).
Under the GIL a forced interleaving is a stricter race detector for this code
than naive threading, and a fixed schedule cannot flap.
"""

from __future__ import annotations

import json
import contextvars
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from .conftest import record_for_schema_lock

pytest.importorskip("google.adk")

from layerlens.models import CreateTracesResponse  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.google_adk import (
    GoogleADKAdapter,
)  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers (copied minimal — do not import private helpers from siblings)
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


def _make_invocation_context(agent_name: str, user_content: str, run_key: str) -> Mock:
    """Mock ADK InvocationContext.

    ``sub_agents`` is a real list (``_agent_tree`` slices it) and the session
    extras are pinned to ``None`` so no Mock leaks into JSON payloads.
    """
    ctx = Mock()
    agent = Mock()
    agent.name = agent_name
    agent.sub_agents = []
    ctx.agent = agent
    ctx.invocation_id = f"inv-{run_key}"
    ctx.user_content = user_content
    session = Mock()
    session.id = f"sess-{run_key}"
    session.user_id = None
    session.app_name = None
    session.state = None
    ctx.session = session
    return ctx


def _make_agent(name: str) -> Mock:
    """Mock ADK sub-agent — attributes pinned so no Mock leaks into payloads."""
    agent = Mock()
    agent.name = name
    agent.description = None
    agent.instruction = None
    agent.model = None
    agent.tools = []
    agent.sub_agents = []
    return agent


def _make_callback_context(agent_name: str, user_content: str) -> Mock:
    ctx = Mock()
    ctx.agent_name = agent_name
    ctx.user_content = user_content
    ctx.function_call_id = None
    del ctx.session
    return ctx


# ---------------------------------------------------------------------------
# Isolation invariant
# ---------------------------------------------------------------------------

# Each schedule is a list of (run_key, action). Every schedule must yield two
# fully isolated traces — that is the post-D1b invariant.
_SCHEDULES = {
    "sequential": [("a", "start"), ("a", "step"), ("a", "end"), ("b", "start"), ("b", "step"), ("b", "end")],
    "interleaved": [("a", "start"), ("b", "start"), ("a", "step"), ("b", "step"), ("a", "end"), ("b", "end")],
    "reversed": [("b", "start"), ("a", "start"), ("b", "step"), ("a", "step"), ("b", "end"), ("a", "end")],
    "nested": [("a", "start"), ("b", "start"), ("b", "step"), ("b", "end"), ("a", "step"), ("a", "end")],
}

# Per-run markers: ask (run-level input), root agent, worker agent (mid-run),
# and the worker's followup input. Distinct strings so contamination is visible.
_MARKERS = {
    "a": {"ask": "ask-alpha", "root": "root-alpha", "worker": "worker-alpha", "followup": "followup-alpha"},
    "b": {"ask": "ask-beta", "root": "root-beta", "worker": "worker-beta", "followup": "followup-beta"},
}


@pytest.mark.parametrize("schedule_name", list(_SCHEDULES))
def test_interleaved_runs_produce_two_isolated_traces(mock_client, schedule_name):
    traces = _collect_traces(mock_client)
    # Full capture so the run-content markers the isolation check searches for are
    # actually captured — the default standard() config redacts content.
    adapter = GoogleADKAdapter(mock_client, capture_config=CaptureConfig.full())
    adapter.connect()

    inv = {k: _make_invocation_context(m["root"], m["ask"], run_key=k) for k, m in _MARKERS.items()}
    workers = {k: _make_agent(m["worker"]) for k, m in _MARKERS.items()}
    cbs = {k: _make_callback_context(m["worker"], m["followup"]) for k, m in _MARKERS.items()}

    # Each run gets its own Context — the deterministic stand-in for concurrent
    # runs on separate threads/tasks, where ContextVar state is independent.
    # Running each handler in its run's context is what isolates them.
    ctxs = {k: contextvars.copy_context() for k in _MARKERS}

    for run_key, action in _SCHEDULES[schedule_name]:
        ctx = ctxs[run_key]
        if action == "start":
            ctx.run(adapter._on_before_run, inv[run_key])
        elif action == "step":
            ctx.run(adapter._on_before_agent, workers[run_key], cbs[run_key])
            ctx.run(adapter._on_after_agent, workers[run_key], cbs[run_key])
        elif action == "end":
            ctx.run(adapter._on_after_run, inv[run_key])

    adapter.disconnect()

    summaries = [[(e["event_type"], e.get("span_name")) for e in t["events"]] for t in traces]
    assert len(traces) == 2, (
        f"[{schedule_name}] each run must flush its own trace: expected 2, got {len(traces)}: {summaries}"
    )
    assert len({t["trace_id"] for t in traces}) == 2, f"[{schedule_name}] runs must not share a trace_id"

    markers = {m["ask"]: (m["root"], m["worker"], m["followup"]) for m in _MARKERS.values()}
    by_ask: Dict[str, Dict[str, Any]] = {}
    for trace in traces:
        run_inputs = [
            e for e in trace["events"] if e["event_type"] == "agent.input" and e["payload"].get("input") in markers
        ]
        assert len(run_inputs) == 1, (
            f"[{schedule_name}] each trace holds exactly one run-level agent.input, got "
            f"{[e['payload'].get('input') for e in run_inputs]}"
        )
        by_ask[run_inputs[0]["payload"]["input"]] = trace
    assert set(by_ask) == set(markers), f"[{schedule_name}] expected one trace per run input, got {sorted(by_ask)}"

    for ask, own_markers in markers.items():
        text = json.dumps(by_ask[ask]["events"])
        for marker in (ask, *own_markers):
            assert marker in text, f"[{schedule_name}] trace for {ask!r} lost its own marker {marker!r}"
        (other_ask,) = [a for a in markers if a != ask]
        for marker in (other_ask, *markers[other_ask]):
            assert marker not in text, f"[{schedule_name}] trace for {ask!r} contaminated by {marker!r}"
