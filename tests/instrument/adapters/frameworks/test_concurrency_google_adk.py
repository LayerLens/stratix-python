"""Interleaved-run isolation guard for the Google ADK adapter (LAY-3576 / T2).

THE INVARIANT: two runs driven through ONE ``GoogleADKAdapter`` instance,
with lifecycle events interleaved (start A, start B, mid-run A, mid-run B,
end A, end B), must upload two traces with distinct trace_ids, each holding
exactly its own run's events — no cross-contamination, no lost events.

The adapter keeps run state in instance scalars
(``src/layerlens/instrument/adapters/frameworks/google_adk.py`` —
``self._collector``, ``self._run_span_id``, ``self._agent_span_ids``), so
run B's before_run replaces run A's collector: run A's opening events are
dropped unflushed, both runs' agent-level events land in run B's collector,
one merged trace flushes when run A's after_run arrives, and run B's
closing events vanish (no collector left).

Expected XFAIL (strict): this is the RED guard for the D1b
collector-convergence work (stability report §3.1), which moves the
self-flushing adapters onto the per-run RunState/ContextVar isolation that
PydanticAIAdapter already uses (see test_concurrency.py). When D1b lands,
the strict xfail turns into a strict XPASS failure and the marker must be
removed. The interleave is single-threaded direct plugin-handler calls, so
the corruption is deterministic and the xfail cannot flap.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

pytest.importorskip("google.adk")

from layerlens.instrument.adapters.frameworks.google_adk import (
    GoogleADKAdapter,
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

XFAIL_REASON = (
    "LAY-3576: GoogleADKAdapter keeps run state in instance scalars "
    "(google_adk.py — self._collector, self._run_span_id, self._agent_span_ids); "
    "interleaved runs on one instance corrupt traces. "
    "RED guard for the D1b collector-convergence work (stability report §3.1) — "
    "NOT fixed in phase 4."
)


@pytest.mark.xfail(strict=True, reason=XFAIL_REASON)
def test_interleaved_runs_produce_two_isolated_traces(mock_client):
    traces = _collect_traces(mock_client)
    adapter = GoogleADKAdapter(mock_client)
    adapter.connect()

    ctx_a = _make_invocation_context("root-alpha", "ask-alpha", run_key="alpha")
    ctx_b = _make_invocation_context("root-beta", "ask-beta", run_key="beta")
    worker_a = _make_agent("worker-alpha")
    worker_b = _make_agent("worker-beta")
    cb_a = _make_callback_context("worker-alpha", "followup-alpha")
    cb_b = _make_callback_context("worker-beta", "followup-beta")

    # Deterministic single-thread interleave: start A, start B, mid A, mid B,
    # end A, end B.
    adapter._on_before_run(ctx_a)
    adapter._on_before_run(ctx_b)
    adapter._on_before_agent(worker_a, cb_a)
    adapter._on_before_agent(worker_b, cb_b)
    adapter._on_after_agent(worker_a, cb_a)
    adapter._on_after_agent(worker_b, cb_b)
    adapter._on_after_run(ctx_a)
    adapter._on_after_run(ctx_b)
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
        "ask-alpha": ("root-alpha", "worker-alpha", "followup-alpha"),
        "ask-beta": ("root-beta", "worker-beta", "followup-beta"),
    }
    by_ask: Dict[str, Dict[str, Any]] = {}
    for trace in traces:
        run_inputs = [
            e for e in trace["events"] if e["event_type"] == "agent.input" and e["payload"].get("input") in markers
        ]
        assert len(run_inputs) == 1, (
            f"Each trace must hold exactly one run-level agent.input, got "
            f"{[e['payload'].get('input') for e in run_inputs]}"
        )
        by_ask[run_inputs[0]["payload"]["input"]] = trace
    assert set(by_ask) == set(markers), f"Expected one trace per run input, got {sorted(by_ask)}"

    for ask, own_markers in markers.items():
        text = json.dumps(by_ask[ask]["events"])
        for marker in (ask, *own_markers):
            assert marker in text, f"Trace for {ask!r} lost its own event marker {marker!r}"
        (other_ask,) = [a for a in markers if a != ask]
        for marker in (other_ask, *markers[other_ask]):
            assert marker not in text, f"Trace for {ask!r} contaminated by other run's marker {marker!r}"
