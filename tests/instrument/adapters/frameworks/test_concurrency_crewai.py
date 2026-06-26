"""Interleaved-run isolation for the CrewAI adapter (LAY-3576 / D1b).

THE INVARIANT: two crew kickoffs driven through ONE ``CrewAIAdapter`` instance
whose lifecycle events interleave must upload two traces with distinct
trace_ids, each holding exactly its own run's events — no cross-contamination,
no lost events.

FIXED (D1b): the adapter no longer keeps run state in instance scalars. CrewAI's
typed event bus dispatches every handler through a fresh
``contextvars.copy_context()`` on a thread-pool worker, so the ContextVar
``RunState`` migration used by smolagents/google_adk/strands is impossible here.
Instead the bus stamps event LINEAGE (``event_id`` / ``parent_event_id`` /
``started_event_id``) SYNCHRONOUSLY in the emitting thread, *before* the
thread-pool dispatch (``crewai/events/event_bus.py`` ``_prepare_event``), so the
lineage lives ON THE EVENT and is immune to ``copy_context``. The adapter keeps a
locked per-run map keyed by the root event_id and resolves each event to its
owning run by walking that lineage.

Because the fix reads lineage off the event, a forced single-thread interleaving
with correct lineage is fully deterministic (and a stricter race detector under
the GIL than naive threading, which cannot flap). We construct the crewai events,
then SET the lineage fields manually exactly as the bus's ``_prepare_event`` would
stamp them, and drive the handlers interleaved across several fixed schedules.
"""

from __future__ import annotations

import sys
import json
from typing import Any, Dict, List
from collections import Counter

import pytest

from .conftest import record_for_schema_lock

if sys.version_info < (3, 10):
    pytest.skip("crewai requires Python >= 3.10", allow_module_level=True)

crewai = pytest.importorskip("crewai")
pytest.importorskip("crewai.events")

# The adapter needs crewai's modern typed event bus (the 1.14 line); the base lock
# resolves whatever the cross-adapter solve allows (0.193.2 or 1.6.1 by platform),
# both too old. Only the pinned matrix (crewai==1.14.6) row is supported; skip below.
from packaging.version import Version  # noqa: E402

if Version(crewai.__version__) < Version("1.14"):
    pytest.skip(
        f"crewai adapter requires >= 1.14; got {crewai.__version__}",
        allow_module_level=True,
    )

from crewai.events import (  # noqa: E402
    TaskStartedEvent,
    TaskCompletedEvent,
    CrewKickoffStartedEvent,
    CrewKickoffCompletedEvent,
    crewai_event_bus,
)
from crewai.tasks.task_output import TaskOutput  # noqa: E402

from layerlens.instrument.adapters.frameworks.crewai import CrewAIAdapter  # noqa: E402

# -- Helpers --


def _collect_traces(mock_client: Any) -> List[Dict[str, Any]]:
    """Set up mock_client to accumulate SEPARATE trace payloads per upload."""
    traces: List[Dict[str, Any]] = []

    def _capture(path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        traces.append(data[0])
        record_for_schema_lock(data[0].get("events", []))

    mock_client.traces.upload.side_effect = _capture
    return traces


def _assert_isolated_traces(
    traces: List[Dict[str, Any]],
    markers: List[str],
    expected_counts: Dict[str, int],
) -> None:
    """The LAY-3576 invariant: one trace per run, keyed by content marker.

    Each marker must appear in exactly one uploaded trace; that trace must
    contain none of the other runs' markers, and exactly its own run's
    events (by event_type counts).
    """
    assert len(traces) == len(markers), (
        f"Expected {len(markers)} uploaded traces (one per run), got {len(traces)} — "
        "interleaved runs merged or lost a trace"
    )
    trace_ids = {t["trace_id"] for t in traces}
    assert len(trace_ids) == len(markers), f"Traces must have distinct trace_ids, got {trace_ids}"

    for marker in markers:
        own = [t for t in traces if marker in json.dumps(t["events"])]
        assert len(own) == 1, f"Run marker {marker!r} must appear in exactly 1 trace, found in {len(own)}"
        trace = own[0]
        blob = json.dumps(trace["events"])
        for other in markers:
            if other != marker:
                assert other not in blob, f"Trace for run {marker!r} is contaminated with run {other!r} events"
        counts = Counter(e["event_type"] for e in trace["events"])
        assert counts == Counter(expected_counts), (
            f"Trace for run {marker!r} lost or gained events: got {dict(counts)}, expected {expected_counts}"
        )
        assert all(e["trace_id"] == trace["trace_id"] for e in trace["events"]), (
            "Events within a trace must share its trace_id"
        )


class _Run:
    """One crew kickoff's four hand-built events, with lineage stamped exactly
    as the crewai bus's ``_prepare_event`` would set it for this schedule.

    * ``crew_started.event_id`` is the run's ROOT (run_key).
    * ``task_started.parent_event_id`` = the crew-started event_id.
    * ``task_completed.parent_event_id`` = crew-started; ``started_event_id`` =
      task-started (it is a scope-ending event).
    * ``crew_completed.parent_event_id`` = None; ``started_event_id`` =
      crew-started (the real bus leaves a completed-crew's parent None and links
      via started_event_id — the case the resolver's fallback covers).
    """

    def __init__(self, marker: str) -> None:
        self.marker = marker
        self.crew_started = CrewKickoffStartedEvent(crew_name=f"{marker}-crew", inputs={"topic": f"{marker}-topic"})

        self.task_started = TaskStartedEvent(
            context=f"{marker} context",
            task_name=f"{marker}-task",
            agent_role=f"{marker}-researcher",
        )
        self.task_started.parent_event_id = self.crew_started.event_id

        task_out = TaskOutput(description=f"{marker}-task", raw=f"{marker}-task-result", agent=f"{marker}-researcher")
        self.task_completed = TaskCompletedEvent(output=task_out, task_name=f"{marker}-task")
        self.task_completed.parent_event_id = self.crew_started.event_id
        self.task_completed.started_event_id = self.task_started.event_id

        crew_out = TaskOutput(description="final", raw=f"{marker}-final-result", agent=f"{marker}-researcher")
        self.crew_completed = CrewKickoffCompletedEvent(crew_name=f"{marker}-crew", output=crew_out)
        self.crew_completed.parent_event_id = None
        self.crew_completed.started_event_id = self.crew_started.event_id

    def fire(self, adapter: CrewAIAdapter, action: str) -> None:
        if action == "start":
            adapter._on_crew_started(None, self.crew_started)
        elif action == "events":
            adapter._on_task_started(None, self.task_started)
            adapter._on_task_completed(None, self.task_completed)
        elif action == "end":
            adapter._on_crew_completed(None, self.crew_completed)
        else:  # pragma: no cover - guard against typos in a schedule
            raise AssertionError(f"unknown action {action!r}")


#: Events one run produces: crew agent.input/agent.output + task agent.input/agent.output.
_EXPECTED_RUN_COUNTS = {
    "agent.input": 2,
    "agent.output": 2,
}

# Each schedule is a list of (run_key, action). Every schedule must yield two
# fully isolated traces — the post-D1b invariant. Because the adapter reads run
# ownership off the per-event lineage, the interleaving is deterministic on one
# thread and cannot flap.
_SCHEDULES = {
    "sequential": [
        ("alpha", "start"),
        ("alpha", "events"),
        ("alpha", "end"),
        ("bravo", "start"),
        ("bravo", "events"),
        ("bravo", "end"),
    ],
    "interleaved": [
        ("alpha", "start"),
        ("bravo", "start"),
        ("alpha", "events"),
        ("bravo", "events"),
        ("alpha", "end"),
        ("bravo", "end"),
    ],
    "reversed": [
        ("bravo", "start"),
        ("alpha", "start"),
        ("bravo", "events"),
        ("alpha", "events"),
        ("bravo", "end"),
        ("alpha", "end"),
    ],
    "nested": [
        ("alpha", "start"),
        ("bravo", "start"),
        ("bravo", "events"),
        ("bravo", "end"),
        ("alpha", "events"),
        ("alpha", "end"),
    ],
}


# -- Fixtures --


@pytest.fixture
def adapter(mock_client):
    """Create a connected CrewAI adapter inside a scoped event-bus context."""
    a = CrewAIAdapter(mock_client)
    with crewai_event_bus.scoped_handlers():
        a.connect()
        yield a
    a.disconnect()


# -- Tests --


class TestInterleavedRunIsolation:
    def test_sequential_runs_are_isolated(self, adapter, mock_client):
        """GREEN baseline: back-to-back runs produce two clean traces.

        Proves the helpers and assertions are sound independent of any
        interleaving, so a failure in the parametrized schedules below can
        only be the interleaving corruption itself.
        """
        traces = _collect_traces(mock_client)

        runs = {"alpha": _Run("alpha"), "bravo": _Run("bravo")}
        for run_key, action in _SCHEDULES["sequential"]:
            runs[run_key].fire(adapter, action)

        _assert_isolated_traces(traces, ["alpha", "bravo"], _EXPECTED_RUN_COUNTS)

    @pytest.mark.parametrize("schedule_name", list(_SCHEDULES))
    def test_interleaved_runs_are_isolated(self, adapter, mock_client, schedule_name):
        """Every schedule (incl. start A, start B, events A, events B, end A,
        end B) yields two clean, fully isolated traces — the lineage on each
        event partitions it to its owning crew with zero cross-bleed."""
        traces = _collect_traces(mock_client)

        runs = {"alpha": _Run("alpha"), "bravo": _Run("bravo")}
        for run_key, action in _SCHEDULES[schedule_name]:
            runs[run_key].fire(adapter, action)

        _assert_isolated_traces(traces, ["alpha", "bravo"], _EXPECTED_RUN_COUNTS)
