"""Interleaved-run isolation for the LangGraph adapter (LAY-3576 / A7).

LangGraph graph orchestration is the prime concurrent fan-out surface, yet it
had NO isolation test (census A7 / CONC-2). It is on the ContextVar path
(``FrameworkAdapter._begin_run`` / ``_current_run``), so its isolation is
*presumed* correct but was unproven — a regression to instance-scalar run state
would corrupt traces silently with nothing turning red.

This guard drives TWO concurrent graph runs through ONE
``LangGraphCallbackHandler`` and asserts the LAY-3576 invariant: one trace per
run, distinct trace_ids, each holding exactly its own run's events and none of
the other run's. It runs the runs both as ``asyncio.gather`` tasks and as worker
threads — the two ways ContextVars get copied per concurrent unit. Events use
the real langchain-core callback typings (``LLMResult`` / ``Generation`` and the
real ``langgraph_node`` metadata markers), so the test cannot drift from the
wire shape.

If the adapter ever moves run state off ``_current_run`` onto an instance
scalar, both concurrent runs would share one collector and these tests go RED.
"""

from __future__ import annotations

import json
import asyncio
import threading
from uuid import uuid4
from typing import Any, Dict, List
from collections import Counter

import pytest

from .conftest import record_for_schema_lock

pytest.importorskip("langgraph.graph")
pytest.importorskip("langchain_core")

from langchain_core.outputs import LLMResult, Generation  # noqa: E402

from layerlens.instrument import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.langgraph import (  # noqa: E402
    LangGraphCallbackHandler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_traces(mock_client: Any) -> List[Dict[str, Any]]:
    """Accumulate each uploaded trace payload separately (one entry per flush)."""
    traces: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def _capture(path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        with lock:
            traces.append(data[0])
            record_for_schema_lock(data[0].get("events", []))

    mock_client.traces.upload.side_effect = _capture
    return traces


def _assert_isolated_traces(
    traces: List[Dict[str, Any]],
    markers: List[str],
    expected_counts: Dict[str, int],
) -> None:
    """One trace per run, keyed by content marker — no contamination, no loss."""
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


def _drive_graph_run(handler: LangGraphCallbackHandler, marker: str) -> None:
    """Fire one full graph run: root chain -> one node (+ LLM) -> end.

    Every payload carries the run's marker (node name, model, state) so the
    isolation assertions can attribute each event to its run.
    """
    root = uuid4()
    node = uuid4()
    llm = uuid4()

    handler.on_chain_start({"name": f"{marker}-graph"}, {"topic": f"{marker}-topic"}, run_id=root)
    handler.on_chain_start(
        {"name": f"{marker}-researcher"},
        {"topic": f"{marker}-topic"},
        run_id=node,
        parent_run_id=root,
        metadata={"langgraph_node": f"{marker}-researcher", "langgraph_step": 1},
    )
    handler.on_llm_start({"name": "ChatOpenAI"}, [f"{marker}-prompt"], run_id=llm, parent_run_id=node)
    result = LLMResult(
        generations=[[Generation(text=f"{marker}-answer")]],
        llm_output={"model_name": f"{marker}-model", "token_usage": {"total_tokens": 10}},
    )
    handler.on_llm_end(result, run_id=llm)
    handler.on_chain_end({"result": f"{marker}-result"}, run_id=node, parent_run_id=root)
    handler.on_chain_end({"result": f"{marker}-result"}, run_id=root)


#: Events one run produces via _drive_graph_run.
_EXPECTED_RUN_COUNTS = {
    "agent.input": 2,  # root chain + node chain
    "agent.output": 2,  # node end + root end
    "agent.node.enter": 1,
    "agent.node.exit": 1,
    "agent.state.change": 1,
    "model.invoke": 1,
    "cost.record": 1,
    "trace.root": 1,  # the captured structural root span, emitted once per trace (trace-root)
    "agent.identity": 1,  # canonical declared-identity marker, once per trace (from the langgraph node)
}


def _handler(mock_client: Any) -> LangGraphCallbackHandler:
    return LangGraphCallbackHandler(
        mock_client,
        capture_config=CaptureConfig(capture_content=True),
        detect_handoffs=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLangGraphRunIsolation:
    def test_sequential_runs_are_isolated(self, mock_client):
        """GREEN baseline: back-to-back runs through one handler stay separate."""
        traces = _collect_traces(mock_client)
        handler = _handler(mock_client)

        _drive_graph_run(handler, "alpha")
        _drive_graph_run(handler, "bravo")

        _assert_isolated_traces(traces, ["alpha", "bravo"], _EXPECTED_RUN_COUNTS)

    def test_threaded_concurrent_runs_are_isolated(self, mock_client):
        """Two graph runs on worker threads through one shared handler."""
        traces = _collect_traces(mock_client)
        handler = _handler(mock_client)
        barrier = threading.Barrier(2)
        errors: List[BaseException] = []

        def run(marker: str) -> None:
            try:
                barrier.wait(timeout=10)
                _drive_graph_run(handler, marker)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        ta = threading.Thread(target=run, args=("alpha",))
        tb = threading.Thread(target=run, args=("bravo",))
        ta.start()
        tb.start()
        ta.join(timeout=30)
        tb.join(timeout=30)

        assert not errors, f"worker thread raised: {errors}"
        _assert_isolated_traces(traces, ["alpha", "bravo"], _EXPECTED_RUN_COUNTS)

    def test_asyncio_gather_runs_are_isolated(self, mock_client):
        """Two graph runs as concurrent asyncio tasks through one shared handler.

        ContextVars are copied per asyncio.Task, so the two runs' _current_run
        must not bleed into each other. Yields between phases force interleave.
        """
        traces = _collect_traces(mock_client)
        handler = _handler(mock_client)

        async def run(marker: str) -> None:
            root = uuid4()
            node = uuid4()
            llm = uuid4()
            handler.on_chain_start({"name": f"{marker}-graph"}, {"topic": f"{marker}-topic"}, run_id=root)
            await asyncio.sleep(0)
            handler.on_chain_start(
                {"name": f"{marker}-researcher"},
                {"topic": f"{marker}-topic"},
                run_id=node,
                parent_run_id=root,
                metadata={"langgraph_node": f"{marker}-researcher", "langgraph_step": 1},
            )
            await asyncio.sleep(0)
            handler.on_llm_start({"name": "ChatOpenAI"}, [f"{marker}-prompt"], run_id=llm, parent_run_id=node)
            result = LLMResult(
                generations=[[Generation(text=f"{marker}-answer")]],
                llm_output={"model_name": f"{marker}-model", "token_usage": {"total_tokens": 10}},
            )
            await asyncio.sleep(0)
            handler.on_llm_end(result, run_id=llm)
            handler.on_chain_end({"result": f"{marker}-result"}, run_id=node, parent_run_id=root)
            await asyncio.sleep(0)
            handler.on_chain_end({"result": f"{marker}-result"}, run_id=root)

        async def main() -> None:
            await asyncio.gather(run("alpha"), run("bravo"))

        asyncio.run(main())

        _assert_isolated_traces(traces, ["alpha", "bravo"], _EXPECTED_RUN_COUNTS)


# ---------------------------------------------------------------------------
# Handoff-detection isolation (detect_handoffs=True — the DEFAULT)
# ---------------------------------------------------------------------------


def _drive_handoff_run(handler: LangGraphCallbackHandler, marker: str) -> None:
    """One graph run that walks TWO named nodes (researcher -> writer), so a real
    intra-run ``agent.handoff`` fires (writer's node-enter, prev=researcher).

    A root chain begins the run (no langgraph_node -> no detect); the two node
    chains live inside it; the root end flushes the trace.
    """
    root = uuid4()
    handler.on_chain_start({"name": f"{marker}-graph"}, {"topic": f"{marker}-topic"}, run_id=root)
    for step, node_name in enumerate((f"{marker}-researcher", f"{marker}-writer"), start=1):
        node = uuid4()
        handler.on_chain_start(
            {"name": node_name},
            {"topic": f"{marker}-topic"},
            run_id=node,
            parent_run_id=root,
            metadata={"langgraph_node": node_name, "langgraph_step": step},
        )
        handler.on_chain_end({"result": f"{marker}-result-{step}"}, run_id=node, parent_run_id=root)
    handler.on_chain_end({"result": f"{marker}-final"}, run_id=root)


def _own_nodes(trace: Dict[str, Any]) -> set:
    """The run's OWN node identities, from its agent.node.enter events — these are
    emitted by the run itself and are NOT polluted by a contaminating handoff."""
    return {e["payload"]["node"] for e in trace["events"] if e["event_type"] == "agent.node.enter"}


def _handoff_endpoints(trace: Dict[str, Any]) -> List[tuple]:
    return [
        (e["payload"].get("from_agent"), e["payload"].get("to_agent"))
        for e in trace["events"]
        if e["event_type"] == "agent.handoff"
    ]


def _assert_no_cross_run_handoffs(traces: List[Dict[str, Any]], n_runs: int) -> None:
    assert len(traces) == n_runs, f"expected {n_runs} traces, got {len(traces)}"
    total = 0
    for trace in traces:
        allowed = _own_nodes(trace)
        assert allowed, "trace has no agent.node.enter events to attribute it to a run"
        for frm, to in _handoff_endpoints(trace):
            total += 1
            assert frm in allowed and to in allowed, (
                f"handoff {frm!r}->{to!r} has an endpoint outside this trace's own nodes "
                f"{sorted(allowed)} — the shared detector leaked another run's node (fabricated edge)"
            )
    assert total >= n_runs, f"expected a real intra-run handoff per run, got {total}"


class TestConcurrentHandoffIsolation:
    """detect_handoffs defaults to True, so real customers reusing one handler
    across graph runs (sequentially or concurrently) hit the shared-detector
    leak. Every handoff edge must stay within its own run's node set.

    Both tests are RED while the detector is a single shared instance on the
    handler and GREEN once the detector state is per-run (RunState-isolated):
    the shared 'previous node' scalar leaks the prior/other run's last node into
    the next run's first handoff.
    """

    def _handler_detecting(self, mock_client: Any) -> LangGraphCallbackHandler:
        return LangGraphCallbackHandler(
            mock_client,
            capture_config=CaptureConfig(capture_content=True),
            detect_handoffs=True,
        )

    def test_sequential_handoff_runs_do_not_leak_last_node(self, mock_client):
        """Back-to-back detecting runs on one handler: the second run must not
        emit a handoff from the first run's last node."""
        traces = _collect_traces(mock_client)
        handler = self._handler_detecting(mock_client)
        _drive_handoff_run(handler, "alpha")
        _drive_handoff_run(handler, "bravo")
        _assert_no_cross_run_handoffs(traces, 2)

    def test_gathered_handoff_runs_do_not_cross_contaminate(self, mock_client):
        """Two detecting graph runs gathered on one handler — the shared
        'previous node' scalar fabricates a handoff whose from_agent belongs to
        the other run."""
        traces = _collect_traces(mock_client)
        handler = self._handler_detecting(mock_client)

        async def run(marker: str) -> None:
            root = uuid4()
            handler.on_chain_start({"name": f"{marker}-graph"}, {"topic": f"{marker}-topic"}, run_id=root)
            for step, node_name in enumerate((f"{marker}-researcher", f"{marker}-writer"), start=1):
                node = uuid4()
                await asyncio.sleep(0)
                handler.on_chain_start(
                    {"name": node_name},
                    {"topic": f"{marker}-topic"},
                    run_id=node,
                    parent_run_id=root,
                    metadata={"langgraph_node": node_name, "langgraph_step": step},
                )
                await asyncio.sleep(0)
                handler.on_chain_end({"result": f"{marker}-result-{step}"}, run_id=node, parent_run_id=root)
            handler.on_chain_end({"result": f"{marker}-final"}, run_id=root)

        async def main() -> None:
            await asyncio.gather(run("alpha"), run("bravo"))

        asyncio.run(main())
        _assert_no_cross_run_handoffs(traces, 2)
