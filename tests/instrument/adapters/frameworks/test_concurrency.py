"""Concurrency test: prove that RunState gives per-task isolation.

Concurrent ``asyncio.gather`` runs on the same PydanticAI adapter must
produce one trace per run — distinct trace_ids, each carrying exactly its
own run's events, with no cross-run content contamination.

(This module was dead from creation until LAY-3567: it importorskip'd on the
fictional ``pydantic_ai.capabilities.hooks`` — see tests/test_skip_hygiene.py.)
"""

from __future__ import annotations

import json
import time
import asyncio
from typing import Any, Dict, List

import pytest

pydantic_ai = pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.models.test import TestModel  # noqa: E402

from layerlens.instrument.adapters.frameworks.pydantic_ai import (
    PydanticAIAdapter,
)  # noqa: E402


def _make_agent(output_text: str = "Hello!", tools: list | None = None) -> Agent:
    agent = Agent(
        model=TestModel(custom_output_text=output_text),
        name="test_agent",
    )
    if tools:
        for fn in tools:
            agent.tool_plain(fn)
    return agent


def _collect_traces(mock_client: Any) -> List[Dict[str, Any]]:
    """Set up mock_client to accumulate individual trace payloads."""
    traces: List[Dict[str, Any]] = []

    def _capture(path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        traces.append(data[0])

    mock_client.traces.upload.side_effect = _capture
    return traces


def _wait_for_uploads(traces: List[Dict[str, Any]], expected: int, timeout: float = 5.0) -> None:
    """Uploads flush through a background channel — wait briefly for them."""
    deadline = time.time() + timeout
    while len(traces) < expected and time.time() < deadline:
        time.sleep(0.05)


class TestConcurrentRunIsolation:
    def test_concurrent_runs_produce_separate_traces(self, mock_client: Any) -> None:
        """Six gather'd runs on one adapter → six isolated traces."""
        traces = _collect_traces(mock_client)

        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"72F in {city}"

        agent = _make_agent(output_text="done", tools=[get_weather])
        adapter = PydanticAIAdapter(mock_client)
        adapter.connect(target=agent)

        prompts = [f"question {i}" for i in range(6)]

        async def run_all() -> None:
            await asyncio.gather(*(agent.run(p) for p in prompts))

        asyncio.run(run_all())
        adapter.disconnect()
        _wait_for_uploads(traces, expected=len(prompts))

        # One trace per run
        assert len(traces) == len(prompts), f"Expected {len(prompts)} traces, got {len(traces)}"

        # Distinct trace_ids
        trace_ids = {t["trace_id"] for t in traces}
        assert len(trace_ids) == len(prompts), f"Traces must have distinct trace_ids, got {trace_ids}"

        seen_inputs = []
        for trace in traces:
            events = trace["events"]
            event_types = [e["event_type"] for e in events]

            # Each trace has the core lifecycle events — exactly one run's worth
            assert event_types.count("agent.input") == 1, f"{event_types}"
            assert event_types.count("agent.output") == 1, f"{event_types}"
            assert "model.invoke" in event_types, f"Missing model.invoke in {event_types}"

            # All events in a single trace share the same trace_id
            assert all(e["trace_id"] == trace["trace_id"] for e in events), "Events within a trace must share trace_id"

            # agent.output completed ok
            output = next(e for e in events if e["event_type"] == "agent.output")
            assert output["payload"]["status"] == "ok"

            seen_inputs.append(next(e for e in events if e["event_type"] == "agent.input")["payload"]["input"])

        # No cross-run content contamination: each submitted prompt appears in
        # exactly one trace's agent.input
        assert sorted(seen_inputs) == sorted(prompts)

    def test_threaded_run_sync_isolated(self, mock_client: Any) -> None:
        """run_sync from worker threads also gets per-thread isolation."""
        import threading

        traces = _collect_traces(mock_client)

        agent = _make_agent(output_text="ok")
        adapter = PydanticAIAdapter(mock_client)
        adapter.connect(target=agent)

        prompts = [f"thread question {i}" for i in range(4)]
        errors: List[BaseException] = []

        def _run(prompt: str) -> None:
            try:
                agent.run_sync(prompt)
            except BaseException as exc:  # surfaced below — threads swallow otherwise
                errors.append(exc)

        threads = [threading.Thread(target=_run, args=(p,)) for p in prompts]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        adapter.disconnect()
        _wait_for_uploads(traces, expected=len(prompts))

        assert not errors, f"run_sync raised in worker threads: {errors!r}"
        assert len(traces) == len(prompts)
        assert len({t["trace_id"] for t in traces}) == len(prompts)
        inputs = sorted(
            next(e for e in t["events"] if e["event_type"] == "agent.input")["payload"]["input"] for t in traces
        )
        assert inputs == sorted(prompts)
