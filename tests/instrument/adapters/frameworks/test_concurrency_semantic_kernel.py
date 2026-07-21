"""Interleaved-run isolation for the Semantic Kernel adapter.

The SK adapter tracks run boundaries with a nesting-depth counter and keeps ALL
per-run state (``depth``, group-chat ``group_prev_agent``) in ``RunState.data``,
which is ContextVar-isolated per asyncio task / thread (semantic_kernel.py
docstring + ``_enter_invocation``/``_leave_invocation``). This guard proves that
claim end-to-end: two ``kernel.invoke()`` runs racing through ONE adapter/kernel
— as gathered asyncio tasks, as worker threads, and driving the function-
invocation filter directly with interleaved scheduling — each produce their own
isolated trace (distinct trace_ids, own events, no content contamination).

Unlike the ms_agent_framework / langgraph handoff detectors (which shared an
instance scalar), SK already keeps this state per-run, so these are GREEN today
— coverage that locks the isolation in against a future regression to instance
state (the failure mode ``run_matrix`` cannot see because these tests were never
CI-wired).
"""

from __future__ import annotations

import json
import asyncio
import threading
from typing import Any, Dict, List, Optional

import pytest

from .conftest import record_for_schema_lock

sk = pytest.importorskip("semantic_kernel")

from semantic_kernel import Kernel  # noqa: E402
from semantic_kernel.functions import kernel_function  # noqa: E402

from layerlens.models import CreateTracesResponse  # noqa: E402
from layerlens.instrument._upload import shutdown_uploads  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.semantic_kernel import (  # noqa: E402
    SemanticKernelAdapter,
)

# ---------------------------------------------------------------------------
# Real marker plugin (no LLM) + local filter doubles
# ---------------------------------------------------------------------------


class EchoPlugin:
    @kernel_function(name="echo", description="Echo the marker back")
    def echo(self, text: str) -> str:
        return f"echoed-{text}"


class _MockFunction:
    def __init__(self, name: str, plugin_name: str) -> None:
        self.name = name
        self.plugin_name = plugin_name


class _MockResult:
    def __init__(self, value: Any) -> None:
        self.value = value


class _MockContext:
    def __init__(self, marker: str) -> None:
        self.function = _MockFunction("echo", "EchoPlugin")
        self.arguments = {"text": marker}
        self.result: Any = None
        self.rendered_prompt: Optional[str] = None
        self.function_call_content = None
        self.function_result = None
        self.request_sequence_index = 0
        self.function_sequence_index = 0


# ---------------------------------------------------------------------------
# Collection + isolation invariant
# ---------------------------------------------------------------------------


def _collect_traces(mock_client: Any) -> List[Dict[str, Any]]:
    traces: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def _capture(path: str) -> CreateTracesResponse:
        with open(path) as f:
            data = json.load(f)
        with lock:
            traces.append(data[0])
            record_for_schema_lock(data[0].get("events", []))
        return CreateTracesResponse(trace_ids=[data[0].get("trace_id") or "mock-trace-id"])

    mock_client.traces.upload.side_effect = _capture
    return traces


def _assert_isolated(traces: List[Dict[str, Any]], markers: List[str]) -> None:
    shutdown_uploads(10.0)
    # SK emits a one-shot ``environment.config`` on first plugin use, flushed as
    # its own marker-less trace; the per-run traces are the marker-bearing ones.
    runs = [t for t in traces if any(m in json.dumps(t["events"]) for m in markers)]
    assert len(runs) == len(markers), (
        f"expected {len(markers)} run traces (one per run), got {len(runs)} of {len(traces)} uploaded — "
        "interleaved runs merged or lost a trace"
    )
    trace_ids = {t["trace_id"] for t in runs}
    assert len(trace_ids) == len(markers), f"run traces must have distinct trace_ids, got {trace_ids}"
    for marker in markers:
        own = [t for t in runs if marker in json.dumps(t["events"])]
        assert len(own) == 1, f"marker {marker!r} must appear in exactly 1 run trace, found {len(own)}"
        blob = json.dumps(own[0]["events"])
        for other in markers:
            if other != marker:
                assert other not in blob, f"trace for {marker!r} is contaminated with {other!r}"
        # Teeth: the run really did capture its own tool call.
        calls = [e for e in own[0]["events"] if e["event_type"] == "tool.call"]
        assert calls, f"trace for {marker!r} has no tool.call"
        assert all(e["trace_id"] == own[0]["trace_id"] for e in own[0]["events"])


@pytest.fixture
def adapter(mock_client):
    return SemanticKernelAdapter(mock_client, capture_config=CaptureConfig.full())


def _kernel() -> Kernel:
    kernel = Kernel()
    kernel.add_plugin(EchoPlugin(), "EchoPlugin")
    return kernel


class TestSemanticKernelRunIsolation:
    def test_gathered_kernel_invokes_are_isolated(self, adapter, mock_client):
        """Two kernel.invoke() runs as concurrent asyncio tasks on one adapter."""
        traces = _collect_traces(mock_client)
        kernel = _kernel()
        adapter.connect(target=kernel)
        markers = ["alpha-sk", "bravo-sk"]

        async def main() -> None:
            await asyncio.gather(
                *(kernel.invoke(plugin_name="EchoPlugin", function_name="echo", text=m) for m in markers)
            )

        asyncio.run(main())
        adapter.disconnect()
        _assert_isolated(traces, markers)

    def test_threaded_kernel_invokes_are_isolated(self, adapter, mock_client):
        """Whole kernel.invoke() runs racing on one adapter from worker threads."""
        traces = _collect_traces(mock_client)
        kernel = _kernel()
        adapter.connect(target=kernel)
        markers = [f"thread-sk-{i}" for i in range(4)]
        barrier = threading.Barrier(len(markers))
        errors: List[BaseException] = []

        def run(marker: str) -> None:
            try:
                barrier.wait(timeout=5)
                asyncio.run(kernel.invoke(plugin_name="EchoPlugin", function_name="echo", text=marker))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(m,)) for m in markers]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        adapter.disconnect()
        assert not errors, f"worker thread raised: {errors!r}"
        _assert_isolated(traces, markers)

    def test_interleaved_filter_calls_are_isolated(self, adapter, mock_client):
        """Drive the function-invocation filter directly with two interleaved
        runs (each next() yields control), the deterministic isolation proof."""
        traces = _collect_traces(mock_client)
        kernel = Kernel()
        adapter.connect(target=kernel)
        markers = ["filter-alpha", "filter-bravo"]

        async def drive(marker: str) -> None:
            ctx = _MockContext(marker)

            async def next_(context: Any) -> None:
                await asyncio.sleep(0)
                context.result = _MockResult(f"echoed-{marker}")

            await adapter._function_invocation_filter(ctx, next_)

        async def main() -> None:
            await asyncio.gather(*(drive(m) for m in markers))

        asyncio.run(main())
        adapter.disconnect()
        _assert_isolated(traces, markers)
