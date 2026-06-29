"""Interleaved-run isolation guard for the AutoGen adapter (LAY-3576 / A6).

THE INVARIANT: two runs (conversations) driven through ONE ``AutoGenAdapter``
instance concurrently must upload two traces with distinct trace_ids, each
holding exactly its own run's events — no cross-contamination, no lost events.

THE BUG (now fixed): the adapter funnelled every run into one lazily-created
``self._collector`` shared across all conversations (autogen logs through the
module-global ``EVENT_LOGGER_NAME`` and its events carry no run/topic/session
id). Two interleaved conversations merged into one trace.

THE FIX: each AgentChat team owns its own ``SingleThreadedAgentRuntime`` whose
message loop drains on its own asyncio task / thread, and the logging handler
runs inline on the emitting task/thread. So the adapter opens a ``RunState``
lazily on the first event and binds it to ``_current_run``; the task/thread
ContextVar copy keeps each concurrent run isolated even for the sender-less
``LLMCallEvent`` (``autogen.py`` — ``_runs`` / ``_ensure_run`` / ``_dispatch``).
Each run flushes as its own trace on ``disconnect()``.

DRIVING: faithful concurrency — each conversation runs on its OWN worker thread
through the REAL ``autogen_core`` event logger, with a barrier forcing the two
runs to interleave. A single-thread interleave is intentionally NOT used: it is
genuinely infeasible (the runtime drains every event from one shared task and
``LLMCallEvent`` has no conversation key), so it would not reflect — or be able
to isolate — a real concurrent workload.

Requires autogen-core >= 0.4 (Python >= 3.10).
"""

from __future__ import annotations

import sys
import json
import logging
import threading
from typing import Any, Dict, List

import pytest

from .conftest import record_for_schema_lock

if sys.version_info < (3, 10):
    pytest.skip("autogen-core requires Python >= 3.10", allow_module_level=True)
try:
    import autogen_core  # noqa: F401
except (ImportError, TypeError):
    pytest.skip("autogen-core not installed or incompatible", allow_module_level=True)

from autogen_core import EVENT_LOGGER_NAME, AgentId  # noqa: E402
from autogen_core.logging import (  # noqa: E402
    MessageKind,
    LLMCallEvent,
    MessageEvent,
    DeliveryStage,
)

from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.autogen import (
    AutoGenAdapter,
)  # noqa: E402

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


def _ask(content: str, user: str, assistant: str) -> MessageEvent:
    return MessageEvent(
        payload=content,
        sender=AgentId(user, "default"),
        receiver=AgentId(assistant, "default"),
        kind=MessageKind.DIRECT,
        delivery_stage=DeliveryStage.SEND,
    )


def _answer(content: str, user: str, assistant: str) -> MessageEvent:
    return MessageEvent(
        payload=content,
        sender=AgentId(assistant, "default"),
        receiver=AgentId(user, "default"),
        kind=MessageKind.RESPOND,
        delivery_stage=DeliveryStage.SEND,
    )


def _llm_call(content: str, model: str) -> LLMCallEvent:
    return LLMCallEvent(
        messages=[{"role": "user", "content": content}],
        response={"model": model},
        prompt_tokens=11,
        completion_tokens=7,
    )


# ---------------------------------------------------------------------------
# Isolation invariant
# ---------------------------------------------------------------------------


def _assert_two_isolated_traces(traces: List[Dict[str, Any]]) -> None:
    summaries = [[(e["event_type"], e["payload"].get("content")) for e in t["events"]] for t in traces]
    assert traces, "SETUP BUG (not the isolation invariant): no trace was uploaded at all"
    assert len(traces) == 2, (
        f"Each conversation must flush its own trace: expected 2 uploads, got {len(traces)}. "
        f"Uploaded traces as (event_type, content): {summaries}"
    )

    trace_ids = {t["trace_id"] for t in traces}
    assert len(trace_ids) == 2, f"The two runs must not share a trace_id: {trace_ids}"

    markers = {
        "ask-alpha": ("user_alpha", "assistant_alpha", "model-alpha", "answer-alpha"),
        "ask-beta": ("user_beta", "assistant_beta", "model-beta", "answer-beta"),
    }
    by_ask: Dict[str, Dict[str, Any]] = {}
    for trace in traces:
        inputs = [e for e in trace["events"] if e["event_type"] == "agent.input"]
        assert len(inputs) == 1, (
            f"Each trace must hold exactly one run's agent.input, got {[e['payload'].get('content') for e in inputs]}"
        )
        by_ask[inputs[0]["payload"]["content"]] = trace
    assert set(by_ask) == set(markers), f"Expected one trace per conversation, got {sorted(by_ask)}"

    for ask, own_markers in markers.items():
        text = json.dumps(by_ask[ask]["events"])
        for marker in (ask, *own_markers):
            assert marker in text, f"Trace for {ask!r} lost its own event marker {marker!r}"
        (other_ask,) = [a for a in markers if a != ask]
        for marker in (other_ask, *markers[other_ask]):
            assert marker not in text, f"Trace for {ask!r} contaminated by other run's marker {marker!r}"


def test_concurrent_conversations_produce_two_isolated_traces(mock_client):
    """Two conversations on two worker threads → two clean, isolated traces.

    The faithful production workload: two ``team.run()`` equivalents racing
    through the one shared adapter and the real autogen event logger. A barrier
    interleaves their phases (ask A/ask B, llm A/llm B, answer A/answer B).
    Reverting the keyed-run fix (back to one shared ``self._collector``) merges
    them into a single trace and turns this RED.
    """
    traces = _collect_traces(mock_client)
    # capture_content=True so the per-run content markers (ask/answer text) are
    # present to prove non-contamination; the sender/receiver/model markers are
    # checked regardless.
    adapter = AutoGenAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
    adapter.connect()
    logger = logging.getLogger(EVENT_LOGGER_NAME)

    barrier = threading.Barrier(2)
    errors: List[BaseException] = []

    def run(user: str, assistant: str, model: str, suffix: str) -> None:
        try:
            logger.info(_ask(f"ask-{suffix}", user, assistant))
            barrier.wait(timeout=10)
            logger.info(_llm_call(f"ask-{suffix}", model=model))
            barrier.wait(timeout=10)
            logger.info(_answer(f"answer-{suffix}", user, assistant))
        except BaseException as exc:  # noqa: BLE001 — surface to the test thread
            errors.append(exc)

    ta = threading.Thread(target=run, args=("user_alpha", "assistant_alpha", "model-alpha", "alpha"))
    tb = threading.Thread(target=run, args=("user_beta", "assistant_beta", "model-beta", "beta"))
    ta.start()
    tb.start()
    ta.join(timeout=30)
    tb.join(timeout=30)

    assert not errors, f"worker thread raised: {errors}"

    adapter.disconnect()

    _assert_two_isolated_traces(traces)
