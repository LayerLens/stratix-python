"""Interleaved-run isolation guard for the AutoGen adapter (LAY-3576 / T2).

THE INVARIANT: two runs (conversations) driven through ONE
``AutoGenAdapter`` instance, with lifecycle events interleaved (start A,
start B, mid-run A, mid-run B, end A, end B), must upload two traces with
distinct trace_ids, each holding exactly its own run's events — no
cross-contamination, no lost events.

The adapter keeps run state in instance scalars
(``src/layerlens/instrument/adapters/frameworks/autogen.py`` — a lazily
created ``self._collector`` plus ``self._root_span_id``): the first event
of ANY run creates one shared collector and one root span, every
subsequent event from every run is appended to it, and the single merged
trace only flushes on ``disconnect()``. Two interleaved conversations can
never come apart into separate traces.

Expected XFAIL (strict): this is the RED guard for the D1b
collector-convergence work (stability report §3.1), which moves the
self-flushing adapters onto the per-run RunState/ContextVar isolation that
PydanticAIAdapter already uses (see test_concurrency.py). When D1b lands,
the strict xfail turns into a strict XPASS failure and the marker must be
removed. Events are dispatched synchronously through the real autogen_core
event logger on one thread, so the corruption is deterministic and the
xfail cannot flap.

Requires autogen-core >= 0.4 (Python >= 3.10).
"""

from __future__ import annotations

import sys
import json
import logging
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

from layerlens.instrument.adapters.frameworks.autogen import (
    AutoGenAdapter,
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

XFAIL_REASON = (
    "LAY-3576: AutoGenAdapter keeps run state in instance scalars "
    "(autogen.py — lazily created self._collector, self._root_span_id); "
    "interleaved runs on one instance corrupt traces. "
    "RED guard for the D1b collector-convergence work (stability report §3.1) — "
    "NOT fixed in phase 4."
)


@pytest.mark.xfail(strict=True, reason=XFAIL_REASON)
def test_interleaved_conversations_produce_two_isolated_traces(mock_client):
    traces = _collect_traces(mock_client)
    adapter = AutoGenAdapter(mock_client)
    adapter.connect()
    logger = logging.getLogger(EVENT_LOGGER_NAME)

    # Deterministic single-thread interleave through the real autogen event
    # logger: start A, start B, mid A, mid B, end A, end B.
    logger.info(_ask("ask-alpha", "user_alpha", "assistant_alpha"))
    logger.info(_ask("ask-beta", "user_beta", "assistant_beta"))
    logger.info(_llm_call("ask-alpha", model="model-alpha"))
    logger.info(_llm_call("ask-beta", model="model-beta"))
    logger.info(_answer("answer-alpha", "user_alpha", "assistant_alpha"))
    logger.info(_answer("answer-beta", "user_beta", "assistant_beta"))

    adapter.disconnect()

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
