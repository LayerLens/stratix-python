"""S20/F16 — emit-path hygiene batch.

(a) _base_protocol.emit_async propagates the caller's contextvars into the
    executor thread (was silently dropping events).
(c) HandoffDetector stamps payload.framework on agent.handoff.
(d) embedding._st_model_id reads the real SentenceTransformer id, not "local".
(f) a2a/mcp per-operation parents nest under the ambient _current_span_id when
    one is set (standalone behavior unchanged).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List

from layerlens.instrument._context import _current_span_id, _current_collector
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.frameworks._handoff import HandoffDetector, _emit_handoff
from layerlens.instrument.adapters.frameworks.embedding import _st_model_id
from layerlens.instrument.adapters.protocols.a2a.adapter import A2AProtocolAdapter


def _events_in(collector) -> List[Dict[str, Any]]:
    return collector.events


# --- (a) emit_async contextvar propagation --------------------------------


def test_emit_async_propagates_collector_context():
    collector = TraceCollector(object(), CaptureConfig())
    adapter = A2AProtocolAdapter()

    async def go():
        token = _current_collector.set(collector)
        try:
            # Before the fix this ran self.emit on a bare executor thread whose
            # _current_collector was empty -> silent drop (and a TypeError on the
            # positional parent_span_id).
            await adapter.emit_async("a2a.task.updated", {"task_id": "t1", "status": "completed"})
        finally:
            _current_collector.reset(token)

    asyncio.run(go())
    updates = [e for e in collector.events if e["event_type"] == "a2a.task.updated"]
    assert len(updates) == 1, "emit_async dropped the event (context not propagated)"
    assert updates[0]["payload"]["task_id"] == "t1"


# --- (c) _handoff framework stamp -----------------------------------------


def test_handoff_detector_stamps_framework():
    collector = TraceCollector(object(), CaptureConfig())
    token = _current_collector.set(collector)
    try:
        det = HandoffDetector(framework="langgraph")
        det.set_current_agent("supervisor")
        assert det.detect("researcher") is True
    finally:
        _current_collector.reset(token)
    handoffs = [e for e in collector.events if e["event_type"] == "agent.handoff"]
    assert handoffs and handoffs[0]["payload"]["framework"] == "langgraph"


def test_handoff_no_framework_when_unset():
    collector = TraceCollector(object(), CaptureConfig())
    token = _current_collector.set(collector)
    try:
        _emit_handoff(from_agent="a", to_agent="b")  # no framework arg
    finally:
        _current_collector.reset(token)
    handoffs = [e for e in collector.events if e["event_type"] == "agent.handoff"]
    assert handoffs and "framework" not in handoffs[0]["payload"]


# --- (d) embedding real model id ------------------------------------------


def test_st_model_id_reads_model_card_base_model():
    fake = SimpleNamespace(
        model_card_data=SimpleNamespace(model_id=None, model_name=None, base_model="all-MiniLM-L6-v2")
    )
    assert _st_model_id(fake) == "all-MiniLM-L6-v2"


def test_st_model_id_none_when_unavailable():
    assert _st_model_id(None) is None
    assert _st_model_id(SimpleNamespace()) is None  # no card, no first_module


# --- (f) a2a per-operation parent nests under the ambient span -------------


def _drive_a2a(status: str):
    adapter = A2AProtocolAdapter()
    target = type("Cli", (), {"send_task": staticmethod(lambda **kw: {"task_id": "t1", "status": status})})()
    adapter.connect(target=target)
    collector = TraceCollector(object(), CaptureConfig())
    ctoken = _current_collector.set(collector)
    try:
        target.send_task(task_id="t1")
    finally:
        _current_collector.reset(ctoken)
    return collector


def test_a2a_parent_nests_under_ambient_span_when_set():
    stoken = _current_span_id.set("ambient-span-abc")
    try:
        collector = _drive_a2a("completed")
    finally:
        _current_span_id.reset(stoken)
    created = [e for e in collector.events if e["event_type"] == "a2a.task.created"]
    assert created and created[0]["parent_span_id"] == "ambient-span-abc"


def test_a2a_parent_is_standalone_when_no_ambient():
    # No ambient span -> a fresh per-operation parent (unchanged standalone behavior).
    collector = _drive_a2a("completed")
    created = [e for e in collector.events if e["event_type"] == "a2a.task.created"]
    assert created
    parent = created[0]["parent_span_id"]
    assert parent and parent != "ambient-span-abc"
