"""S13/F6 — the a2a client-wrap path stamps submitter_agent_id on
a2a.task.created when the caller declared a from_agent (parity with
A2AClientWrapper.send_task, client.py:60). submitter_agent_id is a node-identity
field both graph engines read; before this fix the wrap path dropped it (it was
only ever attached to a2a.delegation, and only when a delegatee was also named).
"""

from __future__ import annotations

from typing import Any, Dict, List

from layerlens.instrument._context import _current_collector
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.protocols.a2a.adapter import A2AProtocolAdapter


def _run(fn: Any) -> List[Dict[str, Any]]:
    collector = TraceCollector(object(), CaptureConfig())
    token = _current_collector.set(collector)
    try:
        fn()
    finally:
        _current_collector.reset(token)
    return collector.events


def _created(events):
    for e in events:
        if e["event_type"] == "a2a.task.created":
            return e["payload"]
    raise AssertionError("no a2a.task.created emitted")


def _connect(**send_result):
    adapter = A2AProtocolAdapter()
    target = type("Cli", (), {"send_task": staticmethod(lambda **kw: dict(send_result))})()
    adapter.connect(target=target)
    return target


def test_submitter_stamped_when_from_agent_declared():
    target = _connect(task_id="t1", status="completed")
    events = _run(lambda: target.send_task(task_id="t1", from_agent="orchestrator-1"))
    assert _created(events)["submitter_agent_id"] == "orchestrator-1"


def test_submitter_omitted_when_absent():
    target = _connect(task_id="t1", status="completed")
    payload = _created(_run(lambda: target.send_task(task_id="t1")))
    # Honest blank: no from_agent -> no fabricated submitter key.
    assert "submitter_agent_id" not in payload
