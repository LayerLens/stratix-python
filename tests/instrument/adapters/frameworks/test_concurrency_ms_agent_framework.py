"""Concurrent-run handoff isolation for the Microsoft Agent Framework adapter.

The adapter tracks the "previous agent" for ``agent.handoff`` detection. If that
state is shared across runs, two ``AgentGroupChat`` invocations racing through
ONE adapter instance cross-contaminate: run B's ``set_current_agent`` lands
between run A's messages, so run A emits a handoff whose ``from_agent`` belongs
to run B — a FABRICATED cross-run edge that poisons the atlas multi-agent DAG
(``.claude/CLAUDE.md`` rule #3: no fake data).

The guard runs two gathered group-chat runs, each transitioning ONLY among its
own agents, and asserts every emitted handoff's ``from_agent``/``to_agent``
belongs to that run's own agent set. It goes RED while the handoff detector is a
single shared instance on the adapter and GREEN once the detector state is
per-run (RunState-isolated).

The runs interleave deterministically: each traced ``invoke`` wrapper seeds the
detector synchronously, then the fake generator's leading ``await sleep(0)``
suspends it — so under ``gather`` the second run's seed always lands before the
first run processes its first message (the exact race the bug needs).
"""

from __future__ import annotations

import json
import asyncio
import threading
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from layerlens.instrument._upload import shutdown_uploads
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.frameworks.ms_agent_framework import (
    MSAgentFrameworkAdapter,
)

from .conftest import record_for_schema_lock

# ---------------------------------------------------------------------------
# Synthetic ChatMessageContent-shaped doubles (no semantic-kernel needed)
# ---------------------------------------------------------------------------


def _msg(agent_name: str):
    """A group-chat message whose speaker is ``agent_name`` (drives handoffs)."""
    return SimpleNamespace(agent_name=agent_name, items=[], metadata=None)


def _make_invoke(messages: List[Any]):
    """A fake ``chat.invoke`` that yields *messages*, yielding control BEFORE the
    first message so two gathered runs interleave at the seed boundary."""

    async def invoke(*_args: Any, **_kwargs: Any):
        for m in messages:
            await asyncio.sleep(0)
            yield m

    return invoke


def _group_chat(name: str) -> SimpleNamespace:
    # AgentGroupChat: no `agent` attr, so the run seeds under the chat name.
    return SimpleNamespace(name=name)


# ---------------------------------------------------------------------------
# Trace collection (one uploaded trace per run)
# ---------------------------------------------------------------------------


def _collect_traces(mock_client: Any) -> List[Dict[str, Any]]:
    traces: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def _capture(path: str) -> Any:
        with open(path) as f:
            data = json.load(f)
        with lock:
            traces.append(data[0])
            record_for_schema_lock(data[0].get("events", []))
        # Non-empty trace_ids or the upload counts as a REJECT (F-L7-002).
        from layerlens.models import CreateTracesResponse

        return CreateTracesResponse(trace_ids=[data[0].get("trace_id") or "mock-trace-id"])

    mock_client.traces.upload.side_effect = _capture
    return traces


#: Each run transitions only among its own agents; the seed is the chat name.
_RUN_A = {"agents": ["a-planner", "a-researcher", "a-writer"], "chat": "GroupA"}
_RUN_B = {"agents": ["b-analyst", "b-reviewer", "b-approver"], "chat": "GroupB"}


def _consume_coro(adapter: MSAgentFrameworkAdapter, chat: SimpleNamespace, agents: List[str]):
    chat.invoke = _make_invoke([_msg(a) for a in agents])
    adapter.instrument_chat(chat)

    async def consume() -> None:
        async for _m in chat.invoke():
            pass

    return consume()


def _handoffs_for(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [e["payload"] for e in trace["events"] if e["event_type"] == "agent.handoff"]


def _run_for(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Attribute a trace to its run by the chat name on its agent.input."""
    names = {e["payload"].get("agent_name") for e in trace["events"] if e["event_type"] == "agent.input"}
    if _RUN_A["chat"] in names:
        return _RUN_A
    if _RUN_B["chat"] in names:
        return _RUN_B
    raise AssertionError(f"trace matches no run: agent.input names={names}")


def _assert_no_cross_run_handoffs(traces: List[Dict[str, Any]]) -> None:
    shutdown_uploads(10.0)
    assert len(traces) == 2, f"expected 2 uploaded traces (one per run), got {len(traces)}"
    total_handoffs = 0
    for trace in traces:
        run = _run_for(trace)
        allowed = set(run["agents"]) | {run["chat"]}
        for ho in _handoffs_for(trace):
            total_handoffs += 1
            frm, to = ho.get("from_agent"), ho.get("to_agent")
            assert frm in allowed, (
                f"run {run['chat']}: handoff from_agent {frm!r} is not one of this run's "
                f"agents {sorted(allowed)} — cross-run contamination (fabricated edge)"
            )
            assert to in allowed, (
                f"run {run['chat']}: handoff to_agent {to!r} is not one of this run's "
                f"agents {sorted(allowed)} — cross-run contamination (fabricated edge)"
            )
    # The test must have teeth: each run really does transition between agents.
    assert total_handoffs >= 2, f"expected real intra-run handoffs, got {total_handoffs}"


@pytest.fixture
def adapter(mock_client):
    return MSAgentFrameworkAdapter(mock_client, capture_config=CaptureConfig.full())


class TestConcurrentHandoffIsolation:
    def test_gathered_group_chats_do_not_cross_contaminate_handoffs(self, adapter, mock_client):
        """Two AgentGroupChat runs gathered on one adapter — every handoff edge
        must stay within its own run's agent set."""
        traces = _collect_traces(mock_client)
        # No connect(): connect() checks the optional semantic-kernel dependency
        # (absent in the py3.9 base suite). Instrumentation is via instrument_chat
        # (in _consume_coro), the same double-driven pattern as test_ms_agent_framework.
        chat_a = _group_chat(_RUN_A["chat"])
        chat_b = _group_chat(_RUN_B["chat"])

        async def main() -> None:
            await asyncio.gather(
                _consume_coro(adapter, chat_a, _RUN_A["agents"]),
                _consume_coro(adapter, chat_b, _RUN_B["agents"]),
            )

        asyncio.run(main())
        adapter.disconnect()
        _assert_no_cross_run_handoffs(traces)

    def test_sequential_group_chats_are_isolated(self, adapter, mock_client):
        """GREEN baseline: back-to-back runs never cross-contaminate."""
        traces = _collect_traces(mock_client)
        # No connect(): connect() checks the optional semantic-kernel dependency
        # (absent in the py3.9 base suite). Instrumentation is via instrument_chat
        # (in _consume_coro), the same double-driven pattern as test_ms_agent_framework.

        for run in (_RUN_A, _RUN_B):
            chat = _group_chat(run["chat"])
            asyncio.run(_consume_coro(adapter, chat, run["agents"]))

        adapter.disconnect()
        _assert_no_cross_run_handoffs(traces)
