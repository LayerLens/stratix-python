"""End-to-end: CrewAI delegation detection on the real event bus.

We don't need an LLM to exercise the delegation path — we just need
crewai's real event bus + real event classes. The adapter subscribes
through ``adapter.connect()``, we kick off a (synthetic) crew lifecycle
by emitting the events that crewai itself would emit, and verify
agent.handoff fires for the "Delegate work to coworker" tool call.

This is "end-to-end" in the sense that:
- the adapter goes through its real connect/subscribe path
- events go through the real crewai event bus, not a mock
- the adapter's _on_tool_started handler runs as a real subscriber
"""

from __future__ import annotations

import sys

import pytest

if sys.version_info < (3, 10):
    pytest.skip("crewai requires Python >= 3.10", allow_module_level=True)

# crewai needs to be importable in full
crewai_events = pytest.importorskip("crewai.events")
pytest.importorskip("crewai.tasks.task_output")

from crewai.events import (
    ToolUsageStartedEvent,
    CrewKickoffStartedEvent,
    CrewKickoffCompletedEvent,
    AgentExecutionStartedEvent,
    crewai_event_bus,
)
from crewai.tasks.task_output import TaskOutput

from layerlens.instrument.adapters.frameworks.crewai import CrewAIAdapter

from .conftest import events_of, first_event


@pytest.fixture
def adapter_in_real_bus(client_and_uploads):
    """Connect the adapter through the real crewai event bus."""
    client, uploads = client_and_uploads
    adapter = CrewAIAdapter(client)
    # scoped_handlers lets us mount handlers cleanly per-test.
    with crewai_event_bus.scoped_handlers():
        adapter.connect()
        yield adapter, uploads
    adapter.disconnect()


def _await(fut, timeout=5):
    """crewai's event bus runs handlers on a ThreadPoolExecutor and returns a
    Future. Block until the handlers finish so test assertions don't race."""
    if fut is not None:
        fut.result(timeout=timeout)


def _emit(event):
    _await(crewai_event_bus.emit(None, event))


def _start_crew_with_manager():
    """Fire the events that crewai would fire at the start of a hierarchical crew."""
    _emit(CrewKickoffStartedEvent(crew_name="research_crew", inputs={}))
    _emit(AgentExecutionStartedEvent.model_construct(agent_role="manager"))


def _finish_crew():
    _emit(
        CrewKickoffCompletedEvent(
            crew_name="research_crew",
            output=TaskOutput(description="t", raw="done", agent="manager"),
        )
    )


def test_real_event_bus_emits_handoff_on_delegation_tool(adapter_in_real_bus):
    adapter, uploads = adapter_in_real_bus
    _start_crew_with_manager()

    crewai_event_bus.emit(
        None,
        ToolUsageStartedEvent(
            tool_name="Delegate work to coworker",
            tool_args={
                "task": "Find recent papers on attention mechanisms",
                "coworker": "researcher",
                "context": "Focus on transformers and LLMs",
            },
            agent_key="manager_1",
        ),
    )

    _finish_crew()

    handoff = first_event(uploads, "agent.handoff")
    p = handoff["payload"]
    assert p["from_agent"] == "manager"
    assert p["to_agent"] == "researcher"
    assert p["reason"] == "delegation"
    assert p["delegation_seq"] == 1
    assert p["tool_name"] == "Delegate work to coworker"
    assert p["handoff_context_hash"].startswith("sha256:")


def test_chain_of_delegations_keeps_sequence(adapter_in_real_bus):
    adapter, uploads = adapter_in_real_bus
    _start_crew_with_manager()

    for to_agent in ["researcher", "writer", "reviewer"]:
        crewai_event_bus.emit(
            None,
            ToolUsageStartedEvent(
                tool_name="Delegate work to coworker",
                tool_args={"task": f"work for {to_agent}", "coworker": to_agent},
                agent_key="manager_1",
            ),
        )

    _finish_crew()

    handoffs = events_of(uploads, "agent.handoff")
    assert len(handoffs) == 3
    assert [h["payload"]["delegation_seq"] for h in handoffs] == [1, 2, 3]
    assert [h["payload"]["to_agent"] for h in handoffs] == ["researcher", "writer", "reviewer"]


def test_ask_question_variant(adapter_in_real_bus):
    adapter, uploads = adapter_in_real_bus
    _start_crew_with_manager()

    crewai_event_bus.emit(
        None,
        ToolUsageStartedEvent(
            tool_name="Ask question to coworker",
            tool_args={"question": "What is the deadline?", "coworker": "researcher"},
            agent_key="manager_1",
        ),
    )
    _finish_crew()

    h = first_event(uploads, "agent.handoff")
    assert h["payload"]["to_agent"] == "researcher"
    assert h["payload"]["reason"] == "delegation"


def test_regular_tool_does_not_fire_handoff(adapter_in_real_bus):
    adapter, uploads = adapter_in_real_bus
    _start_crew_with_manager()

    crewai_event_bus.emit(
        None,
        ToolUsageStartedEvent(
            tool_name="web_search",
            tool_args={"query": "AI safety"},
            agent_key="manager_1",
        ),
    )
    _finish_crew()

    assert events_of(uploads, "agent.handoff") == []
    # The tool call event itself does still fire
    tools = events_of(uploads, "tool.call")
    assert any(t["payload"]["tool_name"] == "web_search" for t in tools)
