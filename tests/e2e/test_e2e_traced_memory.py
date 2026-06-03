"""End-to-end: TracedMemory wrapping a real LangChain ConversationBufferMemory.

Runs a multi-turn conversation through wrap_memory + ConversationBufferMemory
and verifies agent.state.change events fire with distinct sha256 hashes per
turn. Also exercises MemoryMutationTracker as the context-manager variant.
"""

from __future__ import annotations

import pytest

# langchain is optional; skip cleanly when it isn't installed.
langchain_memory = pytest.importorskip("langchain.memory")
ConversationBufferMemory = langchain_memory.ConversationBufferMemory

from layerlens.instrument import trace_context
from layerlens.instrument.adapters.frameworks.langchain import (
    TracedMemory,
    MemoryMutationTracker,
    wrap_memory,
)

from .conftest import events_of, first_event


def test_save_context_emits_state_change_per_turn(client_and_uploads):
    client, uploads = client_and_uploads

    memory = ConversationBufferMemory()
    traced = wrap_memory(memory)
    assert isinstance(traced, TracedMemory)

    with trace_context(client):
        traced.save_context({"input": "hi"}, {"output": "hello"})
        traced.save_context({"input": "what's up?"}, {"output": "not much"})

    changes = events_of(uploads, "agent.state.change")
    assert len(changes) == 2
    # Each turn produces distinct before/after hashes
    for ev in changes:
        p = ev["payload"]
        assert p["memory_type"] == "ConversationBufferMemory"
        assert p["trigger"] == "save_context"
        assert p["before_hash"].startswith("sha256:")
        assert p["after_hash"].startswith("sha256:")
        assert p["before_hash"] != p["after_hash"]
    # The second turn's "before" should equal the first turn's "after"
    assert changes[1]["payload"]["before_hash"] == changes[0]["payload"]["after_hash"]


def test_clear_emits_state_change_when_memory_was_nonempty(client_and_uploads):
    client, uploads = client_and_uploads

    memory = ConversationBufferMemory()
    memory.save_context({"input": "seed"}, {"output": "primed"})
    traced = wrap_memory(memory)

    with trace_context(client):
        traced.clear()

    evt = first_event(uploads, "agent.state.change")
    assert evt["payload"]["trigger"] == "clear"


def test_no_change_no_event(client_and_uploads):
    """Wrapping the load path shouldn't emit anything — load is a read."""
    client, uploads = client_and_uploads

    memory = ConversationBufferMemory()
    memory.save_context({"input": "x"}, {"output": "y"})
    traced = wrap_memory(memory)

    with trace_context(client):
        # Pure reads — should not fire agent.state.change
        _ = traced.load_memory_variables({})
        _ = traced.memory_variables

    assert events_of(uploads, "agent.state.change") == []


def test_mutation_tracker_groups_internal_save_contexts(client_and_uploads):
    """When a third-party agent calls save_context inside an operation we
    don't control, the tracker emits one logical-operation event per ``with``
    block rather than one per save_context."""
    client, uploads = client_and_uploads

    memory = ConversationBufferMemory()
    tracker = MemoryMutationTracker()

    with trace_context(client):
        with tracker.track(memory, operation="agent_turn_1"):
            # Two internal saves still produce ONE tracker mutation (we
            # snapshot before/after the with block).
            memory.save_context({"input": "q1"}, {"output": "a1"})
            memory.save_context({"input": "q1b"}, {"output": "a1b"})

    assert len(tracker.mutations) == 1
    mutation = tracker.mutations[0]
    assert mutation["operation"] == "agent_turn_1"
    assert mutation["before_hash"] != mutation["after_hash"]

    # And we emit one agent.state.change carrying the operation label
    evt = first_event(uploads, "agent.state.change")
    assert evt["payload"]["trigger"] == "agent_turn_1"
