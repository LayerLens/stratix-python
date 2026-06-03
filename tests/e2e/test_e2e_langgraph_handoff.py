"""End-to-end: LangGraph handoff detection + state hashing on a real StateGraph.

Builds a real multi-agent supervisor graph using ``langgraph.graph.StateGraph``,
wires the LangGraphCallbackHandler, and runs it. Verifies that:

- agent.node.enter / agent.node.exit fire for each node
- agent.state.change carries a sha256: hash that changes between nodes
- agent.handoff fires on transitions between named agent nodes
"""

from __future__ import annotations

from typing import TypedDict

import pytest

# Real langgraph or skip.
langgraph_graph = pytest.importorskip("langgraph.graph")
StateGraph = langgraph_graph.StateGraph
END = langgraph_graph.END
START = langgraph_graph.START

from layerlens.instrument.adapters.frameworks.langgraph import LangGraphCallbackHandler

from .conftest import events_of


class AgentState(TypedDict, total=False):
    messages: list
    next_agent: str
    counter: int


def _supervisor(state: AgentState) -> AgentState:
    return {"messages": state.get("messages", []) + ["supervisor: routing"], "counter": state.get("counter", 0) + 1}


def _researcher(state: AgentState) -> AgentState:
    return {"messages": state.get("messages", []) + ["researcher: found data"], "counter": state.get("counter", 0) + 1}


def _writer(state: AgentState) -> AgentState:
    return {"messages": state.get("messages", []) + ["writer: drafted summary"], "counter": state.get("counter", 0) + 1}


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", _supervisor)
    graph.add_node("researcher", _researcher)
    graph.add_node("writer", _writer)
    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", END)
    return graph.compile()


def test_real_supervisor_graph_emits_handoffs_and_state_changes(client_and_uploads):
    client, uploads = client_and_uploads
    handler = LangGraphCallbackHandler(client)

    graph = _build_graph()
    initial: AgentState = {"messages": [], "counter": 0}
    final = graph.invoke(initial, config={"callbacks": [handler]})

    # The graph ran end-to-end
    assert final["counter"] == 3
    assert any("writer" in m for m in final["messages"])

    # Node lifecycle events fired for every node
    node_enters = events_of(uploads, "agent.node.enter")
    node_exits = events_of(uploads, "agent.node.exit")
    visited_nodes = {e["payload"]["node"] for e in node_enters}
    assert {"supervisor", "researcher", "writer"}.issubset(visited_nodes)
    # Each entry has a matching exit
    assert {e["payload"]["node"] for e in node_exits} >= {
        "supervisor",
        "researcher",
        "writer",
    }

    # State hashes are emitted and they actually differ between nodes
    state_changes = events_of(uploads, "agent.state.change")
    assert len(state_changes) >= 3
    hashes = [e["payload"]["state_hash"] for e in state_changes]
    assert all(h.startswith("sha256:") for h in hashes)
    # The counter increments per node so consecutive hashes must differ
    assert len(set(hashes)) >= 2

    # Handoffs between named agents
    handoffs = events_of(uploads, "agent.handoff")
    transitions = [(h["payload"]["from_agent"], h["payload"]["to_agent"]) for h in handoffs]
    # supervisor -> researcher -> writer transitions should show up
    assert ("supervisor", "researcher") in transitions
    assert ("researcher", "writer") in transitions


def test_state_include_keys_scopes_the_hash(client_and_uploads):
    """If we tell the handler to hash only `counter`, two runs that differ
    in `messages` but match in `counter` should produce identical hashes."""
    client, uploads = client_and_uploads
    handler = LangGraphCallbackHandler(client, state_include_keys=["counter"])

    graph = _build_graph()
    graph.invoke({"messages": [], "counter": 0}, config={"callbacks": [handler]})
    graph.invoke({"messages": ["alien"], "counter": 0}, config={"callbacks": [handler]})

    state_changes = events_of(uploads, "agent.state.change")
    # Pair up by node + step where possible
    by_node: dict = {}
    for ev in state_changes:
        node = ev["payload"].get("node")
        by_node.setdefault(node, []).append(ev["payload"]["state_hash"])

    # For every node that was visited on both runs, the hash should be the
    # same across runs (because we're only hashing `counter`, which started
    # at the same value on both runs).
    for node, hashes in by_node.items():
        if len(hashes) >= 2:
            assert hashes[0] == hashes[1], f"hashes for {node} differ: {hashes}"


def test_disabling_handoff_silences_handoff_events(client_and_uploads):
    client, uploads = client_and_uploads
    handler = LangGraphCallbackHandler(client, detect_handoffs=False)

    graph = _build_graph()
    graph.invoke({"messages": [], "counter": 0}, config={"callbacks": [handler]})

    # State hashes still fire; handoffs do not.
    assert events_of(uploads, "agent.handoff") == []
    assert events_of(uploads, "agent.state.change")
