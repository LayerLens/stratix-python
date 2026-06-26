"""Recorded-real-response replay for the LangGraph framework (LAY-3614).

Drives a REAL ``langgraph`` ``StateGraph`` — a single node that calls a real
``langchain_openai.ChatOpenAI`` over ``httpx.MockTransport`` serving the captured
OpenAI response — with the real ``LangGraphCallbackHandler`` attached. This
exercises the full path: real provider response shape -> real LangGraph node
runtime + langchain-core callback objects -> real adapter -> emitted events,
which the matrix layer (fake models) and the hand-built unit doubles never
combine.

LangGraph reuses the langchain-core callback protocol, so the captured OpenAI
chat.completion fixture (the framework consumes the provider's response) drives
both the inherited ``model.invoke`` / ``cost.record`` and LangGraph's own
``agent.node.enter`` / ``agent.node.exit`` pair fired around the node.
"""

from __future__ import annotations

from typing import Any, Dict, List
from typing_extensions import TypedDict

import httpx
import pytest

pytest.importorskip("langgraph.graph")  # skips in envs without langgraph
pytest.importorskip("langchain_openai")  # ...or without langchain-openai

from langgraph.graph import END, START, StateGraph  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

from layerlens.instrument.adapters.frameworks.langgraph import LangGraphCallbackHandler

from .conftest import find_event, find_events, capture_framework_trace
from ..._recorded import load_recorded, mock_transport


class _State(TypedDict):
    messages: List[Any]


def _chat(fixture: Dict[str, Any]) -> ChatOpenAI:
    transport, _ = mock_transport(fixture)
    return ChatOpenAI(model="gpt-4o-mini", api_key="test-key", http_client=httpx.Client(transport=transport))


def _build_graph(fixture: Dict[str, Any]):
    """A minimal single-node graph: ``agent_node`` calls the recorded ChatOpenAI."""
    llm = _chat(fixture)

    def agent_node(state: _State) -> _State:
        reply = llm.invoke(state["messages"])
        return {"messages": state["messages"] + [reply]}

    graph = StateGraph(_State)
    graph.add_node("agent_node", agent_node)
    graph.add_edge(START, "agent_node")
    graph.add_edge("agent_node", END)
    return graph.compile()


class TestLangGraphRecorded:
    def test_graph_node_over_recorded_openai(self, mock_client):
        fixture = load_recorded("openai", "default")
        uploaded = capture_framework_trace(mock_client)
        handler = LangGraphCallbackHandler(mock_client)

        app = _build_graph(fixture)
        result = app.invoke(
            {"messages": [HumanMessage(content="Reply with exactly: pong")]},
            config={"callbacks": [handler]},
        )
        assert result["messages"][-1].content == "pong"

        events = uploaded["events"]

        # Inherited LangChain LLM path: the real callback objects carry the real
        # OpenAI usage/model from the recorded chat.completion body.
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["model"] == "gpt-4o-mini-2024-07-18"
        assert mi["payload"]["tokens_prompt"] == 12
        assert mi["payload"]["tokens_completion"] == 1
        assert mi["payload"]["tokens_total"] == 13
        assert mi["payload"]["finish_reason"] == "stop"

        cost = find_event(events, "cost.record")
        assert cost["payload"]["framework"] == "langgraph"
        assert cost["payload"]["tokens_total"] == 13

        # LangGraph-specific: the node lifecycle emits a paired enter/exit for the
        # real graph node (the langgraph runtime tags the chain with the node name).
        enters = find_events(events, "agent.node.enter")
        exits = find_events(events, "agent.node.exit")
        assert any(e["payload"]["node"] == "agent_node" for e in enters)
        assert any(e["payload"]["node"] == "agent_node" for e in exits)
