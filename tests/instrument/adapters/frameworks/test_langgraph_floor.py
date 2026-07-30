"""Offline node-lifecycle + redaction + cost + attestation + streaming floor for LangGraph.

Closes the W1 census cells that the existing ``test_langgraph`` /
``test_langgraph_recorded`` suites leave open, so a regression fails in plain CI
without provider credentials or network:

* Node-exit-on-error — the ``on_chain_error`` node-exit branch
  (``langgraph.py`` lines 160-173) was UNTESTED: the existing
  ``test_error_handling_inherited`` starts a chain WITHOUT a ``langgraph_node``
  marker, so no pending node exists and that branch never runs. Here a node IS
  entered (real ``langgraph_node`` metadata) and then errors with a REAL
  ``langgraph.errors.GraphRecursionError`` — the framework's own exception, not a
  synthetic ``RuntimeError`` — surfacing as an ``agent.node.exit`` carrying
  ``status="error"`` + the error string + ``latency_ms``.
* Redaction     — ``capture_content=False`` over a node lifecycle + a handoff
  strips node input/output AND the handoff ``context`` while the deterministic
  hashes survive, with a ``True`` vacuity control and a SENTINEL sweep over
  ``json.dumps(events)`` (both directions).
* Cost/Tokens   — a REAL ``langgraph`` ``StateGraph`` node calling a real
  ``langchain_openai.ChatOpenAI`` over the recorded OpenAI response prices a
  ``cost.record`` end-to-end; ``cost_usd`` is non-None, strictly positive, and
  equals the pricing table's recomputation from the emitted tokens+model.
* Attestation   — the hash chain of a captured LangGraph trace verifies offline
  (mirrors the live harness ``_assert_attestation``).
* Streaming     — a streamed LLM span (``on_llm_new_token`` x2) fires INSIDE a
  node context: the merged ``model.invoke`` carries the streamed chunk count and
  its parent span is the node's span, bracketed by ``node.enter`` / ``node.exit``.
"""

from __future__ import annotations

import json
from uuid import uuid4
from typing import Any, List
from typing_extensions import TypedDict

import httpx
import pytest

# LangGraph reuses the langchain-core callback protocol; these imports transitively
# require langchain-core, so gating on the real packages skips the module cleanly
# in environments without the langchain audit venv (and satisfies skip-hygiene).
pytest.importorskip("langgraph.graph")
pytest.importorskip("langchain_openai")

from langgraph.graph import END, START, StateGraph  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.errors import GraphRecursionError  # noqa: E402
from langchain_core.outputs import LLMResult, ChatGeneration  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from layerlens.instrument import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.langgraph import LangGraphCallbackHandler  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402
from ..._recorded import load_recorded, mock_transport  # noqa: E402

SENTINEL = "LL-SENTINEL-7f3a9c2e"


# ---------------------------------------------------------------------------
# Real single-node StateGraph over the recorded OpenAI response (no creds/network)
# ---------------------------------------------------------------------------
class _State(TypedDict):
    messages: List[Any]


def _build_graph(fixture: dict):
    """A minimal single-node graph whose ``agent_node`` calls a recorded ChatOpenAI."""
    transport, _ = mock_transport(fixture)
    llm = ChatOpenAI(model="gpt-4o-mini", api_key="test-key", http_client=httpx.Client(transport=transport))

    def agent_node(state: _State) -> _State:
        reply = llm.invoke(state["messages"])
        return {"messages": state["messages"] + [reply]}

    graph = StateGraph(_State)
    graph.add_node("agent_node", agent_node)
    graph.add_edge(START, "agent_node")
    graph.add_edge("agent_node", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Floor 1 — node-exit on error (exercises langgraph.py:160-173, previously untested)
# ---------------------------------------------------------------------------
class TestNodeExitOnError:
    def test_node_error_emits_node_exit_with_status_and_latency(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        # capture ON so the error string survives (node.exit gates ``error`` as content).
        handler = LangGraphCallbackHandler(mock_client, capture_config=CaptureConfig.full())

        run_id = uuid4()
        handler.on_chain_start(
            {"name": "Seq"},
            {"messages": ["do work"]},
            run_id=run_id,
            metadata={"langgraph_node": "agent"},
        )
        # A REAL LangGraph framework exception — not a synthetic RuntimeError.
        err = GraphRecursionError("Recursion limit of 25 reached at node 'agent'")
        handler.on_chain_error(err, run_id=run_id)

        events = uploaded["events"]
        exits = find_events(events, "agent.node.exit")
        # Exactly one node.exit, and it came from the ERROR branch (no on_chain_end fired).
        assert len(exits) == 1
        payload = exits[0]["payload"]
        assert payload["node"] == "agent"
        assert payload["status"] == "error"
        assert "Recursion limit of 25 reached" in payload["error"]
        assert isinstance(payload["latency_ms"], (int, float))
        assert payload["latency_ms"] >= 0
        # No node.exit ever carries status "ok" here — proves this exit is the
        # error-path emission, not a normal completion.
        assert all(e["payload"].get("status") != "ok" for e in exits)
        # The base handler still surfaces the run-level agent.error alongside it.
        assert "Recursion limit of 25 reached" in find_event(events, "agent.error")["payload"]["error"]


# ---------------------------------------------------------------------------
# Floor 2 — content-absence redaction over a node lifecycle + handoff
# ---------------------------------------------------------------------------
class TestContentAbsenceRedaction:
    def _drive_two_node_handoff(self, handler):
        """supervisor -> researcher transition inside one graph invocation."""
        root = uuid4()
        handler.on_chain_start(
            {"name": "Seq"},
            {"task": f"summarize {SENTINEL}"},
            run_id=root,
            metadata={"langgraph_node": "supervisor"},
        )
        handler.on_chain_start(
            {"name": "Seq"},
            {"task": f"summarize {SENTINEL}"},
            run_id=uuid4(),
            parent_run_id=root,
            metadata={"langgraph_node": "researcher"},
        )
        handler.on_chain_end({"messages": [f"result {SENTINEL}"]}, run_id=root)

    def test_content_stripped_when_capture_off(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangGraphCallbackHandler(mock_client, capture_config=CaptureConfig(capture_content=False))
        self._drive_two_node_handoff(handler)

        events = uploaded["events"]
        # node.enter carries no input; node.exit carries no output.
        for enter in find_events(events, "agent.node.enter"):
            assert "input" not in enter["payload"]
        node_exit = find_event(events, "agent.node.exit")
        assert "output" not in node_exit["payload"]
        # handoff carries no raw context, but the deterministic hash survives.
        handoff = find_event(events, "agent.handoff")
        assert "context" not in handoff["payload"]
        assert handoff["payload"]["handoff_context_hash"].startswith("sha256:")
        # state hash survives even though the raw state is stripped.
        assert find_event(events, "agent.state.change")["payload"]["state_hash"].startswith("sha256:")

    def test_content_present_when_capture_on(self, mock_client):
        """Vacuity control: the SAME path DOES carry content when capture is on."""
        uploaded = capture_framework_trace(mock_client)
        handler = LangGraphCallbackHandler(mock_client, capture_config=CaptureConfig(capture_content=True))
        self._drive_two_node_handoff(handler)

        events = uploaded["events"]
        assert any("input" in e["payload"] for e in find_events(events, "agent.node.enter"))
        assert "output" in find_event(events, "agent.node.exit")["payload"]
        handoff = find_event(events, "agent.handoff")
        assert handoff["payload"]["context"]["task"] == f"summarize {SENTINEL}"

    def test_sentinel_never_leaks_when_redacted(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangGraphCallbackHandler(mock_client, capture_config=CaptureConfig(capture_content=False))
        self._drive_two_node_handoff(handler)

        blob = json.dumps(uploaded["events"])
        assert SENTINEL not in blob, "content redaction leaked the SENTINEL into the stored trace"

    def test_sentinel_present_when_capture_on(self, mock_client):
        """Vacuity control for the sweep above."""
        uploaded = capture_framework_trace(mock_client)
        handler = LangGraphCallbackHandler(mock_client, capture_config=CaptureConfig(capture_content=True))
        self._drive_two_node_handoff(handler)

        assert SENTINEL in json.dumps(uploaded["events"])


# ---------------------------------------------------------------------------
# Floor 3 — a real node's OpenAI call prices a cost.record (gpt-4o-mini is priced)
# ---------------------------------------------------------------------------
class TestCostRecord:
    def test_openai_node_cost_is_priced(self, mock_client):
        from layerlens.instrument.adapters.providers.pricing import PRICING, calculate_cost
        from layerlens.instrument.adapters.providers.token_usage import NormalizedTokenUsage

        fixture = load_recorded("openai", "default")
        uploaded = capture_framework_trace(mock_client)
        handler = LangGraphCallbackHandler(mock_client)

        app = _build_graph(fixture)
        result = app.invoke(
            {"messages": [HumanMessage(content="Reply with exactly: pong")]},
            config={"callbacks": [handler]},
        )
        assert result["messages"][-1].content == "pong"

        cost = find_event(uploaded["events"], "cost.record")["payload"]
        assert cost["cost_usd"] is not None, "priced model gpt-4o-mini produced no cost_usd"
        assert cost["cost_usd"] > 0
        # Strict, non-vacuous bite: the emitted cost must equal the pricing table's
        # recomputation from the emitted model + token counts (a dropped model or
        # lost tokens would make cost_usd None / wrong).
        usage = NormalizedTokenUsage(
            prompt_tokens=cost["tokens_prompt"],
            completion_tokens=cost["tokens_completion"],
            total_tokens=cost["tokens_total"],
        )
        expected = calculate_cost(cost["model"], usage, PRICING)
        assert expected is not None and expected > 0
        assert cost["cost_usd"] == expected


# ---------------------------------------------------------------------------
# Floor 4 — offline attestation-chain verification over a captured LangGraph trace
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_offline(self, mock_client):
        from layerlens.attestation._verify import verify_chain
        from layerlens.attestation._envelope import HashScope, AttestationEnvelope

        uploaded = capture_framework_trace(mock_client)
        handler = LangGraphCallbackHandler(mock_client, capture_config=CaptureConfig.full())

        # One root run == one flush, so the captured attestation covers exactly
        # these events (real adapter + real langchain message objects).
        run_id = uuid4()
        handler.on_chain_start(
            {"name": "Seq"},
            {"messages": [HumanMessage(content="weather in Tokyo?")]},
            run_id=run_id,
            metadata={"langgraph_node": "agent"},
        )
        handler.on_chain_end({"messages": [AIMessage(content="It is rainy.")]}, run_id=run_id)

        events = uploaded["events"]
        raw = (uploaded["attestation"].get("chain") or {}).get("events") or []
        envelopes = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured"
        assert len(envelopes) == len(events)
        result = verify_chain(envelopes)
        assert result.valid, f"attestation chain invalid: {result.error}"


# ---------------------------------------------------------------------------
# Floor 5 — a streamed LLM span fires inside the node context, bracketed by the node
# ---------------------------------------------------------------------------
class TestStreamingInterleave:
    def test_streamed_llm_bracketed_by_node(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangGraphCallbackHandler(mock_client, capture_config=CaptureConfig.full())

        node_id = uuid4()
        llm_id = uuid4()
        handler.on_chain_start(
            {"name": "Seq"},
            {"messages": [HumanMessage(content="stream please")]},
            run_id=node_id,
            metadata={"langgraph_node": "agent"},
        )
        handler.on_llm_start({"name": "ChatOpenAI"}, ["stream please"], run_id=llm_id, parent_run_id=node_id)
        handler.on_llm_new_token("Hel", run_id=llm_id, parent_run_id=node_id)
        handler.on_llm_new_token("lo", run_id=llm_id, parent_run_id=node_id)
        gen = ChatGeneration(
            message=AIMessage(content="Hello"), text="Hello", generation_info={"finish_reason": "stop"}
        )
        response = LLMResult(
            generations=[[gen]],
            llm_output={
                "model_name": "gpt-4o-mini",
                "token_usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )
        handler.on_llm_end(response, run_id=llm_id, parent_run_id=node_id)
        handler.on_chain_end({"messages": [AIMessage(content="Hello")]}, run_id=node_id)

        events = uploaded["events"]
        mi_event = find_event(events, "model.invoke")
        mi = mi_event["payload"]
        # Streaming metrics captured from the streamed tokens.
        assert mi["streaming"] is True
        assert mi["streamed_chunks"] == 2
        assert mi["output_message"] == "Hello"
        assert mi["tokens_total"] == 7

        # The LLM span is nested INSIDE the node: its parent span is the node's span.
        node_enter = find_event(events, "agent.node.enter")
        assert mi_event["parent_span_id"] == node_enter["span_id"]

        # node.enter / node.exit bracket the LLM span in emission order.
        types = [e["event_type"] for e in events]
        assert types.index("agent.node.enter") < types.index("model.invoke") < types.index("agent.node.exit")
