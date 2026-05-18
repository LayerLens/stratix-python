from __future__ import annotations

from uuid import uuid4
from unittest.mock import Mock

from langchain_core.callbacks import BaseCallbackHandler

from layerlens.instrument import CaptureConfig
from layerlens.instrument.adapters.frameworks.langgraph import LangGraphCallbackHandler

from .conftest import find_event, find_events, capture_framework_trace

# ---------------------------------------------------------------------------
# Sanity: real base class
# ---------------------------------------------------------------------------


class TestBaseClass:
    def test_inherits_langchain_base(self):
        assert issubclass(LangGraphCallbackHandler, BaseCallbackHandler)

    def test_name(self):
        handler = LangGraphCallbackHandler(Mock())
        assert handler.name == "langgraph"


# ---------------------------------------------------------------------------
# Inherited LangChain behavior
# ---------------------------------------------------------------------------


class TestInheritedBehavior:
    """LangGraph inherits all LangChain callbacks except on_chain_start."""

    def test_llm_events_inherited(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangGraphCallbackHandler(mock_client)

        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start({"name": "Graph"}, {}, run_id=chain_id)
        handler.on_llm_start(
            {"name": "ChatOpenAI"},
            ["prompt"],
            run_id=llm_id,
            parent_run_id=chain_id,
        )
        llm_response = Mock()
        llm_response.generations = [[Mock(text="output")]]
        llm_response.llm_output = {"model_name": "gpt-4", "token_usage": {"total_tokens": 10}}
        handler.on_llm_end(llm_response, run_id=llm_id)
        handler.on_chain_end({}, run_id=chain_id)

        events = uploaded["events"]
        assert len(find_events(events, "model.invoke")) >= 1
        assert find_event(events, "cost.record")["payload"]["tokens_total"] == 10

    def test_tool_events_inherited(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangGraphCallbackHandler(mock_client)

        chain_id = uuid4()
        tool_id = uuid4()

        handler.on_chain_start({"name": "Graph"}, {}, run_id=chain_id)
        handler.on_tool_start({"name": "search"}, "q", run_id=tool_id, parent_run_id=chain_id)
        handler.on_tool_end("results", run_id=tool_id)
        handler.on_chain_end({}, run_id=chain_id)

        events = uploaded["events"]
        assert find_event(events, "tool.call")["payload"]["name"] == "search"
        assert find_event(events, "tool.result")["payload"]["output"] == "results"

    def test_error_handling_inherited(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangGraphCallbackHandler(mock_client)

        chain_id = uuid4()
        handler.on_chain_start({"name": "Graph"}, {}, run_id=chain_id)
        handler.on_chain_error(RuntimeError("graph failed"), run_id=chain_id)

        error = find_event(uploaded["events"], "agent.error")
        assert error["payload"]["error"] == "graph failed"


# ---------------------------------------------------------------------------
# LangGraph-specific: on_chain_start node extraction
# ---------------------------------------------------------------------------


class TestNodeExtraction:
    def test_extracts_node_from_tags(self, mock_client):
        """LangGraph passes node names as plain tags (no colon)."""
        uploaded = capture_framework_trace(mock_client)
        handler = LangGraphCallbackHandler(mock_client)

        chain_id = uuid4()
        handler.on_chain_start(
            {"name": "RunnableSequence"},
            {"input": "hello"},
            run_id=chain_id,
            tags=["graph:step:1", "retriever_node"],
        )
        handler.on_chain_end({}, run_id=chain_id)

        agent_input = find_event(uploaded["events"], "agent.input")
        assert agent_input["payload"]["name"] == "retriever_node"

    def test_extracts_node_from_metadata(self, mock_client):
        """LangGraph puts node name in metadata.langgraph_node."""
        uploaded = capture_framework_trace(mock_client)
        handler = LangGraphCallbackHandler(mock_client)

        chain_id = uuid4()
        handler.on_chain_start(
            {"name": "RunnableSequence"},
            {"input": "hello"},
            run_id=chain_id,
            metadata={"langgraph_node": "agent_node"},
        )
        handler.on_chain_end({}, run_id=chain_id)

        agent_input = find_event(uploaded["events"], "agent.input")
        assert agent_input["payload"]["name"] == "agent_node"

    def test_metadata_overrides_tags(self, mock_client):
        """When both tags and metadata provide a node name, metadata wins."""
        uploaded = capture_framework_trace(mock_client)
        handler = LangGraphCallbackHandler(mock_client)

        chain_id = uuid4()
        handler.on_chain_start(
            {"name": "Seq"},
            {},
            run_id=chain_id,
            tags=["tag_node"],
            metadata={"langgraph_node": "meta_node"},
        )
        handler.on_chain_end({}, run_id=chain_id)

        agent_input = find_event(uploaded["events"], "agent.input")
        assert agent_input["payload"]["name"] == "meta_node"

    def test_falls_back_to_serialized_name(self, mock_client):
        """Without tags or metadata, falls back to serialized name."""
        uploaded = capture_framework_trace(mock_client)
        handler = LangGraphCallbackHandler(mock_client)

        chain_id = uuid4()
        handler.on_chain_start(
            {"name": "MyCustomChain"},
            {},
            run_id=chain_id,
        )
        handler.on_chain_end({}, run_id=chain_id)

        agent_input = find_event(uploaded["events"], "agent.input")
        assert agent_input["payload"]["name"] == "MyCustomChain"

    def test_skips_graph_step_tags(self, mock_client):
        """Tags starting with 'graph:step:' should be skipped."""
        uploaded = capture_framework_trace(mock_client)
        handler = LangGraphCallbackHandler(mock_client)

        chain_id = uuid4()
        handler.on_chain_start(
            {"name": "Default"},
            {},
            run_id=chain_id,
            tags=["graph:step:0", "graph:step:1"],
        )
        handler.on_chain_end({}, run_id=chain_id)

        agent_input = find_event(uploaded["events"], "agent.input")
        # No usable tags — falls back to serialized name
        assert agent_input["payload"]["name"] == "Default"


# ---------------------------------------------------------------------------
# adapter_info
# ---------------------------------------------------------------------------


class TestAdapterInfo:
    def test_info(self):
        handler = LangGraphCallbackHandler(Mock())
        info = handler.adapter_info()
        assert info.name == "langgraph"
        assert info.adapter_type == "framework"


# ---------------------------------------------------------------------------
# LangGraph-specific: SHA-256 state hashing
# ---------------------------------------------------------------------------


class TestStateHashing:
    def _run_node(self, handler, outputs, *, node="agent_node", tags=None, metadata=None):
        """Drive a single node lifecycle (chain_start -> chain_end) on the handler."""
        chain_id = uuid4()
        handler.on_chain_start(
            {"name": "Seq"},
            {},
            run_id=chain_id,
            tags=tags,
            metadata=metadata or {"langgraph_node": node},
        )
        handler.on_chain_end(outputs, run_id=chain_id)
        return chain_id

    def test_emits_state_change_with_sha256_prefix(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangGraphCallbackHandler(mock_client)

        self._run_node(handler, {"messages": ["hello"], "counter": 1})

        state_change = find_event(uploaded["events"], "agent.state.change")
        assert state_change["payload"]["node"] == "agent_node"
        assert state_change["payload"]["state_hash"].startswith("sha256:")
        # 64 hex chars after the prefix
        assert len(state_change["payload"]["state_hash"]) == len("sha256:") + 64
        assert state_change["payload"]["state_keys"] == ["counter", "messages"]

    def test_same_state_produces_same_hash(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangGraphCallbackHandler(mock_client)

        self._run_node(handler, {"messages": ["a"], "counter": 1}, node="node_1")
        self._run_node(handler, {"counter": 1, "messages": ["a"]}, node="node_2")  # key order swapped

        changes = find_events(uploaded["events"], "agent.state.change")
        assert len(changes) == 2
        assert changes[0]["payload"]["state_hash"] == changes[1]["payload"]["state_hash"]

    def test_different_state_produces_different_hash(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangGraphCallbackHandler(mock_client)

        self._run_node(handler, {"counter": 1}, node="n1")
        self._run_node(handler, {"counter": 2}, node="n2")

        changes = find_events(uploaded["events"], "agent.state.change")
        assert len(changes) == 2
        assert changes[0]["payload"]["state_hash"] != changes[1]["payload"]["state_hash"]

    def test_disabled_via_emit_state_hash_false(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangGraphCallbackHandler(mock_client, emit_state_hash=False)

        self._run_node(handler, {"counter": 1})

        # node.exit still emitted, but no state.change
        assert find_events(uploaded["events"], "agent.state.change") == []
        assert len(find_events(uploaded["events"], "agent.node.exit")) == 1

    def test_state_include_keys_filters(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangGraphCallbackHandler(mock_client, state_include_keys=["counter"])

        # Two runs that differ ONLY in a key excluded from the include set —
        # they should hash identically.
        self._run_node(handler, {"counter": 1, "junk": "a"}, node="n1")
        self._run_node(handler, {"counter": 1, "junk": "b"}, node="n2")

        changes = find_events(uploaded["events"], "agent.state.change")
        assert len(changes) == 2
        assert changes[0]["payload"]["state_hash"] == changes[1]["payload"]["state_hash"]
        assert changes[0]["payload"]["state_keys"] == ["counter"]

    def test_state_exclude_keys_filters(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        handler = LangGraphCallbackHandler(mock_client, state_exclude_keys=["timestamp"])

        self._run_node(handler, {"counter": 1, "timestamp": "2026-01-01"}, node="n1")
        self._run_node(handler, {"counter": 1, "timestamp": "2027-12-31"}, node="n2")

        changes = find_events(uploaded["events"], "agent.state.change")
        assert len(changes) == 2
        assert changes[0]["payload"]["state_hash"] == changes[1]["payload"]["state_hash"]
        assert "timestamp" not in changes[0]["payload"]["state_keys"]

    def test_non_serializable_state_falls_back_to_repr(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        # Disable content capture so the agent.node.exit payload doesn't try
        # to JSON-encode the opaque object; we only want to exercise our
        # state-hash fallback path here.
        config = CaptureConfig(capture_content=False)
        handler = LangGraphCallbackHandler(mock_client, capture_config=config)

        # An object that doesn't survive canonical_json (no to_dict, not dataclass)
        class _Opaque:
            def __repr__(self):
                return "<opaque>"

        self._run_node(handler, {"obj": _Opaque()})

        state_change = find_event(uploaded["events"], "agent.state.change")
        assert state_change["payload"]["state_hash"].startswith("sha256:")
        # state_keys is still set since the outer container is a dict
        assert state_change["payload"]["state_keys"] == ["obj"]
