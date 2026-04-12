"""Tests for LlamaIndex adapter using real LlamaIndex types."""

from __future__ import annotations

import uuid
import threading
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

llama_index_core = pytest.importorskip("llama_index.core")

from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    ChatResponse,
    CompletionResponse,
)
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.base.response.schema import Response as LlamaResponse
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatStartEvent,
    LLMCompletionEndEvent,
)
from llama_index.core.instrumentation.events.agent import (
    AgentToolCallEvent,
    AgentRunStepEndEvent,
    AgentRunStepStartEvent,
)
from llama_index.core.instrumentation.events.query import (
    QueryEndEvent,
    QueryStartEvent,
)
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.instrumentation.events.embedding import (
    EmbeddingEndEvent,
    EmbeddingStartEvent,
)
from llama_index.core.instrumentation.events.exception import ExceptionEvent
from llama_index.core.instrumentation.events.retrieval import (
    RetrievalEndEvent,
    RetrievalStartEvent,
)

from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.frameworks.llamaindex import LlamaIndexAdapter

# -- Fixtures --


@pytest.fixture
def adapter(mock_client):
    return LlamaIndexAdapter(mock_client)


@pytest.fixture(autouse=True)
def clean_dispatcher():
    """Remove our handlers after each test to prevent leaks."""
    yield
    dispatcher = get_dispatcher()
    # Remove any _LayerLens handlers
    dispatcher.event_handlers = [h for h in dispatcher.event_handlers if "LayerLens" not in type(h).__name__]
    dispatcher.span_handlers = [h for h in dispatcher.span_handlers if "LayerLens" not in type(h).__name__]


def _find_events(adapter: LlamaIndexAdapter, event_type: str) -> List[Dict[str, Any]]:
    """Extract events of a given type from the adapter's collectors."""
    events: List[Dict[str, Any]] = []
    for collector in adapter._collectors.values():
        for ev in collector._events:
            if ev["event_type"] == event_type:
                events.append(ev)
    return events


def _all_events(adapter: LlamaIndexAdapter) -> List[Dict[str, Any]]:
    """Get all events from the adapter's collectors."""
    events: List[Dict[str, Any]] = []
    for collector in adapter._collectors.values():
        events.extend(collector._events)
    return events


def _emit_event_via_dispatcher(event: Any, span_id: Optional[str] = None) -> None:
    """Emit an event through the LlamaIndex dispatcher."""
    if span_id is not None:
        # LlamaIndex events have span_id as a field
        object.__setattr__(event, "span_id", span_id)
    dispatcher = get_dispatcher()
    dispatcher.event(event)


def _create_span(adapter: LlamaIndexAdapter, parent_span_id: Optional[str] = None) -> str:
    """Create a span in the adapter's span handler, return span_id."""
    import inspect

    span_id = f"Test.method-{uuid.uuid4().hex}"
    handler = adapter._span_handler
    # Use a mock BoundArguments
    mock_bound = MagicMock(spec=inspect.BoundArguments)
    handler.span_enter(
        id_=span_id,
        bound_args=mock_bound,
        instance=None,
        parent_id=parent_span_id,
    )
    return span_id


def _close_span(adapter: LlamaIndexAdapter, span_id: str) -> None:
    """Close a span, triggering flush if root."""
    import inspect

    handler = adapter._span_handler
    mock_bound = MagicMock(spec=inspect.BoundArguments)
    handler.span_exit(
        id_=span_id,
        bound_args=mock_bound,
        instance=None,
        result=None,
    )


# -- Test Classes --


class TestLlamaIndexAdapterLifecycle:
    def test_connect_sets_connected(self, adapter):
        adapter.connect()
        info = adapter.adapter_info()
        assert info.connected is True
        assert info.name == "llamaindex"

    def test_disconnect_clears_state(self, adapter):
        adapter.connect()
        adapter.disconnect()
        info = adapter.adapter_info()
        assert info.connected is False
        assert adapter._event_handler is None
        assert adapter._span_handler is None

    def test_connect_registers_handlers(self, adapter):
        dispatcher = get_dispatcher()
        initial_event_count = len(dispatcher.event_handlers)
        initial_span_count = len(dispatcher.span_handlers)

        adapter.connect()

        assert len(dispatcher.event_handlers) == initial_event_count + 1
        assert len(dispatcher.span_handlers) == initial_span_count + 1

    def test_disconnect_removes_handlers(self, adapter):
        dispatcher = get_dispatcher()
        initial_event_count = len(dispatcher.event_handlers)
        initial_span_count = len(dispatcher.span_handlers)

        adapter.connect()
        adapter.disconnect()

        assert len(dispatcher.event_handlers) == initial_event_count
        assert len(dispatcher.span_handlers) == initial_span_count

    def test_connect_without_llamaindex_raises(self, mock_client):
        with patch("layerlens.instrument.adapters.frameworks.llamaindex._HAS_LLAMAINDEX", False):
            adapter = LlamaIndexAdapter(mock_client)
            with pytest.raises(ImportError, match="llama-index-core"):
                adapter.connect()


class TestLLMChatEvents:
    def test_chat_end_emits_model_invoke(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        msg = ChatMessage(role=MessageRole.USER, content="What is Python?")
        response = ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="Python is a programming language."),
            raw={"model": "gpt-4", "usage": {"prompt_tokens": 15, "completion_tokens": 10}},
        )

        event = LLMChatEndEvent(messages=[msg], response=response, span_id=root)
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "model.invoke")
        assert len(events) >= 1
        payload = events[0]["payload"]
        assert payload["framework"] == "llamaindex"
        assert payload["model"] == "gpt-4"
        assert payload["tokens_prompt"] == 15
        assert payload["tokens_completion"] == 10
        assert payload["tokens_total"] == 25
        assert "output_message" in payload

    def test_chat_end_emits_cost_record(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        msg = ChatMessage(role=MessageRole.USER, content="hi")
        response = ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="hello"),
            raw={"model": "gpt-4o", "usage": {"prompt_tokens": 5, "completion_tokens": 3}},
        )

        event = LLMChatEndEvent(messages=[msg], response=response, span_id=root)
        _emit_event_via_dispatcher(event, span_id=root)

        cost_events = _find_events(adapter, "cost.record")
        assert len(cost_events) >= 1
        payload = cost_events[0]["payload"]
        assert payload["model"] == "gpt-4o"
        assert payload["tokens_total"] == 8

    def test_chat_latency_tracking(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        # Send start event
        start_event = LLMChatStartEvent(
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            additional_kwargs={},
            model_dict={"model": "gpt-4"},
            span_id=root,
        )
        _emit_event_via_dispatcher(start_event, span_id=root)

        # Brief pause for measurable latency
        import time

        time.sleep(0.01)

        # Send end event
        response = ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="hello"),
            raw={"model": "gpt-4", "usage": {"prompt_tokens": 5, "completion_tokens": 3}},
        )
        end_event = LLMChatEndEvent(
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            response=response,
            span_id=root,
        )
        _emit_event_via_dispatcher(end_event, span_id=root)

        events = _find_events(adapter, "model.invoke")
        assert len(events) >= 1
        payload = events[0]["payload"]
        assert "latency_ms" in payload
        assert payload["latency_ms"] >= 5  # at least 5ms

    def test_chat_with_messages_captured(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are helpful."),
            ChatMessage(role=MessageRole.USER, content="Hello"),
        ]
        response = ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="Hi!"),
            raw={},
        )
        event = LLMChatEndEvent(messages=messages, response=response, span_id=root)
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "model.invoke")
        assert len(events) >= 1
        payload = events[0]["payload"]
        assert "messages" in payload
        assert len(payload["messages"]) == 2

    def test_no_usage_no_cost_event(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        msg = ChatMessage(role=MessageRole.USER, content="hi")
        response = ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="hello"),
            raw={},  # No usage
        )
        event = LLMChatEndEvent(messages=[msg], response=response, span_id=root)
        _emit_event_via_dispatcher(event, span_id=root)

        cost_events = _find_events(adapter, "cost.record")
        assert len(cost_events) == 0


class TestLLMCompletionEvents:
    def test_completion_end_emits_model_invoke(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        response = CompletionResponse(
            text="Python is great!",
            raw={"model": "gpt-3.5-turbo-instruct", "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        )
        event = LLMCompletionEndEvent(prompt="What is Python?", response=response, span_id=root)
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "model.invoke")
        assert len(events) >= 1
        payload = events[0]["payload"]
        assert payload["framework"] == "llamaindex"
        assert payload["model"] == "gpt-3.5-turbo-instruct"
        assert "messages" in payload


class TestToolCallEvents:
    def test_tool_call_emits_event(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        tool = ToolMetadata(name="web_search", description="Search the web")
        event = AgentToolCallEvent(
            arguments='{"query": "Python tutorial"}',
            tool=tool,
            span_id=root,
        )
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "tool.call")
        assert len(events) >= 1
        payload = events[0]["payload"]
        assert payload["framework"] == "llamaindex"
        assert payload["tool_name"] == "web_search"
        assert payload["input"] == '{"query": "Python tutorial"}'
        assert payload["tool_description"] == "Search the web"

    def test_multiple_tool_calls(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        for name in ["search", "calculate", "summarize"]:
            tool = ToolMetadata(name=name, description=f"Tool: {name}")
            event = AgentToolCallEvent(arguments=f'{{"action": "{name}"}}', tool=tool, span_id=root)
            _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "tool.call")
        assert len(events) == 3
        names = [e["payload"]["tool_name"] for e in events]
        assert names == ["search", "calculate", "summarize"]


class TestRetrievalEvents:
    def test_retrieval_start_emits_tool_call(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        event = RetrievalStartEvent(str_or_query_bundle="How does RAG work?", span_id=root)
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "tool.call")
        assert len(events) >= 1
        payload = events[0]["payload"]
        assert payload["tool_name"] == "retrieval"
        assert payload["input"] == "How does RAG work?"

    def test_retrieval_end_emits_tool_result(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        # Create real nodes
        mock_nodes = []
        for i in range(3):
            text_node = TextNode(text=f"Document chunk {i}", id_=f"node-{i}")
            nws = NodeWithScore(node=text_node, score=0.9 - i * 0.1)
            mock_nodes.append(nws)

        event = RetrievalEndEvent(
            str_or_query_bundle="How does RAG work?",
            nodes=mock_nodes,
            span_id=root,
        )
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "tool.result")
        assert len(events) >= 1
        payload = events[0]["payload"]
        assert payload["tool_name"] == "retrieval"
        assert payload["num_results"] == 3
        assert len(payload["output"]) == 3
        assert payload["output"][0]["score"] == 0.9


class TestEmbeddingEvents:
    def test_embedding_start_emits_model_invoke(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        event = EmbeddingStartEvent(
            model_dict={"model_name": "text-embedding-ada-002"},
            span_id=root,
        )
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "model.invoke")
        assert len(events) >= 1
        payload = events[0]["payload"]
        assert payload["framework"] == "llamaindex"
        assert payload["model"] == "text-embedding-ada-002"
        assert payload["embedding"] is True

    def test_embedding_end_emits_dimensions(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        event = EmbeddingEndEvent(
            chunks=["chunk1", "chunk2", "chunk3"],
            embeddings=[[0.1] * 1536, [0.2] * 1536, [0.3] * 1536],
            span_id=root,
        )
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "model.invoke")
        assert len(events) >= 1
        payload = events[0]["payload"]
        assert payload["num_chunks"] == 3
        assert payload["num_embeddings"] == 3
        assert payload["embedding_dim"] == 1536


class TestQueryEvents:
    def test_query_start_emits_agent_input(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        event = QueryStartEvent(query="What is the meaning of life?", span_id=root)
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "agent.input")
        assert len(events) >= 1
        payload = events[0]["payload"]
        assert payload["input"] == "What is the meaning of life?"

    def test_query_end_emits_agent_output(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        event = QueryEndEvent(
            query="What is the meaning of life?",
            response=LlamaResponse(response="42"),
            span_id=root,
        )
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "agent.output")
        assert len(events) >= 1
        payload = events[0]["payload"]
        assert payload["status"] == "ok"
        assert payload["output"] == "42"


class TestAgentStepEvents:
    def test_agent_step_start(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        event = AgentRunStepStartEvent(
            task_id="task-123",
            step=MagicMock(),
            input="Do the thing",
            span_id=root,
        )
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "agent.input")
        assert len(events) >= 1
        payload = events[0]["payload"]
        assert payload["task_id"] == "task-123"

    def test_agent_step_end(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        event = AgentRunStepEndEvent(
            step_output="Step completed successfully",
            span_id=root,
        )
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "agent.output")
        assert len(events) >= 1
        payload = events[0]["payload"]
        assert payload["status"] == "ok"


class TestReRankEvents:
    def test_rerank_start_emits_tool_call(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        event = ReRankStartEvent(
            query="test query",
            nodes=[NodeWithScore(node=TextNode(text="test", id_="n1"), score=0.9)],
            top_n=5,
            model_name="cross-encoder/ms-marco",
            span_id=root,
        )
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "tool.call")
        assert len(events) >= 1
        payload = events[0]["payload"]
        assert payload["tool_name"] == "rerank"
        assert payload["model"] == "cross-encoder/ms-marco"
        assert payload["top_n"] == 5

    def test_rerank_end_emits_tool_result(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        event = ReRankEndEvent(
            nodes=[
                NodeWithScore(node=TextNode(text="a", id_="n1"), score=0.9),
                NodeWithScore(node=TextNode(text="b", id_="n2"), score=0.8),
            ],
            span_id=root,
        )
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "tool.result")
        assert len(events) >= 1
        payload = events[0]["payload"]
        assert payload["tool_name"] == "rerank"
        assert payload["num_results"] == 2


class TestExceptionEvents:
    def test_exception_emits_agent_error(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        event = ExceptionEvent(exception=ValueError("Something went wrong"), span_id=root)
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "agent.error")
        assert len(events) >= 1
        payload = events[0]["payload"]
        assert "Something went wrong" in payload["error"]
        assert payload["error_type"] == "ValueError"

    def test_runtime_error(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        event = ExceptionEvent(exception=RuntimeError("connection timeout"), span_id=root)
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "agent.error")
        assert len(events) >= 1
        payload = events[0]["payload"]
        assert "connection timeout" in payload["error"]
        assert payload["error_type"] == "RuntimeError"


class TestFullFlow:
    def test_complete_query_flow(self, adapter, mock_client):
        """Simulate a full RAG query flow: query → retrieval → LLM → response."""
        adapter.connect()
        root = _create_span(adapter)

        # 1. Query start
        _emit_event_via_dispatcher(
            QueryStartEvent(query="What is RAG?", span_id=root),
            span_id=root,
        )

        # 2. Retrieval
        _emit_event_via_dispatcher(
            RetrievalStartEvent(str_or_query_bundle="What is RAG?", span_id=root),
            span_id=root,
        )
        mock_node = NodeWithScore(
            node=TextNode(text="RAG stands for Retrieval-Augmented Generation...", id_="doc-1"),
            score=0.95,
        )
        _emit_event_via_dispatcher(
            RetrievalEndEvent(str_or_query_bundle="What is RAG?", nodes=[mock_node], span_id=root),
            span_id=root,
        )

        # 3. LLM call
        msgs = [ChatMessage(role=MessageRole.USER, content="What is RAG?")]
        response = ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="RAG is a technique..."),
            raw={"model": "gpt-4", "usage": {"prompt_tokens": 50, "completion_tokens": 30}},
        )
        _emit_event_via_dispatcher(
            LLMChatEndEvent(messages=msgs, response=response, span_id=root),
            span_id=root,
        )

        # 4. Query end
        _emit_event_via_dispatcher(
            QueryEndEvent(query="What is RAG?", response=LlamaResponse(response="RAG is a technique..."), span_id=root),
            span_id=root,
        )

        all_evts = _all_events(adapter)
        types = [e["event_type"] for e in all_evts]
        assert "agent.input" in types
        assert "tool.call" in types
        assert "tool.result" in types
        assert "model.invoke" in types
        assert "cost.record" in types
        assert "agent.output" in types
        assert len(all_evts) >= 6


class TestCaptureConfigGating:
    def test_minimal_config_suppresses_model_invoke(self, mock_client):
        config = CaptureConfig.minimal()
        adapter = LlamaIndexAdapter(mock_client, capture_config=config)
        adapter.connect()
        root = _create_span(adapter)

        # LLM event should be gated by L3
        msg = ChatMessage(role=MessageRole.USER, content="hi")
        response = ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="hello"),
            raw={"model": "gpt-4", "usage": {"prompt_tokens": 5, "completion_tokens": 3}},
        )
        _emit_event_via_dispatcher(
            LLMChatEndEvent(messages=[msg], response=response, span_id=root),
            span_id=root,
        )

        # model.invoke should be suppressed (L3 off)
        model_events = _find_events(adapter, "model.invoke")
        assert len(model_events) == 0

        # cost.record should still exist (always enabled)
        cost_events = _find_events(adapter, "cost.record")
        assert len(cost_events) >= 1

    def test_minimal_config_allows_agent_io(self, mock_client):
        config = CaptureConfig.minimal()
        adapter = LlamaIndexAdapter(mock_client, capture_config=config)
        adapter.connect()
        root = _create_span(adapter)

        _emit_event_via_dispatcher(
            QueryStartEvent(query="test", span_id=root),
            span_id=root,
        )

        events = _find_events(adapter, "agent.input")
        assert len(events) >= 1


class TestSpanHierarchy:
    def test_root_span_creates_collector(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        assert root in adapter._collectors

    def test_child_span_uses_parent_collector(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)
        child = _create_span(adapter, parent_span_id=root)

        assert child not in adapter._collectors
        # Child should find parent's collector
        collector = adapter._collector_for(child)
        assert collector is adapter._collectors[root]

    def test_root_span_close_flushes(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        # Emit an event
        _emit_event_via_dispatcher(
            QueryStartEvent(query="test", span_id=root),
            span_id=root,
        )

        # Close root span
        _close_span(adapter, root)

        assert root not in adapter._collectors
        # Verify flush happened (upload called)
        assert mock_client.traces.upload.called


class TestConcurrency:
    def test_concurrent_queries(self, adapter, mock_client):
        adapter.connect()
        errors = []
        results = {"events_per_thread": {}}

        def run_query(thread_id: int) -> None:
            try:
                root = _create_span(adapter)
                msg = ChatMessage(role=MessageRole.USER, content=f"Query {thread_id}")
                response = ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=f"Answer {thread_id}"),
                    raw={"model": "gpt-4", "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
                )
                _emit_event_via_dispatcher(
                    LLMChatEndEvent(messages=[msg], response=response, span_id=root),
                    span_id=root,
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_query, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestErrorIsolation:
    def test_broken_collector_does_not_crash(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        # Break the collector
        collector = adapter._collectors[root]
        collector.emit = MagicMock(side_effect=RuntimeError("collector broken"))

        # This should not raise
        msg = ChatMessage(role=MessageRole.USER, content="hi")
        response = ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="hello"),
            raw={},
        )
        _emit_event_via_dispatcher(
            LLMChatEndEvent(messages=[msg], response=response, span_id=root),
            span_id=root,
        )
        # If we get here without raising, the test passes

    def test_none_event_does_not_crash(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        # Directly call handle with various None scenarios
        event_handler = adapter._event_handler
        event_handler.handle(MagicMock(__class__=type("UnknownEvent", (), {})))
        # Should not crash


class TestEdgeCases:
    def test_no_raw_usage(self, adapter, mock_client):
        """Response with no raw usage data."""
        adapter.connect()
        root = _create_span(adapter)

        response = ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="hello"),
            raw=None,
        )
        event = LLMChatEndEvent(
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            response=response,
            span_id=root,
        )
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "model.invoke")
        assert len(events) >= 1
        payload = events[0]["payload"]
        assert "tokens_prompt" not in payload

    def test_usage_in_additional_kwargs(self, adapter, mock_client):
        """Some providers put usage in additional_kwargs."""
        adapter.connect()
        root = _create_span(adapter)

        response = ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="hello"),
            raw={},  # empty raw
            additional_kwargs={"usage": {"prompt_tokens": 20, "completion_tokens": 10}},
        )
        event = LLMChatEndEvent(
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            response=response,
            span_id=root,
        )
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "model.invoke")
        assert len(events) >= 1
        payload = events[0]["payload"]
        assert payload["tokens_prompt"] == 20
        assert payload["tokens_completion"] == 10

    def test_model_from_raw_object(self, adapter, mock_client):
        """Model name from a raw response object (not dict)."""
        adapter.connect()
        root = _create_span(adapter)

        raw_obj = MagicMock()
        raw_obj.model = "claude-3-opus"
        raw_obj.usage = None

        response = ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="hello"),
            raw=raw_obj,
        )
        event = LLMChatEndEvent(
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            response=response,
            span_id=root,
        )
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "model.invoke")
        assert len(events) >= 1
        assert events[0]["payload"]["model"] == "claude-3-opus"

    def test_empty_embedding(self, adapter, mock_client):
        """Embedding with no results."""
        adapter.connect()
        root = _create_span(adapter)

        event = EmbeddingEndEvent(chunks=[], embeddings=[], span_id=root)
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "model.invoke")
        assert len(events) >= 1
        payload = events[0]["payload"]
        assert payload["num_chunks"] == 0
        assert payload["num_embeddings"] == 0
        assert "embedding_dim" not in payload  # empty list, no dimension

    def test_disconnect_flushes_remaining(self, adapter, mock_client):
        """Disconnect should flush all open collectors."""
        adapter.connect()
        root = _create_span(adapter)

        _emit_event_via_dispatcher(
            QueryStartEvent(query="test", span_id=root),
            span_id=root,
        )

        # Don't close the span — just disconnect
        adapter.disconnect()

        # Should have flushed
        assert mock_client.traces.upload.called
