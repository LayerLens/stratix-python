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

# AgentWorkflow event classes (the multi-agent path). Guard the import: older
# llama-index-core releases predate the ``agent.workflow`` package.
try:
    from llama_index.core.agent.workflow import (
        ToolCall as _WFToolCall,
        AgentInput as _WFAgentInput,
        AgentOutput as _WFAgentOutput,
    )

    _HAS_AGENT_WORKFLOW = True
except ImportError:  # pragma: no cover - depends on llama-index version
    _HAS_AGENT_WORKFLOW = False

# -- Fixtures --


@pytest.fixture
def adapter(mock_client):
    return LlamaIndexAdapter(mock_client)


#: The adapter installs these handler classes on the global dispatcher
#: (see ``llamaindex.py`` ``_make_span_handler`` / ``_make_event_handler``).
#: The old filter matched ``"LayerLens"`` in the class name, which never hit
#: these names — so cleanup was a no-op and handlers leaked across this module.
_ADAPTER_HANDLER_NAMES = {"_SpanHandler", "_EventHandler"}


def _drop_adapter_handlers(dispatcher: Any) -> None:
    """Remove the LlamaIndex adapter's own handlers from the global dispatcher."""
    dispatcher.event_handlers = [h for h in dispatcher.event_handlers if type(h).__name__ not in _ADAPTER_HANDLER_NAMES]
    dispatcher.span_handlers = [h for h in dispatcher.span_handlers if type(h).__name__ not in _ADAPTER_HANDLER_NAMES]


@pytest.fixture(autouse=True)
def clean_dispatcher():
    """Remove our handlers after each test to prevent leaks across this module."""
    yield
    _drop_adapter_handlers(get_dispatcher())


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
    def test_chat_end_emits_model_invoke(self, mock_client):
        adapter = LlamaIndexAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
        adapter.connect()
        root = _create_span(adapter)

        msg = ChatMessage(role=MessageRole.USER, content="What is Python?")
        response = ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="Python is a programming language."),
            raw={
                "model": "gpt-4",
                "usage": {"prompt_tokens": 15, "completion_tokens": 10},
            },
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

    def test_chat_end_emits_response_id_when_raw_has_id(self, mock_client):
        """S18/F11: surface the provider's own response id, never fabricated."""
        adapter = LlamaIndexAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
        adapter.connect()
        root = _create_span(adapter)

        msg = ChatMessage(role=MessageRole.USER, content="What is Python?")
        response = ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="Python is a programming language."),
            raw={
                "id": "chatcmpl-abc123",
                "model": "gpt-4",
                "usage": {"prompt_tokens": 15, "completion_tokens": 10},
            },
        )

        event = LLMChatEndEvent(messages=[msg], response=response, span_id=root)
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "model.invoke")
        assert events[0]["payload"]["response_id"] == "chatcmpl-abc123"

    def test_chat_end_no_response_id_when_raw_lacks_id(self, mock_client):
        """No 'id' on raw must stay honestly blank, not fabricated."""
        adapter = LlamaIndexAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
        adapter.connect()
        root = _create_span(adapter)

        msg = ChatMessage(role=MessageRole.USER, content="hi")
        response = ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="hello"),
            raw={"model": "gpt-4"},
        )

        event = LLMChatEndEvent(messages=[msg], response=response, span_id=root)
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "model.invoke")
        assert "response_id" not in events[0]["payload"]

    def test_chat_end_emits_cost_record(self, adapter, mock_client):
        adapter.connect()
        root = _create_span(adapter)

        msg = ChatMessage(role=MessageRole.USER, content="hi")
        response = ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="hello"),
            raw={
                "model": "gpt-4o",
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            },
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
            raw={
                "model": "gpt-4",
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            },
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

    def test_chat_with_messages_captured(self, mock_client):
        adapter = LlamaIndexAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
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
    def test_completion_end_emits_model_invoke(self, mock_client):
        adapter = LlamaIndexAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
        adapter.connect()
        root = _create_span(adapter)

        response = CompletionResponse(
            text="Python is great!",
            raw={
                "model": "gpt-3.5-turbo-instruct",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
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
    def test_tool_call_emits_event(self, mock_client):
        adapter = LlamaIndexAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
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
    def test_retrieval_start_emits_tool_call(self, mock_client):
        adapter = LlamaIndexAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
        adapter.connect()
        root = _create_span(adapter)

        event = RetrievalStartEvent(str_or_query_bundle="How does RAG work?", span_id=root)
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "tool.call")
        assert len(events) >= 1
        payload = events[0]["payload"]
        assert payload["tool_name"] == "retrieval"
        assert payload["input"] == "How does RAG work?"

    def test_retrieval_end_emits_tool_result(self, mock_client):
        adapter = LlamaIndexAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
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

    def test_embedding_end_computes_chunk_char_metrics(self, adapter, mock_client):
        # The chunk-length metrics (llamaindex.py:_on_embedding_end) are a real
        # computed delta — total length summed over NON-EMPTY chunks, avg via
        # integer floor-division — and were untested (grep chunk_chars = source
        # only). Chunk lengths 5, 8, 0 -> total 13 over 2 non-empty -> avg 13//2=6.
        adapter.connect()
        root = _create_span(adapter)

        event = EmbeddingEndEvent(
            chunks=["abcde", "fghijklm", ""],  # lengths 5, 8, 0
            embeddings=[[0.1] * 8, [0.2] * 8, [0.3] * 8],
            span_id=root,
        )
        _emit_event_via_dispatcher(event, span_id=root)

        payload = _find_events(adapter, "model.invoke")[0]["payload"]
        assert payload["num_chunks"] == 3, "num_chunks counts the whole list (empty included)"
        assert payload["num_embeddings"] == 3
        assert payload["embedding_dim"] == 8
        assert payload["chunk_chars_total"] == 13, "total must sum len over non-empty chunks (5+8)"
        # 13//2 == 6 bites: an empty chunk in the divisor -> 13//3==4; num_chunks as
        # divisor -> 13//3==4; float instead of floor -> 6.5. 6 is distinct from
        # num_chunks(3), total(13) and any single chunk length (5/8).
        assert payload["chunk_chars_avg"] == 6, "avg must be floor-division over NON-EMPTY chunk count"

    def test_l3_disabled_suppresses_embedding_fire(self, mock_client):
        # The l3_model_metadata=False early-returns (_on_embedding_start/_end) are
        # a real adapter-side suppression, but a collector-event assertion is
        # VACUOUS: model.invoke maps to l3_model_metadata, so the collector drops
        # it regardless. Bite the ADAPTER's own early-return with a _fire spy.
        adapter = LlamaIndexAdapter(mock_client, capture_config=CaptureConfig(l3_model_metadata=False))
        adapter.connect()
        root = _create_span(adapter)

        with patch.object(adapter, "_fire") as spy:
            _emit_event_via_dispatcher(
                EmbeddingStartEvent(model_dict={"model_name": "text-embedding-ada-002"}, span_id=root),
                span_id=root,
            )
            _emit_event_via_dispatcher(
                EmbeddingEndEvent(chunks=["abcde", "fghijklm", ""], embeddings=[[0.1] * 8] * 3, span_id=root),
                span_id=root,
            )

        # Deleting either early-return calls _fire("model.invoke", ...) -> >0.
        # test_embedding_end_computes_chunk_char_metrics proves the l3=True path
        # DOES fire, so this "not called" assertion is meaningful, not trivial.
        assert spy.call_count == 0, "l3_model_metadata=False did not suppress the adapter's embedding _fire"


class TestQueryEvents:
    def test_query_start_emits_agent_input(self, mock_client):
        adapter = LlamaIndexAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
        adapter.connect()
        root = _create_span(adapter)

        event = QueryStartEvent(query="What is the meaning of life?", span_id=root)
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "agent.input")
        assert len(events) >= 1
        payload = events[0]["payload"]
        assert payload["input"] == "What is the meaning of life?"

    def test_query_end_emits_agent_output(self, mock_client):
        adapter = LlamaIndexAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
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
    def test_exception_emits_agent_error(self, mock_client):
        adapter = LlamaIndexAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
        adapter.connect()
        root = _create_span(adapter)

        event = ExceptionEvent(exception=ValueError("Something went wrong"), span_id=root)
        _emit_event_via_dispatcher(event, span_id=root)

        events = _find_events(adapter, "agent.error")
        assert len(events) >= 1
        payload = events[0]["payload"]
        assert "Something went wrong" in payload["error"]
        assert payload["error_type"] == "ValueError"

    def test_runtime_error(self, mock_client):
        adapter = LlamaIndexAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
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
            raw={
                "model": "gpt-4",
                "usage": {"prompt_tokens": 50, "completion_tokens": 30},
            },
        )
        _emit_event_via_dispatcher(
            LLMChatEndEvent(messages=msgs, response=response, span_id=root),
            span_id=root,
        )

        # 4. Query end
        _emit_event_via_dispatcher(
            QueryEndEvent(
                query="What is RAG?",
                response=LlamaResponse(response="RAG is a technique..."),
                span_id=root,
            ),
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
            raw={
                "model": "gpt-4",
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            },
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


class TestCaptureContentRedaction:
    """W5/G10: with ``CaptureConfig(capture_content=False)`` the framework's
    content (agent.input/agent.output/tool.call args + LLM messages) must be
    SCRUBBED via the per-adapter ``_set_if_capturing`` gating. A control run
    with ``capture_content=True`` proves the gate is real, not vacuous."""

    SENTINEL = "SENTINEL_llamaindex_pii_4f7a9c2e"

    def _drive(self, adapter: LlamaIndexAdapter) -> str:
        """Drive content-bearing query/tool/LLM events carrying the SENTINEL."""
        root = _create_span(adapter)

        # agent.input — query text
        _emit_event_via_dispatcher(
            QueryStartEvent(query=f"query {self.SENTINEL}", span_id=root),
            span_id=root,
        )

        # tool.call — argument string
        tool = ToolMetadata(name="web_search", description="Search the web")
        _emit_event_via_dispatcher(
            AgentToolCallEvent(
                arguments=f'{{"q": "{self.SENTINEL}"}}',
                tool=tool,
                span_id=root,
            ),
            span_id=root,
        )

        # model.invoke — chat messages + output_message
        msg = ChatMessage(role=MessageRole.USER, content=f"prompt {self.SENTINEL}")
        response = ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=f"answer {self.SENTINEL}"),
            raw={"model": "gpt-4", "usage": {"prompt_tokens": 5, "completion_tokens": 3}},
        )
        _emit_event_via_dispatcher(
            LLMChatEndEvent(messages=[msg], response=response, span_id=root),
            span_id=root,
        )

        # agent.output — query response text
        _emit_event_via_dispatcher(
            QueryEndEvent(
                query=f"query {self.SENTINEL}",
                response=LlamaResponse(response=f"final {self.SENTINEL}"),
                span_id=root,
            ),
            span_id=root,
        )
        return root

    def test_content_scrubbed_when_capture_content_false(self, mock_client):
        import json as _json

        adapter = LlamaIndexAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        adapter.connect()
        self._drive(adapter)

        agent_in = _find_events(adapter, "agent.input")
        assert len(agent_in) >= 1
        assert "input" not in agent_in[0]["payload"]

        agent_out = _find_events(adapter, "agent.output")
        assert len(agent_out) >= 1
        assert "output" not in agent_out[0]["payload"]

        tool_call = _find_events(adapter, "tool.call")
        assert len(tool_call) >= 1
        assert "input" not in tool_call[0]["payload"]

        model_invoke = _find_events(adapter, "model.invoke")
        assert len(model_invoke) >= 1
        assert "messages" not in model_invoke[0]["payload"]
        assert "output_message" not in model_invoke[0]["payload"]

        # The SENTINEL must not appear anywhere in the emitted payloads.
        blob = _json.dumps(_all_events(adapter), default=str)
        assert self.SENTINEL not in blob

        adapter.disconnect()

    def test_content_present_when_capture_content_true(self, mock_client):
        """Control: the SAME drive WITH capture_content=True must include the
        SENTINEL — otherwise the redaction assertion above is vacuous."""
        import json as _json

        adapter = LlamaIndexAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
        adapter.connect()
        self._drive(adapter)

        agent_in = _find_events(adapter, "agent.input")
        assert agent_in[0]["payload"]["input"] == f"query {self.SENTINEL}"

        agent_out = _find_events(adapter, "agent.output")
        assert agent_out[0]["payload"]["output"] == f"final {self.SENTINEL}"

        tool_call = _find_events(adapter, "tool.call")
        assert self.SENTINEL in tool_call[0]["payload"]["input"]

        model_invoke = _find_events(adapter, "model.invoke")
        assert "messages" in model_invoke[0]["payload"]

        blob = _json.dumps(_all_events(adapter), default=str)
        assert self.SENTINEL in blob

        adapter.disconnect()


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
                    raw={
                        "model": "gpt-4",
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    },
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


# -- Disconnect leave-no-trace invariants (LAY-3577 / T3) --


def _make_sentinel_event_handler():
    """Create a third-party event handler registered via the same dispatcher API."""
    from llama_index.core.instrumentation.event_handlers import BaseEventHandler

    hits: List[Any] = []

    class _SentinelEventHandler(BaseEventHandler):
        @classmethod
        def class_name(cls) -> str:
            return "SentinelEventHandler"

        def handle(self, event: Any, **kwargs: Any) -> None:
            hits.append(event)

    return _SentinelEventHandler(), hits


class TestDisconnectLeaveNoTrace:
    """disconnect() must remove only the adapter's own handlers from the
    global dispatcher, leaving user handlers registered and functional,
    and must be safe to call repeatedly."""

    def test_user_handler_survives_disconnect(self, adapter):
        dispatcher = get_dispatcher()
        sentinel, hits = _make_sentinel_event_handler()
        dispatcher.add_event_handler(sentinel)
        try:
            adapter.connect()
            adapter.disconnect()

            # Still registered on the dispatcher...
            assert sentinel in dispatcher.event_handlers
            # ...and still functional.
            _emit_event_via_dispatcher(QueryStartEvent(query="probe"))
            assert any(isinstance(e, QueryStartEvent) for e in hits)
        finally:
            if sentinel in dispatcher.event_handlers:
                dispatcher.event_handlers.remove(sentinel)

    def test_adapter_handlers_removed_after_disconnect(self, adapter):
        dispatcher = get_dispatcher()
        adapter.connect()
        event_handler = adapter._event_handler
        span_handler = adapter._span_handler
        assert event_handler in dispatcher.event_handlers
        assert span_handler in dispatcher.span_handlers

        adapter.disconnect()

        assert event_handler not in dispatcher.event_handlers
        assert span_handler not in dispatcher.span_handlers
        assert adapter._event_handler is None
        assert adapter._span_handler is None

        # Events dispatched after disconnect must not reach the adapter.
        _emit_event_via_dispatcher(QueryStartEvent(query="ghost"))
        assert adapter._collectors == {}

    def test_double_disconnect_is_safe(self, adapter):
        dispatcher = get_dispatcher()
        sentinel, _ = _make_sentinel_event_handler()
        dispatcher.add_event_handler(sentinel)
        try:
            adapter.connect()
            adapter.disconnect()
            adapter.disconnect()  # must not raise

            assert adapter._event_handler is None
            assert adapter._span_handler is None
            assert sentinel in dispatcher.event_handlers
        finally:
            if sentinel in dispatcher.event_handlers:
                dispatcher.event_handlers.remove(sentinel)

    def test_reconnect_cycle(self, adapter):
        dispatcher = get_dispatcher()
        baseline_events = len(dispatcher.event_handlers)
        baseline_spans = len(dispatcher.span_handlers)

        adapter.connect()
        adapter.disconnect()
        adapter.connect()

        # Re-registered after reconnect.
        assert len(dispatcher.event_handlers) == baseline_events + 1
        assert len(dispatcher.span_handlers) == baseline_spans + 1

        # Events flow again after reconnect.
        root = _create_span(adapter)
        _emit_event_via_dispatcher(QueryStartEvent(query="cycle"), span_id=root)
        assert len(_find_events(adapter, "agent.input")) >= 1

        # Second disconnect cleans up again.
        adapter.disconnect()
        assert len(dispatcher.event_handlers) == baseline_events
        assert len(dispatcher.span_handlers) == baseline_spans


class TestDispatcherHandlerHygiene:
    """The adapter's handlers must not leak onto the global dispatcher across
    this module's tests. ``clean_dispatcher`` is the autouse safety net for any
    test that connects without disconnecting; pin that it actually removes the
    adapter's ``_SpanHandler`` / ``_EventHandler`` (the old ``"LayerLens"``-name
    filter never matched those class names, so cleanup was a no-op)."""

    @staticmethod
    def _ours(handlers: List[Any]) -> List[Any]:
        return [h for h in handlers if type(h).__name__ in {"_SpanHandler", "_EventHandler"}]

    def test_clean_dispatcher_removes_adapter_handlers(self, mock_client):
        dispatcher = get_dispatcher()

        adapter = LlamaIndexAdapter(mock_client)
        adapter.connect()
        # connect() installed our span + event handler on the global dispatcher.
        assert self._ours(dispatcher.span_handlers), "adapter did not register a span handler"
        assert self._ours(dispatcher.event_handlers), "adapter did not register an event handler"

        foreign_spans = [h for h in dispatcher.span_handlers if h not in self._ours(dispatcher.span_handlers)]
        foreign_events = [h for h in dispatcher.event_handlers if h not in self._ours(dispatcher.event_handlers)]

        # Teardown cleanup must remove every adapter handler (none may leak)
        # while leaving foreign handlers untouched.
        _drop_adapter_handlers(dispatcher)
        assert self._ours(dispatcher.span_handlers) == []
        assert self._ours(dispatcher.event_handlers) == []
        assert dispatcher.span_handlers == foreign_spans
        assert dispatcher.event_handlers == foreign_events


# ---------------------------------------------------------------------------
# Honest graph contract (Lever A)
# ---------------------------------------------------------------------------
# AgentWorkflow / FunctionAgent multi-agent runs carry the developer-declared
# agent name only inside the *workflow event stream* — the structured
# ``AgentInput``/``AgentOutput`` (``current_agent_name``) and the built-in
# ``handoff`` ``ToolCall`` (``to_agent``). Those events reach the adapter as the
# ``ev`` argument of each workflow step span (``bound_args.arguments["ev"]``),
# NOT as instrumentation events. The adapter must surface a producer-honest
# ``agent_name`` on agent.input/agent.output/model.invoke and emit a real
# ``agent.handoff{from_agent,to_agent}`` so the server graph engine renders the
# multi-agent topology. A pure RAG query (no named agent) must stay blank, and
# the framework's own unnamed default ("Agent") must never be fabricated into a
# node.


def _bound_ev(ev: Any) -> Any:
    """Build a real ``inspect.BoundArguments`` binding the workflow step's
    ``ev`` parameter — exactly what LlamaIndex's dispatcher hands the span
    handler for an ``AgentWorkflow`` step (``def step(self, ctx, ev): ...``)."""
    import inspect

    return inspect.signature(lambda ev: None).bind(ev=ev)


def _workflow_step(adapter: LlamaIndexAdapter, ev: Any, parent_span_id: str, span_id: Optional[str] = None) -> str:
    """Drive a single AgentWorkflow step span carrying a real workflow ``ev``."""
    span_id = span_id or f"AgentWorkflow.step-{uuid.uuid4().hex}"
    adapter._span_handler.span_enter(
        id_=span_id,
        bound_args=_bound_ev(ev),
        instance=None,
        parent_id=parent_span_id,
    )
    return span_id


@pytest.mark.skipif(not _HAS_AGENT_WORKFLOW, reason="llama-index AgentWorkflow events unavailable")
class TestHonestGraphContract:
    def _run_span(self, adapter: LlamaIndexAdapter) -> str:
        """The AgentWorkflow.run root span that owns the trace collector."""
        return _create_span(adapter)

    def test_agent_input_carries_honest_agent_name(self, mock_client):
        adapter = LlamaIndexAdapter(mock_client)
        adapter.connect()
        root = self._run_span(adapter)

        _workflow_step(
            adapter,
            _WFAgentInput(
                input=[ChatMessage(role=MessageRole.USER, content="route this")],
                current_agent_name="router",
            ),
            parent_span_id=root,
        )

        inp = _find_events(adapter, "agent.input")
        assert len(inp) >= 1
        assert inp[0]["payload"]["agent_name"] == "router"

    def test_agent_output_carries_honest_agent_name(self, mock_client):
        adapter = LlamaIndexAdapter(mock_client)
        adapter.connect()
        root = self._run_span(adapter)

        _workflow_step(
            adapter,
            _WFAgentOutput(
                response=ChatMessage(role=MessageRole.ASSISTANT, content="done"),
                tool_calls=[],
                raw=None,
                current_agent_name="fulfillment",
            ),
            parent_span_id=root,
        )

        out = _find_events(adapter, "agent.output")
        assert len(out) >= 1
        assert out[0]["payload"]["agent_name"] == "fulfillment"

    def test_model_invoke_stamped_with_current_agent(self, mock_client):
        """The LLM call happens under the agent's turn — model.invoke must carry
        the same honest agent_name so its graph node isn't orphaned."""
        adapter = LlamaIndexAdapter(mock_client)
        adapter.connect()
        root = self._run_span(adapter)

        # Router turn begins (sets the current agent for the run)...
        _workflow_step(
            adapter,
            _WFAgentInput(
                input=[ChatMessage(role=MessageRole.USER, content="hi")],
                current_agent_name="router",
            ),
            parent_span_id=root,
        )
        # ...then the LLM is invoked within that turn.
        response = ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="ok"),
            raw={"model": "gpt-4o", "usage": {"prompt_tokens": 5, "completion_tokens": 3}},
        )
        _emit_event_via_dispatcher(
            LLMChatEndEvent(
                messages=[ChatMessage(role=MessageRole.USER, content="hi")], response=response, span_id=root
            ),
            span_id=root,
        )

        mi = _find_events(adapter, "model.invoke")
        assert len(mi) >= 1
        assert mi[0]["payload"]["agent_name"] == "router"

    def test_handoff_emitted_on_transfer(self, mock_client):
        adapter = LlamaIndexAdapter(mock_client)
        adapter.connect()
        root = self._run_span(adapter)

        # router is active, then calls the built-in handoff tool -> fulfillment.
        _workflow_step(
            adapter,
            _WFAgentInput(
                input=[ChatMessage(role=MessageRole.USER, content="route")],
                current_agent_name="router",
            ),
            parent_span_id=root,
        )
        _workflow_step(
            adapter,
            _WFToolCall(tool_name="handoff", tool_kwargs={"to_agent": "fulfillment", "reason": "x"}, tool_id="c1"),
            parent_span_id=root,
        )

        ho = _find_events(adapter, "agent.handoff")
        assert len(ho) == 1
        assert ho[0]["payload"]["from_agent"] == "router"
        assert ho[0]["payload"]["to_agent"] == "fulfillment"

    def test_handoff_deduped_per_run(self, mock_client):
        adapter = LlamaIndexAdapter(mock_client)
        adapter.connect()
        root = self._run_span(adapter)

        _workflow_step(
            adapter,
            _WFAgentInput(
                input=[ChatMessage(role=MessageRole.USER, content="route")],
                current_agent_name="router",
            ),
            parent_span_id=root,
        )
        # The same router->fulfillment handoff edge appearing twice in one run
        # (e.g. a retried step) must still yield exactly one honest edge. The
        # companion ``ToolCallResult`` span for the same tool is ignored by the
        # adapter (only the ``ToolCall`` transfer is an edge), so it cannot
        # double-count either.
        _workflow_step(
            adapter,
            _WFToolCall(tool_name="handoff", tool_kwargs={"to_agent": "fulfillment"}, tool_id="c1"),
            parent_span_id=root,
        )
        _workflow_step(
            adapter,
            _WFToolCall(tool_name="handoff", tool_kwargs={"to_agent": "fulfillment"}, tool_id="c2"),
            parent_span_id=root,
        )

        assert len(_find_events(adapter, "agent.handoff")) == 1

    def test_full_two_agent_handoff_flow(self, mock_client):
        adapter = LlamaIndexAdapter(mock_client)
        adapter.connect()
        root = self._run_span(adapter)

        for ev in (
            _WFAgentInput(input=[ChatMessage(role=MessageRole.USER, content="q")], current_agent_name="router"),
            _WFToolCall(tool_name="handoff", tool_kwargs={"to_agent": "fulfillment"}, tool_id="c1"),
            _WFAgentOutput(
                response=ChatMessage(role=MessageRole.ASSISTANT, content=""),
                tool_calls=[],
                raw=None,
                current_agent_name="router",
            ),
            _WFAgentInput(input=[ChatMessage(role=MessageRole.USER, content="q")], current_agent_name="fulfillment"),
            _WFAgentOutput(
                response=ChatMessage(role=MessageRole.ASSISTANT, content="answer"),
                tool_calls=[],
                raw=None,
                current_agent_name="fulfillment",
            ),
        ):
            _workflow_step(adapter, ev, parent_span_id=root)

        names_in = {e["payload"].get("agent_name") for e in _find_events(adapter, "agent.input")}
        names_out = {e["payload"].get("agent_name") for e in _find_events(adapter, "agent.output")}
        assert names_in == {"router", "fulfillment"}
        assert names_out == {"router", "fulfillment"}

        ho = _find_events(adapter, "agent.handoff")
        assert len(ho) == 1
        assert (ho[0]["payload"]["from_agent"], ho[0]["payload"]["to_agent"]) == ("router", "fulfillment")

    def test_generic_default_agent_renders_verbatim_ateam_parity(self, mock_client):
        """An unnamed FunctionAgent gets llama-index's DEFAULT_AGENT_NAME
        ("Agent") — a generic class-default, never a producer identity. It must
        NOT be fabricated into a node, and no handoff to/from it is emitted."""
        adapter = LlamaIndexAdapter(mock_client)
        adapter.connect()
        root = self._run_span(adapter)

        _workflow_step(
            adapter,
            _WFAgentInput(
                input=[ChatMessage(role=MessageRole.USER, content="hi")],
                current_agent_name="Agent",
            ),
            parent_span_id=root,
        )
        _workflow_step(
            adapter,
            _WFAgentOutput(
                response=ChatMessage(role=MessageRole.ASSISTANT, content="done"),
                tool_calls=[],
                raw=None,
                current_agent_name="Agent",
            ),
            parent_span_id=root,
        )

        # ateam parity (#3): the generic default "Agent" is now surfaced VERBATIM
        # as agent_name so the workflow trace renders like ateam, instead of blank.
        for ev in _find_events(adapter, "agent.input") + _find_events(adapter, "agent.output"):
            assert ev["payload"].get("agent_name") == "Agent"
        # Handoff endpoints stay honest — no fabricated edge for a generic agent.
        assert _find_events(adapter, "agent.handoff") == []

    def test_rag_query_path_stays_blank(self, mock_client):
        """The pure RAG query-engine path has no named agent — agent.input/output
        from QueryStart/End must not carry a fabricated agent_name."""
        adapter = LlamaIndexAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
        adapter.connect()
        root = _create_span(adapter)

        _emit_event_via_dispatcher(QueryStartEvent(query="what is RAG?", span_id=root), span_id=root)
        _emit_event_via_dispatcher(
            QueryEndEvent(query="what is RAG?", response=LlamaResponse(response="..."), span_id=root),
            span_id=root,
        )

        for ev in _find_events(adapter, "agent.input") + _find_events(adapter, "agent.output"):
            assert "agent_name" not in ev["payload"]
        assert _find_events(adapter, "agent.handoff") == []
