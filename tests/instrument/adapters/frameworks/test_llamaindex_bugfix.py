"""Regression tests for LlamaIndex adapter source bugs (W1 coverage).

BUG-8 — standalone streaming drops events.

On the no-bound-collector path (``LlamaIndexAdapter(client).connect()`` WITHOUT
``@trace`` / ``instrument()``), a STREAMING LLM call used to drop its
``model.invoke`` + ``cost.record``. Mechanism: LlamaIndex exits the
``OpenAI.stream_chat`` span (``prepare_to_exit_span`` -> ``_on_span_exit``, which
pops AND flushes that root's per-span collector) the moment ``stream_chat``
RETURNS THE GENERATOR — BEFORE ``LLMChatEndEvent`` fires (that only fires after
the generator is fully consumed). So ``_on_llm_chat_end`` later called
``_fire('model.invoke'/'cost.record', span_id=<the stream span>)`` but the owning
collector was already popped + flushed-empty -> ``_collector_for`` returned
``None`` -> the events were silently dropped.

These tests drive a REAL ``llama_index.llms.openai.OpenAI.stream_chat`` over an
SSE ``httpx.MockTransport`` body through the REAL adapter (its span + event
handlers on the root dispatcher) with NO bound ``@trace`` collector — exactly
the standalone path. Only the network boundary is mocked; the real client does
the real SSE parsing and the real instrumentation fires.
"""

from __future__ import annotations

import json

import httpx
import pytest

llama_index_core = pytest.importorskip("llama_index.core")
pytest.importorskip("llama_index.llms.openai")

from llama_index.core import Document, VectorStoreIndex
from llama_index.llms.openai import OpenAI as LIOpenAI
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.instrumentation import get_dispatcher

from layerlens.instrument._context import _current_collector
from layerlens.instrument.adapters.frameworks.llamaindex import LlamaIndexAdapter

from .conftest import capture_framework_trace
from ...conftest import find_event, find_events

#: The model id echoed in the streamed chunks — asserting on it proves the value
#: was parsed off the REAL streamed ``ChatResponse.raw`` (a ``ChatCompletionChunk``),
#: not the ``gpt-4o-mini`` we *requested*.
_STREAM_MODEL = "gpt-4o-mini-2024-07-18"

#: The adapter installs these handler classes on the GLOBAL dispatcher.
_ADAPTER_HANDLER_NAMES = {"_SpanHandler", "_EventHandler"}


@pytest.fixture(autouse=True)
def _clean_dispatcher():
    yield
    dispatcher = get_dispatcher()
    dispatcher.event_handlers = [h for h in dispatcher.event_handlers if type(h).__name__ not in _ADAPTER_HANDLER_NAMES]
    dispatcher.span_handlers = [h for h in dispatcher.span_handlers if type(h).__name__ not in _ADAPTER_HANDLER_NAMES]


def _sse_body(*, with_usage: bool) -> str:
    """A real OpenAI ``chat.completion.chunk`` SSE stream. When ``with_usage`` the
    final (choice-less) chunk carries the usage block OpenAI only sends under
    ``stream_options={"include_usage": True}`` — mirroring the wire exactly."""
    chunks = [
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": _STREAM_MODEL,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": _STREAM_MODEL,
            "choices": [{"index": 0, "delta": {"content": "pong"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": _STREAM_MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    if with_usage:
        chunks.append(
            {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": _STREAM_MODEL,
                "choices": [],
                "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13},
            }
        )
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def _stream_llm(requests: list, *, with_usage: bool) -> LIOpenAI:
    """A real LlamaIndex OpenAI LLM whose transport serves our SSE stream. Only
    the network boundary is mocked — the real client does the real SSE parsing."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text=_sse_body(with_usage=with_usage),
            headers={"content-type": "text/event-stream"},
        )

    return LIOpenAI(
        model="gpt-4o-mini",
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )


def _drive_standalone_stream(mock_client, requests: list, *, with_usage: bool) -> str:
    """Drive a real streamed chat through the connected adapter on the STANDALONE
    path — NO bound ``@trace`` collector. LlamaIndex fires the ``LLMChatEndEvent``
    only after the generator is exhausted, i.e. AFTER the stream span has exited."""
    # Guard: this must be the no-bound path or the bug is masked by the bound
    # collector fallback in _fire.
    assert _current_collector.get() is None, "standalone path must have no bound collector"

    adapter = LlamaIndexAdapter(mock_client)
    adapter.connect()
    llm = _stream_llm(requests, with_usage=with_usage)

    kwargs = {"stream_options": {"include_usage": True}} if with_usage else {}
    text = ""
    try:
        for chunk in llm.stream_chat([ChatMessage(role="user", content="say pong")], **kwargs):
            text = chunk.message.content
    finally:
        adapter.disconnect()
    return text


class TestStandaloneStreamingCapture:
    def test_standalone_stream_captures_model_invoke_and_cost(self, mock_client):
        """BUG-8: standalone streaming must NOT drop model.invoke + cost.record."""
        captured = capture_framework_trace(mock_client)
        requests: list = []

        out = _drive_standalone_stream(mock_client, requests, with_usage=True)
        assert out == "pong"

        # Genuinely the streaming wire path (not .chat).
        body = json.loads(requests[0].content)
        assert body["stream"] is True
        assert body["stream_options"]["include_usage"] is True

        events = captured["events"]

        # The end-of-stream LLMChatEndEvent's model.invoke must have landed in the
        # standalone per-span collector (it used to be dropped).
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["framework"] == "llamaindex"
        # Parsed off the real streamed chunk raw, not the requested alias.
        assert mi["payload"]["model"] == _STREAM_MODEL
        assert mi["payload"]["tokens_prompt"] == 12
        assert mi["payload"]["tokens_completion"] == 1
        assert mi["payload"]["tokens_total"] == 13

        # ... and its priced cost.record.
        cost = find_event(events, "cost.record")
        assert cost["payload"]["model"] == _STREAM_MODEL
        assert cost["payload"]["tokens_total"] == 13

    def test_standalone_stream_without_usage_still_captures_model_invoke(self, mock_client):
        """Paired control: a plain stream (no usage chunk) still surfaces the
        model.invoke — tokens honestly omitted, no cost.record — proving the
        capture is real and not vacuously token-gated."""
        captured = capture_framework_trace(mock_client)
        requests: list = []

        out = _drive_standalone_stream(mock_client, requests, with_usage=False)
        assert out == "pong"

        body = json.loads(requests[0].content)
        assert body["stream"] is True

        events = captured["events"]
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["model"] == _STREAM_MODEL
        assert "tokens_prompt" not in mi["payload"]
        assert "tokens_total" not in mi["payload"]
        assert find_events(events, "cost.record") == []


def _stream_query_engine(index: VectorStoreIndex, embed: MockEmbedding, requests: list):
    """A real streaming RAG query engine whose synthesis LLM serves our SSE stream
    over MockTransport. ``response_mode="tree_summarize"`` returns a genuinely LAZY
    ``StreamingResponse`` — the synthesizer does NOT consume the LLM generator
    inside ``get_response`` (unlike the default ``compact``/``refine`` modes, which
    drain it to check ``query_satisfied``), so the trailing ``LLMChatEndEvent``
    fires only when the CALLER consumes ``response_gen``."""
    llm = _stream_llm(requests, with_usage=True)
    return index.as_query_engine(llm=llm, embed_model=embed, streaming=True, response_mode="tree_summarize")


class TestStandaloneStreamingQueryEngineCapture:
    """BUG-8, generalized one call-frame higher — a real streaming RAG query engine.

    ``index.as_query_engine(streaming=True, response_mode="tree_summarize").query()``
    returns a ``StreamingResponse`` synchronously WITHOUT consuming the LLM
    generator. The OUTER ``RetrieverQueryEngine.query`` span — the span that owns
    the standalone per-root collector — therefore EXITS while the nested LLM chat
    call is still in flight, and the ``LLMChatEndEvent`` (carrying
    ``model.invoke`` + ``cost.record``) fires one-or-more frames later, once the
    caller drives ``response_gen``, against an already torn-down span tree.

    The earlier partial fix keyed in-flight LLM tracking off the LLM call's OWN
    span, so the query span (a different id) exited seeing an in-flight count of
    zero → it flushed immediately → the streamed synthesis ``model.invoke`` /
    ``cost.record`` were still dropped. This drives the REAL chain (real
    ``VectorStoreIndex`` retrieval over ``MockEmbedding``, real OpenAI SSE
    synthesis over ``MockTransport``, no bound ``@trace``) and asserts the streamed
    synthesis events survive."""

    def test_standalone_streaming_query_engine_captures_synthesis_model_invoke(self, mock_client):
        captured = capture_framework_trace(mock_client)
        requests: list = []

        # Build the index BEFORE connecting so the offline doc-embedding pass is
        # not captured — only the query trace flushes. Standalone path only.
        assert _current_collector.get() is None, "standalone path must have no bound collector"
        embed = MockEmbedding(embed_dim=8)
        docs = [
            Document(text="Grass is green because of chlorophyll."),
            Document(text="The sky is blue due to Rayleigh scattering."),
        ]
        index = VectorStoreIndex.from_documents(docs, embed_model=embed)

        adapter = LlamaIndexAdapter(mock_client)
        adapter.connect()
        text = ""
        try:
            engine = _stream_query_engine(index, embed, requests)
            response = engine.query("Why is grass green?")
            # The query engine returned a lazy StreamingResponse; the LLM end event
            # fires only as we drain response_gen here — AFTER query() (and its
            # collector-owning span) has already exited.
            for token in response.response_gen:
                text += token
        finally:
            adapter.disconnect()

        assert text == "pong"

        # The synthesis leg genuinely streamed over the wire (not .chat).
        assert requests, "no HTTP request reached the synthesis LLM"
        body = json.loads(requests[-1].content)
        assert body["stream"] is True

        events = captured["events"]

        # The streamed synthesis model.invoke (parsed off the real streamed chunk
        # raw) must have survived the deferred flush of the query root collector.
        # It used to be dropped because the query span flushed before the end event.
        synth = [e for e in find_events(events, "model.invoke") if e["payload"].get("model") == _STREAM_MODEL]
        assert synth, (
            "streamed synthesis model.invoke was dropped — the query-engine root "
            "flushed before the deferred LLMChatEndEvent fired"
        )
        assert synth[0]["payload"]["framework"] == "llamaindex"
        assert synth[0]["payload"]["tokens_prompt"] == 12
        assert synth[0]["payload"]["tokens_completion"] == 1
        assert synth[0]["payload"]["tokens_total"] == 13

        # ... and its priced cost.record.
        cost = [e for e in find_events(events, "cost.record") if e["payload"].get("model") == _STREAM_MODEL]
        assert cost, "streamed synthesis cost.record was dropped alongside model.invoke"
        assert cost[0]["payload"]["tokens_total"] == 13

        # The retrieval leg (emitted BEFORE the query span exit) is still present —
        # i.e. the deferred trace is the SAME trace, not a fragment.
        retr = [e for e in find_events(events, "tool.result") if e["payload"].get("tool_name") == "retrieval"]
        assert retr, "retrieval tool.result missing from the deferred query trace"
