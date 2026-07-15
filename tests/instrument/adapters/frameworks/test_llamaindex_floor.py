"""Offline streaming + attestation + recorded-RAG floor for the LlamaIndex adapter.

Closes three W1 census gaps the existing ``test_llamaindex`` suite never covered,
without provider credentials or network:

* Streaming    — a REAL ``llama_index.llms.openai.OpenAI.stream_chat`` call over an
                 SSE ``httpx.MockTransport`` is driven through the REAL adapter (its
                 span + event handlers on the root dispatcher) under a bound
                 ``@trace`` collector. The instrumentation fires
                 ``LLMChatStartEvent``/``LLMChatEndEvent`` around the consumed
                 generator, and ``model.invoke`` must surface the streamed model.
                 Paired control: a plain stream (no ``stream_options``) carries NO
                 usage in its chunks, so tokens are honestly OMITTED and no
                 ``cost.record`` is emitted; the SAME streaming path WITH
                 ``stream_options={"include_usage": True}`` surfaces the real
                 per-call token counts (and a priced ``cost.record``). This is the
                 non-vacuity guarantee — the "omits" branch is only meaningful
                 because the identical path DOES surface tokens when the stream
                 carries them. There was zero streaming coverage before.

* Attestation  — the captured streaming trace's attestation chain is reconstructed
                 from ``attestation.chain.events`` and verifies offline
                 (mirrors the live harness ``_assert_attestation``). Non-vacuous:
                 a single tampered envelope hash flips ``verify_chain(...).valid``
                 to ``False``, and the envelope count must equal the event count
                 (so a streaming event that skipped the chain would fail).

* Recorded RAG — a REAL ``VectorStoreIndex.from_documents(...).as_query_engine()
                 .query(...)`` runs offline (``MockEmbedding`` for vectors) with the
                 synthesis LLM replaying the committed recorded OpenAI corpus over
                 ``httpx.MockTransport``. The real retrieval path must emit a
                 ``retrieval`` ``tool.call`` + ``tool.result`` whose ``num_results``
                 and node text reflect the two real indexed documents. This extends
                 the recorded tier (``test_llamaindex_recorded`` only drove ``.chat``).
"""

from __future__ import annotations

import copy
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

from layerlens.instrument import trace
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.frameworks.llamaindex import LlamaIndexAdapter

from ...conftest import find_event, find_events
from ..._recorded import load_recorded, mock_transport

#: The model id echoed in the streamed chunks — asserting on it proves the value
#: was parsed off the REAL streamed ``ChatResponse.raw`` (a ``ChatCompletionChunk``),
#: not the ``gpt-4o-mini`` we *requested*.
_STREAM_MODEL = "gpt-4o-mini-2024-07-18"

#: The adapter installs these handler classes on the GLOBAL dispatcher; the old
#: "LayerLens"-name filter never matched them, so cleanup used to be a no-op.
_ADAPTER_HANDLER_NAMES = {"_SpanHandler", "_EventHandler"}


@pytest.fixture(autouse=True)
def _clean_dispatcher():
    """Safety net: drop any adapter handlers left on the global dispatcher so a
    test that connects without a clean disconnect cannot leak into siblings."""
    yield
    dispatcher = get_dispatcher()
    dispatcher.event_handlers = [h for h in dispatcher.event_handlers if type(h).__name__ not in _ADAPTER_HANDLER_NAMES]
    dispatcher.span_handlers = [h for h in dispatcher.span_handlers if type(h).__name__ not in _ADAPTER_HANDLER_NAMES]


# ---------------------------------------------------------------------------
# Streaming helpers — a real OpenAI SSE stream served over MockTransport
# ---------------------------------------------------------------------------
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


def _drive_stream(mock_client, requests: list, *, with_usage: bool) -> str:
    """Drive a real streamed chat through the connected adapter under a bound
    ``@trace`` collector (the production capture path). LlamaIndex fires the
    ``LLMChatEndEvent`` only after the generator is exhausted."""
    adapter = LlamaIndexAdapter(mock_client, capture_config=CaptureConfig.full())
    adapter.connect()
    llm = _stream_llm(requests, with_usage=with_usage)

    @trace(mock_client, capture_config=CaptureConfig.full())
    def agent() -> str:
        kwargs = {"stream_options": {"include_usage": True}} if with_usage else {}
        text = ""
        for chunk in llm.stream_chat([ChatMessage(role="user", content="say pong")], **kwargs):
            text = chunk.message.content
        return text

    try:
        return agent()
    finally:
        adapter.disconnect()


# ---------------------------------------------------------------------------
# Streaming floor
# ---------------------------------------------------------------------------
class TestStreamingFloor:
    def test_streaming_surfaces_model_and_omits_tokens_without_usage(self, mock_client, capture_trace):
        requests: list = []
        out = _drive_stream(mock_client, requests, with_usage=False)
        assert out == "pong"

        # Genuinely the streaming wire path (not .chat).
        body = json.loads(requests[0].content)
        assert body["stream"] is True

        mi = find_event(capture_trace["events"], "model.invoke")
        assert mi["payload"]["framework"] == "llamaindex"
        # Parsed off the real streamed chunk raw, not the requested alias.
        assert mi["payload"]["model"] == _STREAM_MODEL
        # A plain OpenAI stream carries NO usage — tokens must be honestly omitted,
        # never fabricated, and there must be no priced cost.record.
        assert "tokens_prompt" not in mi["payload"]
        assert "tokens_completion" not in mi["payload"]
        assert "tokens_total" not in mi["payload"]
        assert find_events(capture_trace["events"], "cost.record") == []

    def test_streaming_surfaces_tokens_when_stream_carries_usage(self, mock_client, capture_trace):
        """Paired control (the non-vacuity guarantee for the test above): the SAME
        real streaming path, once the stream carries a usage chunk
        (``include_usage``), surfaces the real per-call token counts + cost."""
        requests: list = []
        _drive_stream(mock_client, requests, with_usage=True)

        body = json.loads(requests[0].content)
        assert body["stream"] is True
        assert body["stream_options"]["include_usage"] is True

        mi = find_event(capture_trace["events"], "model.invoke")
        assert mi["payload"]["model"] == _STREAM_MODEL
        assert mi["payload"]["tokens_prompt"] == 12
        assert mi["payload"]["tokens_completion"] == 1
        assert mi["payload"]["tokens_total"] == 13

        cost = find_event(capture_trace["events"], "cost.record")
        assert cost["payload"]["model"] == _STREAM_MODEL
        assert cost["payload"]["tokens_total"] == 13


# ---------------------------------------------------------------------------
# Offline attestation-chain verification (over a real streaming trace)
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_offline(self, mock_client, capture_trace):
        from layerlens.attestation._verify import verify_chain
        from layerlens.attestation._envelope import HashScope, AttestationEnvelope

        requests: list = []
        _drive_stream(mock_client, requests, with_usage=True)

        events = capture_trace["events"]
        raw = (capture_trace["attestation"].get("chain") or {}).get("events") or []

        def _envelopes(chain_events):
            return [
                AttestationEnvelope(
                    hash=e["hash"],
                    scope=HashScope(e["scope"]),
                    previous_hash=e.get("previous_hash"),
                )
                for e in chain_events
            ]

        envelopes = _envelopes(raw)
        assert envelopes, "no attestation envelopes captured for the streaming trace"
        # Every emitted event was chained (a streaming event that skipped the chain
        # would break this).
        assert len(envelopes) == len(events)
        result = verify_chain(envelopes)
        assert result.valid, f"attestation chain invalid: {result.error}"

        # Non-vacuity control: corrupting a single envelope hash must invalidate the
        # chain — proving verify_chain actually validates the linkage.
        tampered = copy.deepcopy(raw)
        mid = len(tampered) // 2
        original = tampered[mid]["hash"]
        tampered[mid]["hash"] = ("0" if original[0] != "0" else "1") + original[1:]
        assert not verify_chain(_envelopes(tampered)).valid


# ---------------------------------------------------------------------------
# Recorded RAG / retrieval replay (real VectorStoreIndex query, recorded LLM)
# ---------------------------------------------------------------------------
class TestRecordedRagRetrieval:
    def test_vectorstore_query_captures_real_retrieval(self, mock_client, capture_trace):
        # Build the index BEFORE connecting so the offline doc-embedding pass is not
        # captured — only the query trace flushes.
        embed = MockEmbedding(embed_dim=8)
        docs = [
            Document(text="Grass is green because of chlorophyll."),
            Document(text="The sky is blue due to Rayleigh scattering."),
        ]
        index = VectorStoreIndex.from_documents(docs, embed_model=embed)

        # Synthesis LLM replays the committed recorded OpenAI response shape.
        fixture = load_recorded("openai", "default")
        transport, _ = mock_transport(fixture)
        llm = LIOpenAI(
            model="gpt-4o-mini",
            api_key="test-key",
            http_client=httpx.Client(transport=transport),
            max_retries=0,
        )

        adapter = LlamaIndexAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        try:
            engine = index.as_query_engine(llm=llm, embed_model=embed)
            answer = str(engine.query("Why is grass green?"))
        finally:
            adapter.disconnect()

        # The recorded synthesis response flowed through the real query engine.
        assert answer == "pong"

        events = capture_trace["events"]

        # The real VectorStoreIndex retrieval fired a retrieval tool.call ...
        retr_calls = [e for e in find_events(events, "tool.call") if e["payload"].get("tool_name") == "retrieval"]
        assert retr_calls, "no retrieval tool.call captured from the real query path"
        assert retr_calls[0]["span_name"] == "retrieval"

        # ... and a tool.result whose node count + content reflect the two REAL
        # indexed documents (not a hand-built event).
        retr_results = [e for e in find_events(events, "tool.result") if e["payload"].get("tool_name") == "retrieval"]
        assert retr_results, "no retrieval tool.result captured from the real query path"
        assert retr_results[0]["payload"]["num_results"] == 2
        node_blob = json.dumps(retr_results[0]["payload"].get("output", [])).lower()
        assert "chlorophyll" in node_blob, "retrieved node text (real doc content) missing from tool.result"

        # The synthesis step read the recorded provider's real model id + usage.
        synth = [e for e in find_events(events, "model.invoke") if e["payload"].get("model") == _STREAM_MODEL]
        assert synth, "no synthesis model.invoke carrying the recorded model id"
        assert synth[0]["payload"]["tokens_total"] == 13
