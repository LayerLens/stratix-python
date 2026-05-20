"""Streaming-aggregation tests for the OpenAI + Anthropic providers.

Covers the chunk/event accumulators that turn an SSE stream into a single
response object (which the rest of the emit path then treats the same as a
non-streaming response):

* ``_StreamedChatResponse.from_chunks`` (OpenAI)         -> LAY-3326
* ``_StreamedMessage.from_events``    (Anthropic)        -> LAY-3329

Tests also exercise the end-to-end ``extract_meta`` path on streamed responses,
which is how thinking-token estimates (LAY-3330) and cache-token capture
(LAY-2881) reach the OTel attribute mapper.

We use ``SimpleNamespace`` shims rather than real SDK types because both
adapters access fields exclusively via ``getattr``; this keeps the tests
decoupled from SDK class internals.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

from layerlens.instrument.adapters.providers.openai import (
    OpenAIProvider,
    _StreamedChatResponse,
)
from layerlens.instrument.adapters.providers.anthropic import (
    AnthropicProvider,
    _StreamedMessage,
)

# ---------------------------------------------------------------------------
# OpenAI chunk helpers
# ---------------------------------------------------------------------------


def _openai_chunk(
    *,
    content: str | None = None,
    role: str | None = None,
    finish_reason: str | None = None,
    tool_calls: List[Any] | None = None,
    model: str | None = None,
    response_id: str | None = None,
    system_fingerprint: str | None = None,
    service_tier: str | None = None,
    usage: Any = None,
) -> SimpleNamespace:
    delta = SimpleNamespace(content=content, role=role, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason, index=0)
    return SimpleNamespace(
        choices=[choice],
        model=model,
        id=response_id,
        system_fingerprint=system_fingerprint,
        service_tier=service_tier,
        usage=usage,
    )


def _openai_tool_call_fragment(
    *, index: int, id: str | None = None, name: str | None = None, arguments: str = ""
) -> SimpleNamespace:
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=id, type="function", function=fn)


# ---------------------------------------------------------------------------
# OpenAI streaming -- LAY-3326
# ---------------------------------------------------------------------------


class TestOpenAIStreamingAggregation:
    def test_text_content_concatenated_across_chunks(self):
        chunks = [
            _openai_chunk(role="assistant", content="Hel"),
            _openai_chunk(content="lo, "),
            _openai_chunk(content="world!"),
            _openai_chunk(finish_reason="stop"),
        ]
        agg = _StreamedChatResponse.from_chunks(chunks)
        assert agg.choices[0].message.role == "assistant"
        assert agg.choices[0].message.content == "Hello, world!"
        assert agg.choices[0].finish_reason == "stop"

    def test_response_metadata_carried_through_from_first_chunk(self):
        chunks = [
            _openai_chunk(
                content="hi",
                model="gpt-4o-2024-11-20",
                response_id="chatcmpl-abc",
                system_fingerprint="fp_test123",
                service_tier="scale",
            ),
            _openai_chunk(finish_reason="stop"),
        ]
        agg = _StreamedChatResponse.from_chunks(chunks)
        assert agg.model == "gpt-4o-2024-11-20"
        assert agg.id == "chatcmpl-abc"
        assert agg.system_fingerprint == "fp_test123"
        assert agg.service_tier == "scale"

    def test_usage_taken_from_last_chunk_that_provides_it(self):
        usage = SimpleNamespace(prompt_tokens=12, completion_tokens=34, total_tokens=46)
        chunks = [
            _openai_chunk(content="x"),
            _openai_chunk(usage=usage, finish_reason="stop"),
        ]
        agg = _StreamedChatResponse.from_chunks(chunks)
        assert agg.usage is usage

    def test_tool_call_fragments_assembled_by_index(self):
        # OpenAI streams tool calls as deltas across multiple chunks, keyed by
        # ``index``. The aggregator must concatenate ``arguments`` and pick up
        # ``id`` / ``name`` from whichever chunk first carried them.
        chunks = [
            _openai_chunk(
                tool_calls=[_openai_tool_call_fragment(index=0, id="call_a", name="lookup", arguments='{"q"')]
            ),
            _openai_chunk(tool_calls=[_openai_tool_call_fragment(index=0, arguments=': "weather"')]),
            _openai_chunk(tool_calls=[_openai_tool_call_fragment(index=0, arguments=', "city": "sf"}')]),
            _openai_chunk(finish_reason="tool_calls"),
        ]
        agg = _StreamedChatResponse.from_chunks(chunks)
        tool_calls = OpenAIProvider.extract_tool_calls(agg)
        assert len(tool_calls) == 1
        tc = tool_calls[0]
        assert tc["id"] == "call_a"
        assert tc["tool_name"] == "lookup"
        # JSON arguments are concatenated and then parsed.
        assert tc["arguments"] == {"q": "weather", "city": "sf"}

    def test_parallel_tool_calls_kept_separate_by_index(self):
        chunks = [
            _openai_chunk(tool_calls=[_openai_tool_call_fragment(index=0, id="call_a", name="a", arguments="{}")]),
            _openai_chunk(tool_calls=[_openai_tool_call_fragment(index=1, id="call_b", name="b", arguments="{}")]),
            _openai_chunk(finish_reason="tool_calls"),
        ]
        agg = _StreamedChatResponse.from_chunks(chunks)
        tool_calls = OpenAIProvider.extract_tool_calls(agg)
        assert [tc["tool_name"] for tc in tool_calls] == ["a", "b"]
        assert [tc["id"] for tc in tool_calls] == ["call_a", "call_b"]

    def test_empty_chunk_list_returns_none(self):
        assert OpenAIProvider.aggregate_stream([]) is None

    def test_extract_meta_on_streamed_response_includes_finish_reason_and_fingerprint(self):
        # End-to-end: aggregate -> extract_meta should expose finish_reason and
        # system_fingerprint exactly the way the non-streaming path does, so
        # downstream OTel mapping (gen_ai.response.finish_reasons, etc.) works.
        usage = SimpleNamespace(
            prompt_tokens=5,
            completion_tokens=7,
            total_tokens=12,
            prompt_tokens_details=SimpleNamespace(cached_tokens=2),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=128),
        )
        chunks = [
            _openai_chunk(
                role="assistant",
                content="hi",
                model="gpt-4o",
                response_id="chatcmpl-z",
                system_fingerprint="fp_abc",
                service_tier="scale",
            ),
            _openai_chunk(usage=usage, finish_reason="stop"),
        ]
        agg = _StreamedChatResponse.from_chunks(chunks)
        meta = OpenAIProvider.extract_meta(agg)
        assert meta["finish_reason"] == "stop"
        assert meta["system_fingerprint"] == "fp_abc"
        assert meta["service_tier"] == "scale"
        assert meta["response_id"] == "chatcmpl-z"
        assert meta["usage"]["prompt_tokens"] == 5
        assert meta["usage"]["cached_tokens"] == 2
        assert meta["usage"]["reasoning_tokens"] == 128


# ---------------------------------------------------------------------------
# Anthropic event helpers
# ---------------------------------------------------------------------------


def _message_start_event(
    *,
    id: str = "msg_abc",
    model: str = "claude-3-7-sonnet-20250219",
    input_tokens: int = 10,
    cache_read_input_tokens: int | None = None,
    cache_creation_input_tokens: int | None = None,
    thinking_tokens: int | None = None,
) -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        thinking_tokens=thinking_tokens,
    )
    message = SimpleNamespace(id=id, model=model, role="assistant", usage=usage)
    return SimpleNamespace(type="message_start", message=message)


def _content_block_start_event(*, block_type: str, id: str | None = None, name: str | None = None) -> SimpleNamespace:
    block = SimpleNamespace(type=block_type, id=id, name=name)
    return SimpleNamespace(type="content_block_start", content_block=block, index=0)


def _content_block_delta_event(
    *,
    delta_type: str,
    text: str | None = None,
    thinking: str | None = None,
    partial_json: str | None = None,
    index: int = 0,
) -> SimpleNamespace:
    delta = SimpleNamespace(type=delta_type, text=text, thinking=thinking, partial_json=partial_json)
    return SimpleNamespace(type="content_block_delta", delta=delta, index=index)


def _message_delta_event(
    *,
    stop_reason: str | None = "end_turn",
    stop_sequence: str | None = None,
    output_tokens: int | None = 20,
    thinking_tokens: int | None = None,
) -> SimpleNamespace:
    delta = SimpleNamespace(stop_reason=stop_reason, stop_sequence=stop_sequence)
    usage = SimpleNamespace(output_tokens=output_tokens, thinking_tokens=thinking_tokens)
    return SimpleNamespace(type="message_delta", delta=delta, usage=usage)


# ---------------------------------------------------------------------------
# Anthropic streaming -- LAY-3329 + LAY-3330
# ---------------------------------------------------------------------------


class TestAnthropicStreamingAggregation:
    def test_basic_message_text_flow(self):
        events = [
            _message_start_event(input_tokens=12),
            _content_block_start_event(block_type="text"),
            _content_block_delta_event(delta_type="text_delta", text="Hello"),
            _content_block_delta_event(delta_type="text_delta", text=", world!"),
            _message_delta_event(stop_reason="end_turn", output_tokens=15),
        ]
        msg = _StreamedMessage.from_events(events)
        assert msg.id == "msg_abc"
        assert msg.model == "claude-3-7-sonnet-20250219"
        assert msg.role == "assistant"
        assert msg.stop_reason == "end_turn"
        assert len(msg.content) == 1
        assert msg.content[0].type == "text"
        assert msg.content[0].text == "Hello, world!"
        assert msg.usage.input_tokens == 12
        assert msg.usage.output_tokens == 15

    def test_thinking_block_accumulated(self):
        events = [
            _message_start_event(),
            _content_block_start_event(block_type="thinking"),
            _content_block_delta_event(delta_type="thinking_delta", thinking="Let me "),
            _content_block_delta_event(delta_type="thinking_delta", thinking="think about this..."),
            _content_block_start_event(block_type="text"),
            _content_block_delta_event(delta_type="text_delta", text="Answer."),
            _message_delta_event(stop_reason="end_turn", output_tokens=8),
        ]
        msg = _StreamedMessage.from_events(events)
        # Both blocks captured in order.
        types = [b.type for b in msg.content]
        assert types == ["thinking", "text"]
        thinking_block = msg.content[0]
        assert thinking_block.thinking == "Let me think about this..."
        # End-to-end through extract_meta: thinking tokens estimated from
        # accumulated thinking content (chars / 4 fallback).
        meta = AnthropicProvider.extract_meta(msg)
        expected_thinking_chars = len("Let me think about this...")
        assert meta["usage"]["thinking_tokens"] == expected_thinking_chars // 4
        # ``reasoning_tokens`` aliases ``thinking_tokens`` so the unified OTel
        # ``gen_ai.usage.reasoning_tokens`` attribute can be derived from it.
        assert meta["usage"]["reasoning_tokens"] == meta["usage"]["thinking_tokens"]

    def test_thinking_tokens_from_api_preferred_over_estimate(self):
        # If the Anthropic SDK ever surfaces ``thinking_tokens`` in
        # ``message_delta.usage``, the streaming aggregator must capture it
        # and ``extract_meta`` must prefer it to the char-count estimate.
        events = [
            _message_start_event(),
            _content_block_start_event(block_type="thinking"),
            _content_block_delta_event(delta_type="thinking_delta", thinking="a" * 800),
            _message_delta_event(stop_reason="end_turn", output_tokens=10, thinking_tokens=42),
        ]
        msg = _StreamedMessage.from_events(events)
        assert msg.usage.thinking_tokens == 42
        meta = AnthropicProvider.extract_meta(msg)
        # API-reported value wins over chars/4 (which would be 200).
        assert meta["usage"]["thinking_tokens"] == 42

    def test_cache_tokens_captured_from_message_start(self):
        events = [
            _message_start_event(
                input_tokens=8,
                cache_read_input_tokens=120,
                cache_creation_input_tokens=300,
            ),
            _content_block_start_event(block_type="text"),
            _content_block_delta_event(delta_type="text_delta", text="ok"),
            _message_delta_event(stop_reason="end_turn", output_tokens=5),
        ]
        msg = _StreamedMessage.from_events(events)
        assert msg.usage.cache_read_input_tokens == 120
        assert msg.usage.cache_creation_input_tokens == 300
        meta = AnthropicProvider.extract_meta(msg)
        # Both Anthropic-native and OpenAI-style aliases populated for the
        # OTel mapper to pick up.
        assert meta["usage"]["cache_read_input_tokens"] == 120
        assert meta["usage"]["cache_creation_input_tokens"] == 300
        assert meta["usage"]["cached_tokens"] == 120

    def test_tool_use_json_fragments_assembled(self):
        events = [
            _message_start_event(),
            _content_block_start_event(block_type="tool_use", id="tool_abc", name="get_weather"),
            _content_block_delta_event(delta_type="input_json_delta", partial_json='{"city":', index=0),
            _content_block_delta_event(delta_type="input_json_delta", partial_json=' "sf"}', index=0),
            _message_delta_event(stop_reason="tool_use", output_tokens=12),
        ]
        msg = _StreamedMessage.from_events(events)
        assert len(msg.content) == 1
        tool_block = msg.content[0]
        assert tool_block.type == "tool_use"
        assert tool_block.id == "tool_abc"
        assert tool_block.name == "get_weather"
        assert tool_block.input == {"city": "sf"}
        # extract_tool_calls reads the assembled block.
        tool_calls = AnthropicProvider.extract_tool_calls(msg)
        assert tool_calls == [
            {"id": "tool_abc", "type": "tool_use", "tool_name": "get_weather", "arguments": {"city": "sf"}}
        ]

    def test_empty_event_list_yields_empty_message(self):
        msg = _StreamedMessage.from_events([])
        assert msg.id is None
        assert msg.content == []
        assert msg.usage.input_tokens == 0
        assert msg.usage.output_tokens == 0

    def test_message_stop_event_marks_stream_complete(self):
        # LAY-3328 / LAY-3332 ACs literally name ``message_stop`` as a
        # required SSE event type. Verify the aggregator honours it as the
        # lifecycle terminator (the SDK carries no payload on it).
        events = [
            _message_start_event(),
            _content_block_start_event(block_type="text"),
            _content_block_delta_event(delta_type="text_delta", text="done"),
            _message_delta_event(stop_reason="end_turn", output_tokens=4),
            SimpleNamespace(type="message_stop"),
        ]
        msg = _StreamedMessage.from_events(events)
        assert msg.stopped is True
        # And the aggregator still produces the same content + usage shape.
        assert msg.content[0].text == "done"
        assert msg.usage.output_tokens == 4

    def test_no_message_stop_means_stopped_is_false(self):
        # Without an explicit message_stop event, ``stopped`` stays False —
        # callers can distinguish a torn-down iterator from a clean finish.
        events = [
            _message_start_event(),
            _content_block_start_event(block_type="text"),
            _content_block_delta_event(delta_type="text_delta", text="partial"),
            _message_delta_event(stop_reason="end_turn", output_tokens=2),
            # No message_stop sentinel.
        ]
        msg = _StreamedMessage.from_events(events)
        assert msg.stopped is False

    def test_stop_reason_flows_into_finish_reasons_via_extract_meta(self):
        # The OTel mapper unifies OpenAI ``finish_reason`` and Anthropic
        # ``stop_reason`` under ``gen_ai.response.finish_reasons``. The
        # streaming response must surface ``stop_reason`` in meta so the
        # mapper can pick it up.
        events = [
            _message_start_event(),
            _content_block_start_event(block_type="text"),
            _content_block_delta_event(delta_type="text_delta", text="bye"),
            _message_delta_event(stop_reason="max_tokens", output_tokens=4),
        ]
        msg = _StreamedMessage.from_events(events)
        meta = AnthropicProvider.extract_meta(msg)
        assert meta["stop_reason"] == "max_tokens"


# ---------------------------------------------------------------------------
# TTFT + streaming duration -- LAY-3327 / LAY-3329 / LAY-3328 / LAY-3332
# ---------------------------------------------------------------------------
#
# These tests drive a fake stream end-to-end through the wrapped iterator so
# they exercise the actual TTFT capture path rather than the aggregator only.


import time as _time

import pytest

from layerlens.instrument import trace
from tests.instrument.conftest import find_event as _find_event
from layerlens.instrument.adapters.providers.openai import OpenAIProvider as _OP
from layerlens.instrument.adapters.providers.anthropic import AnthropicProvider as _AP


class TestOpenAIStreamingTTFT:
    def test_ttft_and_streaming_duration_in_model_invoke(self, mock_client, capture_trace):
        # Build a generator that delays a measurable amount before the first
        # chunk, then yields the rest immediately. The TTFT capture must
        # reflect the pre-first-chunk delay; total streaming_duration_ms must
        # be >= TTFT.
        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8)

        def fake_stream():
            _time.sleep(0.03)  # ~30ms before first chunk
            yield _openai_chunk(role="assistant", content="hi", model="gpt-4o", response_id="chatcmpl-1")
            yield _openai_chunk(content=" there", usage=usage, finish_reason="stop")

        openai_client = SimpleNamespace()
        openai_client.chat = SimpleNamespace()
        openai_client.chat.completions = SimpleNamespace(create=lambda **kwargs: fake_stream())

        provider = _OP()
        provider.connect(openai_client)

        @trace(mock_client)
        def my_agent():
            stream = openai_client.chat.completions.create(model="gpt-4o", messages=[], stream=True)
            # Drain without returning chunks — ``@trace`` emits the return
            # value, and our SimpleNamespace shims aren't JSON-serializable.
            for _ in stream:
                pass
            return "done"

        my_agent()
        events = capture_trace["events"]
        model_invoke = _find_event(events, "model.invoke")
        payload = model_invoke["payload"]

        assert "ttft_ms" in payload, "TTFT missing from model.invoke per LAY-3329 AC"
        assert "streaming_duration_ms" in payload, "streaming_duration_ms missing per LAY-3329 AC"
        assert payload["ttft_ms"] >= 20.0  # at least the ~30ms sleep minus jitter
        assert payload["streaming_duration_ms"] >= payload["ttft_ms"]
        assert payload["streaming_duration_ms"] == pytest.approx(payload["latency_ms"], abs=1e-6)

    def test_iterator_contract_preserved(self, mock_client, capture_trace):
        # LAY-3329 DoD: "Iterator contract preserved — downstream consumers
        # see identical chunks". The wrapper must yield exactly what the
        # underlying stream yielded.
        chunks_yielded = [
            _openai_chunk(role="assistant", content="a"),
            _openai_chunk(content="b"),
            _openai_chunk(content="c", finish_reason="stop"),
        ]

        openai_client = SimpleNamespace()
        openai_client.chat = SimpleNamespace()
        openai_client.chat.completions = SimpleNamespace(create=lambda **kwargs: iter(chunks_yielded))

        provider = _OP()
        provider.connect(openai_client)

        observed: List[Any] = []

        @trace(mock_client)
        def my_agent():
            stream = openai_client.chat.completions.create(model="gpt-4o", messages=[], stream=True)
            for c in stream:
                observed.append(c)
            return "done"

        my_agent()
        # The wrapper yielded back exactly the same chunk objects, in order.
        assert len(observed) == len(chunks_yielded)
        for got, want in zip(observed, chunks_yielded):
            assert got is want


class TestNoToolCallsNoEvents:
    """LAY-3331 DoD: "Test no tool_calls in response (no events emitted)".

    A response without ``tool_calls`` must NOT produce any ``tool.call`` events.
    """

    def test_no_tool_calls_emits_no_tool_call_events(self, mock_client, capture_trace):
        from openai.types import CompletionUsage
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice

        openai_client = SimpleNamespace()
        openai_client.chat = SimpleNamespace()
        openai_client.chat.completions = SimpleNamespace(
            create=lambda **kwargs: ChatCompletion(
                id="chatcmpl-no-tools",
                model="gpt-4o",
                object="chat.completion",
                created=1700000000,
                choices=[
                    Choice(
                        index=0,
                        finish_reason="stop",
                        message=ChatCompletionMessage(role="assistant", content="plain answer"),
                        # message.tool_calls is None / omitted.
                    )
                ],
                usage=CompletionUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
            )
        )

        provider = _OP()
        provider.connect(openai_client)

        @trace(mock_client)
        def my_agent():
            openai_client.chat.completions.create(model="gpt-4o", messages=[])
            return "done"

        my_agent()
        events = capture_trace["events"]
        tool_events = [e for e in events if e["event_type"] == "tool.call"]
        assert tool_events == [], (
            f"Expected zero tool.call events when response has no tool_calls; got {len(tool_events)}: {tool_events}"
        )
        # And model.invoke + cost.record still fire normally.
        kinds = {e["event_type"] for e in events}
        assert "model.invoke" in kinds
        assert "cost.record" in kinds


class TestMalformedToolCallJSON:
    """LAY-3331 DoD: malformed tool-call arguments JSON logs at WARNING
    (does not crash, does not silently swallow)."""

    def test_malformed_args_logged_at_warning(self, caplog):
        import logging

        chunk = _openai_chunk(
            tool_calls=[_openai_tool_call_fragment(index=0, id="call_x", name="get_x", arguments="{not valid json")],
            finish_reason="tool_calls",
        )
        agg = _StreamedChatResponse.from_chunks([chunk])
        with caplog.at_level(logging.WARNING, logger="layerlens.instrument.adapters.providers.openai"):
            tool_calls = _OP.extract_tool_calls(agg)
        # Raw string returned on parse failure — caller still gets *something*.
        assert tool_calls[0]["arguments"] == "{not valid json"
        # And we logged a WARNING containing the raw arguments snippet.
        assert any("malformed tool_call JSON" in r.message and r.levelname == "WARNING" for r in caplog.records), (
            f"Expected WARNING log; got: {[(r.levelname, r.message) for r in caplog.records]}"
        )

    def test_valid_args_does_not_log(self, caplog):
        import logging

        chunk = _openai_chunk(
            tool_calls=[_openai_tool_call_fragment(index=0, id="call_x", name="get_x", arguments='{"x": 1}')],
            finish_reason="tool_calls",
        )
        agg = _StreamedChatResponse.from_chunks([chunk])
        with caplog.at_level(logging.WARNING, logger="layerlens.instrument.adapters.providers.openai"):
            tool_calls = _OP.extract_tool_calls(agg)
        assert tool_calls[0]["arguments"] == {"x": 1}
        # No warnings emitted for valid JSON.
        assert not any("malformed tool_call JSON" in r.message for r in caplog.records)


class TestPartialEventOnMidStreamError:
    """LAY-3329 / LAY-3332 DoD: when a stream raises mid-iteration, the
    ``agent.error`` event must carry whatever was accumulated so far so
    consumers can reason about partial completion."""

    def test_openai_mid_stream_exception_carries_partial_meta(self, mock_client, capture_trace):
        # Yield two chunks (usage in second), then raise. The error event
        # should have partial_chunks=2 and partial_meta surfacing the usage
        # we managed to extract.
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14)

        def fake_stream():
            yield _openai_chunk(role="assistant", content="part", model="gpt-4o", response_id="chatcmpl-p")
            yield _openai_chunk(content="ial", usage=usage)
            raise RuntimeError("upstream blew up mid-stream")

        openai_client = SimpleNamespace()
        openai_client.chat = SimpleNamespace()
        openai_client.chat.completions = SimpleNamespace(create=lambda **kwargs: fake_stream())

        provider = _OP()
        provider.connect(openai_client)

        @trace(mock_client)
        def my_agent():
            try:
                stream = openai_client.chat.completions.create(model="gpt-4o", messages=[], stream=True)
                for _ in stream:
                    pass
            except RuntimeError:
                pass
            return "recovered"

        my_agent()
        events = capture_trace["events"]
        err = _find_event(events, "agent.error")
        payload = err["payload"]

        assert payload["error"] == "upstream blew up mid-stream"
        assert payload["error_type"] == "RuntimeError"
        assert payload["partial_chunks"] == 2
        assert "partial_meta" in payload
        # Whatever was extractable should be present.
        partial = payload["partial_meta"]
        assert partial["response_id"] == "chatcmpl-p"
        assert partial["usage"]["prompt_tokens"] == 10
        assert partial["usage"]["completion_tokens"] == 4

    def test_openai_error_before_any_chunk_has_no_partial_meta(self, mock_client, capture_trace):
        # Raise immediately at iteration start — no chunks accumulated.
        def fake_stream():
            raise RuntimeError("immediate failure")
            yield  # unreachable

        openai_client = SimpleNamespace()
        openai_client.chat = SimpleNamespace()
        openai_client.chat.completions = SimpleNamespace(create=lambda **kwargs: fake_stream())

        provider = _OP()
        provider.connect(openai_client)

        @trace(mock_client)
        def my_agent():
            try:
                stream = openai_client.chat.completions.create(model="gpt-4o", messages=[], stream=True)
                for _ in stream:
                    pass
            except RuntimeError:
                pass
            return "recovered"

        my_agent()
        err = _find_event(capture_trace["events"], "agent.error")
        # Zero-chunks case must still emit; just no partial_meta.
        assert err["payload"]["partial_chunks"] == 0
        assert "partial_meta" not in err["payload"]


class TestOpenAIAsyncStreamingTTFT:
    """LAY-3329 DoD: "Both sync and async streaming paths work".

    The sync TTFT path is covered in :class:`TestOpenAIStreamingTTFT`. This
    class drives the async wrapper (``_wrap_async_stream_iterator``) end-to-end.
    """

    @pytest.mark.asyncio
    async def test_async_stream_ttft_and_duration_captured(self, mock_client, capture_trace):
        import asyncio

        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8)

        async def fake_async_stream():
            await asyncio.sleep(0.03)  # ~30ms before first chunk
            yield _openai_chunk(role="assistant", content="hi", model="gpt-4o", response_id="chatcmpl-async")
            yield _openai_chunk(content=" there", usage=usage, finish_reason="stop")

        async def fake_create(**kwargs):
            return fake_async_stream()

        openai_client = SimpleNamespace()
        openai_client.chat = SimpleNamespace()
        openai_client.chat.completions = SimpleNamespace()
        openai_client.chat.completions.create = fake_create
        # The wrapper detects async by checking the bound `acreate` attribute;
        # provide it pointing at the same function for parity with real SDKs.
        openai_client.chat.completions.acreate = fake_create

        provider = _OP()
        provider.connect(openai_client)

        @trace(mock_client)
        async def my_agent():
            stream = await openai_client.chat.completions.acreate(model="gpt-4o", messages=[], stream=True)
            async for _ in stream:
                pass
            return "done"

        await my_agent()
        events = capture_trace["events"]
        model_invoke = _find_event(events, "model.invoke")
        payload = model_invoke["payload"]
        assert "ttft_ms" in payload
        assert "streaming_duration_ms" in payload
        assert payload["ttft_ms"] >= 20.0


class TestAnthropicMidStreamError:
    """LAY-3332 DoD: "Test mid-stream error handling".

    When the Anthropic stream raises mid-iteration, ``agent.error`` must fire
    with partial-meta surfacing whatever state was accumulated.
    """

    def test_anthropic_mid_stream_exception_carries_partial_meta(self, mock_client, capture_trace):
        # Build an event stream that yields a message_start + a content block
        # delta and then explodes.
        good_events = [
            _message_start_event(input_tokens=15, cache_read_input_tokens=4),
            _content_block_start_event(block_type="text"),
            _content_block_delta_event(delta_type="text_delta", text="partial"),
        ]

        class _FakeMessagesStream:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                # Propagate the exception (don't swallow); the wrapper's
                # __exit__ runs _emit which captures the error.
                return False

            def __iter__(self):
                for e in good_events:
                    yield e
                raise RuntimeError("anthropic upstream blew up mid-stream")

        fake_messages = SimpleNamespace()
        fake_messages.stream = lambda **kwargs: _FakeMessagesStream()
        fake_messages.create = lambda **kwargs: None
        anthropic_client = SimpleNamespace(messages=fake_messages)

        provider = _AP()
        provider.connect(anthropic_client)

        @trace(mock_client)
        def my_agent():
            try:
                with anthropic_client.messages.stream(model="claude-3-7-sonnet-20250219", messages=[]) as s:
                    for _ in s:
                        pass
            except RuntimeError:
                pass
            return "recovered"

        my_agent()
        events = capture_trace["events"]
        err = _find_event(events, "agent.error")
        payload = err["payload"]

        assert payload["error"] == "anthropic upstream blew up mid-stream"
        assert payload["error_type"] == "RuntimeError"
        assert payload["partial_chunks"] == 3  # message_start + content_block_start + content_block_delta
        # Partial meta should expose what we managed to extract before the
        # raise — at minimum, the id and the partial usage with cache tokens.
        assert "partial_meta" in payload
        partial = payload["partial_meta"]
        assert partial["response_id"] == "msg_abc"
        assert partial["usage"]["input_tokens"] == 15
        assert partial["usage"]["cache_read_input_tokens"] == 4


class TestAnthropicStreamingTTFT:
    def test_ttft_anchored_on_first_content_block_delta(self, mock_client, capture_trace):
        # Build an event stream where ``message_start`` and
        # ``content_block_start`` fire immediately but the first
        # ``content_block_delta`` is delayed. TTFT must reflect the delay,
        # not the start of streaming overall — that is what
        # "time-to-first-token" means.
        events = [
            _message_start_event(input_tokens=8),
            _content_block_start_event(block_type="text"),
        ]

        def delayed_delta():
            _time.sleep(0.03)
            return _content_block_delta_event(delta_type="text_delta", text="hello")

        # Build the stream lazily so the sleep happens during iteration.
        def event_stream():
            for e in events:
                yield e
            yield delayed_delta()
            yield _message_delta_event(stop_reason="end_turn", output_tokens=4)

        class _FakeMessagesStream:
            def __init__(self, gen):
                self._gen = gen

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def __iter__(self):
                return self._gen

        fake_messages = SimpleNamespace()
        fake_messages.stream = lambda **kwargs: _FakeMessagesStream(event_stream())
        # Anthropic adapter also patches `create`; provide a no-op so connect doesn't crash.
        fake_messages.create = lambda **kwargs: None
        anthropic_client = SimpleNamespace(messages=fake_messages)

        provider = _AP()
        provider.connect(anthropic_client)

        @trace(mock_client)
        def my_agent():
            with anthropic_client.messages.stream(model="claude-3-7-sonnet-20250219", messages=[]) as s:
                for _ in s:
                    pass

        my_agent()
        events_out = capture_trace["events"]
        model_invoke = _find_event(events_out, "model.invoke")
        payload = model_invoke["payload"]

        assert "ttft_ms" in payload, "TTFT missing from Anthropic model.invoke per LAY-3332 AC"
        assert "streaming_duration_ms" in payload
        assert payload["ttft_ms"] >= 20.0  # the ~30ms sleep before first delta
        # And TTFT < streaming_duration (delta wasn't the last event).
        assert payload["ttft_ms"] <= payload["streaming_duration_ms"]
