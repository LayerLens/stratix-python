"""Async-client provider coverage (LAY-3575 / T1, fixes N5).

Modern ``AsyncOpenAI``/``AsyncAnthropic`` clients (openai/anthropic >= 1.x)
expose a coroutine ``create`` on the SAME attribute as the sync clients —
there is no ``acreate``. Patching must detect coroutine functions and route
them through the async wrapper; wrapping them with the sync wrapper measures
a coroutine construction (~0 ms), runs the extractors on the un-awaited
coroutine, and hands async streams to the sync iterator.
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any, Optional

from layerlens.instrument import trace
from layerlens.instrument.adapters.providers.ollama import OllamaProvider
from layerlens.instrument.adapters.providers.openai import OpenAIProvider
from layerlens.instrument.adapters.providers.anthropic import AnthropicProvider

from .conftest import make_openai_response, make_anthropic_response
from ...conftest import find_event
from .test_streaming import _openai_chunk


def _async_openai_client(
    response: Any = None,
    delay: float = 0.0,
    error: Optional[Exception] = None,
    stream_factory: Any = None,
) -> SimpleNamespace:
    async def create(**kwargs: Any) -> Any:
        if delay:
            await asyncio.sleep(delay)
        if error is not None:
            raise error
        if kwargs.get("stream") is True and stream_factory is not None:
            return stream_factory()
        return response

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _async_anthropic_client(response: Any = None, delay: float = 0.0) -> SimpleNamespace:
    async def create(**kwargs: Any) -> Any:
        if delay:
            await asyncio.sleep(delay)
        return response

    return SimpleNamespace(messages=SimpleNamespace(create=create))


# ---------------------------------------------------------------------------
# OpenAI async client
# ---------------------------------------------------------------------------


class TestAsyncOpenAICreate:
    def test_wrapped_create_remains_coroutine_function(self) -> None:
        client = _async_openai_client(make_openai_response())
        OpenAIProvider().connect(client)
        assert inspect.iscoroutinefunction(client.chat.completions.create), (
            "async create was wrapped with the sync wrapper (N5)"
        )

    def test_async_create_emits_real_usage_and_latency(self, mock_client, capture_trace) -> None:
        client = _async_openai_client(make_openai_response(), delay=0.02)
        OpenAIProvider().connect(client)

        @trace(mock_client)
        async def my_agent() -> str:
            r = await client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": "Hi"}])
            return r.choices[0].message.content

        asyncio.run(my_agent())

        model_invoke = find_event(capture_trace["events"], "model.invoke")
        payload = model_invoke["payload"]
        assert payload["model"] == "gpt-4"
        assert payload["output_message"]["content"] == "Hello!"
        assert payload["usage"]["prompt_tokens"] == 10
        assert payload["usage"]["completion_tokens"] == 5
        # The sync mis-wrap measures coroutine construction: ~0 ms.
        assert payload["latency_ms"] >= 15

    def test_async_create_passthrough_outside_trace(self) -> None:
        client = _async_openai_client(make_openai_response())
        OpenAIProvider().connect(client)

        async def run() -> Any:
            return await client.chat.completions.create(model="gpt-4", messages=[])

        response = asyncio.run(run())
        assert response.choices[0].message.content == "Hello!"

    def test_async_create_error_emits_agent_error(self, mock_client, capture_trace) -> None:
        client = _async_openai_client(error=RuntimeError("API error"))
        OpenAIProvider().connect(client)

        @trace(mock_client)
        async def my_agent() -> str:
            try:
                await client.chat.completions.create(model="gpt-4", messages=[])
            except RuntimeError:
                pass
            return "recovered"

        asyncio.run(my_agent())

        error = find_event(capture_trace["events"], "agent.error")
        assert error["payload"]["error"] == "API error"
        assert "latency_ms" in error["payload"]

    def test_async_streaming_chunks_flow_through(self, mock_client, capture_trace) -> None:
        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8)

        def make_stream() -> Any:
            async def gen() -> Any:
                yield _openai_chunk(role="assistant", content="hi", model="gpt-4o", response_id="c1")
                yield _openai_chunk(content=" there", usage=usage, finish_reason="stop")

            return gen()

        client = _async_openai_client(stream_factory=make_stream)
        OpenAIProvider().connect(client)

        received: list = []

        @trace(mock_client)
        async def my_agent() -> str:
            stream = await client.chat.completions.create(model="gpt-4o", messages=[], stream=True)
            async for chunk in stream:
                received.append(chunk)
            return "done"

        asyncio.run(my_agent())

        assert len(received) == 2, "async stream chunks must pass through unchanged"
        model_invoke = find_event(capture_trace["events"], "model.invoke")
        assert model_invoke["payload"]["usage"]["total_tokens"] == 8
        assert "ttft_ms" in model_invoke["payload"]


# ---------------------------------------------------------------------------
# Anthropic async client
# ---------------------------------------------------------------------------


class TestAsyncAnthropicMessages:
    def test_wrapped_create_remains_coroutine_function(self) -> None:
        client = _async_anthropic_client(make_anthropic_response())
        AnthropicProvider().connect(client)
        assert inspect.iscoroutinefunction(client.messages.create), (
            "async messages.create was wrapped with the sync wrapper (N5)"
        )

    def test_async_create_emits_real_usage(self, mock_client, capture_trace) -> None:
        client = _async_anthropic_client(make_anthropic_response(), delay=0.02)
        AnthropicProvider().connect(client)

        @trace(mock_client)
        async def my_agent() -> Any:
            return await client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=100,
                messages=[{"role": "user", "content": "Hi"}],
            )

        asyncio.run(my_agent())

        model_invoke = find_event(capture_trace["events"], "model.invoke")
        payload = model_invoke["payload"]
        assert payload["usage"]["prompt_tokens"] == 20
        assert payload["usage"]["completion_tokens"] == 10
        assert payload["latency_ms"] >= 15


class TestAsyncAnthropicMessagesStream:
    def test_async_stream_context_manager_taps_events(self, mock_client, capture_trace) -> None:
        """``messages.stream`` returns a context manager (not a coroutine);
        the traced proxy must support ``async with`` + ``async for`` and emit
        the aggregated events on close (pins the existing __aenter__/__aiter__/
        __aexit__ support against regressions)."""

        events = [
            SimpleNamespace(type="message_start"),
            SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="hi")),
        ]

        class _FakeAsyncStream:
            def __init__(self, items: Any) -> None:
                self._items = items

            def __aiter__(self) -> Any:
                async def gen() -> Any:
                    for item in self._items:
                        yield item

                return gen()

        class _FakeAsyncManager:
            async def __aenter__(self) -> Any:
                return _FakeAsyncStream(events)

            async def __aexit__(self, *exc: Any) -> bool:
                return False

        def stream(**kwargs: Any) -> Any:
            return _FakeAsyncManager()

        async def create(**kwargs: Any) -> Any:  # presence marks the client async
            return make_anthropic_response()

        client = SimpleNamespace(messages=SimpleNamespace(create=create, stream=stream))
        AnthropicProvider().connect(client)

        received: list = []

        @trace(mock_client)
        async def my_agent() -> str:
            async with client.messages.stream(model="claude-3-opus-20240229", max_tokens=10, messages=[]) as s:
                async for ev in s:
                    received.append(ev)
            return "done"

        asyncio.run(my_agent())

        assert len(received) == 2, "async stream events must pass through unchanged"
        model_invoke = find_event(capture_trace["events"], "model.invoke")
        assert model_invoke["payload"]["name"] == "anthropic.messages.stream"


# ---------------------------------------------------------------------------
# Ollama async client
# ---------------------------------------------------------------------------


class TestAsyncOllama:
    def test_wrapped_chat_remains_coroutine_function(self) -> None:
        async def chat(**kwargs: Any) -> Any:
            return {"message": {"role": "assistant", "content": "hi"}}

        client = SimpleNamespace(chat=chat)
        OllamaProvider().connect(client)
        assert inspect.iscoroutinefunction(client.chat), (
            "ollama AsyncClient.chat was wrapped with the sync wrapper (N5)"
        )


# ---------------------------------------------------------------------------
# Legacy acreate clients (regression pin — must stay async-wrapped)
# ---------------------------------------------------------------------------


class TestLegacyAcreate:
    def test_legacy_acreate_still_wrapped_async(self, mock_client, capture_trace) -> None:
        response = make_openai_response()

        def create(**kwargs: Any) -> Any:
            return response

        async def acreate(**kwargs: Any) -> Any:
            return response

        completions = SimpleNamespace(create=create, acreate=acreate)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        OpenAIProvider().connect(client)

        assert inspect.iscoroutinefunction(client.chat.completions.acreate)
        assert not inspect.iscoroutinefunction(client.chat.completions.create)

        @trace(mock_client)
        async def my_agent() -> Any:
            return await client.chat.completions.acreate(model="gpt-4", messages=[])

        asyncio.run(my_agent())
        assert find_event(capture_trace["events"], "model.invoke") is not None


class TestDecoratedAsyncClients:
    """The real openai/anthropic SDKs wrap their async ``create`` methods in
    plain-function decorators (e.g. openai's ``required_args``), so the
    outermost callable is NOT a coroutine function — routing must look
    through ``__wrapped__`` chains exactly like the real clients require
    (found live: AsyncOpenAI got the sync wrapper and fabricated ~0ms
    latencies despite the patch-time routing)."""

    @staticmethod
    def _decorated_async(fn: Any) -> Any:
        import functools

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)  # returns the coroutine, like required_args

        return wrapper

    def test_decorator_wrapped_async_create_routes_async(self) -> None:
        async def create(**kwargs: Any) -> Any:
            return make_openai_response()

        completions = SimpleNamespace(create=self._decorated_async(create))
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        OpenAIProvider().connect(client)
        assert inspect.iscoroutinefunction(inspect.unwrap(client.chat.completions.create)), (
            "decorator-wrapped async create was routed to the sync wrapper (N5 on real SDK clients)"
        )

    def test_decorator_wrapped_async_create_emits_real_latency(self, mock_client, capture_trace) -> None:
        async def create(**kwargs: Any) -> Any:
            await asyncio.sleep(0.02)
            return make_openai_response()

        completions = SimpleNamespace(create=self._decorated_async(create))
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        OpenAIProvider().connect(client)

        @trace(mock_client)
        async def my_agent() -> Any:
            return await client.chat.completions.create(model="gpt-4", messages=[])

        asyncio.run(my_agent())
        model_invoke = find_event(capture_trace["events"], "model.invoke")
        assert model_invoke["payload"]["latency_ms"] >= 15
        assert model_invoke["payload"]["usage"]["prompt_tokens"] == 10
