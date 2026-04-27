"""Tests for the LangGraph LLM wrapper.

Ported from ``ateam/tests/adapters/langgraph/test_llm.py``.

All public symbols (``LLMCallNode``, ``TracedLLM``,
``wrap_llm_for_langgraph``) exist in stratix-python under
``layerlens.instrument.adapters.frameworks.langgraph.llm``.
"""

from __future__ import annotations

import asyncio
from typing import Any
from collections.abc import Iterator

import pytest

from layerlens.instrument.adapters.frameworks.langgraph.llm import (
    TracedLLM,
    LLMCallNode,
    wrap_llm_for_langgraph,
)


class _MockStratix:
    """Mock STRATIX instance for testing."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append({"type": event_type, "payload": payload})

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


class _MockResponse:
    """Mock LLM response."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.type = "ai"
        self.response_metadata: dict[str, Any] = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }


class _MockChunk:
    """Mock streaming chunk."""

    def __init__(self, content: str) -> None:
        self.content = content


class _MockLLM:
    """Mock LLM for testing."""

    def __init__(self, model_name: str = "mock-model") -> None:
        self.model_name = model_name
        self._invocations: list[dict[str, Any]] = []

    def invoke(self, messages: Any, **kwargs: Any) -> _MockResponse:
        self._invocations.append({"messages": messages, "kwargs": kwargs})
        return _MockResponse("Mock LLM response")

    async def ainvoke(self, messages: Any, **kwargs: Any) -> _MockResponse:
        self._invocations.append({"messages": messages, "kwargs": kwargs})
        return _MockResponse("Async mock response")

    def stream(self, messages: Any, **kwargs: Any) -> Iterator[_MockChunk]:
        self._invocations.append({"messages": messages, "kwargs": kwargs})
        yield _MockChunk("Hello")
        yield _MockChunk(" World")


class TestTracedLLM:
    """Tests for TracedLLM."""

    def test_initialization(self) -> None:
        """Test TracedLLM initialization."""
        llm = _MockLLM()
        traced = TracedLLM(llm)

        assert traced._llm is llm
        assert traced._model_name == "mock-model"

    def test_initialization_with_stratix(self) -> None:
        """Test initialization with STRATIX instance."""
        stratix = _MockStratix()
        llm = _MockLLM()
        traced = TracedLLM(llm, stratix_instance=stratix)

        assert traced._stratix is stratix

    def test_invoke_calls_underlying_llm(self) -> None:
        """Test invoke calls the underlying LLM."""
        llm = _MockLLM()
        traced = TracedLLM(llm)

        response = traced.invoke([{"role": "user", "content": "Hello"}])

        assert response.content == "Mock LLM response"
        assert len(llm._invocations) == 1

    def test_invoke_emits_model_invoke_event(self) -> None:
        """Test invoke emits model.invoke event."""
        stratix = _MockStratix()
        llm = _MockLLM()
        traced = TracedLLM(llm, stratix_instance=stratix)

        traced.invoke([{"role": "user", "content": "Hello"}])

        model_events = stratix.get_events("model.invoke")
        assert len(model_events) == 1
        assert model_events[0]["payload"]["model"] == "mock-model"

    def test_invoke_captures_input_messages(self) -> None:
        """Test invoke captures input messages."""
        stratix = _MockStratix()
        llm = _MockLLM()
        traced = TracedLLM(llm, stratix_instance=stratix)

        traced.invoke(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
        )

        model_events = stratix.get_events("model.invoke")
        input_msgs = model_events[0]["payload"]["input_messages"]
        assert len(input_msgs) == 2

    def test_invoke_captures_output(self) -> None:
        """Test invoke captures output."""
        stratix = _MockStratix()
        llm = _MockLLM()
        traced = TracedLLM(llm, stratix_instance=stratix)

        traced.invoke("Hello")

        model_events = stratix.get_events("model.invoke")
        assert model_events[0]["payload"]["output_message"]["content"] == "Mock LLM response"

    def test_invoke_captures_token_usage(self) -> None:
        """Test invoke captures token usage."""
        stratix = _MockStratix()
        llm = _MockLLM()
        traced = TracedLLM(llm, stratix_instance=stratix)

        traced.invoke("Hello")

        model_events = stratix.get_events("model.invoke")
        token_usage = model_events[0]["payload"]["token_usage"]
        assert token_usage["prompt_tokens"] == 10
        assert token_usage["completion_tokens"] == 5

    def test_invoke_handles_exception(self) -> None:
        """Test invoke handles LLM exceptions."""
        stratix = _MockStratix()

        class FailingLLM:
            model_name = "failing"

            def invoke(self, messages: Any, **kwargs: Any) -> Any:
                raise ValueError("LLM error")

        traced = TracedLLM(FailingLLM(), stratix_instance=stratix)

        with pytest.raises(ValueError, match="LLM error"):
            traced.invoke("Hello")

        model_events = stratix.get_events("model.invoke")
        assert model_events[0]["payload"]["error"] == "LLM error"

    def test_invoke_records_duration(self) -> None:
        """Test invoke records duration."""
        stratix = _MockStratix()
        llm = _MockLLM()
        traced = TracedLLM(llm, stratix_instance=stratix)

        traced.invoke("Hello")

        model_events = stratix.get_events("model.invoke")
        assert isinstance(model_events[0]["payload"]["duration_ns"], int)

    def test_stream_combines_chunks(self) -> None:
        """Test stream combines chunks in event."""
        stratix = _MockStratix()
        llm = _MockLLM()
        traced = TracedLLM(llm, stratix_instance=stratix)

        chunks = list(traced.stream("Hello"))

        assert len(chunks) == 2
        model_events = stratix.get_events("model.invoke")
        assert model_events[0]["payload"]["output_message"]["content"] == "Hello World"

    def test_custom_model_name(self) -> None:
        """Test custom model name override."""
        stratix = _MockStratix()
        llm = _MockLLM()
        traced = TracedLLM(llm, stratix_instance=stratix, model_name="custom-model")

        traced.invoke("Hello")

        model_events = stratix.get_events("model.invoke")
        assert model_events[0]["payload"]["model"] == "custom-model"

    def test_custom_provider(self) -> None:
        """Test custom provider override."""
        stratix = _MockStratix()
        llm = _MockLLM()
        traced = TracedLLM(llm, stratix_instance=stratix, provider="custom-provider")

        traced.invoke("Hello")

        model_events = stratix.get_events("model.invoke")
        assert model_events[0]["payload"]["provider"] == "custom-provider"

    def test_attribute_proxying(self) -> None:
        """Test attribute access is proxied to underlying LLM."""
        llm = _MockLLM()
        # Dynamically set custom attribute
        llm.custom_attr = "custom_value"  # type: ignore[attr-defined]
        traced = TracedLLM(llm)

        assert traced.custom_attr == "custom_value"

    def test_provider_detection_openai(self) -> None:
        """Test OpenAI provider detection."""

        class ChatOpenAI:
            model_name = "gpt-4"

        traced = TracedLLM(ChatOpenAI())
        assert traced._provider == "openai"

    def test_provider_detection_anthropic(self) -> None:
        """Test Anthropic provider detection."""

        class ChatAnthropic:
            model_name = "claude-3"

        traced = TracedLLM(ChatAnthropic())
        assert traced._provider == "anthropic"


class TestWrapLLMForLanggraph:
    """Tests for wrap_llm_for_langgraph function."""

    def test_creates_traced_llm(self) -> None:
        """Test creates TracedLLM instance."""
        llm = _MockLLM()
        traced = wrap_llm_for_langgraph(llm)

        assert isinstance(traced, TracedLLM)

    def test_passes_stratix_instance(self) -> None:
        """Test passes STRATIX instance to TracedLLM."""
        stratix = _MockStratix()
        llm = _MockLLM()
        traced = wrap_llm_for_langgraph(llm, stratix_instance=stratix)

        assert traced._stratix is stratix

    def test_passes_custom_model_name(self) -> None:
        """Test passes custom model name."""
        llm = _MockLLM()
        traced = wrap_llm_for_langgraph(llm, model_name="custom")

        assert traced._model_name == "custom"


class TestLLMCallNode:
    """Tests for LLMCallNode."""

    def test_node_initialization(self) -> None:
        """Test node initializes correctly."""
        llm = _MockLLM()
        node = LLMCallNode(llm)

        assert node._messages_key == "messages"
        assert node._response_key == "messages"

    def test_node_callable(self) -> None:
        """Test node is callable."""
        llm = _MockLLM()
        node = LLMCallNode(llm)

        result = node({"messages": [{"role": "user", "content": "Hello"}]})

        assert "messages" in result

    def test_node_emits_model_invoke(self) -> None:
        """Test node emits model.invoke event."""
        stratix = _MockStratix()
        llm = _MockLLM()
        node = LLMCallNode(llm, stratix_instance=stratix)

        node({"messages": [{"role": "user", "content": "Hello"}]})

        model_events = stratix.get_events("model.invoke")
        assert len(model_events) == 1

    def test_node_custom_keys(self) -> None:
        """Test node with custom state keys."""
        llm = _MockLLM()
        node = LLMCallNode(
            llm,
            messages_key="chat_history",
            response_key="response",
        )

        result = node({"chat_history": [{"role": "user", "content": "Hi"}]})

        assert "response" in result


class TestTracedLLMAsync:
    """Async tests for TracedLLM.

    Uses ``asyncio.run`` rather than ``@pytest.mark.asyncio`` because
    ``pytest-asyncio`` is not in the project's dev requirements; the
    existing ``test_langgraph.py`` smoke test follows the same pattern.
    """

    def test_ainvoke_calls_underlying_llm(self) -> None:
        """Test ainvoke calls underlying LLM."""
        llm = _MockLLM()
        traced = TracedLLM(llm)

        response = asyncio.run(traced.ainvoke("Hello"))

        assert response.content == "Async mock response"

    def test_ainvoke_emits_event(self) -> None:
        """Test ainvoke emits model.invoke event."""
        stratix = _MockStratix()
        llm = _MockLLM()
        traced = TracedLLM(llm, stratix_instance=stratix)

        asyncio.run(traced.ainvoke("Hello"))

        model_events = stratix.get_events("model.invoke")
        assert len(model_events) == 1

    def test_ainvoke_handles_exception(self) -> None:
        """Test ainvoke handles exceptions."""
        stratix = _MockStratix()

        class FailingAsyncLLM:
            model_name = "failing"

            async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
                raise ValueError("Async error")

        traced = TracedLLM(FailingAsyncLLM(), stratix_instance=stratix)

        async def _run() -> None:
            await traced.ainvoke("Hello")

        with pytest.raises(ValueError, match="Async error"):
            asyncio.run(_run())
