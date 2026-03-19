"""Tests for STRATIX LangGraph LLM Wrapper."""

import pytest

from layerlens.instrument.adapters.langgraph.llm import (
    TracedLLM,
    wrap_llm_for_langgraph,
    LLMCallNode,
    LLMInvocation,
)


class MockStratix:
    """Mock STRATIX instance for testing."""

    def __init__(self):
        self.events = []

    def emit(self, event_type: str, payload: dict):
        self.events.append({"type": event_type, "payload": payload})

    def get_events(self, event_type: str = None):
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


class MockLLM:
    """Mock LLM for testing."""

    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
        self._invocations = []

    def invoke(self, messages, **kwargs):
        self._invocations.append({"messages": messages, "kwargs": kwargs})
        return MockResponse("Mock LLM response")

    async def ainvoke(self, messages, **kwargs):
        self._invocations.append({"messages": messages, "kwargs": kwargs})
        return MockResponse("Async mock response")

    def stream(self, messages, **kwargs):
        self._invocations.append({"messages": messages, "kwargs": kwargs})
        yield MockChunk("Hello")
        yield MockChunk(" World")


class MockResponse:
    """Mock LLM response."""

    def __init__(self, content: str):
        self.content = content
        self.type = "ai"
        self.response_metadata = {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class MockChunk:
    """Mock streaming chunk."""

    def __init__(self, content: str):
        self.content = content


class TestTracedLLM:
    """Tests for TracedLLM."""

    def test_initialization(self):
        """Test TracedLLM initialization."""
        llm = MockLLM()
        traced = TracedLLM(llm)

        assert traced._llm is llm
        assert traced._model_name == "mock-model"

    def test_initialization_with_stratix(self):
        """Test initialization with STRATIX instance."""
        stratix = MockStratix()
        llm = MockLLM()
        traced = TracedLLM(llm, stratix_instance=stratix)

        assert traced._stratix is stratix

    def test_invoke_calls_underlying_llm(self):
        """Test invoke calls the underlying LLM."""
        llm = MockLLM()
        traced = TracedLLM(llm)

        response = traced.invoke([{"role": "user", "content": "Hello"}])

        assert response.content == "Mock LLM response"
        assert len(llm._invocations) == 1

    def test_invoke_emits_model_invoke_event(self):
        """Test invoke emits model.invoke event."""
        stratix = MockStratix()
        llm = MockLLM()
        traced = TracedLLM(llm, stratix_instance=stratix)

        traced.invoke([{"role": "user", "content": "Hello"}])

        model_events = stratix.get_events("model.invoke")
        assert len(model_events) == 1
        assert model_events[0]["payload"]["model"] == "mock-model"

    def test_invoke_captures_input_messages(self):
        """Test invoke captures input messages."""
        stratix = MockStratix()
        llm = MockLLM()
        traced = TracedLLM(llm, stratix_instance=stratix)

        traced.invoke([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ])

        model_events = stratix.get_events("model.invoke")
        input_msgs = model_events[0]["payload"]["input_messages"]
        assert len(input_msgs) == 2

    def test_invoke_captures_output(self):
        """Test invoke captures output."""
        stratix = MockStratix()
        llm = MockLLM()
        traced = TracedLLM(llm, stratix_instance=stratix)

        traced.invoke("Hello")

        model_events = stratix.get_events("model.invoke")
        assert model_events[0]["payload"]["output_message"]["content"] == "Mock LLM response"

    def test_invoke_captures_token_usage(self):
        """Test invoke captures token usage."""
        stratix = MockStratix()
        llm = MockLLM()
        traced = TracedLLM(llm, stratix_instance=stratix)

        traced.invoke("Hello")

        model_events = stratix.get_events("model.invoke")
        token_usage = model_events[0]["payload"]["token_usage"]
        assert token_usage["prompt_tokens"] == 10
        assert token_usage["completion_tokens"] == 5

    def test_invoke_handles_exception(self):
        """Test invoke handles LLM exceptions."""
        stratix = MockStratix()

        class FailingLLM:
            model_name = "failing"

            def invoke(self, messages, **kwargs):
                raise ValueError("LLM error")

        traced = TracedLLM(FailingLLM(), stratix_instance=stratix)

        with pytest.raises(ValueError, match="LLM error"):
            traced.invoke("Hello")

        model_events = stratix.get_events("model.invoke")
        assert model_events[0]["payload"]["error"] == "LLM error"

    def test_invoke_records_duration(self):
        """Test invoke records duration."""
        stratix = MockStratix()
        llm = MockLLM()
        traced = TracedLLM(llm, stratix_instance=stratix)

        traced.invoke("Hello")

        model_events = stratix.get_events("model.invoke")
        assert isinstance(model_events[0]["payload"]["duration_ns"], int)

    def test_stream_combines_chunks(self):
        """Test stream combines chunks in event."""
        stratix = MockStratix()
        llm = MockLLM()
        traced = TracedLLM(llm, stratix_instance=stratix)

        chunks = list(traced.stream("Hello"))

        assert len(chunks) == 2
        model_events = stratix.get_events("model.invoke")
        assert model_events[0]["payload"]["output_message"]["content"] == "Hello World"

    def test_custom_model_name(self):
        """Test custom model name override."""
        stratix = MockStratix()
        llm = MockLLM()
        traced = TracedLLM(llm, stratix_instance=stratix, model_name="custom-model")

        traced.invoke("Hello")

        model_events = stratix.get_events("model.invoke")
        assert model_events[0]["payload"]["model"] == "custom-model"

    def test_custom_provider(self):
        """Test custom provider override."""
        stratix = MockStratix()
        llm = MockLLM()
        traced = TracedLLM(llm, stratix_instance=stratix, provider="custom-provider")

        traced.invoke("Hello")

        model_events = stratix.get_events("model.invoke")
        assert model_events[0]["payload"]["provider"] == "custom-provider"

    def test_attribute_proxying(self):
        """Test attribute access is proxied to underlying LLM."""
        llm = MockLLM()
        llm.custom_attr = "custom_value"
        traced = TracedLLM(llm)

        assert traced.custom_attr == "custom_value"

    def test_provider_detection_openai(self):
        """Test OpenAI provider detection."""

        class ChatOpenAI:
            model_name = "gpt-4"

        traced = TracedLLM(ChatOpenAI())
        assert traced._provider == "openai"

    def test_provider_detection_anthropic(self):
        """Test Anthropic provider detection."""

        class ChatAnthropic:
            model_name = "claude-3"

        traced = TracedLLM(ChatAnthropic())
        assert traced._provider == "anthropic"


class TestWrapLLMForLanggraph:
    """Tests for wrap_llm_for_langgraph function."""

    def test_creates_traced_llm(self):
        """Test creates TracedLLM instance."""
        llm = MockLLM()
        traced = wrap_llm_for_langgraph(llm)

        assert isinstance(traced, TracedLLM)

    def test_passes_stratix_instance(self):
        """Test passes STRATIX instance to TracedLLM."""
        stratix = MockStratix()
        llm = MockLLM()
        traced = wrap_llm_for_langgraph(llm, stratix_instance=stratix)

        assert traced._stratix is stratix

    def test_passes_custom_model_name(self):
        """Test passes custom model name."""
        llm = MockLLM()
        traced = wrap_llm_for_langgraph(llm, model_name="custom")

        assert traced._model_name == "custom"


class TestLLMCallNode:
    """Tests for LLMCallNode."""

    def test_node_initialization(self):
        """Test node initializes correctly."""
        llm = MockLLM()
        node = LLMCallNode(llm)

        assert node._messages_key == "messages"
        assert node._response_key == "messages"

    def test_node_callable(self):
        """Test node is callable."""
        llm = MockLLM()
        node = LLMCallNode(llm)

        result = node({"messages": [{"role": "user", "content": "Hello"}]})

        assert "messages" in result

    def test_node_emits_model_invoke(self):
        """Test node emits model.invoke event."""
        stratix = MockStratix()
        llm = MockLLM()
        node = LLMCallNode(llm, stratix_instance=stratix)

        node({"messages": [{"role": "user", "content": "Hello"}]})

        model_events = stratix.get_events("model.invoke")
        assert len(model_events) == 1

    def test_node_custom_keys(self):
        """Test node with custom state keys."""
        llm = MockLLM()
        node = LLMCallNode(
            llm,
            messages_key="chat_history",
            response_key="response",
        )

        result = node({"chat_history": [{"role": "user", "content": "Hi"}]})

        assert "response" in result


@pytest.mark.asyncio
class TestTracedLLMAsync:
    """Async tests for TracedLLM."""

    async def test_ainvoke_calls_underlying_llm(self):
        """Test ainvoke calls underlying LLM."""
        llm = MockLLM()
        traced = TracedLLM(llm)

        response = await traced.ainvoke("Hello")

        assert response.content == "Async mock response"

    async def test_ainvoke_emits_event(self):
        """Test ainvoke emits model.invoke event."""
        stratix = MockStratix()
        llm = MockLLM()
        traced = TracedLLM(llm, stratix_instance=stratix)

        await traced.ainvoke("Hello")

        model_events = stratix.get_events("model.invoke")
        assert len(model_events) == 1

    async def test_ainvoke_handles_exception(self):
        """Test ainvoke handles exceptions."""
        stratix = MockStratix()

        class FailingAsyncLLM:
            model_name = "failing"

            async def ainvoke(self, messages, **kwargs):
                raise ValueError("Async error")

        traced = TracedLLM(FailingAsyncLLM(), stratix_instance=stratix)

        with pytest.raises(ValueError, match="Async error"):
            await traced.ainvoke("Hello")
