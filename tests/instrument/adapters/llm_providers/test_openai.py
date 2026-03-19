"""Tests for OpenAI LLM Provider Adapter."""

import json
import pytest

from layerlens.instrument.adapters._capture import CaptureConfig
from layerlens.instrument.adapters.llm_providers.openai_adapter import OpenAIAdapter


class MockStratix:
    def __init__(self):
        self.events = []

    def emit(self, event_type, payload):
        self.events.append({"type": event_type, "payload": payload})

    def get_events(self, event_type=None):
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


class MockUsageDetails:
    def __init__(self, cached_tokens=None, reasoning_tokens=None):
        self.cached_tokens = cached_tokens
        self.reasoning_tokens = reasoning_tokens


class MockUsage:
    def __init__(self, prompt=100, completion=50, total=150, cached=None, reasoning=None):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total
        self.prompt_tokens_details = MockUsageDetails(cached_tokens=cached) if cached else None
        self.completion_tokens_details = MockUsageDetails(reasoning_tokens=reasoning) if reasoning else None


class MockFunction:
    def __init__(self, name="get_weather", arguments='{"city": "NYC"}'):
        self.name = name
        self.arguments = arguments


class MockToolCall:
    def __init__(self, name="get_weather", arguments='{"city": "NYC"}', tc_id="tc_1"):
        self.id = tc_id
        self.function = MockFunction(name=name, arguments=arguments)


class MockMessage:
    def __init__(self, content="Hello", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class MockChoice:
    def __init__(self, message=None):
        self.message = message or MockMessage()


class MockResponse:
    def __init__(self, usage=None, choices=None, model="gpt-4o"):
        self.usage = usage or MockUsage()
        self.choices = choices or [MockChoice()]
        self.model = model


class MockEmbeddingUsage:
    def __init__(self, prompt=10, total=10):
        self.prompt_tokens = prompt
        self.completion_tokens = 0
        self.total_tokens = total
        self.prompt_tokens_details = None
        self.completion_tokens_details = None


class MockEmbeddingResponse:
    def __init__(self):
        self.usage = MockEmbeddingUsage()
        self.data = [{"embedding": [0.1, 0.2]}]


class MockClient:
    """Mock OpenAI client."""
    def __init__(self):
        self.chat = MockChat()
        self.embeddings = MockEmbeddings()


class MockCompletions:
    def create(self, *args, **kwargs):
        return MockResponse()


class MockChat:
    def __init__(self):
        self.completions = MockCompletions()


class MockEmbeddings:
    def create(self, *args, **kwargs):
        return MockEmbeddingResponse()


class TestOpenAIAdapter:
    """Tests for OpenAIAdapter."""

    def test_adapter_framework(self):
        adapter = OpenAIAdapter()
        assert adapter.FRAMEWORK == "openai"
        assert adapter.VERSION == "0.1.0"

    def test_connect_and_disconnect(self):
        adapter = OpenAIAdapter()
        adapter.connect()
        assert adapter.is_connected
        adapter.disconnect()
        assert not adapter.is_connected

    def test_connect_client_wraps_methods(self):
        stratix = MockStratix()
        adapter = OpenAIAdapter(stratix=stratix)
        adapter.connect()

        client = MockClient()
        adapter.connect_client(client)

        assert "chat.completions.create" in adapter._originals
        assert "embeddings.create" in adapter._originals

    def test_chat_completion_emits_model_invoke(self):
        stratix = MockStratix()
        adapter = OpenAIAdapter(stratix=stratix)
        adapter.connect()

        client = MockClient()
        adapter.connect_client(client)

        result = client.chat.completions.create(model="gpt-4o", temperature=0.7)

        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["provider"] == "openai"
        assert events[0]["payload"]["model"] == "gpt-4o"
        assert events[0]["payload"]["prompt_tokens"] == 100
        assert events[0]["payload"]["completion_tokens"] == 50

    def test_chat_completion_emits_cost_record(self):
        stratix = MockStratix()
        adapter = OpenAIAdapter(stratix=stratix)
        adapter.connect()

        client = MockClient()
        adapter.connect_client(client)

        client.chat.completions.create(model="gpt-4o")

        events = stratix.get_events("cost.record")
        assert len(events) == 1
        assert "api_cost_usd" in events[0]["payload"]

    def test_embeddings_emits_events(self):
        stratix = MockStratix()
        adapter = OpenAIAdapter(stratix=stratix)
        adapter.connect()

        client = MockClient()
        adapter.connect_client(client)

        client.embeddings.create(model="text-embedding-3-small", input=["test"])

        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["request_type"] == "embedding"

    def test_function_calling_emits_tool_call(self):
        stratix = MockStratix()
        adapter = OpenAIAdapter(stratix=stratix)
        adapter.connect()

        # Override create to return tool calls
        def create_with_tools(*args, **kwargs):
            msg = MockMessage(tool_calls=[MockToolCall()])
            return MockResponse(choices=[MockChoice(message=msg)])

        client = MockClient()
        adapter.connect_client(client)
        client.chat.completions.create = adapter._wrap_chat_create(create_with_tools)

        client.chat.completions.create(model="gpt-4o")

        tool_events = stratix.get_events("tool.call")
        assert len(tool_events) == 1
        assert tool_events[0]["payload"]["tool_name"] == "get_weather"

    def test_error_propagation(self):
        stratix = MockStratix()
        adapter = OpenAIAdapter(stratix=stratix)
        adapter.connect()

        def failing_create(*args, **kwargs):
            raise ValueError("API error")

        client = MockClient()
        adapter.connect_client(client)
        client.chat.completions.create = adapter._wrap_chat_create(failing_create)

        with pytest.raises(ValueError, match="API error"):
            client.chat.completions.create(model="gpt-4o")

        # Error events should still be emitted
        invoke_events = stratix.get_events("model.invoke")
        assert len(invoke_events) == 1
        assert invoke_events[0]["payload"]["error"] == "API error"

    def test_error_emits_policy_violation(self):
        stratix = MockStratix()
        adapter = OpenAIAdapter(stratix=stratix)
        adapter.connect()

        def failing_create(*args, **kwargs):
            raise ValueError("rate limit")

        client = MockClient()
        adapter.connect_client(client)
        client.chat.completions.create = adapter._wrap_chat_create(failing_create)

        with pytest.raises(ValueError):
            client.chat.completions.create(model="gpt-4o")

        violation_events = stratix.get_events("policy.violation")
        assert len(violation_events) == 1

    def test_adapter_error_does_not_break_call(self):
        """Adapter emit errors should not break the original API call."""
        class FailingSTRATIX:
            def emit(self, *args, **kwargs):
                raise RuntimeError("emit failed")

        adapter = OpenAIAdapter(stratix=FailingSTRATIX())
        adapter.connect()

        client = MockClient()
        adapter.connect_client(client)

        result = client.chat.completions.create(model="gpt-4o")
        assert result is not None  # original still works

    def test_capture_config_minimal_gates_model_invoke(self):
        stratix = MockStratix()
        adapter = OpenAIAdapter(stratix=stratix, capture_config=CaptureConfig.minimal())
        adapter.connect()

        client = MockClient()
        adapter.connect_client(client)

        client.chat.completions.create(model="gpt-4o")

        # model.invoke gated by L3
        assert len(stratix.get_events("model.invoke")) == 0
        # cost.record is cross-cutting, should still emit
        assert len(stratix.get_events("cost.record")) == 1

    def test_cached_tokens_extracted(self):
        stratix = MockStratix()
        adapter = OpenAIAdapter(stratix=stratix)
        adapter.connect()

        def create_with_cache(*args, **kwargs):
            return MockResponse(usage=MockUsage(cached=30))

        client = MockClient()
        adapter.connect_client(client)
        client.chat.completions.create = adapter._wrap_chat_create(create_with_cache)

        client.chat.completions.create(model="gpt-4o")

        events = stratix.get_events("model.invoke")
        assert events[0]["payload"]["cached_tokens"] == 30

    def test_reasoning_tokens_extracted(self):
        stratix = MockStratix()
        adapter = OpenAIAdapter(stratix=stratix)
        adapter.connect()

        def create_with_reasoning(*args, **kwargs):
            return MockResponse(usage=MockUsage(reasoning=200))

        client = MockClient()
        adapter.connect_client(client)
        client.chat.completions.create = adapter._wrap_chat_create(create_with_reasoning)

        client.chat.completions.create(model="o1")

        events = stratix.get_events("model.invoke")
        assert events[0]["payload"]["reasoning_tokens"] == 200

    def test_latency_captured(self):
        stratix = MockStratix()
        adapter = OpenAIAdapter(stratix=stratix)
        adapter.connect()

        client = MockClient()
        adapter.connect_client(client)

        client.chat.completions.create(model="gpt-4o")

        events = stratix.get_events("model.invoke")
        assert "latency_ms" in events[0]["payload"]
        assert events[0]["payload"]["latency_ms"] >= 0

    def test_disconnect_restores_originals(self):
        adapter = OpenAIAdapter()
        adapter.connect()

        client = MockClient()
        adapter.connect_client(client)

        # Should be wrapped now
        assert hasattr(client.chat.completions.create, '_stratix_original')

        adapter.disconnect()
        # Should be restored — no longer a wrapper
        assert not hasattr(client.chat.completions.create, '_stratix_original')

    def test_streaming_wraps_iterator(self):
        stratix = MockStratix()
        adapter = OpenAIAdapter(stratix=stratix)
        adapter.connect()

        class MockDelta:
            def __init__(self, content=None, tool_calls=None):
                self.content = content
                self.tool_calls = tool_calls

        class MockStreamChoice:
            def __init__(self, delta):
                self.delta = delta

        class MockChunk:
            def __init__(self, content=None, usage=None):
                self.choices = [MockStreamChoice(MockDelta(content=content))]
                self.usage = usage

        chunks = [MockChunk("Hello"), MockChunk(" world")]

        def create_streaming(*args, **kwargs):
            if kwargs.get("stream"):
                return iter(chunks)
            return MockResponse()

        client = MockClient()
        adapter.connect_client(client)
        client.chat.completions.create = adapter._wrap_chat_create(create_streaming)

        stream = client.chat.completions.create(model="gpt-4o", stream=True)
        collected = list(stream)

        assert len(collected) == 2
        # Events emitted after stream completes
        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"].get("streaming") is True

    def test_multiple_tool_calls(self):
        stratix = MockStratix()
        adapter = OpenAIAdapter(stratix=stratix)
        adapter.connect()

        def create_multi_tools(*args, **kwargs):
            msg = MockMessage(tool_calls=[
                MockToolCall(name="get_weather", tc_id="tc1"),
                MockToolCall(name="search", arguments='{"q":"test"}', tc_id="tc2"),
            ])
            return MockResponse(choices=[MockChoice(message=msg)])

        client = MockClient()
        adapter.connect_client(client)
        client.chat.completions.create = adapter._wrap_chat_create(create_multi_tools)

        client.chat.completions.create(model="gpt-4o")

        tool_events = stratix.get_events("tool.call")
        assert len(tool_events) == 2

    def test_parameters_captured(self):
        stratix = MockStratix()
        adapter = OpenAIAdapter(stratix=stratix)
        adapter.connect()

        client = MockClient()
        adapter.connect_client(client)

        client.chat.completions.create(model="gpt-4o", temperature=0.5, max_tokens=100)

        events = stratix.get_events("model.invoke")
        params = events[0]["payload"].get("parameters", {})
        assert params.get("temperature") == 0.5
        assert params.get("max_tokens") == 100

    def test_no_usage_handled_gracefully(self):
        stratix = MockStratix()
        adapter = OpenAIAdapter(stratix=stratix)
        adapter.connect()

        def create_no_usage(*args, **kwargs):
            resp = MockResponse()
            resp.usage = None
            return resp

        client = MockClient()
        adapter.connect_client(client)
        client.chat.completions.create = adapter._wrap_chat_create(create_no_usage)

        result = client.chat.completions.create(model="gpt-4o")
        assert result is not None

    def test_finish_reason_captured(self):
        """Test that finish_reason is extracted from the response."""
        stratix = MockStratix()
        adapter = OpenAIAdapter(stratix=stratix)
        adapter.connect()

        class MockChoiceWithFinish:
            def __init__(self):
                self.message = MockMessage()
                self.finish_reason = "stop"

        def create_with_finish(*args, **kwargs):
            resp = MockResponse(choices=[MockChoiceWithFinish()])
            return resp

        client = MockClient()
        adapter.connect_client(client)
        client.chat.completions.create = adapter._wrap_chat_create(create_with_finish)

        client.chat.completions.create(model="gpt-4o")

        events = stratix.get_events("model.invoke")
        assert events[0]["payload"].get("finish_reason") == "stop"

    def test_response_id_captured(self):
        """Test that response_id is extracted from the response."""
        stratix = MockStratix()
        adapter = OpenAIAdapter(stratix=stratix)
        adapter.connect()

        def create_with_id(*args, **kwargs):
            resp = MockResponse()
            resp.id = "chatcmpl-abc123"
            return resp

        client = MockClient()
        adapter.connect_client(client)
        client.chat.completions.create = adapter._wrap_chat_create(create_with_id)

        client.chat.completions.create(model="gpt-4o")

        events = stratix.get_events("model.invoke")
        assert events[0]["payload"].get("response_id") == "chatcmpl-abc123"

    def test_response_model_captured(self):
        """Test that the actual model from response is captured."""
        stratix = MockStratix()
        adapter = OpenAIAdapter(stratix=stratix)
        adapter.connect()

        def create_with_model(*args, **kwargs):
            return MockResponse(model="gpt-4o-2024-05-13")

        client = MockClient()
        adapter.connect_client(client)
        client.chat.completions.create = adapter._wrap_chat_create(create_with_model)

        client.chat.completions.create(model="gpt-4o")

        events = stratix.get_events("model.invoke")
        assert events[0]["payload"].get("response_model") == "gpt-4o-2024-05-13"

    def test_system_fingerprint_captured(self):
        """Test that system_fingerprint is captured in metadata."""
        stratix = MockStratix()
        adapter = OpenAIAdapter(stratix=stratix)
        adapter.connect()

        def create_with_fingerprint(*args, **kwargs):
            resp = MockResponse()
            resp.system_fingerprint = "fp_44709d6fcb"
            return resp

        client = MockClient()
        adapter.connect_client(client)
        client.chat.completions.create = adapter._wrap_chat_create(create_with_fingerprint)

        client.chat.completions.create(model="gpt-4o")

        events = stratix.get_events("model.invoke")
        assert events[0]["payload"].get("system_fingerprint") == "fp_44709d6fcb"
