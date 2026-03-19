"""Tests for Anthropic LLM Provider Adapter."""

import pytest
from layerlens.instrument.adapters._capture import CaptureConfig
from layerlens.instrument.adapters.llm_providers.anthropic_adapter import AnthropicAdapter


class MockStratix:
    def __init__(self):
        self.events = []

    def emit(self, event_type, payload):
        self.events.append({"type": event_type, "payload": payload})

    def get_events(self, event_type=None):
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


class MockUsage:
    def __init__(self, input_tokens=100, output_tokens=50, cache_read=None, thinking=None):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read
        self.thinking_tokens = thinking


class MockTextBlock:
    def __init__(self, text="Hello"):
        self.type = "text"
        self.text = text


class MockToolUseBlock:
    def __init__(self, name="get_weather", input_data=None, block_id="tu_1"):
        self.type = "tool_use"
        self.name = name
        self.input = input_data or {"city": "NYC"}
        self.id = block_id


class MockResponse:
    def __init__(self, usage=None, content=None, model="claude-sonnet-4-5-20250929"):
        self.usage = usage or MockUsage()
        self.content = content or [MockTextBlock()]
        self.model = model
        self.stop_reason = "end_turn"


class MockMessages:
    def create(self, *args, **kwargs):
        return MockResponse()


class MockClient:
    def __init__(self):
        self.messages = MockMessages()


class TestAnthropicAdapter:
    """Tests for AnthropicAdapter."""

    def test_adapter_framework(self):
        adapter = AnthropicAdapter()
        assert adapter.FRAMEWORK == "anthropic"
        assert adapter.VERSION == "0.1.0"

    def test_connect_and_disconnect(self):
        adapter = AnthropicAdapter()
        adapter.connect()
        assert adapter.is_connected
        adapter.disconnect()
        assert not adapter.is_connected

    def test_connect_client_wraps_methods(self):
        adapter = AnthropicAdapter()
        adapter.connect()

        client = MockClient()
        adapter.connect_client(client)

        assert "messages.create" in adapter._originals

    def test_messages_create_emits_model_invoke(self):
        stratix = MockStratix()
        adapter = AnthropicAdapter(stratix=stratix)
        adapter.connect()

        client = MockClient()
        adapter.connect_client(client)

        client.messages.create(model="claude-sonnet-4-5-20250929", max_tokens=1024)

        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["provider"] == "anthropic"
        assert events[0]["payload"]["model"] == "claude-sonnet-4-5-20250929"
        assert events[0]["payload"]["prompt_tokens"] == 100
        assert events[0]["payload"]["completion_tokens"] == 50

    def test_messages_create_emits_cost_record(self):
        stratix = MockStratix()
        adapter = AnthropicAdapter(stratix=stratix)
        adapter.connect()

        client = MockClient()
        adapter.connect_client(client)

        client.messages.create(model="claude-sonnet-4-5-20250929", max_tokens=1024)

        events = stratix.get_events("cost.record")
        assert len(events) == 1
        assert events[0]["payload"]["api_cost_usd"] is not None

    def test_tool_use_emits_tool_call(self):
        stratix = MockStratix()
        adapter = AnthropicAdapter(stratix=stratix)
        adapter.connect()

        def create_with_tools(*args, **kwargs):
            return MockResponse(content=[
                MockTextBlock("I'll check the weather"),
                MockToolUseBlock(),
            ])

        client = MockClient()
        adapter.connect_client(client)
        client.messages.create = adapter._wrap_messages_create(create_with_tools)

        client.messages.create(model="claude-sonnet-4-5-20250929", max_tokens=1024)

        tool_events = stratix.get_events("tool.call")
        assert len(tool_events) == 1
        assert tool_events[0]["payload"]["tool_name"] == "get_weather"

    def test_error_propagation(self):
        stratix = MockStratix()
        adapter = AnthropicAdapter(stratix=stratix)
        adapter.connect()

        def failing_create(*args, **kwargs):
            raise ValueError("API error")

        client = MockClient()
        adapter.connect_client(client)
        client.messages.create = adapter._wrap_messages_create(failing_create)

        with pytest.raises(ValueError, match="API error"):
            client.messages.create(model="claude-sonnet-4-5-20250929", max_tokens=1024)

        invoke_events = stratix.get_events("model.invoke")
        assert len(invoke_events) == 1
        assert invoke_events[0]["payload"]["error"] == "API error"

    def test_error_emits_policy_violation(self):
        stratix = MockStratix()
        adapter = AnthropicAdapter(stratix=stratix)
        adapter.connect()

        def failing_create(*args, **kwargs):
            raise ValueError("rate limit")

        client = MockClient()
        adapter.connect_client(client)
        client.messages.create = adapter._wrap_messages_create(failing_create)

        with pytest.raises(ValueError):
            client.messages.create(model="claude-sonnet-4-5-20250929", max_tokens=1024)

        assert len(stratix.get_events("policy.violation")) == 1

    def test_adapter_error_does_not_break_call(self):
        class FailingSTRATIX:
            def emit(self, *args, **kwargs):
                raise RuntimeError("emit failed")

        adapter = AnthropicAdapter(stratix=FailingSTRATIX())
        adapter.connect()

        client = MockClient()
        adapter.connect_client(client)

        result = client.messages.create(model="claude-sonnet-4-5-20250929", max_tokens=1024)
        assert result is not None

    def test_capture_config_minimal_gates_model_invoke(self):
        stratix = MockStratix()
        adapter = AnthropicAdapter(stratix=stratix, capture_config=CaptureConfig.minimal())
        adapter.connect()

        client = MockClient()
        adapter.connect_client(client)

        client.messages.create(model="claude-sonnet-4-5-20250929", max_tokens=1024)

        assert len(stratix.get_events("model.invoke")) == 0
        assert len(stratix.get_events("cost.record")) == 1

    def test_cached_tokens_extracted(self):
        stratix = MockStratix()
        adapter = AnthropicAdapter(stratix=stratix)
        adapter.connect()

        def create_with_cache(*args, **kwargs):
            return MockResponse(usage=MockUsage(cache_read=30))

        client = MockClient()
        adapter.connect_client(client)
        client.messages.create = adapter._wrap_messages_create(create_with_cache)

        client.messages.create(model="claude-sonnet-4-5-20250929", max_tokens=1024)

        events = stratix.get_events("model.invoke")
        assert events[0]["payload"]["cached_tokens"] == 30

    def test_thinking_tokens_extracted(self):
        stratix = MockStratix()
        adapter = AnthropicAdapter(stratix=stratix)
        adapter.connect()

        def create_with_thinking(*args, **kwargs):
            return MockResponse(usage=MockUsage(thinking=500))

        client = MockClient()
        adapter.connect_client(client)
        client.messages.create = adapter._wrap_messages_create(create_with_thinking)

        client.messages.create(model="claude-sonnet-4-5-20250929", max_tokens=1024)

        events = stratix.get_events("model.invoke")
        assert events[0]["payload"]["reasoning_tokens"] == 500

    def test_system_presence_captured(self):
        stratix = MockStratix()
        adapter = AnthropicAdapter(stratix=stratix)
        adapter.connect()

        client = MockClient()
        adapter.connect_client(client)

        client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            system="You are a helpful assistant",
        )

        events = stratix.get_events("model.invoke")
        params = events[0]["payload"].get("parameters", {})
        assert params.get("has_system") is True

    def test_tools_count_captured(self):
        stratix = MockStratix()
        adapter = AnthropicAdapter(stratix=stratix)
        adapter.connect()

        client = MockClient()
        adapter.connect_client(client)

        client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            tools=[{"name": "tool1"}, {"name": "tool2"}],
        )

        events = stratix.get_events("model.invoke")
        params = events[0]["payload"].get("parameters", {})
        assert params.get("tools_count") == 2

    def test_disconnect_restores_originals(self):
        adapter = AnthropicAdapter()
        adapter.connect()

        client = MockClient()
        adapter.connect_client(client)
        assert hasattr(client.messages.create, '_stratix_original')

        adapter.disconnect()
        assert not hasattr(client.messages.create, '_stratix_original')

    def test_multiple_tool_use_blocks(self):
        stratix = MockStratix()
        adapter = AnthropicAdapter(stratix=stratix)
        adapter.connect()

        def create_multi_tools(*args, **kwargs):
            return MockResponse(content=[
                MockToolUseBlock(name="get_weather", block_id="tu_1"),
                MockToolUseBlock(name="search", input_data={"q": "test"}, block_id="tu_2"),
            ])

        client = MockClient()
        adapter.connect_client(client)
        client.messages.create = adapter._wrap_messages_create(create_multi_tools)

        client.messages.create(model="claude-sonnet-4-5-20250929", max_tokens=1024)

        tool_events = stratix.get_events("tool.call")
        assert len(tool_events) == 2

    def test_latency_captured(self):
        stratix = MockStratix()
        adapter = AnthropicAdapter(stratix=stratix)
        adapter.connect()

        client = MockClient()
        adapter.connect_client(client)

        client.messages.create(model="claude-sonnet-4-5-20250929", max_tokens=1024)

        events = stratix.get_events("model.invoke")
        assert "latency_ms" in events[0]["payload"]
        assert events[0]["payload"]["latency_ms"] >= 0

    def test_capture_config_gates_tool_calls(self):
        stratix = MockStratix()
        adapter = AnthropicAdapter(stratix=stratix, capture_config=CaptureConfig.minimal())
        adapter.connect()

        def create_with_tools(*args, **kwargs):
            return MockResponse(content=[MockToolUseBlock()])

        client = MockClient()
        adapter.connect_client(client)
        client.messages.create = adapter._wrap_messages_create(create_with_tools)

        client.messages.create(model="claude-sonnet-4-5-20250929", max_tokens=1024)

        assert len(stratix.get_events("tool.call")) == 0

    def test_no_usage_handled_gracefully(self):
        stratix = MockStratix()
        adapter = AnthropicAdapter(stratix=stratix)
        adapter.connect()

        def create_no_usage(*args, **kwargs):
            resp = MockResponse()
            resp.usage = None
            return resp

        client = MockClient()
        adapter.connect_client(client)
        client.messages.create = adapter._wrap_messages_create(create_no_usage)

        result = client.messages.create(model="claude-sonnet-4-5-20250929", max_tokens=1024)
        assert result is not None

    def test_finish_reason_captured(self):
        """Test that stop_reason is captured as finish_reason."""
        stratix = MockStratix()
        adapter = AnthropicAdapter(stratix=stratix)
        adapter.connect()

        def create_with_stop(*args, **kwargs):
            resp = MockResponse()
            resp.stop_reason = "end_turn"
            return resp

        client = MockClient()
        adapter.connect_client(client)
        client.messages.create = adapter._wrap_messages_create(create_with_stop)

        client.messages.create(model="claude-sonnet-4-5-20250929", max_tokens=1024)

        events = stratix.get_events("model.invoke")
        assert events[0]["payload"].get("finish_reason") == "end_turn"

    def test_response_id_captured(self):
        """Test that response id is captured."""
        stratix = MockStratix()
        adapter = AnthropicAdapter(stratix=stratix)
        adapter.connect()

        def create_with_id(*args, **kwargs):
            resp = MockResponse()
            resp.id = "msg_abc123"
            return resp

        client = MockClient()
        adapter.connect_client(client)
        client.messages.create = adapter._wrap_messages_create(create_with_id)

        client.messages.create(model="claude-sonnet-4-5-20250929", max_tokens=1024)

        events = stratix.get_events("model.invoke")
        assert events[0]["payload"].get("response_id") == "msg_abc123"

    def test_response_model_captured(self):
        """Test that the actual model from response is captured."""
        stratix = MockStratix()
        adapter = AnthropicAdapter(stratix=stratix)
        adapter.connect()

        client = MockClient()
        adapter.connect_client(client)

        client.messages.create(model="claude-sonnet-4-5-20250929", max_tokens=1024)

        events = stratix.get_events("model.invoke")
        assert events[0]["payload"].get("response_model") == "claude-sonnet-4-5-20250929"
