from __future__ import annotations

import sys
import types
from unittest.mock import Mock

from layerlens.instrument import trace
from .conftest import find_events, find_event


def _openai_response():
    r = Mock()
    r.choices = [Mock()]
    r.choices[0].message = Mock()
    r.choices[0].message.role = "assistant"
    r.choices[0].message.content = "Hello!"
    r.usage = Mock()
    r.usage.prompt_tokens = 10
    r.usage.completion_tokens = 5
    r.usage.total_tokens = 15
    r.model = "gpt-4"
    return r


def _anthropic_response():
    r = Mock()
    block = Mock()
    block.type = "text"
    block.text = "I'm Claude!"
    r.content = [block]
    r.usage = Mock()
    r.usage.input_tokens = 20
    r.usage.output_tokens = 10
    r.model = "claude-3-opus"
    r.stop_reason = "end_turn"
    return r


class TestOpenAIProvider:
    def test_instrument_emits_events(self, mock_client, capture_trace):
        from layerlens.instrument.adapters.providers.openai import OpenAIProvider

        openai_client = Mock()
        openai_client.chat.completions.create = Mock(return_value=_openai_response())

        provider = OpenAIProvider()
        provider.connect(openai_client)

        @trace(mock_client)
        def my_agent():
            return (
                openai_client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": "Hi"}])
                .choices[0]
                .message.content
            )

        my_agent()
        events = capture_trace["events"]
        model_invoke = find_event(events, "model.invoke")
        assert model_invoke["payload"]["name"] == "openai.chat.completions.create"
        assert model_invoke["payload"]["parameters"]["model"] == "gpt-4"
        assert model_invoke["payload"]["usage"]["total_tokens"] == 15
        assert model_invoke["payload"]["output_message"]["content"] == "Hello!"

        cost = find_event(events, "cost.record")
        assert cost["payload"]["provider"] == "openai"
        assert cost["payload"]["total_tokens"] == 15

    def test_passthrough_without_trace(self):
        from layerlens.instrument.adapters.providers.openai import OpenAIProvider

        openai_client = Mock()
        openai_client.chat.completions.create = Mock(return_value=_openai_response())

        provider = OpenAIProvider()
        provider.connect(openai_client)

        result = openai_client.chat.completions.create(model="gpt-4", messages=[])
        assert result.choices[0].message.content == "Hello!"

    def test_disconnect_restores(self):
        from layerlens.instrument.adapters.providers.openai import OpenAIProvider

        openai_client = Mock()
        original = openai_client.chat.completions.create

        provider = OpenAIProvider()
        provider.connect(openai_client)
        assert openai_client.chat.completions.create is not original

        provider.disconnect()
        assert openai_client.chat.completions.create is original

    def test_instrument_convenience_function(self):
        from layerlens.instrument.adapters.providers.openai import instrument_openai, uninstrument_openai

        openai_client = Mock()
        original = openai_client.chat.completions.create
        instrument_openai(openai_client)
        assert openai_client.chat.completions.create is not original
        uninstrument_openai()


class TestAnthropicProvider:
    def test_instrument_emits_events(self, mock_client, capture_trace):
        from layerlens.instrument.adapters.providers.anthropic import AnthropicProvider

        anthropic_client = Mock()
        anthropic_client.messages.create = Mock(return_value=_anthropic_response())

        provider = AnthropicProvider()
        provider.connect(anthropic_client)

        @trace(mock_client)
        def my_agent():
            return (
                anthropic_client.messages.create(
                    model="claude-3-opus", max_tokens=1024, messages=[{"role": "user", "content": "Hi"}]
                )
                .content[0]
                .text
            )

        my_agent()
        events = capture_trace["events"]
        model_invoke = find_event(events, "model.invoke")
        assert model_invoke["payload"]["output_message"]["text"] == "I'm Claude!"
        assert model_invoke["payload"]["usage"]["input_tokens"] == 20
        assert model_invoke["payload"]["response_model"] == "claude-3-opus"
        assert model_invoke["payload"]["stop_reason"] == "end_turn"

    def test_disconnect_restores(self):
        from layerlens.instrument.adapters.providers.anthropic import AnthropicProvider

        anthropic_client = Mock()
        original = anthropic_client.messages.create

        provider = AnthropicProvider()
        provider.connect(anthropic_client)
        provider.disconnect()
        assert anthropic_client.messages.create is original


class TestLiteLLMProvider:
    def setup_method(self):
        self.mock_litellm = types.ModuleType("litellm")
        self.mock_litellm.completion = Mock(return_value=_openai_response())
        self.mock_litellm.acompletion = Mock()
        sys.modules["litellm"] = self.mock_litellm

    def teardown_method(self):
        from layerlens.instrument.adapters.providers.litellm import uninstrument_litellm

        uninstrument_litellm()
        for key in list(sys.modules.keys()):
            if key.startswith("litellm"):
                del sys.modules[key]

    def test_instrument_emits_events(self, mock_client, capture_trace):
        from layerlens.instrument.adapters.providers.litellm import instrument_litellm

        instrument_litellm()

        @trace(mock_client)
        def my_agent():
            import litellm

            return (
                litellm.completion(model="gpt-4", messages=[{"role": "user", "content": "Hi"}])
                .choices[0]
                .message.content
            )

        my_agent()
        events = capture_trace["events"]
        model_invoke = find_event(events, "model.invoke")
        assert model_invoke["payload"]["name"] == "litellm.completion"
        assert model_invoke["payload"]["parameters"]["model"] == "gpt-4"

    def test_passthrough_without_trace(self):
        from layerlens.instrument.adapters.providers.litellm import instrument_litellm

        instrument_litellm()
        import litellm

        result = litellm.completion(model="gpt-4", messages=[])
        assert result.choices[0].message.content == "Hello!"

    def test_uninstrument(self):
        from layerlens.instrument.adapters.providers.litellm import instrument_litellm, uninstrument_litellm

        original = self.mock_litellm.completion
        instrument_litellm()
        assert self.mock_litellm.completion is not original
        uninstrument_litellm()
        assert self.mock_litellm.completion is original


class TestProviderErrorHandling:
    def test_error_emits_event(self, mock_client, capture_trace):
        from layerlens.instrument.adapters.providers.openai import OpenAIProvider

        openai_client = Mock()
        openai_client.chat.completions.create = Mock(side_effect=RuntimeError("API error"))

        provider = OpenAIProvider()
        provider.connect(openai_client)

        @trace(mock_client)
        def my_agent():
            try:
                openai_client.chat.completions.create(model="gpt-4", messages=[])
            except RuntimeError:
                pass
            return "recovered"

        my_agent()
        events = capture_trace["events"]
        error = find_event(events, "agent.error")
        assert error["payload"]["error"] == "API error"
