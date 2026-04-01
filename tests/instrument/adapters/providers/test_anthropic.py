from __future__ import annotations

from unittest.mock import Mock

from layerlens.instrument import trace
from layerlens.instrument.adapters.providers.anthropic import (
    AnthropicProvider,
    instrument_anthropic,
    uninstrument_anthropic,
)

from ...conftest import find_event
from .conftest import make_anthropic_response, make_anthropic_response_empty_content


# ---------------------------------------------------------------------------
# Emit events
# ---------------------------------------------------------------------------


class TestEmitsEvents:
    def test_model_invoke_and_cost_record(self, mock_client, capture_trace):
        anthropic_client = Mock()
        anthropic_client.messages.create = Mock(return_value=make_anthropic_response())

        provider = AnthropicProvider()
        provider.connect(anthropic_client)

        @trace(mock_client)
        def my_agent():
            r = anthropic_client.messages.create(
                model="claude-3-opus-20240229", max_tokens=1024,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return r.content[0].text

        my_agent()
        events = capture_trace["events"]

        model_invoke = find_event(events, "model.invoke")
        assert model_invoke["payload"]["name"] == "anthropic.messages.create"
        assert model_invoke["payload"]["response_model"] == "claude-3-opus-20240229"
        assert model_invoke["payload"]["output_message"]["type"] == "text"
        assert model_invoke["payload"]["output_message"]["text"] == "I'm Claude!"
        assert model_invoke["payload"]["usage"]["input_tokens"] == 20
        assert model_invoke["payload"]["usage"]["output_tokens"] == 10
        assert model_invoke["payload"]["stop_reason"] == "end_turn"
        assert "latency_ms" in model_invoke["payload"]

        cost = find_event(events, "cost.record")
        assert cost["payload"]["provider"] == "anthropic"
        assert cost["payload"]["input_tokens"] == 20
        assert cost["payload"]["output_tokens"] == 10

    def test_error_emits_agent_error(self, mock_client, capture_trace):
        anthropic_client = Mock()
        anthropic_client.messages.create = Mock(side_effect=RuntimeError("overloaded"))

        provider = AnthropicProvider()
        provider.connect(anthropic_client)

        @trace(mock_client)
        def my_agent():
            try:
                anthropic_client.messages.create(model="claude-3-opus-20240229", max_tokens=1024, messages=[])
            except RuntimeError:
                pass
            return "recovered"

        my_agent()
        events = capture_trace["events"]
        error = find_event(events, "agent.error")
        assert error["payload"]["error"] == "overloaded"
        assert "latency_ms" in error["payload"]


# ---------------------------------------------------------------------------
# Passthrough / no-op behavior
# ---------------------------------------------------------------------------


class TestPassthrough:
    def test_no_op_outside_trace(self):
        response = make_anthropic_response()
        anthropic_client = Mock()
        anthropic_client.messages.create = Mock(return_value=response)

        provider = AnthropicProvider()
        provider.connect(anthropic_client)

        result = anthropic_client.messages.create(model="claude-3-opus-20240229", max_tokens=1024, messages=[])
        assert result.content[0].text == "I'm Claude!"


# ---------------------------------------------------------------------------
# Connect / disconnect lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_disconnect_restores_original(self):
        anthropic_client = Mock()
        original = anthropic_client.messages.create

        provider = AnthropicProvider()
        provider.connect(anthropic_client)
        assert anthropic_client.messages.create is not original

        provider.disconnect()
        assert anthropic_client.messages.create is original

    def test_disconnect_when_not_connected(self):
        provider = AnthropicProvider()
        provider.disconnect()  # should not raise

    def test_double_connect_replaces_wrapper(self):
        anthropic_client = Mock()
        provider = AnthropicProvider()
        provider.connect(anthropic_client)
        first_wrapper = anthropic_client.messages.create

        provider2 = AnthropicProvider()
        provider2.connect(anthropic_client)
        assert anthropic_client.messages.create is not first_wrapper


# ---------------------------------------------------------------------------
# adapter_info
# ---------------------------------------------------------------------------


class TestAdapterInfo:
    def test_info_before_connect(self):
        provider = AnthropicProvider()
        info = provider.adapter_info()
        assert info.name == "anthropic"
        assert info.adapter_type == "provider"
        assert info.connected is False

    def test_info_after_connect(self):
        provider = AnthropicProvider()
        provider.connect(Mock())
        info = provider.adapter_info()
        assert info.connected is True

    def test_info_after_disconnect(self):
        provider = AnthropicProvider()
        provider.connect(Mock())
        provider.disconnect()
        assert provider.adapter_info().connected is False


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------


class TestConvenienceAPI:
    def test_instrument_and_uninstrument(self):
        anthropic_client = Mock()
        original = anthropic_client.messages.create
        instrument_anthropic(anthropic_client)
        assert anthropic_client.messages.create is not original
        uninstrument_anthropic()


# ---------------------------------------------------------------------------
# capture_params filtering
# ---------------------------------------------------------------------------


class TestCaptureParams:
    def test_captured_params_included(self, mock_client, capture_trace):
        anthropic_client = Mock()
        anthropic_client.messages.create = Mock(return_value=make_anthropic_response())

        provider = AnthropicProvider()
        provider.connect(anthropic_client)

        @trace(mock_client)
        def my_agent():
            anthropic_client.messages.create(
                model="claude-3-opus-20240229", max_tokens=1024, temperature=0.5, top_k=40,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return "done"

        my_agent()
        params = find_event(capture_trace["events"], "model.invoke")["payload"]["parameters"]
        assert params["model"] == "claude-3-opus-20240229"
        assert params["max_tokens"] == 1024
        assert params["temperature"] == 0.5
        assert params["top_k"] == 40

    def test_non_captured_params_excluded(self, mock_client, capture_trace):
        anthropic_client = Mock()
        anthropic_client.messages.create = Mock(return_value=make_anthropic_response())

        provider = AnthropicProvider()
        provider.connect(anthropic_client)

        @trace(mock_client)
        def my_agent():
            anthropic_client.messages.create(
                model="claude-3-opus-20240229", max_tokens=1024,
                messages=[], stream=True, metadata={"user_id": "abc"},
            )
            return "done"

        my_agent()
        params = find_event(capture_trace["events"], "model.invoke")["payload"]["parameters"]
        assert "stream" not in params
        assert "metadata" not in params
        assert "messages" not in params


# ---------------------------------------------------------------------------
# Extractor edge cases (using real SDK types)
# ---------------------------------------------------------------------------


class TestExtractors:
    def test_extract_output_normal(self):
        r = make_anthropic_response(text="Hello world")
        output = AnthropicProvider.extract_output(r)
        assert output == {"type": "text", "text": "Hello world"}

    def test_extract_output_empty_content(self):
        r = make_anthropic_response_empty_content()
        assert AnthropicProvider.extract_output(r) is None

    def test_extract_meta_normal(self):
        r = make_anthropic_response(
            model="claude-3-5-sonnet-20241022",
            input_tokens=100, output_tokens=50,
            stop_reason="max_tokens",
        )
        meta = AnthropicProvider.extract_meta(r)
        assert meta["response_model"] == "claude-3-5-sonnet-20241022"
        assert meta["usage"]["input_tokens"] == 100
        assert meta["usage"]["output_tokens"] == 50
        assert meta["stop_reason"] == "max_tokens"
