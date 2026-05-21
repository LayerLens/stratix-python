from __future__ import annotations

from unittest.mock import Mock

from layerlens.instrument import trace
from layerlens.instrument.adapters.providers.openai import (
    OpenAIProvider,
    instrument_openai,
    uninstrument_openai,
)

from .conftest import (
    make_openai_response,
    make_openai_response_no_usage,
    make_openai_response_empty_choices,
)
from ...conftest import find_event

# ---------------------------------------------------------------------------
# Emit events
# ---------------------------------------------------------------------------


class TestEmitsEvents:
    def test_model_invoke_and_cost_record(self, mock_client, capture_trace):
        openai_client = Mock()
        openai_client.chat.completions.create = Mock(return_value=make_openai_response())

        provider = OpenAIProvider()
        provider.connect(openai_client)

        @trace(mock_client)
        def my_agent():
            r = openai_client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": "Hi"}])
            return r.choices[0].message.content

        my_agent()
        events = capture_trace["events"]

        model_invoke = find_event(events, "model.invoke")
        assert model_invoke["payload"]["name"] == "openai.chat.completions.create"
        assert model_invoke["payload"]["model"] == "gpt-4"
        assert model_invoke["payload"]["output_message"]["role"] == "assistant"
        assert model_invoke["payload"]["output_message"]["content"] == "Hello!"
        assert model_invoke["payload"]["usage"]["prompt_tokens"] == 10
        assert model_invoke["payload"]["usage"]["completion_tokens"] == 5
        assert model_invoke["payload"]["usage"]["total_tokens"] == 15
        assert "latency_ms" in model_invoke["payload"]

        cost = find_event(events, "cost.record")
        assert cost["payload"]["provider"] == "openai"
        assert cost["payload"]["total_tokens"] == 15

    def test_error_emits_agent_error(self, mock_client, capture_trace):
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
        assert "latency_ms" in error["payload"]


# ---------------------------------------------------------------------------
# Passthrough / no-op behavior
# ---------------------------------------------------------------------------


class TestPassthrough:
    def test_no_op_outside_trace(self):
        response = make_openai_response()
        openai_client = Mock()
        openai_client.chat.completions.create = Mock(return_value=response)

        provider = OpenAIProvider()
        provider.connect(openai_client)

        result = openai_client.chat.completions.create(model="gpt-4", messages=[])
        assert result.choices[0].message.content == "Hello!"


# ---------------------------------------------------------------------------
# Connect / disconnect lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_disconnect_restores_original(self):
        openai_client = Mock()
        original = openai_client.chat.completions.create

        provider = OpenAIProvider()
        provider.connect(openai_client)
        assert openai_client.chat.completions.create is not original

        provider.disconnect()
        assert openai_client.chat.completions.create is original

    def test_disconnect_when_not_connected(self):
        provider = OpenAIProvider()
        provider.disconnect()  # should not raise

    def test_double_connect_replaces_wrapper(self):
        openai_client = Mock()
        provider = OpenAIProvider()
        provider.connect(openai_client)
        first_wrapper = openai_client.chat.completions.create

        provider2 = OpenAIProvider()
        provider2.connect(openai_client)
        assert openai_client.chat.completions.create is not first_wrapper


# ---------------------------------------------------------------------------
# adapter_info
# ---------------------------------------------------------------------------


class TestAdapterInfo:
    def test_info_before_connect(self):
        provider = OpenAIProvider()
        info = provider.adapter_info()
        assert info.name == "openai"
        assert info.adapter_type == "provider"
        assert info.connected is False

    def test_info_after_connect(self):
        provider = OpenAIProvider()
        provider.connect(Mock())
        info = provider.adapter_info()
        assert info.connected is True

    def test_info_after_disconnect(self):
        provider = OpenAIProvider()
        provider.connect(Mock())
        provider.disconnect()
        assert provider.adapter_info().connected is False


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------


class TestConvenienceAPI:
    def test_instrument_and_uninstrument(self):
        openai_client = Mock()
        original = openai_client.chat.completions.create
        instrument_openai(openai_client)
        assert openai_client.chat.completions.create is not original
        uninstrument_openai()


# ---------------------------------------------------------------------------
# capture_params filtering
# ---------------------------------------------------------------------------


class TestCaptureParams:
    def test_captured_params_included(self, mock_client, capture_trace):
        openai_client = Mock()
        openai_client.chat.completions.create = Mock(return_value=make_openai_response())

        provider = OpenAIProvider()
        provider.connect(openai_client)

        @trace(mock_client)
        def my_agent():
            openai_client.chat.completions.create(
                model="gpt-4",
                temperature=0.7,
                top_p=0.9,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return "done"

        my_agent()
        params = find_event(capture_trace["events"], "model.invoke")["payload"]["parameters"]
        assert params["model"] == "gpt-4"
        assert params["temperature"] == 0.7
        assert params["top_p"] == 0.9

    def test_non_captured_params_excluded(self, mock_client, capture_trace):
        openai_client = Mock()
        openai_client.chat.completions.create = Mock(return_value=make_openai_response())

        provider = OpenAIProvider()
        provider.connect(openai_client)

        @trace(mock_client)
        def my_agent():
            openai_client.chat.completions.create(
                model="gpt-4",
                messages=[],
                user="test-user",
            )
            return "done"

        my_agent()
        params = find_event(capture_trace["events"], "model.invoke")["payload"]["parameters"]
        assert "user" not in params
        assert "messages" not in params


# ---------------------------------------------------------------------------
# Extractor edge cases (using real SDK types)
# ---------------------------------------------------------------------------


class TestExtractors:
    def test_extract_output_normal(self):
        r = make_openai_response(content="Hi there", role="assistant")
        output = OpenAIProvider.extract_output(r)
        assert output == {"role": "assistant", "content": "Hi there"}

    def test_extract_output_empty_choices(self):
        r = make_openai_response_empty_choices()
        assert OpenAIProvider.extract_output(r) is None

    def test_extract_meta_normal(self):
        r = make_openai_response(model="gpt-4o", prompt_tokens=5, completion_tokens=3, total_tokens=8)
        meta = OpenAIProvider.extract_meta(r)
        assert meta["response_model"] == "gpt-4o"
        assert meta["usage"]["prompt_tokens"] == 5
        assert meta["usage"]["completion_tokens"] == 3
        assert meta["usage"]["total_tokens"] == 8

    def test_extract_meta_no_usage(self):
        r = make_openai_response_no_usage(model="gpt-4")
        meta = OpenAIProvider.extract_meta(r)
        assert "usage" not in meta
        assert meta["response_model"] == "gpt-4"
