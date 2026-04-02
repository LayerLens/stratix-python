from __future__ import annotations

import sys
import types
from unittest.mock import Mock

from layerlens.instrument import trace
from layerlens.instrument.adapters.providers.litellm import (
    LiteLLMProvider,
    instrument_litellm,
    uninstrument_litellm,
)

from ...conftest import find_event
from .conftest import make_openai_response, make_openai_response_empty_choices, make_openai_response_no_usage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_mock_litellm(response=None):
    """Inject a fake litellm module into sys.modules with real OpenAI response types."""
    mock_mod = types.ModuleType("litellm")
    mock_mod.completion = Mock(return_value=response or make_openai_response())
    mock_mod.acompletion = Mock()
    sys.modules["litellm"] = mock_mod
    return mock_mod


def _remove_mock_litellm():
    uninstrument_litellm()
    for key in list(sys.modules.keys()):
        if key.startswith("litellm"):
            del sys.modules[key]


# ---------------------------------------------------------------------------
# Emit events
# ---------------------------------------------------------------------------


class TestEmitsEvents:
    def setup_method(self):
        self.mock_litellm = _install_mock_litellm()

    def teardown_method(self):
        _remove_mock_litellm()

    def test_model_invoke_and_cost_record(self, mock_client, capture_trace):
        instrument_litellm()

        @trace(mock_client)
        def my_agent():
            import litellm
            r = litellm.completion(
                model="gpt-4", messages=[{"role": "user", "content": "Hi"}]
            )
            return r.choices[0].message.content

        my_agent()
        events = capture_trace["events"]

        model_invoke = find_event(events, "model.invoke")
        assert model_invoke["payload"]["name"] == "litellm.completion"
        assert model_invoke["payload"]["model"] == "gpt-4"
        assert model_invoke["payload"]["output_message"]["content"] == "Hello!"
        assert model_invoke["payload"]["usage"]["total_tokens"] == 15
        assert "latency_ms" in model_invoke["payload"]

        cost = find_event(events, "cost.record")
        assert cost["payload"]["provider"] == "litellm"
        assert cost["payload"]["total_tokens"] == 15

    def test_error_emits_agent_error(self, mock_client, capture_trace):
        self.mock_litellm.completion = Mock(side_effect=RuntimeError("rate limited"))
        instrument_litellm()

        @trace(mock_client)
        def my_agent():
            import litellm
            try:
                litellm.completion(model="gpt-4", messages=[])
            except RuntimeError:
                pass
            return "recovered"

        my_agent()
        events = capture_trace["events"]
        error = find_event(events, "agent.error")
        assert error["payload"]["error"] == "rate limited"
        assert "latency_ms" in error["payload"]


# ---------------------------------------------------------------------------
# Passthrough / no-op behavior
# ---------------------------------------------------------------------------


class TestPassthrough:
    def setup_method(self):
        self.mock_litellm = _install_mock_litellm()

    def teardown_method(self):
        _remove_mock_litellm()

    def test_no_op_outside_trace(self):
        instrument_litellm()
        import litellm
        result = litellm.completion(model="gpt-4", messages=[])
        assert result.choices[0].message.content == "Hello!"


# ---------------------------------------------------------------------------
# Connect / disconnect lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def setup_method(self):
        self.mock_litellm = _install_mock_litellm()

    def teardown_method(self):
        _remove_mock_litellm()

    def test_uninstrument_restores_original(self):
        original = self.mock_litellm.completion
        instrument_litellm()
        assert self.mock_litellm.completion is not original
        uninstrument_litellm()
        assert self.mock_litellm.completion is original

    def test_disconnect_when_not_connected(self):
        provider = LiteLLMProvider()
        provider.disconnect()  # should not raise


# ---------------------------------------------------------------------------
# adapter_info
# ---------------------------------------------------------------------------


class TestAdapterInfo:
    def setup_method(self):
        self.mock_litellm = _install_mock_litellm()

    def teardown_method(self):
        _remove_mock_litellm()

    def test_info_before_connect(self):
        provider = LiteLLMProvider()
        info = provider.adapter_info()
        assert info.name == "litellm"
        assert info.adapter_type == "provider"
        assert info.connected is False

    def test_info_after_connect(self):
        provider = LiteLLMProvider()
        provider.connect()
        info = provider.adapter_info()
        assert info.connected is True

    def test_info_after_disconnect(self):
        provider = LiteLLMProvider()
        provider.connect()
        provider.disconnect()
        assert provider.adapter_info().connected is False


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------


class TestConvenienceAPI:
    def setup_method(self):
        self.mock_litellm = _install_mock_litellm()

    def teardown_method(self):
        _remove_mock_litellm()

    def test_instrument_and_uninstrument(self):
        original = self.mock_litellm.completion
        instrument_litellm()
        assert self.mock_litellm.completion is not original
        uninstrument_litellm()
        assert self.mock_litellm.completion is original


# ---------------------------------------------------------------------------
# capture_params filtering
# ---------------------------------------------------------------------------


class TestCaptureParams:
    def setup_method(self):
        self.mock_litellm = _install_mock_litellm()

    def teardown_method(self):
        _remove_mock_litellm()

    def test_captured_params_included(self, mock_client, capture_trace):
        instrument_litellm()

        @trace(mock_client)
        def my_agent():
            import litellm
            litellm.completion(
                model="gpt-4", temperature=0.7, top_p=0.9,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return "done"

        my_agent()
        params = find_event(capture_trace["events"], "model.invoke")["payload"]["parameters"]
        assert params["model"] == "gpt-4"
        assert params["temperature"] == 0.7
        assert params["top_p"] == 0.9

    def test_non_captured_params_excluded(self, mock_client, capture_trace):
        instrument_litellm()

        @trace(mock_client)
        def my_agent():
            import litellm
            litellm.completion(
                model="gpt-4", messages=[], stream=True, api_key="sk-123",
            )
            return "done"

        my_agent()
        params = find_event(capture_trace["events"], "model.invoke")["payload"]["parameters"]
        assert "stream" not in params
        assert "api_key" not in params
        assert "messages" not in params


# ---------------------------------------------------------------------------
# Extractor edge cases (LiteLLM reuses OpenAI extractors, real types)
# ---------------------------------------------------------------------------


class TestExtractors:
    def test_extract_output_normal(self):
        r = make_openai_response(content="LiteLLM response")
        output = LiteLLMProvider.extract_output(r)
        assert output == {"role": "assistant", "content": "LiteLLM response"}

    def test_extract_output_empty_choices(self):
        r = make_openai_response_empty_choices()
        assert LiteLLMProvider.extract_output(r) is None

    def test_extract_meta_normal(self):
        r = make_openai_response(model="gpt-4o", prompt_tokens=5, completion_tokens=3, total_tokens=8)
        meta = LiteLLMProvider.extract_meta(r)
        assert meta["response_model"] == "gpt-4o"
        assert meta["usage"]["total_tokens"] == 8

    def test_extract_meta_no_usage(self):
        r = make_openai_response_no_usage()
        meta = LiteLLMProvider.extract_meta(r)
        assert "usage" not in meta
