"""Tests for LiteLLM Provider Adapter."""

import pytest
from datetime import datetime
from layerlens.instrument.adapters.llm_providers.litellm_adapter import (
    STRATIXLiteLLMCallback,
    LiteLLMAdapter,
    detect_provider,
)


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
    def __init__(self, prompt=100, completion=50):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion


class MockResponseObj:
    def __init__(self, usage=None):
        self.usage = usage or MockUsage()


class TestDetectProvider:
    """Tests for LiteLLM provider detection."""

    def test_openai_prefix(self):
        assert detect_provider("openai/gpt-4o") == "openai"

    def test_anthropic_prefix(self):
        assert detect_provider("anthropic/claude-sonnet-4-5-20250929") == "anthropic"

    def test_azure_prefix(self):
        assert detect_provider("azure/gpt-4o") == "azure_openai"

    def test_bedrock_prefix(self):
        assert detect_provider("bedrock/anthropic.claude-v2") == "aws_bedrock"

    def test_vertex_prefix(self):
        assert detect_provider("vertex_ai/gemini-2.0-flash") == "google_vertex"

    def test_ollama_prefix(self):
        assert detect_provider("ollama/llama3.1") == "ollama"

    def test_gpt_without_prefix(self):
        assert detect_provider("gpt-4o") == "openai"

    def test_claude_without_prefix(self):
        assert detect_provider("claude-sonnet-4-5-20250929") == "anthropic"

    def test_gemini_without_prefix(self):
        assert detect_provider("gemini-2.0-flash") == "google_vertex"

    def test_unknown_model(self):
        assert detect_provider("some-random-model") == "unknown"

    def test_empty_string(self):
        assert detect_provider("") == "unknown"

    def test_groq_prefix(self):
        assert detect_provider("groq/llama-3.1-70b") == "groq"


class TestSTRATIXLiteLLMCallback:
    """Tests for STRATIXLiteLLMCallback."""

    def test_log_success_emits_model_invoke(self):
        stratix = MockStratix()
        adapter = LiteLLMAdapter(stratix=stratix)
        adapter.connect()
        callback = STRATIXLiteLLMCallback(adapter)

        now = datetime.now()
        callback.log_success_event(
            kwargs={"model": "openai/gpt-4o", "temperature": 0.7},
            response_obj=MockResponseObj(),
            start_time=now,
            end_time=now,
        )

        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["provider"] == "openai"
        assert events[0]["payload"]["model"] == "openai/gpt-4o"

    def test_log_success_emits_cost_record(self):
        stratix = MockStratix()
        adapter = LiteLLMAdapter(stratix=stratix)
        adapter.connect()
        callback = STRATIXLiteLLMCallback(adapter)

        now = datetime.now()
        callback.log_success_event(
            kwargs={"model": "gpt-4o"},
            response_obj=MockResponseObj(),
            start_time=now,
            end_time=now,
        )

        events = stratix.get_events("cost.record")
        assert len(events) == 1

    def test_log_failure_emits_error(self):
        stratix = MockStratix()
        adapter = LiteLLMAdapter(stratix=stratix)
        adapter.connect()
        callback = STRATIXLiteLLMCallback(adapter)

        now = datetime.now()
        callback.log_failure_event(
            kwargs={"model": "gpt-4o", "exception": "rate limit exceeded"},
            response_obj=None,
            start_time=now,
            end_time=now,
        )

        invoke_events = stratix.get_events("model.invoke")
        assert len(invoke_events) == 1
        assert invoke_events[0]["payload"]["error"] == "rate limit exceeded"

        violation_events = stratix.get_events("policy.violation")
        assert len(violation_events) == 1

    def test_log_stream_emits_events(self):
        stratix = MockStratix()
        adapter = LiteLLMAdapter(stratix=stratix)
        adapter.connect()
        callback = STRATIXLiteLLMCallback(adapter)

        now = datetime.now()
        callback.log_stream_event(
            kwargs={"model": "anthropic/claude-sonnet-4-5-20250929"},
            response_obj=MockResponseObj(),
            start_time=now,
            end_time=now,
        )

        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"].get("streaming") is True

    def test_latency_calculation(self):
        stratix = MockStratix()
        adapter = LiteLLMAdapter(stratix=stratix)
        adapter.connect()
        callback = STRATIXLiteLLMCallback(adapter)

        start = datetime(2025, 1, 1, 0, 0, 0)
        end = datetime(2025, 1, 1, 0, 0, 1)  # 1 second later

        callback.log_success_event(
            kwargs={"model": "gpt-4o"},
            response_obj=MockResponseObj(),
            start_time=start,
            end_time=end,
        )

        events = stratix.get_events("model.invoke")
        assert events[0]["payload"]["latency_ms"] == pytest.approx(1000.0, abs=1)

    def test_params_extraction(self):
        params = STRATIXLiteLLMCallback._extract_params({
            "model": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
        })
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 100
        assert params["top_p"] == 0.9

    def test_params_from_optional_params(self):
        params = STRATIXLiteLLMCallback._extract_params({
            "model": "gpt-4o",
            "optional_params": {"temperature": 0.5},
        })
        assert params["temperature"] == 0.5

    def test_callback_error_does_not_propagate(self):
        """Callback errors should be swallowed."""
        class FailingSTRATIX:
            def emit(self, *args, **kwargs):
                raise RuntimeError("emit failed")

        adapter = LiteLLMAdapter(stratix=FailingSTRATIX())
        adapter.connect()
        callback = STRATIXLiteLLMCallback(adapter)

        now = datetime.now()
        # Should not raise
        callback.log_success_event(
            kwargs={"model": "gpt-4o"},
            response_obj=MockResponseObj(),
            start_time=now,
            end_time=now,
        )


class TestLiteLLMAdapter:
    """Tests for LiteLLMAdapter."""

    def test_adapter_framework(self):
        adapter = LiteLLMAdapter()
        assert adapter.FRAMEWORK == "litellm"

    def test_connect_and_disconnect(self):
        adapter = LiteLLMAdapter()
        adapter.connect()
        assert adapter.is_connected
        assert adapter._callback is not None

        adapter.disconnect()
        assert not adapter.is_connected
        assert adapter._callback is None

    def test_connect_client_is_noop(self):
        adapter = LiteLLMAdapter()
        adapter.connect()

        client = object()
        result = adapter.connect_client(client)
        assert result is client

    def test_no_usage_handled_gracefully(self):
        stratix = MockStratix()
        adapter = LiteLLMAdapter(stratix=stratix)
        adapter.connect()
        callback = STRATIXLiteLLMCallback(adapter)

        now = datetime.now()
        resp = MockResponseObj()
        resp.usage = None

        callback.log_success_event(
            kwargs={"model": "gpt-4o"},
            response_obj=resp,
            start_time=now,
            end_time=now,
        )

        events = stratix.get_events("model.invoke")
        assert len(events) == 1

    def test_null_timestamps_handled(self):
        stratix = MockStratix()
        adapter = LiteLLMAdapter(stratix=stratix)
        adapter.connect()
        callback = STRATIXLiteLLMCallback(adapter)

        callback.log_success_event(
            kwargs={"model": "gpt-4o"},
            response_obj=MockResponseObj(),
            start_time=None,
            end_time=None,
        )

        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"].get("latency_ms") is None
