"""Tests for LLMProviderAdapter base class."""

import pytest
from layerlens.instrument.adapters._base import (
    AdapterCapability,
    AdapterStatus,
    ReplayableTrace,
)
from layerlens.instrument.adapters._capture import CaptureConfig
from layerlens.instrument.adapters.llm_providers.base_provider import LLMProviderAdapter
from layerlens.instrument.adapters.llm_providers.token_usage import NormalizedTokenUsage


class MockStratix:
    def __init__(self):
        self.events = []

    def emit(self, event_type, payload):
        self.events.append({"type": event_type, "payload": payload})

    def get_events(self, event_type=None):
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


class ConcreteProvider(LLMProviderAdapter):
    """Concrete implementation for testing the abstract base."""
    FRAMEWORK = "test_provider"
    VERSION = "0.1.0"

    def connect_client(self, client):
        self._client = client
        return client


class TestLLMProviderAdapter:
    """Tests for the LLMProviderAdapter abstract base."""

    def test_initialization(self):
        adapter = ConcreteProvider()
        assert adapter.FRAMEWORK == "test_provider"
        assert adapter.adapter_type == "llm_provider"
        assert adapter._client is None
        assert adapter._originals == {}

    def test_connect_sets_healthy(self):
        adapter = ConcreteProvider()
        adapter.connect()
        assert adapter.is_connected
        assert adapter.status == AdapterStatus.HEALTHY

    def test_disconnect_sets_disconnected(self):
        adapter = ConcreteProvider()
        adapter.connect()
        adapter.disconnect()
        assert not adapter.is_connected
        assert adapter.status == AdapterStatus.DISCONNECTED

    def test_health_check(self):
        adapter = ConcreteProvider()
        adapter.connect()
        health = adapter.health_check()
        assert health.status == AdapterStatus.HEALTHY
        assert health.framework_name == "test_provider"

    def test_get_adapter_info(self):
        adapter = ConcreteProvider()
        info = adapter.get_adapter_info()
        assert info.name == "ConcreteProvider"
        assert info.framework == "test_provider"
        assert AdapterCapability.TRACE_MODELS in info.capabilities
        assert AdapterCapability.TRACE_TOOLS in info.capabilities

    def test_serialize_for_replay(self):
        adapter = ConcreteProvider()
        trace = adapter.serialize_for_replay()
        assert isinstance(trace, ReplayableTrace)
        assert trace.framework == "test_provider"
        assert trace.trace_id

    def test_emit_model_invoke(self):
        stratix = MockStratix()
        adapter = ConcreteProvider(stratix=stratix)
        adapter.connect()

        usage = NormalizedTokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        adapter._emit_model_invoke(
            provider="test",
            model="test-model",
            usage=usage,
            latency_ms=123.4,
        )

        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["provider"] == "test"
        assert events[0]["payload"]["model"] == "test-model"
        assert events[0]["payload"]["prompt_tokens"] == 100
        assert events[0]["payload"]["latency_ms"] == 123.4

    def test_emit_cost_record(self):
        stratix = MockStratix()
        adapter = ConcreteProvider(stratix=stratix)
        adapter.connect()

        usage = NormalizedTokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        adapter._emit_cost_record(model="gpt-4o", usage=usage, provider="openai")

        events = stratix.get_events("cost.record")
        assert len(events) == 1
        assert events[0]["payload"]["provider"] == "openai"
        assert "api_cost_usd" in events[0]["payload"]

    def test_emit_cost_record_unknown_model(self):
        stratix = MockStratix()
        adapter = ConcreteProvider(stratix=stratix)
        adapter.connect()

        usage = NormalizedTokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        adapter._emit_cost_record(model="unknown-model", usage=usage)

        events = stratix.get_events("cost.record")
        assert len(events) == 1
        assert events[0]["payload"]["api_cost_usd"] is None
        assert events[0]["payload"]["pricing_unavailable"] is True

    def test_emit_tool_calls(self):
        stratix = MockStratix()
        adapter = ConcreteProvider(stratix=stratix)
        adapter.connect()

        adapter._emit_tool_calls(
            [
                {"name": "get_weather", "arguments": {"city": "NYC"}, "id": "tc1"},
                {"name": "search", "arguments": {"q": "test"}, "id": "tc2"},
            ],
            parent_model="gpt-4o",
        )

        events = stratix.get_events("tool.call")
        assert len(events) == 2
        assert events[0]["payload"]["tool_name"] == "get_weather"
        assert events[1]["payload"]["tool_name"] == "search"

    def test_emit_provider_error(self):
        stratix = MockStratix()
        adapter = ConcreteProvider(stratix=stratix)
        adapter.connect()

        adapter._emit_provider_error("test", "rate limit exceeded", model="gpt-4o")

        events = stratix.get_events("policy.violation")
        assert len(events) == 1
        assert events[0]["payload"]["error"] == "rate limit exceeded"

    def test_capture_config_gates_model_invoke(self):
        stratix = MockStratix()
        adapter = ConcreteProvider(stratix=stratix, capture_config=CaptureConfig.minimal())
        adapter.connect()

        adapter._emit_model_invoke(provider="test", model="test-model")

        events = stratix.get_events("model.invoke")
        assert len(events) == 0

    def test_capture_config_allows_cost_record(self):
        """cost.record is cross-cutting, should always emit."""
        stratix = MockStratix()
        adapter = ConcreteProvider(stratix=stratix, capture_config=CaptureConfig.minimal())
        adapter.connect()

        usage = NormalizedTokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        adapter._emit_cost_record(model="gpt-4o", usage=usage)

        events = stratix.get_events("cost.record")
        assert len(events) == 1


class TestTraceContextPropagation:
    """Tests for W3C Trace Context propagation."""

    def test_inject_trace_context_empty_headers(self):
        """Test injecting trace context into empty headers."""
        adapter = ConcreteProvider()
        adapter.connect()

        headers = adapter._inject_trace_context()

        assert isinstance(headers, dict)

    def test_inject_trace_context_preserves_existing(self):
        """Test that existing headers are preserved."""
        adapter = ConcreteProvider()
        adapter.connect()

        headers = {"Authorization": "Bearer token123"}
        result = adapter._inject_trace_context(headers)

        assert result["Authorization"] == "Bearer token123"

    def test_inject_trace_context_with_fallback_ids(self, monkeypatch):
        """Test fallback traceparent generation when OTel SDK is not available."""
        # Block OTel import to test the fallback path
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name.startswith("opentelemetry"):
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        adapter = ConcreteProvider()
        adapter.connect()
        adapter._current_trace_id = "0af7651916cd43dd8448eb211c80319c"
        adapter._current_span_id = "b7ad6b7169203331"

        headers = adapter._inject_trace_context()

        assert "traceparent" in headers
        assert headers["traceparent"] == "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"

    def test_extract_trace_context_valid_traceparent(self):
        """Test extracting a valid traceparent header."""
        adapter = ConcreteProvider()

        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        }

        result = adapter._extract_trace_context(headers)

        assert result["trace_id"] == "0af7651916cd43dd8448eb211c80319c"
        assert result["parent_span_id"] == "b7ad6b7169203331"
        assert result["trace_flags"] == "01"

    def test_extract_trace_context_with_tracestate(self):
        """Test extracting tracestate alongside traceparent."""
        adapter = ConcreteProvider()

        headers = {
            "traceparent": "00-trace123-span456-01",
            "tracestate": "congo=lZWRzIGhqIHNkZmRs,rojo=00f067aa0ba902b7",
        }

        result = adapter._extract_trace_context(headers)

        assert "tracestate" in result
        assert "congo=" in result["tracestate"]

    def test_extract_trace_context_no_headers(self):
        """Test extracting from empty headers."""
        adapter = ConcreteProvider()

        result = adapter._extract_trace_context({})

        assert result == {}

    def test_extract_trace_context_invalid_traceparent(self):
        """Test extracting from invalid traceparent format."""
        adapter = ConcreteProvider()

        headers = {"traceparent": "invalid"}
        result = adapter._extract_trace_context(headers)

        # Should not crash, just return empty
        assert result == {}
