"""Tests for Ollama LLM Provider Adapter."""

import pytest
from layerlens.instrument.adapters.llm_providers.ollama_adapter import OllamaAdapter


class MockStratix:
    def __init__(self):
        self.events = []

    def emit(self, event_type, payload):
        self.events.append({"type": event_type, "payload": payload})

    def get_events(self, event_type=None):
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


class MockOllamaModule:
    """Mock ollama module."""

    def chat(self, *args, **kwargs):
        return {
            "model": "llama3.1:70b",
            "message": {"role": "assistant", "content": "Hello!"},
            "prompt_eval_count": 50,
            "eval_count": 30,
            "eval_duration": 1_000_000_000,  # 1 second
            "prompt_eval_duration": 500_000_000,  # 0.5 seconds
        }

    def generate(self, *args, **kwargs):
        return {
            "model": "llama3.1:70b",
            "response": "Hello!",
            "prompt_eval_count": 40,
            "eval_count": 20,
            "eval_duration": 800_000_000,
            "prompt_eval_duration": 400_000_000,
        }

    def embeddings(self, *args, **kwargs):
        return {
            "model": "llama3.1:70b",
            "embedding": [0.1, 0.2, 0.3],
            "prompt_eval_count": 10,
            "eval_count": 0,
        }


class TestOllamaAdapter:
    """Tests for OllamaAdapter."""

    def test_adapter_framework(self):
        adapter = OllamaAdapter()
        assert adapter.FRAMEWORK == "ollama"

    def test_connect_client_wraps_methods(self):
        adapter = OllamaAdapter()
        adapter.connect()

        client = MockOllamaModule()
        adapter.connect_client(client)

        assert "chat" in adapter._originals
        assert "generate" in adapter._originals
        assert "embeddings" in adapter._originals

    def test_chat_emits_model_invoke(self):
        stratix = MockStratix()
        adapter = OllamaAdapter(stratix=stratix)
        adapter.connect()

        client = MockOllamaModule()
        adapter.connect_client(client)

        client.chat(model="llama3.1:70b", messages=[])

        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["provider"] == "ollama"
        assert events[0]["payload"]["prompt_tokens"] == 50
        assert events[0]["payload"]["completion_tokens"] == 30

    def test_chat_emits_zero_cost(self):
        stratix = MockStratix()
        adapter = OllamaAdapter(stratix=stratix)
        adapter.connect()

        client = MockOllamaModule()
        adapter.connect_client(client)

        client.chat(model="llama3.1:70b", messages=[])

        events = stratix.get_events("cost.record")
        assert len(events) == 1
        assert events[0]["payload"]["api_cost_usd"] == 0.0

    def test_generate_emits_model_invoke(self):
        stratix = MockStratix()
        adapter = OllamaAdapter(stratix=stratix)
        adapter.connect()

        client = MockOllamaModule()
        adapter.connect_client(client)

        client.generate(model="llama3.1:70b", prompt="Hello")

        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["prompt_tokens"] == 40
        assert events[0]["payload"]["completion_tokens"] == 20

    def test_embeddings_emits_events(self):
        stratix = MockStratix()
        adapter = OllamaAdapter(stratix=stratix)
        adapter.connect()

        client = MockOllamaModule()
        adapter.connect_client(client)

        client.embeddings(model="llama3.1:70b", prompt="Hello")

        events = stratix.get_events("model.invoke")
        assert len(events) == 1

    def test_infra_cost_calculated_with_config(self):
        stratix = MockStratix()
        adapter = OllamaAdapter(stratix=stratix, cost_per_second=0.001)
        adapter.connect()

        client = MockOllamaModule()
        adapter.connect_client(client)

        client.chat(model="llama3.1:70b", messages=[])

        events = stratix.get_events("cost.record")
        assert len(events) == 1
        assert "infra_cost_usd" in events[0]["payload"]
        # 1.5 seconds * 0.001 = 0.0015
        assert events[0]["payload"]["infra_cost_usd"] == pytest.approx(0.0015, abs=1e-6)

    def test_no_infra_cost_without_config(self):
        stratix = MockStratix()
        adapter = OllamaAdapter(stratix=stratix)
        adapter.connect()

        client = MockOllamaModule()
        adapter.connect_client(client)

        client.chat(model="llama3.1:70b", messages=[])

        events = stratix.get_events("cost.record")
        assert "infra_cost_usd" not in events[0]["payload"]

    def test_error_propagation(self):
        stratix = MockStratix()
        adapter = OllamaAdapter(stratix=stratix)
        adapter.connect()

        def failing_chat(*args, **kwargs):
            raise ConnectionError("Ollama not running")

        client = MockOllamaModule()
        adapter.connect_client(client)
        client.chat = adapter._wrap_call(failing_chat, "chat")

        with pytest.raises(ConnectionError, match="Ollama not running"):
            client.chat(model="llama3.1:70b", messages=[])

    def test_adapter_error_does_not_break_call(self):
        class FailingSTRATIX:
            def emit(self, *args, **kwargs):
                raise RuntimeError("emit failed")

        adapter = OllamaAdapter(stratix=FailingSTRATIX())
        adapter.connect()

        client = MockOllamaModule()
        adapter.connect_client(client)

        result = client.chat(model="llama3.1:70b", messages=[])
        assert result is not None

    def test_disconnect_restores_originals(self):
        adapter = OllamaAdapter()
        adapter.connect()

        client = MockOllamaModule()
        adapter.connect_client(client)
        assert hasattr(client.chat, '_stratix_original')

        adapter.disconnect()
        assert not hasattr(client.chat, '_stratix_original')

    def test_endpoint_detection(self):
        adapter = OllamaAdapter()
        adapter.connect()
        assert adapter._endpoint is not None

    def test_method_name_in_metadata(self):
        stratix = MockStratix()
        adapter = OllamaAdapter(stratix=stratix)
        adapter.connect()

        client = MockOllamaModule()
        adapter.connect_client(client)

        client.chat(model="llama3.1:70b", messages=[])

        events = stratix.get_events("model.invoke")
        assert events[0]["payload"]["method"] == "chat"

    def test_latency_captured(self):
        stratix = MockStratix()
        adapter = OllamaAdapter(stratix=stratix)
        adapter.connect()

        client = MockOllamaModule()
        adapter.connect_client(client)

        client.chat(model="llama3.1:70b", messages=[])

        events = stratix.get_events("model.invoke")
        assert "latency_ms" in events[0]["payload"]
