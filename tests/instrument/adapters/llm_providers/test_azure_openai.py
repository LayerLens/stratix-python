"""Tests for Azure OpenAI LLM Provider Adapter."""

import pytest
from layerlens.instrument.adapters.llm_providers.azure_openai_adapter import AzureOpenAIAdapter


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
    def __init__(self, prompt=100, completion=50, total=150):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total
        self.prompt_tokens_details = None
        self.completion_tokens_details = None


class MockMessage:
    def __init__(self, content="Hello", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class MockChoice:
    def __init__(self):
        self.message = MockMessage()


class MockResponse:
    def __init__(self):
        self.usage = MockUsage()
        self.choices = [MockChoice()]
        self.model = "gpt-4o"


class MockEmbeddingUsage:
    def __init__(self):
        self.prompt_tokens = 10
        self.completion_tokens = 0
        self.total_tokens = 10
        self.prompt_tokens_details = None
        self.completion_tokens_details = None


class MockEmbeddingResponse:
    def __init__(self):
        self.usage = MockEmbeddingUsage()


class MockCompletions:
    def create(self, *args, **kwargs):
        return MockResponse()


class MockChat:
    def __init__(self):
        self.completions = MockCompletions()


class MockEmbeddings:
    def create(self, *args, **kwargs):
        return MockEmbeddingResponse()


class MockAzureClient:
    def __init__(self):
        self.chat = MockChat()
        self.embeddings = MockEmbeddings()
        self._base_url = "https://myresource.openai.azure.com"
        self._api_version = "2024-02-01"


class TestAzureOpenAIAdapter:
    """Tests for AzureOpenAIAdapter."""

    def test_adapter_framework(self):
        adapter = AzureOpenAIAdapter()
        assert adapter.FRAMEWORK == "azure_openai"

    def test_connect_client_captures_azure_metadata(self):
        stratix = MockStratix()
        adapter = AzureOpenAIAdapter(stratix=stratix)
        adapter.connect()

        client = MockAzureClient()
        adapter.connect_client(client)

        assert adapter._azure_metadata["azure_endpoint"] == "https://myresource.openai.azure.com"
        assert adapter._azure_metadata["api_version"] == "2024-02-01"

    def test_chat_completion_emits_model_invoke(self):
        stratix = MockStratix()
        adapter = AzureOpenAIAdapter(stratix=stratix)
        adapter.connect()

        client = MockAzureClient()
        adapter.connect_client(client)

        client.chat.completions.create(model="gpt-4o")

        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["provider"] == "azure_openai"
        assert events[0]["payload"]["azure_endpoint"] == "https://myresource.openai.azure.com"

    def test_chat_completion_uses_azure_pricing(self):
        stratix = MockStratix()
        adapter = AzureOpenAIAdapter(stratix=stratix)
        adapter.connect()

        client = MockAzureClient()
        adapter.connect_client(client)

        client.chat.completions.create(model="gpt-4o")

        events = stratix.get_events("cost.record")
        assert len(events) == 1
        assert events[0]["payload"]["provider"] == "azure_openai"
        assert "api_cost_usd" in events[0]["payload"]

    def test_embeddings_emits_events(self):
        stratix = MockStratix()
        adapter = AzureOpenAIAdapter(stratix=stratix)
        adapter.connect()

        client = MockAzureClient()
        adapter.connect_client(client)

        client.embeddings.create(model="text-embedding-3-small", input=["test"])

        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["request_type"] == "embedding"

    def test_error_propagation(self):
        stratix = MockStratix()
        adapter = AzureOpenAIAdapter(stratix=stratix)
        adapter.connect()

        def failing_create(*args, **kwargs):
            raise ValueError("Azure API error")

        client = MockAzureClient()
        adapter.connect_client(client)
        client.chat.completions.create = adapter._wrap_chat_create(failing_create)

        with pytest.raises(ValueError, match="Azure API error"):
            client.chat.completions.create(model="gpt-4o")

    def test_adapter_error_does_not_break_call(self):
        class FailingSTRATIX:
            def emit(self, *args, **kwargs):
                raise RuntimeError("emit failed")

        adapter = AzureOpenAIAdapter(stratix=FailingSTRATIX())
        adapter.connect()

        client = MockAzureClient()
        adapter.connect_client(client)

        result = client.chat.completions.create(model="gpt-4o")
        assert result is not None

    def test_disconnect_restores_originals(self):
        adapter = AzureOpenAIAdapter()
        adapter.connect()

        client = MockAzureClient()
        adapter.connect_client(client)
        assert hasattr(client.chat.completions.create, '_stratix_original')

        adapter.disconnect()
        assert not hasattr(client.chat.completions.create, '_stratix_original')

    def test_embedding_error_propagation(self):
        stratix = MockStratix()
        adapter = AzureOpenAIAdapter(stratix=stratix)
        adapter.connect()

        def failing_embed(*args, **kwargs):
            raise ValueError("Embedding error")

        client = MockAzureClient()
        adapter.connect_client(client)
        client.embeddings.create = adapter._wrap_embeddings_create(failing_embed)

        with pytest.raises(ValueError, match="Embedding error"):
            client.embeddings.create(model="text-embedding-3-small")

    def test_parameters_captured(self):
        stratix = MockStratix()
        adapter = AzureOpenAIAdapter(stratix=stratix)
        adapter.connect()

        client = MockAzureClient()
        adapter.connect_client(client)

        client.chat.completions.create(model="gpt-4o", temperature=0.5)

        events = stratix.get_events("model.invoke")
        params = events[0]["payload"].get("parameters", {})
        assert params.get("temperature") == 0.5

    def test_latency_captured(self):
        stratix = MockStratix()
        adapter = AzureOpenAIAdapter(stratix=stratix)
        adapter.connect()

        client = MockAzureClient()
        adapter.connect_client(client)

        client.chat.completions.create(model="gpt-4o")

        events = stratix.get_events("model.invoke")
        assert "latency_ms" in events[0]["payload"]

    def test_token_extraction(self):
        stratix = MockStratix()
        adapter = AzureOpenAIAdapter(stratix=stratix)
        adapter.connect()

        client = MockAzureClient()
        adapter.connect_client(client)

        client.chat.completions.create(model="gpt-4o")

        events = stratix.get_events("model.invoke")
        assert events[0]["payload"]["prompt_tokens"] == 100
        assert events[0]["payload"]["completion_tokens"] == 50

    def test_api_version_from_custom_query(self):
        stratix = MockStratix()
        adapter = AzureOpenAIAdapter(stratix=stratix)
        adapter.connect()

        client = MockAzureClient()
        client._custom_query = {"api-version": "2025-01-01"}
        adapter.connect_client(client)

        assert adapter._azure_metadata["api_version"] == "2025-01-01"

    def test_connect_wraps_both_methods(self):
        adapter = AzureOpenAIAdapter()
        adapter.connect()

        client = MockAzureClient()
        adapter.connect_client(client)

        assert "chat.completions.create" in adapter._originals
        assert "embeddings.create" in adapter._originals

    def test_error_emits_policy_violation(self):
        stratix = MockStratix()
        adapter = AzureOpenAIAdapter(stratix=stratix)
        adapter.connect()

        def failing_create(*args, **kwargs):
            raise ValueError("forbidden")

        client = MockAzureClient()
        adapter.connect_client(client)
        client.chat.completions.create = adapter._wrap_chat_create(failing_create)

        with pytest.raises(ValueError):
            client.chat.completions.create(model="gpt-4o")

        assert len(stratix.get_events("policy.violation")) == 1
