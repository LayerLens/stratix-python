"""LAY-3326 (ADP-070) acceptance-criteria tests.

The ticket requires that the legacy import path::

    from layerlens.adapters.providers import AzureOpenAIAdapter
    from layerlens.adapters.providers import VertexAIAdapter
    from layerlens.adapters.providers import BedrockAdapter
    from layerlens.adapters.providers import OllamaAdapter

succeeds, that each adapter exposes ``connect_client(client)`` and that
``health_check()`` returns an ``AdapterHealth`` snapshot with a sensible
status.

These tests intentionally do NOT depend on the real provider SDKs being
installed — they exercise the shim contract using mocks.
"""

from __future__ import annotations

from unittest.mock import Mock


class TestLegacyImportPath:
    def test_azure_openai_adapter_importable(self):
        from layerlens.adapters.providers import AzureOpenAIAdapter

        assert AzureOpenAIAdapter is not None

    def test_vertex_ai_adapter_importable(self):
        from layerlens.adapters.providers import VertexAIAdapter

        assert VertexAIAdapter is not None

    def test_bedrock_adapter_importable(self):
        from layerlens.adapters.providers import BedrockAdapter

        assert BedrockAdapter is not None

    def test_ollama_adapter_importable(self):
        from layerlens.adapters.providers import OllamaAdapter

        assert OllamaAdapter is not None

    def test_openai_anthropic_litellm_also_importable(self):
        # Not enumerated in the bullets but implied by the story scope.
        from layerlens.adapters.providers import (
            OpenAIAdapter,
            LiteLLMAdapter,
            AnthropicAdapter,
        )

        assert OpenAIAdapter is not None
        assert AnthropicAdapter is not None
        assert LiteLLMAdapter is not None


class TestHealthCheckShape:
    def test_health_check_returns_adapter_health(self):
        from layerlens.adapters.providers import AdapterHealth, AdapterStatus, AzureOpenAIAdapter

        adapter = AzureOpenAIAdapter()
        health = adapter.health_check()
        assert isinstance(health, AdapterHealth)
        # Before connect_client, status is DISCONNECTED.
        assert health.status == AdapterStatus.DISCONNECTED
        assert health.framework_name == "azure-openai"
        assert isinstance(health.adapter_version, str)

    def test_health_status_flips_to_healthy_after_connect(self):
        from layerlens.adapters.providers import AdapterStatus, AzureOpenAIAdapter

        client = Mock()
        # Mock the parts the underlying provider patches so .connect doesn't
        # crash; the legacy shim's job is just to delegate.
        client.chat.completions.create = Mock()
        client.responses.create = Mock()
        client.embeddings.create = Mock()

        adapter = AzureOpenAIAdapter()
        adapter.connect_client(client)
        assert adapter.is_connected is True
        assert adapter.health_check().status == AdapterStatus.HEALTHY


class TestConnectClientTracesLLMCalls:
    """LAY-3326 AC: ``LLM calls are traced with token usage, latency, and cost``.

    The shim delegates to the canonical provider, whose tracing is already
    covered by ``test_openai.py``, ``test_anthropic.py`` etc — these tests
    verify the delegation actually wires through.
    """

    def _find_events(self, events, etype):
        return [e for e in events if e["event_type"] == etype]

    def test_connect_client_wraps_openai(self, mock_client, capture_trace):
        # Reuse the existing test infrastructure: when ``connect_client``
        # wires up an OpenAI client, model.invoke + cost.record events fire.
        from openai.types import CompletionUsage
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from layerlens.instrument import trace
        from layerlens.adapters.providers import OpenAIAdapter
        from openai.types.chat.chat_completion import Choice

        openai_client = Mock()
        openai_client.chat.completions.create = Mock(
            return_value=ChatCompletion(
                id="chatcmpl-x",
                model="gpt-4o",
                object="chat.completion",
                created=1700000000,
                choices=[
                    Choice(
                        index=0,
                        finish_reason="stop",
                        message=ChatCompletionMessage(role="assistant", content="hi"),
                    )
                ],
                usage=CompletionUsage(prompt_tokens=3, completion_tokens=1, total_tokens=4),
            )
        )

        adapter = OpenAIAdapter()
        adapter.connect_client(openai_client)

        @trace(mock_client)
        def my_agent():
            openai_client.chat.completions.create(model="gpt-4o", messages=[])
            return "done"

        my_agent()
        events = capture_trace["events"]
        types = {e["event_type"] for e in events}
        # Per AC: token usage + latency captured (model.invoke) + cost emitted.
        assert "model.invoke" in types
        assert "cost.record" in types

        adapter.disconnect()
        assert adapter.is_connected is False

    def test_connect_client_wraps_anthropic(self, mock_client, capture_trace):
        from anthropic.types import Usage, Message, TextBlock

        from layerlens.instrument import trace
        from layerlens.adapters.providers import AnthropicAdapter

        anthropic_client = Mock()
        anthropic_client.messages.create = Mock(
            return_value=Message(
                id="msg-x",
                type="message",
                role="assistant",
                model="claude-3-5-sonnet-20241022",
                content=[TextBlock(type="text", text="hi")],
                usage=Usage(input_tokens=5, output_tokens=2),
                stop_reason="end_turn",
            )
        )

        adapter = AnthropicAdapter()
        adapter.connect_client(anthropic_client)

        @trace(mock_client)
        def my_agent():
            anthropic_client.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=50, messages=[])
            return "done"

        my_agent()
        types = {e["event_type"] for e in capture_trace["events"]}
        assert "model.invoke" in types
        assert "cost.record" in types

    def test_connect_client_wraps_bedrock(self, mock_client, capture_trace):
        # Bedrock's adapter doesn't use the MonkeyPatchProvider flow — it wraps
        # boto3 invoke_model directly. Verify connect_client wires that up.
        import json as _json

        from layerlens.instrument import trace
        from layerlens.adapters.providers import BedrockAdapter

        boto_client = Mock()

        anthropic_body = _json.dumps(
            {
                "content": [{"text": "hello from bedrock"}],
                "usage": {"input_tokens": 4, "output_tokens": 2},
                "stop_reason": "end_turn",
            }
        ).encode("utf-8")

        def fake_invoke_model(**kwargs):
            return {
                "ResponseMetadata": {"RequestId": "aws-req-1"},
                "body": _MockStreamingBody(anthropic_body),
            }

        boto_client.invoke_model = Mock(side_effect=fake_invoke_model)
        # Don't provide converse / streaming variants so connect only wraps invoke_model.
        del boto_client.converse
        del boto_client.invoke_model_with_response_stream
        del boto_client.converse_stream

        adapter = BedrockAdapter()
        adapter.connect_client(boto_client)

        @trace(mock_client)
        def my_agent():
            boto_client.invoke_model(
                modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
                body=_json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
            )
            return "done"

        my_agent()
        events = capture_trace["events"]
        kinds = {e["event_type"] for e in events}
        assert "model.invoke" in kinds, f"got: {kinds}"
        # Cost should be emitted because BEDROCK_PRICING has the anthropic model.
        assert "cost.record" in kinds
        # And the OTel attrs reach the model.invoke payload (via the bespoke
        # Bedrock _emit_invoke we wired up).
        from tests.instrument.conftest import find_event

        invoke = find_event(events, "model.invoke")
        otel = invoke["payload"]["otel_gen_ai"]
        assert otel["gen_ai.system"] == "aws.bedrock"
        assert otel["gen_ai.response.id"] == "aws-req-1"
        assert otel["gen_ai.response.finish_reasons"] == ["end_turn"]

    def test_connect_client_wraps_vertex(self, mock_client, capture_trace):
        from types import SimpleNamespace as _SN

        from layerlens.instrument import trace
        from layerlens.adapters.providers import VertexAIAdapter

        # Vertex GenerativeModel client: needs generate_content method.
        vertex_client = Mock()

        def fake_generate_content(**kwargs):
            return _SN(
                candidates=[
                    _SN(
                        content=_SN(parts=[_SN(text="hi", function_call=None)]),
                        finish_reason=_SN(name="STOP"),
                    )
                ],
                usage_metadata=_SN(
                    prompt_token_count=4,
                    candidates_token_count=2,
                    total_token_count=6,
                    thoughts_token_count=None,
                ),
                response_id="vertex-resp-1",
            )

        vertex_client.generate_content = Mock(side_effect=fake_generate_content)
        # Don't expose async path so connect only wraps the sync one.
        del vertex_client.generate_content_async

        adapter = VertexAIAdapter()
        adapter.connect_client(vertex_client)

        @trace(mock_client)
        def my_agent():
            vertex_client.generate_content(contents=[{"role": "user", "parts": ["hi"]}])
            return "done"

        my_agent()
        kinds = {e["event_type"] for e in capture_trace["events"]}
        # Vertex has no per-model entry in default PRICING so cost is None —
        # but the cost.record event still emits with cost_usd=None per
        # _emit_cost semantics.
        assert "model.invoke" in kinds

    def test_connect_client_wraps_ollama(self, mock_client, capture_trace):
        from layerlens.instrument import trace
        from layerlens.adapters.providers import OllamaAdapter

        ollama_client = Mock()
        ollama_client.chat = Mock(
            return_value={
                "model": "llama3.1:8b",
                "message": {"role": "assistant", "content": "hi from ollama"},
                "done_reason": "stop",
                "prompt_eval_count": 7,
                "eval_count": 3,
                "total_duration": 1_500_000,  # 1.5ms
            }
        )
        # Don't expose generate/embeddings/embed.
        for attr in ("generate", "embeddings", "embed"):
            if hasattr(ollama_client, attr):
                delattr(ollama_client, attr)

        adapter = OllamaAdapter()
        adapter.connect_client(ollama_client)

        @trace(mock_client)
        def my_agent():
            ollama_client.chat(model="llama3.1:8b", messages=[{"role": "user", "content": "hi"}])
            return "done"

        my_agent()
        events = capture_trace["events"]
        kinds = {e["event_type"] for e in events}
        # Ollama doesn't have a default pricing table entry; no cost.record
        # expected unless ``cost_per_second`` was set on the provider, but
        # model.invoke must still fire.
        assert "model.invoke" in kinds
        from tests.instrument.conftest import find_event

        invoke = find_event(events, "model.invoke")
        # finish_reason normalised from done_reason.
        assert invoke["payload"].get("finish_reason") == "stop"


class _MockStreamingBody:
    """Mimics boto3's botocore.response.StreamingBody for the test fixture."""

    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data
