"""Tests for AWS Bedrock LLM Provider Adapter."""

import json
import pytest
from layerlens.instrument.adapters.llm_providers.bedrock_adapter import (
    AWSBedrockAdapter,
    _detect_provider_family,
    _RereadableBody,
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


class MockStreamBody:
    """Simulates boto3 StreamingBody."""
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data


class MockBedrockClient:
    """Mock Bedrock runtime client."""

    def invoke_model(self, **kwargs):
        model_id = kwargs.get("modelId", "")
        family = _detect_provider_family(model_id)
        if family == "anthropic":
            body = json.dumps({
                "usage": {"input_tokens": 100, "output_tokens": 50},
                "content": [{"type": "text", "text": "Hello"}],
            }).encode()
        elif family == "meta":
            body = json.dumps({
                "prompt_token_count": 80,
                "generation_token_count": 40,
                "generation": "Hello",
            }).encode()
        elif family == "cohere":
            body = json.dumps({
                "meta": {"billed_units": {"input_tokens": 90, "output_tokens": 45}},
                "generations": [{"text": "Hello"}],
            }).encode()
        else:
            body = json.dumps({"result": "Hello"}).encode()
        return {"body": MockStreamBody(body)}

    def converse(self, **kwargs):
        return {
            "output": {"message": {"content": [{"text": "Hello"}]}},
            "usage": {"inputTokens": 100, "outputTokens": 50},
        }

    def invoke_model_with_response_stream(self, **kwargs):
        return {"body": iter([])}

    def converse_stream(self, **kwargs):
        return {"stream": iter([])}


class TestDetectProviderFamily:
    """Tests for _detect_provider_family."""

    def test_anthropic(self):
        assert _detect_provider_family("anthropic.claude-3-5-sonnet-20241022-v2:0") == "anthropic"

    def test_meta(self):
        assert _detect_provider_family("meta.llama3-1-70b-instruct-v1:0") == "meta"

    def test_cohere(self):
        assert _detect_provider_family("cohere.command-r-plus-v1:0") == "cohere"

    def test_amazon(self):
        assert _detect_provider_family("amazon.titan-text-express-v1") == "amazon"

    def test_ai21(self):
        assert _detect_provider_family("ai21.jamba-1-5-large-v1:0") == "ai21"

    def test_mistral(self):
        assert _detect_provider_family("mistral.mistral-large-2407-v1:0") == "mistral"

    def test_unknown(self):
        assert _detect_provider_family("unknown.model") == "unknown"

    def test_empty_string(self):
        assert _detect_provider_family("") == "unknown"


class TestAWSBedrockAdapter:
    """Tests for AWSBedrockAdapter."""

    def test_adapter_framework(self):
        adapter = AWSBedrockAdapter()
        assert adapter.FRAMEWORK == "aws_bedrock"

    def test_connect_client_wraps_methods(self):
        adapter = AWSBedrockAdapter()
        adapter.connect()

        client = MockBedrockClient()
        adapter.connect_client(client)

        assert "invoke_model" in adapter._originals
        assert "converse" in adapter._originals
        assert "invoke_model_with_response_stream" in adapter._originals
        assert "converse_stream" in adapter._originals

    def test_invoke_model_anthropic_tokens(self):
        stratix = MockStratix()
        adapter = AWSBedrockAdapter(stratix=stratix)
        adapter.connect()

        client = MockBedrockClient()
        adapter.connect_client(client)

        result = client.invoke_model(modelId="anthropic.claude-3-5-sonnet-20241022-v2:0", body=b'{}')

        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["prompt_tokens"] == 100
        assert events[0]["payload"]["completion_tokens"] == 50
        assert events[0]["payload"]["provider_family"] == "anthropic"

    def test_invoke_model_meta_tokens(self):
        stratix = MockStratix()
        adapter = AWSBedrockAdapter(stratix=stratix)
        adapter.connect()

        client = MockBedrockClient()
        adapter.connect_client(client)

        client.invoke_model(modelId="meta.llama3-1-70b-instruct-v1:0", body=b'{}')

        events = stratix.get_events("model.invoke")
        assert events[0]["payload"]["prompt_tokens"] == 80
        assert events[0]["payload"]["completion_tokens"] == 40

    def test_invoke_model_cohere_tokens(self):
        stratix = MockStratix()
        adapter = AWSBedrockAdapter(stratix=stratix)
        adapter.connect()

        client = MockBedrockClient()
        adapter.connect_client(client)

        client.invoke_model(modelId="cohere.command-r-plus-v1:0", body=b'{}')

        events = stratix.get_events("model.invoke")
        assert events[0]["payload"]["prompt_tokens"] == 90
        assert events[0]["payload"]["completion_tokens"] == 45

    def test_converse_tokens(self):
        stratix = MockStratix()
        adapter = AWSBedrockAdapter(stratix=stratix)
        adapter.connect()

        client = MockBedrockClient()
        adapter.connect_client(client)

        client.converse(modelId="anthropic.claude-3-5-sonnet-20241022-v2:0", messages=[])

        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["prompt_tokens"] == 100
        assert events[0]["payload"]["completion_tokens"] == 50

    def test_invoke_model_emits_cost_record(self):
        stratix = MockStratix()
        adapter = AWSBedrockAdapter(stratix=stratix)
        adapter.connect()

        client = MockBedrockClient()
        adapter.connect_client(client)

        client.invoke_model(modelId="anthropic.claude-3-5-sonnet-20241022-v2:0", body=b'{}')

        events = stratix.get_events("cost.record")
        assert len(events) == 1
        assert events[0]["payload"]["provider"] == "aws_bedrock"

    def test_converse_emits_cost_record(self):
        stratix = MockStratix()
        adapter = AWSBedrockAdapter(stratix=stratix)
        adapter.connect()

        client = MockBedrockClient()
        adapter.connect_client(client)

        client.converse(modelId="anthropic.claude-3-5-sonnet-20241022-v2:0", messages=[])

        events = stratix.get_events("cost.record")
        assert len(events) == 1

    def test_error_propagation(self):
        stratix = MockStratix()
        adapter = AWSBedrockAdapter(stratix=stratix)
        adapter.connect()

        def failing_invoke(**kwargs):
            raise ValueError("Bedrock error")

        client = MockBedrockClient()
        adapter.connect_client(client)
        client.invoke_model = adapter._wrap_invoke_model(failing_invoke)

        with pytest.raises(ValueError, match="Bedrock error"):
            client.invoke_model(modelId="anthropic.claude-3-5-sonnet-20241022-v2:0")

    def test_adapter_error_does_not_break_call(self):
        class FailingSTRATIX:
            def emit(self, *args, **kwargs):
                raise RuntimeError("emit failed")

        adapter = AWSBedrockAdapter(stratix=FailingSTRATIX())
        adapter.connect()

        client = MockBedrockClient()
        adapter.connect_client(client)

        result = client.invoke_model(modelId="anthropic.claude-3-5-sonnet-20241022-v2:0", body=b'{}')
        assert result is not None

    def test_rereadable_body(self):
        body = _RereadableBody(b'{"test": true}')
        assert body.read() == b'{"test": true}'
        assert body.read() == b'{"test": true}'  # can read again

    def test_invoke_body_remains_readable(self):
        stratix = MockStratix()
        adapter = AWSBedrockAdapter(stratix=stratix)
        adapter.connect()

        client = MockBedrockClient()
        adapter.connect_client(client)

        result = client.invoke_model(modelId="anthropic.claude-3-5-sonnet-20241022-v2:0", body=b'{}')
        # Body should still be readable after adapter consumed it
        body_data = json.loads(result["body"].read())
        assert "usage" in body_data

    def test_disconnect_restores_originals(self):
        adapter = AWSBedrockAdapter()
        adapter.connect()

        client = MockBedrockClient()
        adapter.connect_client(client)
        assert hasattr(client.invoke_model, '_stratix_original')

        adapter.disconnect()
        assert not hasattr(client.invoke_model, '_stratix_original')

    def test_streaming_invoke_emits_event(self):
        stratix = MockStratix()
        adapter = AWSBedrockAdapter(stratix=stratix)
        adapter.connect()

        client = MockBedrockClient()
        adapter.connect_client(client)

        client.invoke_model_with_response_stream(modelId="anthropic.claude-3-5-sonnet-20241022-v2:0")

        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"].get("streaming") is True

    def test_converse_stream_emits_event(self):
        stratix = MockStratix()
        adapter = AWSBedrockAdapter(stratix=stratix)
        adapter.connect()

        client = MockBedrockClient()
        adapter.connect_client(client)

        client.converse_stream(modelId="anthropic.claude-3-5-sonnet-20241022-v2:0")

        events = stratix.get_events("model.invoke")
        assert len(events) == 1

    def test_latency_captured(self):
        stratix = MockStratix()
        adapter = AWSBedrockAdapter(stratix=stratix)
        adapter.connect()

        client = MockBedrockClient()
        adapter.connect_client(client)

        client.converse(modelId="anthropic.claude-3-5-sonnet-20241022-v2:0", messages=[])

        events = stratix.get_events("model.invoke")
        assert "latency_ms" in events[0]["payload"]

    def test_converse_error_propagation(self):
        stratix = MockStratix()
        adapter = AWSBedrockAdapter(stratix=stratix)
        adapter.connect()

        def failing_converse(**kwargs):
            raise ValueError("Converse error")

        client = MockBedrockClient()
        adapter.connect_client(client)
        client.converse = adapter._wrap_converse(failing_converse)

        with pytest.raises(ValueError, match="Converse error"):
            client.converse(modelId="anthropic.claude-3-5-sonnet-20241022-v2:0")

    def test_unknown_family_fallback(self):
        stratix = MockStratix()
        adapter = AWSBedrockAdapter(stratix=stratix)
        adapter.connect()

        def invoke_unknown(**kwargs):
            body = json.dumps({"result": "Hello"}).encode()
            return {"body": MockStreamBody(body)}

        client = MockBedrockClient()
        adapter.connect_client(client)
        client.invoke_model = adapter._wrap_invoke_model(invoke_unknown)

        client.invoke_model(modelId="unknown.model-v1:0")

        events = stratix.get_events("model.invoke")
        assert len(events) == 1


# ============================================================
# Output extraction tests
# ============================================================


class TestExtractInvokeOutput:
    """Tests for AWSBedrockAdapter._extract_invoke_output()."""

    def test_anthropic_output(self):
        body = {
            "content": [{"type": "text", "text": "Hello world"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = AWSBedrockAdapter._extract_invoke_output(body, "anthropic")
        assert result == {"role": "assistant", "content": "Hello world"}

    def test_anthropic_multi_block(self):
        body = {
            "content": [
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ],
        }
        result = AWSBedrockAdapter._extract_invoke_output(body, "anthropic")
        assert result == {"role": "assistant", "content": "Part 1\nPart 2"}

    def test_anthropic_non_text_blocks_skipped(self):
        body = {
            "content": [
                {"type": "tool_use", "id": "t1"},
                {"type": "text", "text": "Only text"},
            ],
        }
        result = AWSBedrockAdapter._extract_invoke_output(body, "anthropic")
        assert result == {"role": "assistant", "content": "Only text"}

    def test_meta_output(self):
        body = {"generation": "Meta response", "prompt_token_count": 10}
        result = AWSBedrockAdapter._extract_invoke_output(body, "meta")
        assert result == {"role": "assistant", "content": "Meta response"}

    def test_mistral_output(self):
        body = {"generation": "Mistral response"}
        result = AWSBedrockAdapter._extract_invoke_output(body, "mistral")
        assert result == {"role": "assistant", "content": "Mistral response"}

    def test_cohere_output(self):
        body = {"generations": [{"text": "Cohere response"}]}
        result = AWSBedrockAdapter._extract_invoke_output(body, "cohere")
        assert result == {"role": "assistant", "content": "Cohere response"}

    def test_cohere_empty_generations(self):
        body = {"generations": []}
        result = AWSBedrockAdapter._extract_invoke_output(body, "cohere")
        assert result is None

    def test_amazon_output(self):
        body = {"results": [{"outputText": "Amazon Titan response"}]}
        result = AWSBedrockAdapter._extract_invoke_output(body, "amazon")
        assert result == {"role": "assistant", "content": "Amazon Titan response"}

    def test_amazon_empty_results(self):
        body = {"results": []}
        result = AWSBedrockAdapter._extract_invoke_output(body, "amazon")
        assert result is None

    def test_unknown_family_generation(self):
        body = {"generation": "Generic response"}
        result = AWSBedrockAdapter._extract_invoke_output(body, "unknown")
        assert result == {"role": "assistant", "content": "Generic response"}

    def test_unknown_family_completion(self):
        body = {"completion": "Completion response"}
        result = AWSBedrockAdapter._extract_invoke_output(body, "unknown")
        assert result == {"role": "assistant", "content": "Completion response"}

    def test_unknown_family_output_text(self):
        body = {"outputText": "Output text"}
        result = AWSBedrockAdapter._extract_invoke_output(body, "unknown")
        assert result == {"role": "assistant", "content": "Output text"}

    def test_empty_body(self):
        assert AWSBedrockAdapter._extract_invoke_output({}, "anthropic") is None

    def test_none_body(self):
        assert AWSBedrockAdapter._extract_invoke_output(None, "anthropic") is None

    def test_truncation_at_10k(self):
        long_text = "x" * 20_000
        body = {"generation": long_text}
        result = AWSBedrockAdapter._extract_invoke_output(body, "meta")
        assert len(result["content"]) == 10_000

    def test_anthropic_empty_content(self):
        body = {"content": []}
        result = AWSBedrockAdapter._extract_invoke_output(body, "anthropic")
        assert result is None


class TestExtractConverseOutput:
    """Tests for AWSBedrockAdapter._extract_converse_output()."""

    def test_single_text_block(self):
        response = {
            "output": {"message": {"content": [{"text": "Hello"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }
        result = AWSBedrockAdapter._extract_converse_output(response)
        assert result == {"role": "assistant", "content": "Hello"}

    def test_multi_text_blocks(self):
        response = {
            "output": {"message": {"content": [
                {"text": "Part A"},
                {"text": "Part B"},
            ]}},
        }
        result = AWSBedrockAdapter._extract_converse_output(response)
        assert result == {"role": "assistant", "content": "Part A\nPart B"}

    def test_non_text_blocks_skipped(self):
        response = {
            "output": {"message": {"content": [
                {"toolUse": {"name": "search"}},
                {"text": "Text only"},
            ]}},
        }
        result = AWSBedrockAdapter._extract_converse_output(response)
        assert result == {"role": "assistant", "content": "Text only"}

    def test_empty_content(self):
        response = {"output": {"message": {"content": []}}}
        result = AWSBedrockAdapter._extract_converse_output(response)
        assert result is None

    def test_no_message(self):
        response = {"output": {}}
        result = AWSBedrockAdapter._extract_converse_output(response)
        assert result is None

    def test_no_output(self):
        response = {}
        result = AWSBedrockAdapter._extract_converse_output(response)
        assert result is None

    def test_truncation_at_10k(self):
        long_text = "y" * 20_000
        response = {"output": {"message": {"content": [{"text": long_text}]}}}
        result = AWSBedrockAdapter._extract_converse_output(response)
        assert len(result["content"]) == 10_000


# ============================================================
# Input message extraction on streaming methods
# ============================================================


class TestFinishReasonAndResponseId:
    """Tests for finish_reason and response_id extraction."""

    def test_anthropic_stop_reason_captured(self):
        """Test that Anthropic stop_reason is extracted as finish_reason."""
        stratix = MockStratix()
        adapter = AWSBedrockAdapter(stratix=stratix)
        adapter.connect()

        def invoke_with_stop(**kwargs):
            body = json.dumps({
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "Hello"}],
                "stop_reason": "end_turn",
            }).encode()
            return {"body": MockStreamBody(body)}

        client = MockBedrockClient()
        adapter.connect_client(client)
        client.invoke_model = adapter._wrap_invoke_model(invoke_with_stop)

        client.invoke_model(modelId="anthropic.claude-3-5-sonnet-20241022-v2:0")

        events = stratix.get_events("model.invoke")
        assert events[0]["payload"].get("finish_reason") == "end_turn"

    def test_converse_stop_reason_captured(self):
        """Test that Converse API stopReason is extracted as finish_reason."""
        stratix = MockStratix()
        adapter = AWSBedrockAdapter(stratix=stratix)
        adapter.connect()

        def converse_with_stop(**kwargs):
            return {
                "output": {"message": {"content": [{"text": "Hello"}]}},
                "usage": {"inputTokens": 100, "outputTokens": 50},
                "stopReason": "end_turn",
            }

        client = MockBedrockClient()
        adapter.connect_client(client)
        client.converse = adapter._wrap_converse(converse_with_stop)

        client.converse(modelId="anthropic.claude-3-5-sonnet-20241022-v2:0", messages=[])

        events = stratix.get_events("model.invoke")
        assert events[0]["payload"].get("finish_reason") == "end_turn"

    def test_response_id_from_metadata(self):
        """Test that response_id is extracted from ResponseMetadata."""
        stratix = MockStratix()
        adapter = AWSBedrockAdapter(stratix=stratix)
        adapter.connect()

        def converse_with_metadata(**kwargs):
            return {
                "output": {"message": {"content": [{"text": "Hello"}]}},
                "usage": {"inputTokens": 100, "outputTokens": 50},
                "ResponseMetadata": {"RequestId": "req-abc-123"},
            }

        client = MockBedrockClient()
        adapter.connect_client(client)
        client.converse = adapter._wrap_converse(converse_with_metadata)

        client.converse(modelId="anthropic.claude-3-5-sonnet-20241022-v2:0", messages=[])

        events = stratix.get_events("model.invoke")
        assert events[0]["payload"].get("response_id") == "req-abc-123"


class TestStreamingInputMessages:
    """Tests that streaming methods pass input_messages through."""

    def test_invoke_stream_passes_input_messages(self):
        stratix = MockStratix()
        adapter = AWSBedrockAdapter(stratix=stratix)
        adapter.connect()

        client = MockBedrockClient()
        adapter.connect_client(client)

        body = json.dumps({
            "messages": [{"role": "user", "content": "Hello"}],
        }).encode()
        client.invoke_model_with_response_stream(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            body=body,
        )

        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload.get("streaming") is True
        assert payload.get("messages") is not None

    def test_converse_stream_passes_input_messages(self):
        stratix = MockStratix()
        adapter = AWSBedrockAdapter(stratix=stratix)
        adapter.connect()

        client = MockBedrockClient()
        adapter.connect_client(client)

        client.converse_stream(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": [{"text": "Hello"}]}],
        )

        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload.get("streaming") is True


# ============================================================
# Output message in invoke_model and converse events
# ============================================================


class TestOutputMessageInEvents:
    """Tests that output_message appears in emitted events."""

    def test_invoke_model_includes_output_message(self):
        stratix = MockStratix()
        adapter = AWSBedrockAdapter(stratix=stratix)
        adapter.connect()

        client = MockBedrockClient()
        adapter.connect_client(client)

        client.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            body=b'{}',
        )

        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload.get("output_message") == {"role": "assistant", "content": "Hello"}

    def test_converse_includes_output_message(self):
        stratix = MockStratix()
        adapter = AWSBedrockAdapter(stratix=stratix)
        adapter.connect()

        client = MockBedrockClient()
        adapter.connect_client(client)

        client.converse(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[],
        )

        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload.get("output_message") == {"role": "assistant", "content": "Hello"}
