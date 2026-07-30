"""TEL-029 / LAY-2883 coverage: ``finish_reason`` + ``response_id`` across all
seven LLM provider adapters reach ``gen_ai.response.finish_reasons`` and
``gen_ai.response.id``.

These tests inspect each adapter's ``extract_meta`` or equivalent in isolation
to keep them independent of the underlying SDKs (boto3, ollama, etc.).
"""

from __future__ import annotations

from types import SimpleNamespace

from layerlens.instrument._w3c import gen_ai_attributes
from layerlens.instrument.adapters.providers.ollama import OllamaProvider
from layerlens.instrument.adapters.providers.openai import OpenAIProvider
from layerlens.instrument.adapters.providers.bedrock import (
    _bedrock_response_id,
    _extract_invoke_stop_reason,
)
from layerlens.instrument.adapters.providers.litellm import LiteLLMProvider
from layerlens.instrument.adapters.providers.anthropic import AnthropicProvider
from layerlens.instrument.adapters.providers.google_vertex import GoogleVertexProvider


def _otel(provider: str, response_meta: dict) -> dict:
    return gen_ai_attributes(provider=provider, operation="chat", parameters={}, response_meta=response_meta)


class TestOpenAI:
    def test_finish_reason_and_response_id_in_otel(self):
        # Use a real ChatCompletion via the existing conftest helper rather
        # than a SimpleNamespace, since OpenAI's extract_meta has strict
        # field checks.
        from .conftest import make_openai_response

        resp = make_openai_response()
        meta = OpenAIProvider.extract_meta(resp)
        otel = _otel("openai", meta)
        assert otel["gen_ai.response.finish_reasons"] == ["stop"]
        assert otel["gen_ai.response.id"] == "chatcmpl-test"


class TestAnthropic:
    def test_stop_reason_maps_to_finish_reasons(self):
        from .conftest import make_anthropic_response

        resp = make_anthropic_response(stop_reason="end_turn")
        meta = AnthropicProvider.extract_meta(resp)
        otel = _otel("anthropic", meta)
        assert otel["gen_ai.response.finish_reasons"] == ["end_turn"]
        assert otel["gen_ai.response.id"] == "msg-test"


class TestAzureOpenAI:
    def test_inherits_openai_extraction(self):
        # Azure's adapter subclasses OpenAIProvider with no extract_meta
        # override, so coverage is the same shape as OpenAI's.
        from layerlens.instrument.adapters.providers.azure_openai import AzureOpenAIProvider

        from .conftest import make_openai_response

        resp = make_openai_response()
        meta = AzureOpenAIProvider.extract_meta(resp)
        otel = _otel("azure_openai", meta)
        assert otel["gen_ai.response.finish_reasons"] == ["stop"]
        assert otel["gen_ai.response.id"] == "chatcmpl-test"


class TestGoogleVertex:
    def test_finish_reason_and_response_id_in_otel(self):
        # Vertex finish_reason is an enum-like; mock its ``.name`` attr.
        finish_reason = SimpleNamespace(name="STOP")
        cand = SimpleNamespace(finish_reason=finish_reason, content=None)
        response = SimpleNamespace(
            candidates=[cand],
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=20,
                total_token_count=30,
            ),
            response_id="vertex-resp-abc",
        )
        meta = GoogleVertexProvider.extract_meta(response)
        otel = _otel("google_vertex", meta)
        assert otel["gen_ai.response.finish_reasons"] == ["STOP"]
        assert otel["gen_ai.response.id"] == "vertex-resp-abc"


class TestBedrock:
    def test_request_id_extracted_from_response_metadata(self):
        boto3_response = {
            "ResponseMetadata": {"RequestId": "aws-req-id-xyz", "HTTPStatusCode": 200},
        }
        assert _bedrock_response_id(boto3_response) == "aws-req-id-xyz"

    def test_request_id_missing_metadata_returns_none(self):
        assert _bedrock_response_id({}) is None
        assert _bedrock_response_id({"ResponseMetadata": {}}) is None

    def test_stop_reason_extracted_per_family(self):
        # Anthropic body shape.
        assert _extract_invoke_stop_reason({"stop_reason": "end_turn"}, "anthropic") == "end_turn"
        # Cohere body shape.
        assert _extract_invoke_stop_reason({"generations": [{"finish_reason": "COMPLETE"}]}, "cohere") == "COMPLETE"
        # Amazon body shape.
        assert _extract_invoke_stop_reason({"results": [{"completionReason": "FINISH"}]}, "amazon") == "FINISH"
        # Mistral.
        assert _extract_invoke_stop_reason({"outputs": [{"stop_reason": "stop"}]}, "mistral") == "stop"
        # Unknown family.
        assert _extract_invoke_stop_reason({"stop_reason": "x"}, "unknown") is None

    def test_otel_attrs_reach_finish_reasons_and_response_id(self):
        # Simulate the response_meta that _emit_invoke builds before calling
        # gen_ai_attributes.
        meta = {
            "response_id": "aws-req-id-xyz",
            "stop_reason": "end_turn",
            "response_model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        }
        otel = _otel("bedrock", meta)
        assert otel["gen_ai.response.finish_reasons"] == ["end_turn"]
        assert otel["gen_ai.response.id"] == "aws-req-id-xyz"
        assert otel["gen_ai.response.model"] == "anthropic.claude-3-5-sonnet-20241022-v2:0"


class TestOllama:
    def test_done_reason_maps_to_finish_reasons(self):
        # Ollama uses ``done_reason`` instead of ``finish_reason``. Its
        # extract_meta normalises that into the meta dict.
        response = {
            "model": "llama3.1:8b",
            "done_reason": "stop",
            "prompt_eval_count": 12,
            "eval_count": 34,
        }
        meta = OllamaProvider.extract_meta(response)
        otel = _otel("ollama", meta)
        assert otel["gen_ai.response.finish_reasons"] == ["stop"]


class TestLiteLLM:
    def test_delegates_to_openai_extraction(self):
        # LiteLLM reuses OpenAI's extract_meta — same fields surface.
        from .conftest import make_openai_response

        resp = make_openai_response()
        meta = LiteLLMProvider.extract_meta(resp)
        otel = _otel("litellm", meta)
        assert otel["gen_ai.response.finish_reasons"] == ["stop"]
        assert otel["gen_ai.response.id"] == "chatcmpl-test"
