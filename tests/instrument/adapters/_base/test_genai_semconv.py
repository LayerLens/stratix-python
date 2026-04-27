"""OpenTelemetry GenAI semantic-convention helper tests.

Pin the spec ``07-otel-genai-semantic-conventions.md`` contract for the
shared helper module
``layerlens.instrument.adapters._base.genai_semconv``. Coverage:

* Every ``gen_ai.*`` attribute key matches the upstream OTel semconv
  spelling exactly (no LayerLens-side renaming).
* :func:`detect_gen_ai_system` maps every major provider's adapter
  class / framework name to the canonical ``gen_ai.system`` value.
* :func:`stamp_genai_attributes` produces value types matching the
  spec (ints for token counts, list-of-string for finish reasons,
  string for system / model, etc.).
* The helper is **additive** — it never removes existing custom keys
  (CLAUDE.md "complete means complete": dashboards built on the legacy
  attribute set must keep working).
* The helper is **safe** — payloads with missing fields, malformed
  response objects, or unknown providers do not crash.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from layerlens.instrument.adapters._base.genai_semconv import (
    GEN_AI_SYSTEM,
    SYSTEM_COHERE,
    SYSTEM_OLLAMA,
    SYSTEM_OPENAI,
    OPERATION_CHAT,
    SYSTEM_LITELLM,
    SYSTEM_MISTRAL,
    GEN_AI_AGENT_ID,
    OPERATION_EMBED,
    GEN_AI_TOOL_NAME,
    SYSTEM_ANTHROPIC,
    GEN_AI_AGENT_NAME,
    SYSTEM_GCP_GEMINI,
    SYSTEM_GCP_VERTEX,
    GEN_AI_RESPONSE_ID,
    SYSTEM_AWS_BEDROCK,
    GEN_AI_TOOL_CALL_ID,
    SYSTEM_AZURE_OPENAI,
    AWS_BEDROCK_AGENT_ID,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_OPERATION_NAME,
    GEN_AI_RESPONSE_MODEL,
    SYSTEM_FALLBACK_OTHER,
    AWS_BEDROCK_GUARDRAIL_ID,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    AWS_BEDROCK_KNOWLEDGE_BASE_ID,
    GEN_AI_REQUEST_STOP_SEQUENCES,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_OPENAI_REQUEST_RESPONSE_FORMAT,
    GEN_AI_ANTHROPIC_CACHE_READ_INPUT_TOKENS,
    GEN_AI_OPENAI_RESPONSE_SYSTEM_FINGERPRINT,
    GEN_AI_ANTHROPIC_CACHE_CREATION_INPUT_TOKENS,
    detect_operation,
    detect_gen_ai_system,
    stamp_genai_attributes,
)

# ---------------------------------------------------------------------------
# 1. Constants verbatim from the OTel GenAI semantic-conventions spec.
# ---------------------------------------------------------------------------


class TestGenAiAttributeNamesMatchSpec:
    """Pin every ``gen_ai.*`` constant to the upstream spec spelling.

    The OTel GenAI semconv defines exact attribute names; LayerLens MUST
    NOT diverge. If the upstream spec renames a key, the test breaks
    here and the constant + downstream consumers update together.
    """

    def test_system_and_provider_keys(self) -> None:
        assert GEN_AI_SYSTEM == "gen_ai.system"
        assert GEN_AI_PROVIDER_NAME == "gen_ai.provider.name"
        assert GEN_AI_OPERATION_NAME == "gen_ai.operation.name"

    def test_request_keys(self) -> None:
        assert GEN_AI_REQUEST_MODEL == "gen_ai.request.model"
        assert GEN_AI_REQUEST_MAX_TOKENS == "gen_ai.request.max_tokens"
        assert GEN_AI_REQUEST_TEMPERATURE == "gen_ai.request.temperature"
        assert GEN_AI_REQUEST_TOP_P == "gen_ai.request.top_p"
        assert GEN_AI_REQUEST_STOP_SEQUENCES == "gen_ai.request.stop_sequences"

    def test_response_keys(self) -> None:
        assert GEN_AI_RESPONSE_ID == "gen_ai.response.id"
        assert GEN_AI_RESPONSE_MODEL == "gen_ai.response.model"
        assert GEN_AI_RESPONSE_FINISH_REASONS == "gen_ai.response.finish_reasons"

    def test_usage_keys(self) -> None:
        assert GEN_AI_USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"
        assert GEN_AI_USAGE_OUTPUT_TOKENS == "gen_ai.usage.output_tokens"

    def test_tool_keys(self) -> None:
        assert GEN_AI_TOOL_NAME == "gen_ai.tool.name"
        assert GEN_AI_TOOL_CALL_ID == "gen_ai.tool.call.id"

    def test_agent_keys(self) -> None:
        assert GEN_AI_AGENT_ID == "gen_ai.agent.id"
        assert GEN_AI_AGENT_NAME == "gen_ai.agent.name"

    def test_provider_specific_keys(self) -> None:
        assert (
            GEN_AI_OPENAI_RESPONSE_SYSTEM_FINGERPRINT
            == "gen_ai.openai.response.system_fingerprint"
        )
        assert GEN_AI_OPENAI_REQUEST_RESPONSE_FORMAT == "gen_ai.openai.request.response_format"
        assert (
            GEN_AI_ANTHROPIC_CACHE_CREATION_INPUT_TOKENS
            == "gen_ai.anthropic.cache_creation_input_tokens"
        )
        assert (
            GEN_AI_ANTHROPIC_CACHE_READ_INPUT_TOKENS
            == "gen_ai.anthropic.cache_read_input_tokens"
        )
        # AWS Bedrock per spec §4.3 lives under aws.bedrock.* (not gen_ai.*)
        assert AWS_BEDROCK_GUARDRAIL_ID == "aws.bedrock.guardrail.id"
        assert AWS_BEDROCK_KNOWLEDGE_BASE_ID == "aws.bedrock.knowledge_base.id"
        assert AWS_BEDROCK_AGENT_ID == "aws.bedrock.agent.id"


# ---------------------------------------------------------------------------
# 2. System detection covers every adapter currently shipped.
# ---------------------------------------------------------------------------


class TestSystemDetection:
    """Every LLM provider / framework adapter currently in the SDK must
    map to a canonical ``gen_ai.system`` value via
    :func:`detect_gen_ai_system`.
    """

    @pytest.mark.parametrize(
        "needle, expected",
        [
            # Provider class names.
            ("OpenAIAdapter", SYSTEM_OPENAI),
            ("openai", SYSTEM_OPENAI),
            ("AzureOpenAIAdapter", SYSTEM_AZURE_OPENAI),
            ("azure_openai", SYSTEM_AZURE_OPENAI),
            ("AnthropicAdapter", SYSTEM_ANTHROPIC),
            ("anthropic", SYSTEM_ANTHROPIC),
            ("BedrockAdapter", SYSTEM_AWS_BEDROCK),
            ("aws_bedrock", SYSTEM_AWS_BEDROCK),
            ("bedrock_agents", SYSTEM_AWS_BEDROCK),
            ("GoogleVertexAdapter", SYSTEM_GCP_VERTEX),
            ("google_vertex", SYSTEM_GCP_VERTEX),
            ("vertex_ai", SYSTEM_GCP_VERTEX),
            ("google_adk", SYSTEM_GCP_GEMINI),
            ("CohereAdapter", SYSTEM_COHERE),
            ("cohere", SYSTEM_COHERE),
            ("MistralAdapter", SYSTEM_MISTRAL),
            ("mistral", SYSTEM_MISTRAL),
            ("OllamaAdapter", SYSTEM_OLLAMA),
            ("ollama", SYSTEM_OLLAMA),
            ("LiteLLMAdapter", SYSTEM_LITELLM),
            ("litellm", SYSTEM_LITELLM),
        ],
    )
    def test_known_systems(self, needle: str, expected: str) -> None:
        assert detect_gen_ai_system(needle) == expected

    def test_unknown_returns_other_fallback(self) -> None:
        assert detect_gen_ai_system("totally_unknown_runtime") == SYSTEM_FALLBACK_OTHER

    def test_none_and_empty_return_other_fallback(self) -> None:
        assert detect_gen_ai_system(None) == SYSTEM_FALLBACK_OTHER
        assert detect_gen_ai_system("") == SYSTEM_FALLBACK_OTHER

    def test_fallback_value_matches_spec(self) -> None:
        # Per spec, ``_OTHER`` is the documented enum fallback when the
        # system is not enumerated in the registry.
        assert SYSTEM_FALLBACK_OTHER == "_OTHER"


# ---------------------------------------------------------------------------
# 3. detect_operation infers the operation from request shape.
# ---------------------------------------------------------------------------


class TestDetectOperation:
    def test_chat_default(self) -> None:
        assert detect_operation({}, request_kwargs={"messages": []}) == OPERATION_CHAT

    def test_explicit_operation_in_payload_wins(self) -> None:
        assert (
            detect_operation({"operation": "text_completion"}, request_kwargs=None)
            == "text_completion"
        )

    def test_embedding_kwargs_routed_to_embed(self) -> None:
        assert (
            detect_operation({}, request_kwargs={"input": ["text"], "encoding_format": "float"})
            == OPERATION_EMBED
        )

    def test_default_when_no_kwargs(self) -> None:
        assert detect_operation({}) == OPERATION_CHAT


# ---------------------------------------------------------------------------
# 4. stamp_genai_attributes produces correct value types and is additive.
# ---------------------------------------------------------------------------


def _sample_response_obj() -> Any:
    """Tiny duck-typed response object stand-in for OpenAI / Anthropic."""

    class Response:
        id = "chatcmpl-abc123"
        model = "gpt-4-0613"
        system_fingerprint = "fp_abc123"
        service_tier = "default"

    return Response()


class TestStampGenAiAttributes:
    def test_chat_payload_stamped_with_full_set(self) -> None:
        payload: Dict[str, Any] = {
            "provider": "openai",
            "model": "gpt-4",
            "parameters": {"temperature": 0.7, "max_tokens": 1000, "top_p": 0.9},
            "prompt_tokens": 150,
            "completion_tokens": 42,
        }
        stamp_genai_attributes(
            payload,
            request_kwargs={"model": "gpt-4", "messages": []},
            response_obj=_sample_response_obj(),
            system=SYSTEM_OPENAI,
        )

        # System / provider / operation always required.
        assert payload[GEN_AI_SYSTEM] == SYSTEM_OPENAI
        assert payload[GEN_AI_PROVIDER_NAME] == SYSTEM_OPENAI
        assert payload[GEN_AI_OPERATION_NAME] == OPERATION_CHAT

        # Request attributes.
        assert payload[GEN_AI_REQUEST_MODEL] == "gpt-4"
        assert payload[GEN_AI_REQUEST_TEMPERATURE] == 0.7
        assert payload[GEN_AI_REQUEST_MAX_TOKENS] == 1000
        assert payload[GEN_AI_REQUEST_TOP_P] == 0.9

        # Usage (must be ints per spec).
        assert payload[GEN_AI_USAGE_INPUT_TOKENS] == 150
        assert payload[GEN_AI_USAGE_OUTPUT_TOKENS] == 42
        assert isinstance(payload[GEN_AI_USAGE_INPUT_TOKENS], int)

        # Response attributes.
        assert payload[GEN_AI_RESPONSE_ID] == "chatcmpl-abc123"
        assert payload[GEN_AI_RESPONSE_MODEL] == "gpt-4-0613"

        # OpenAI-specific.
        assert payload[GEN_AI_OPENAI_RESPONSE_SYSTEM_FINGERPRINT] == "fp_abc123"

    def test_finish_reasons_always_array_of_strings(self) -> None:
        payload: Dict[str, Any] = {"provider": "openai", "finish_reason": "stop"}
        stamp_genai_attributes(payload, system=SYSTEM_OPENAI)
        # Spec mandates list-of-string even for a single reason.
        assert payload[GEN_AI_RESPONSE_FINISH_REASONS] == ["stop"]
        assert isinstance(payload[GEN_AI_RESPONSE_FINISH_REASONS], list)

    def test_stop_sequences_string_coerced_to_list(self) -> None:
        payload: Dict[str, Any] = {"provider": "openai"}
        stamp_genai_attributes(
            payload,
            request_kwargs={"stop": "END"},
            system=SYSTEM_OPENAI,
        )
        assert payload[GEN_AI_REQUEST_STOP_SEQUENCES] == ["END"]

    def test_additive_does_not_remove_existing_keys(self) -> None:
        """CLAUDE.md "complete means complete" — legacy dashboards keep
        reading their custom keys; the stamper never removes them.
        """
        payload: Dict[str, Any] = {
            "provider": "anthropic",
            "model": "claude-3-opus",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "custom_layerlens_field": "preserve_me",
        }
        original_keys = set(payload.keys())
        stamp_genai_attributes(payload, system=SYSTEM_ANTHROPIC)
        # Every original key still present.
        assert original_keys.issubset(set(payload.keys()))
        assert payload["custom_layerlens_field"] == "preserve_me"
        assert payload["provider"] == "anthropic"
        # gen_ai.* keys added alongside.
        assert payload[GEN_AI_SYSTEM] == SYSTEM_ANTHROPIC
        assert payload[GEN_AI_USAGE_INPUT_TOKENS] == 100

    def test_anthropic_cache_tokens_extracted_from_response(self) -> None:
        class _Usage:
            cache_creation_input_tokens = 10
            cache_read_input_tokens = 5

        class _Resp:
            id = "msg_abc"
            usage = _Usage()

        payload: Dict[str, Any] = {"provider": "anthropic", "model": "claude-3-opus"}
        stamp_genai_attributes(payload, response_obj=_Resp(), system=SYSTEM_ANTHROPIC)
        assert payload[GEN_AI_ANTHROPIC_CACHE_CREATION_INPUT_TOKENS] == 10
        assert payload[GEN_AI_ANTHROPIC_CACHE_READ_INPUT_TOKENS] == 5

    def test_handles_missing_fields_gracefully(self) -> None:
        """Adapters frequently emit on the error path with neither
        usage nor response object available — the stamper must not
        crash and must omit (not invent) missing keys.
        """
        payload: Dict[str, Any] = {"provider": "openai"}
        stamp_genai_attributes(payload, request_kwargs=None, response_obj=None)
        # Required keys always present.
        assert GEN_AI_SYSTEM in payload
        assert GEN_AI_OPERATION_NAME in payload
        # Optional keys absent (not None — the helper omits them).
        assert GEN_AI_USAGE_INPUT_TOKENS not in payload
        assert GEN_AI_RESPONSE_ID not in payload
        assert GEN_AI_REQUEST_TEMPERATURE not in payload

    def test_handles_completely_empty_payload(self) -> None:
        payload: Dict[str, Any] = {}
        stamp_genai_attributes(payload)
        # Falls back to _OTHER system + chat operation.
        assert payload[GEN_AI_SYSTEM] == SYSTEM_FALLBACK_OTHER
        assert payload[GEN_AI_OPERATION_NAME] == OPERATION_CHAT

    def test_explicit_operation_override(self) -> None:
        payload: Dict[str, Any] = {"provider": "openai", "model": "text-embedding-3-small"}
        stamp_genai_attributes(
            payload,
            system=SYSTEM_OPENAI,
            operation=OPERATION_EMBED,
        )
        assert payload[GEN_AI_OPERATION_NAME] == OPERATION_EMBED

    def test_langchain_nested_model_dict(self) -> None:
        """LangChain emits ``model`` as a nested dict ``{name, provider}``.
        The stamper must read both layouts so framework adapters work.
        """
        payload: Dict[str, Any] = {
            "model": {"name": "gpt-4", "provider": "openai"},
            "token_usage": {"prompt_tokens": 200, "completion_tokens": 55},
        }
        stamp_genai_attributes(payload)
        assert payload[GEN_AI_REQUEST_MODEL] == "gpt-4"
        assert payload[GEN_AI_SYSTEM] == SYSTEM_OPENAI
        assert payload[GEN_AI_USAGE_INPUT_TOKENS] == 200
        assert payload[GEN_AI_USAGE_OUTPUT_TOKENS] == 55

    def test_aws_bedrock_guardrail_extraction(self) -> None:
        payload: Dict[str, Any] = {"provider": "aws_bedrock", "model": "anthropic.claude-3"}
        stamp_genai_attributes(
            payload,
            request_kwargs={
                "guardrailConfig": {"guardrailIdentifier": "gr-abc"},
                "knowledgeBaseId": "kb-xyz",
            },
            system=SYSTEM_AWS_BEDROCK,
        )
        assert payload[AWS_BEDROCK_GUARDRAIL_ID] == "gr-abc"
        assert payload[AWS_BEDROCK_KNOWLEDGE_BASE_ID] == "kb-xyz"

    def test_returns_payload_for_chaining(self) -> None:
        payload: Dict[str, Any] = {"provider": "cohere"}
        result = stamp_genai_attributes(payload, system=SYSTEM_COHERE)
        # Mutates in place AND returns same reference.
        assert result is payload


# ---------------------------------------------------------------------------
# 5. Centralized stamping via BaseAdapter.emit_dict_event covers every
#    framework adapter that goes through the standard emission path.
# ---------------------------------------------------------------------------


class TestCentralizedStampingHook:
    """Pin the contract that every emission of a GenAI-shaped event
    type through :meth:`BaseAdapter.emit_dict_event` is automatically
    stamped — no per-adapter call is required for coverage.
    """

    def _make_adapter(self) -> Any:
        """Build a minimal BaseAdapter subclass for the hook test."""
        from layerlens.instrument.adapters._base import (
            AdapterInfo,
            BaseAdapter,
            AdapterHealth,
            AdapterStatus,
            ReplayableTrace,
        )

        class _RecordingStratix:
            def __init__(self) -> None:
                self.events: list[tuple[str, Dict[str, Any]]] = []

            def emit(self, event_type: str, payload: Dict[str, Any]) -> None:
                self.events.append((event_type, dict(payload)))

        class _OpenAILikeAdapter(BaseAdapter):
            FRAMEWORK = "openai"
            VERSION = "0.1.0"

            def connect(self) -> None:
                self._connected = True

            def disconnect(self) -> None:
                self._connected = False

            def health_check(self) -> AdapterHealth:
                return AdapterHealth(
                    status=AdapterStatus.HEALTHY,
                    framework_name=self.FRAMEWORK,
                    adapter_version=self.VERSION,
                )

            def get_adapter_info(self) -> AdapterInfo:
                return AdapterInfo(
                    name=type(self).__name__, version=self.VERSION, framework=self.FRAMEWORK
                )

            def serialize_for_replay(self) -> ReplayableTrace:
                return ReplayableTrace(
                    adapter_name=type(self).__name__,
                    framework=self.FRAMEWORK,
                    trace_id="t",
                    events=[],
                    state_snapshots=[],
                    config={},
                )

        client = _RecordingStratix()
        adapter = _OpenAILikeAdapter(stratix=client, org_id="org-test")
        return adapter, client

    def test_model_invoke_auto_stamped(self) -> None:
        adapter, client = self._make_adapter()
        adapter.emit_dict_event(
            "model.invoke",
            {"provider": "openai", "model": "gpt-4", "prompt_tokens": 10},
        )
        assert client.events, "expected emission to reach stratix client"
        _, payload = client.events[-1]
        assert payload[GEN_AI_SYSTEM] == SYSTEM_OPENAI
        assert payload[GEN_AI_REQUEST_MODEL] == "gpt-4"
        assert payload[GEN_AI_USAGE_INPUT_TOKENS] == 10

    def test_embedding_create_auto_stamped_as_embed(self) -> None:
        adapter, client = self._make_adapter()
        adapter.emit_dict_event(
            "embedding.create",
            {"provider": "openai", "model": "text-embedding-3-small", "total_tokens": 5},
        )
        _, payload = client.events[-1]
        assert payload[GEN_AI_OPERATION_NAME] == OPERATION_EMBED

    def test_non_genai_event_types_untouched(self) -> None:
        adapter, client = self._make_adapter()
        adapter.emit_dict_event("agent.input", {"agent_name": "foo", "input": "bar"})
        _, payload = client.events[-1]
        assert GEN_AI_SYSTEM not in payload
        assert GEN_AI_OPERATION_NAME not in payload
