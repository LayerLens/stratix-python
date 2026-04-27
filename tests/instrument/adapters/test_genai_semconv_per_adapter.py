"""Per-adapter OTel GenAI semconv stamping verification.

For every shipped LLM-call adapter (provider + framework), assert that
emissions of ``model.invoke`` (and other LLM-call event types) carry the
canonical ``gen_ai.*`` attribute set defined by spec
``07-otel-genai-semantic-conventions.md``.

The strategy is uniform across adapters: instantiate the adapter with a
recording stratix double, drive a single ``_emit_model_invoke`` /
``emit_dict_event("model.invoke", ...)`` call, and assert that both the
canonical request fields (``gen_ai.system``, ``gen_ai.request.model``,
``gen_ai.usage.input_tokens`` etc.) AND the legacy LayerLens fields
(``provider``, ``model``, ``prompt_tokens`` etc.) are present
side-by-side. CLAUDE.md "complete means complete" — every adapter
listed in spec 07 must be wired.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from layerlens.instrument.adapters._base import BaseAdapter
from layerlens.instrument.adapters._base.genai_semconv import (
    GEN_AI_SYSTEM,
    SYSTEM_COHERE,
    SYSTEM_OLLAMA,
    SYSTEM_OPENAI,
    OPERATION_CHAT,
    SYSTEM_LITELLM,
    SYSTEM_MISTRAL,
    OPERATION_EMBED,
    SYSTEM_ANTHROPIC,
    SYSTEM_GCP_GEMINI,
    SYSTEM_GCP_VERTEX,
    SYSTEM_AWS_BEDROCK,
    SYSTEM_AZURE_OPENAI,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_OPERATION_NAME,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from layerlens.instrument.adapters.providers._base.provider import LLMProviderAdapter


class _RecordingStratix:
    """Stratix client double that records every emitted event."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            event_type, payload = args
            self.events.append({"event_type": event_type, "payload": payload})

    @property
    def org_id(self) -> str:
        return "org-test"


# ---------------------------------------------------------------------------
# Provider adapter wiring matrix.
# ---------------------------------------------------------------------------
#
# Each row is (adapter_module_path, adapter_class_name,
# provider_string_passed_to_emit_model_invoke, expected_gen_ai_system).
# The provider string mirrors what the adapter's wrapper actually passes
# (see e.g. ``openai_adapter.py: provider="openai"``). The expected
# system is the canonical OTel value the helper resolves to.
_PROVIDER_MATRIX = [
    (
        "layerlens.instrument.adapters.providers.openai_adapter",
        "OpenAIAdapter",
        "openai",
        SYSTEM_OPENAI,
    ),
    (
        "layerlens.instrument.adapters.providers.azure_openai_adapter",
        "AzureOpenAIAdapter",
        "azure_openai",
        SYSTEM_AZURE_OPENAI,
    ),
    (
        "layerlens.instrument.adapters.providers.anthropic_adapter",
        "AnthropicAdapter",
        "anthropic",
        SYSTEM_ANTHROPIC,
    ),
    (
        "layerlens.instrument.adapters.providers.bedrock_adapter",
        "AWSBedrockAdapter",
        "aws_bedrock",
        SYSTEM_AWS_BEDROCK,
    ),
    (
        "layerlens.instrument.adapters.providers.google_vertex_adapter",
        "GoogleVertexAdapter",
        "google_vertex",
        SYSTEM_GCP_VERTEX,
    ),
    (
        "layerlens.instrument.adapters.providers.cohere_adapter",
        "CohereAdapter",
        "cohere",
        SYSTEM_COHERE,
    ),
    (
        "layerlens.instrument.adapters.providers.mistral_adapter",
        "MistralAdapter",
        "mistral",
        SYSTEM_MISTRAL,
    ),
    (
        "layerlens.instrument.adapters.providers.ollama_adapter",
        "OllamaAdapter",
        "ollama",
        SYSTEM_OLLAMA,
    ),
    (
        "layerlens.instrument.adapters.providers.litellm_adapter",
        "LiteLLMAdapter",
        "litellm",
        SYSTEM_LITELLM,
    ),
]


def _import_adapter(module_path: str, class_name: str) -> type:
    """Import the adapter class, skipping if its provider SDK is not installed."""
    import importlib

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        pytest.skip(f"{class_name}: provider SDK missing ({exc})")
    return getattr(module, class_name)


@pytest.mark.parametrize(
    "module_path, class_name, provider_arg, expected_system",
    _PROVIDER_MATRIX,
    ids=[row[1] for row in _PROVIDER_MATRIX],
)
class TestProviderAdapterGenAiStamping:
    """Every shipped LLM provider adapter must stamp gen_ai.* attributes
    when ``_emit_model_invoke`` runs.
    """

    def test_emit_model_invoke_stamps_gen_ai_system(
        self,
        module_path: str,
        class_name: str,
        provider_arg: str,
        expected_system: str,
    ) -> None:
        adapter_cls = _import_adapter(module_path, class_name)
        # Skip non-LLMProviderAdapter classes (defensive).
        if not issubclass(adapter_cls, LLMProviderAdapter):
            pytest.skip(f"{class_name} is not an LLMProviderAdapter")

        stratix = _RecordingStratix()
        adapter = adapter_cls(stratix=stratix, org_id="org-test")  # type: ignore[call-arg]
        adapter._connected = True  # bypass connect() — we only want emission

        adapter._emit_model_invoke(
            provider=provider_arg,
            model="some-model",
            parameters={"temperature": 0.7, "max_tokens": 256},
        )

        events = [e for e in stratix.events if e["event_type"] == "model.invoke"]
        assert events, f"{class_name}: no model.invoke event emitted"
        payload = events[-1]["payload"]

        # Canonical gen_ai.* attributes (per-adapter wiring guarantee).
        assert payload[GEN_AI_SYSTEM] == expected_system, (
            f"{class_name}: expected gen_ai.system={expected_system!r} "
            f"got {payload.get(GEN_AI_SYSTEM)!r}"
        )
        assert payload[GEN_AI_PROVIDER_NAME] == expected_system
        assert payload[GEN_AI_OPERATION_NAME] == OPERATION_CHAT
        assert payload[GEN_AI_REQUEST_MODEL] == "some-model"

        # Legacy LayerLens fields preserved alongside (additive contract).
        assert payload["provider"] == provider_arg
        assert payload["model"] == "some-model"
        assert "parameters" in payload

    def test_token_usage_propagates_to_gen_ai_usage(
        self,
        module_path: str,
        class_name: str,
        provider_arg: str,
        expected_system: str,
    ) -> None:
        adapter_cls = _import_adapter(module_path, class_name)
        if not issubclass(adapter_cls, LLMProviderAdapter):
            pytest.skip(f"{class_name} is not an LLMProviderAdapter")

        from layerlens.instrument.adapters.providers._base.tokens import (
            NormalizedTokenUsage,
        )

        stratix = _RecordingStratix()
        adapter = adapter_cls(stratix=stratix, org_id="org-test")  # type: ignore[call-arg]
        adapter._connected = True

        adapter._emit_model_invoke(
            provider=provider_arg,
            model="some-model",
            usage=NormalizedTokenUsage(
                prompt_tokens=42,
                completion_tokens=7,
                total_tokens=49,
            ),
        )

        events = [e for e in stratix.events if e["event_type"] == "model.invoke"]
        assert events
        payload = events[-1]["payload"]
        assert payload[GEN_AI_USAGE_INPUT_TOKENS] == 42
        assert payload[GEN_AI_USAGE_OUTPUT_TOKENS] == 7
        # Legacy fields still present.
        assert payload["prompt_tokens"] == 42
        assert payload["completion_tokens"] == 7


# ---------------------------------------------------------------------------
# Framework adapter wiring matrix.
# ---------------------------------------------------------------------------
#
# Framework adapters call ``self.emit_dict_event("model.invoke", ...)``
# directly. The centralized hook in BaseAdapter.emit_dict_event then
# stamps gen_ai.* using the FRAMEWORK constant. Each row pins the
# adapter's framework string to the expected canonical system.
_FRAMEWORK_MATRIX = [
    ("agno", SYSTEM_OPENAI),  # default — agno is provider-agnostic, runtime detection
    ("autogen", SYSTEM_OPENAI),
    ("bedrock_agents", SYSTEM_AWS_BEDROCK),
    ("crewai", SYSTEM_OPENAI),
    ("google_adk", SYSTEM_GCP_GEMINI),
    ("langchain", "_OTHER"),  # langchain is provider-agnostic
    ("langfuse", "_OTHER"),
    ("langgraph", "_OTHER"),
    ("llama_index", "_OTHER"),
    ("ms_agent_framework", "_OTHER"),
    ("openai_agents", SYSTEM_OPENAI),
    ("pydantic_ai", "_OTHER"),
    ("semantic_kernel", "_OTHER"),
    ("smolagents", "_OTHER"),
    ("strands", "_OTHER"),
    ("salesforce_agentforce", "_OTHER"),
]


class _MinimalFrameworkAdapter(BaseAdapter):
    """Minimal BaseAdapter subclass parameterized by FRAMEWORK string.

    Used to exercise the centralized stamping hook without importing
    optional framework SDKs.
    """

    def __init__(self, framework: str, **kwargs: Any) -> None:
        type(self).FRAMEWORK = framework  # type: ignore[misc]
        super().__init__(**kwargs)

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def health_check(self) -> Any:
        from layerlens.instrument.adapters._base import AdapterHealth, AdapterStatus

        return AdapterHealth(
            status=AdapterStatus.HEALTHY,
            framework_name=self.FRAMEWORK,
            adapter_version=self.VERSION,
        )

    def get_adapter_info(self) -> Any:
        from layerlens.instrument.adapters._base import AdapterInfo

        return AdapterInfo(
            name=type(self).__name__, version=self.VERSION, framework=self.FRAMEWORK
        )

    def serialize_for_replay(self) -> Any:
        from layerlens.instrument.adapters._base import ReplayableTrace

        return ReplayableTrace(
            adapter_name=type(self).__name__,
            framework=self.FRAMEWORK,
            trace_id="t",
            events=[],
            state_snapshots=[],
            config={},
        )


@pytest.mark.parametrize(
    "framework, expected_system",
    _FRAMEWORK_MATRIX,
    ids=[row[0] for row in _FRAMEWORK_MATRIX],
)
def test_framework_adapter_emit_model_invoke_is_stamped(
    framework: str, expected_system: str
) -> None:
    """Every framework adapter that calls emit_dict_event("model.invoke", ...)
    receives gen_ai.* stamping via the centralized hook in BaseAdapter.
    """
    stratix = _RecordingStratix()
    adapter = _MinimalFrameworkAdapter(framework=framework, stratix=stratix)

    adapter.emit_dict_event(
        "model.invoke",
        {
            "framework": framework,
            "model": "some-model",
            "provider": "openai" if expected_system == SYSTEM_OPENAI else None,
            "prompt_tokens": 11,
            "completion_tokens": 3,
        },
    )

    events = [e for e in stratix.events if e["event_type"] == "model.invoke"]
    assert events, f"{framework}: no model.invoke event reached stratix client"
    payload = events[-1]["payload"]

    # gen_ai.system always present (even when fallback is _OTHER).
    assert GEN_AI_SYSTEM in payload, f"{framework}: gen_ai.system missing"
    # Operation name always set.
    assert payload[GEN_AI_OPERATION_NAME] == OPERATION_CHAT
    # Token counts propagated to gen_ai namespace.
    assert payload[GEN_AI_USAGE_INPUT_TOKENS] == 11
    assert payload[GEN_AI_USAGE_OUTPUT_TOKENS] == 3
    # Original framework key preserved.
    assert payload["framework"] == framework


def test_embedding_create_event_routed_to_embed_operation() -> None:
    """The embedding adapter emits ``embedding.create``, which the
    centralized hook stamps as ``gen_ai.operation.name = embeddings``
    per spec §3.1 span naming.
    """
    stratix = _RecordingStratix()
    adapter = _MinimalFrameworkAdapter(framework="openai", stratix=stratix)

    adapter.emit_dict_event(
        "embedding.create",
        {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "total_tokens": 8,
            "batch_size": 4,
        },
    )

    events = [e for e in stratix.events if e["event_type"] == "embedding.create"]
    assert events
    payload = events[-1]["payload"]
    assert payload[GEN_AI_OPERATION_NAME] == OPERATION_EMBED
    assert payload[GEN_AI_SYSTEM] == SYSTEM_OPENAI
    assert payload[GEN_AI_REQUEST_MODEL] == "text-embedding-3-small"
