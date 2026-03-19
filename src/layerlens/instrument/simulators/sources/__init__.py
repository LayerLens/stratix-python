"""Source formatters for 12 ingestion sources."""

from .base import BaseSourceFormatter, ProviderProfile

_SOURCE_REGISTRY: dict[str, type[BaseSourceFormatter]] = {}


def register_source(name: str, formatter_class: type[BaseSourceFormatter]) -> None:
    """Register a source formatter."""
    _SOURCE_REGISTRY[name] = formatter_class


def get_source_formatter(name: str) -> BaseSourceFormatter:
    """Get a source formatter instance by name."""
    if not _SOURCE_REGISTRY:
        _load_sources()
    cls = _SOURCE_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown source format: {name}. Available: {list(_SOURCE_REGISTRY.keys())}"
        )
    return cls()


def list_sources() -> list[str]:
    """List all registered source format names."""
    if not _SOURCE_REGISTRY:
        _load_sources()
    return sorted(_SOURCE_REGISTRY.keys())


def _load_sources() -> None:
    """Lazy-load all source formatters to populate registry."""
    from .agentforce_otlp import AgentForceOTLPFormatter
    from .agentforce_soql import AgentForceSOQLFormatter
    from .anthropic_provider import AnthropicSourceFormatter
    from .azure_openai_provider import AzureOpenAISourceFormatter
    from .bedrock_provider import BedrockSourceFormatter
    from .generic_otel import GenericOTelFormatter
    from .google_vertex_provider import GoogleVertexSourceFormatter
    from .jsonl_provider import JSONLSourceFormatter
    from .langfuse_provider import LangfuseSourceFormatter
    from .litellm_provider import LiteLLMSourceFormatter
    from .ollama_provider import OllamaSourceFormatter
    from .openai_provider import OpenAISourceFormatter

    register_source("generic_otel", GenericOTelFormatter)
    register_source("agentforce_otlp", AgentForceOTLPFormatter)
    register_source("agentforce_soql", AgentForceSOQLFormatter)
    register_source("openai", OpenAISourceFormatter)
    register_source("anthropic", AnthropicSourceFormatter)
    register_source("azure_openai", AzureOpenAISourceFormatter)
    register_source("bedrock", BedrockSourceFormatter)
    register_source("google_vertex", GoogleVertexSourceFormatter)
    register_source("ollama", OllamaSourceFormatter)
    register_source("litellm", LiteLLMSourceFormatter)
    register_source("langfuse", LangfuseSourceFormatter)
    register_source("jsonl", JSONLSourceFormatter)


__all__ = [
    "BaseSourceFormatter",
    "ProviderProfile",
    "get_source_formatter",
    "list_sources",
    "register_source",
]
