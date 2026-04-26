"""LLM provider adapters for the LayerLens Instrument layer.

Each provider adapter wraps a vendor SDK client to intercept API calls
and emit ``model.invoke``, ``cost.record``, ``tool.call``, and
``policy.violation`` events through the LayerLens telemetry pipeline.

Adapters available:

* ``openai_adapter`` — OpenAI Python SDK (``openai >= 1.30``)
* ``anthropic_adapter`` — Anthropic Python SDK (``anthropic >= 0.30``)
* ``azure_openai_adapter`` — Azure OpenAI (``openai >= 1.30``)
* ``bedrock`` — AWS Bedrock (``boto3``) — package layout, the legacy
  ``bedrock_adapter`` module is preserved as a thin shim
* ``google_vertex_adapter`` — Google Vertex AI (``google-cloud-aiplatform``)
* ``ollama_adapter`` — Ollama (``ollama``)
* ``litellm_adapter`` — LiteLLM proxy (``litellm``)
* ``cohere_adapter`` — Cohere (``cohere`` >= 5)
* ``mistral_adapter`` — Mistral AI (``mistralai`` >= 1)

Importing this package does NOT import any vendor SDK; modules are
loaded on demand via :class:`AdapterRegistry` or via PEP 562 lazy
attribute access on this package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type-only imports so lazy-import guards (test_lazy_imports) and
    # default-install guards (test_default_install) keep passing without
    # forcing every consumer to install boto3 just to do
    # ``layerlens.instrument.adapters.providers.AWSBedrockAdapter``.
    from layerlens.instrument.adapters.providers.bedrock import AWSBedrockAdapter

__all__ = ["AWSBedrockAdapter"]


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access for vendor-coupled adapter classes.

    Resolves ``providers.AWSBedrockAdapter`` on first access without
    pulling the boto3 import chain at package-import time.
    """
    if name == "AWSBedrockAdapter":
        from layerlens.instrument.adapters.providers.bedrock import AWSBedrockAdapter

        return AWSBedrockAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
