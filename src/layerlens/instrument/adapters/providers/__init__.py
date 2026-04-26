"""LLM provider adapters for the LayerLens Instrument layer.

Each provider adapter wraps a vendor SDK client (or, for routers like
LiteLLM, a callback registry) to intercept API calls and emit
``model.invoke``, ``cost.record``, ``tool.call``, and
``policy.violation`` events through the LayerLens telemetry pipeline.

Adapters available:

* ``openai_adapter`` — OpenAI Python SDK (``openai >= 1.30``)
* ``anthropic_adapter`` — Anthropic Python SDK (``anthropic >= 0.30``)
* ``azure_openai_adapter`` — Azure OpenAI (``openai >= 1.30``)
* ``bedrock_adapter`` — AWS Bedrock (``boto3``)
* ``google_vertex_adapter`` — Google Vertex AI (``google-cloud-aiplatform``)
* ``ollama_adapter`` — Ollama (``ollama``)
* ``litellm`` — LiteLLM multi-provider router (``litellm``); also
  importable as the legacy flat ``litellm_adapter`` module.
* ``cohere_adapter`` — Cohere (``cohere`` >= 5)
* ``mistral_adapter`` — Mistral AI (``mistralai`` >= 1)

Importing this package does NOT import any vendor SDK; modules are
loaded on demand via :class:`AdapterRegistry` or via the lazy
``__getattr__`` shim below.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - pure typing aid, no runtime cost
    # Re-exported lazily at runtime via ``__getattr__`` (see below). The
    # eager import here exists solely so static analysers (mypy, pyright,
    # IDE imports) can resolve ``providers.LiteLLMAdapter`` without
    # forcing the vendor SDK to be importable.
    from layerlens.instrument.adapters.providers.litellm import LiteLLMAdapter as LiteLLMAdapter

# Public re-exports surfaced at the ``providers`` package level. Names are
# mapped to ``(submodule, attribute)`` and resolved on first access via
# :func:`__getattr__` (PEP 562) so that ``import layerlens.instrument.adapters.providers``
# stays free of vendor-SDK imports — the lazy-import contract enforced
# by ``tests/instrument/test_lazy_imports.py``.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "LiteLLMAdapter": ("layerlens.instrument.adapters.providers.litellm", "LiteLLMAdapter"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute resolver for vendor-SDK-backed adapters."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    from importlib import import_module

    module = import_module(module_name)
    return getattr(module, attr_name)
