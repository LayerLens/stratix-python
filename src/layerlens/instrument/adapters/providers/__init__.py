"""LLM provider adapters for the LayerLens Instrument layer.

Each provider adapter wraps a vendor SDK client to intercept API calls
and emit ``model.invoke``, ``cost.record``, ``tool.call``, and
``policy.violation`` events through the LayerLens telemetry pipeline.

Adapters available:

* ``openai_adapter`` — OpenAI Python SDK (``openai >= 1.30``)
* ``anthropic_adapter`` — Anthropic Python SDK (``anthropic >= 0.30``)
* ``azure_openai_adapter`` — Azure OpenAI (``openai >= 1.30``)
* ``bedrock_adapter`` — AWS Bedrock (``boto3``)
* ``google_vertex_adapter`` — Google Vertex AI Gemini-only legacy adapter
  (``google-cloud-aiplatform``)
* ``vertex`` — Google Vertex AI multi-vendor adapter (Gemini, Anthropic
  on Vertex, Llama on Vertex). New M3 sub-package; preferred for new
  integrations. (``google-cloud-aiplatform``)
* ``ollama_adapter`` — Ollama (``ollama``)
* ``litellm_adapter`` — LiteLLM proxy (``litellm``)
* ``cohere_adapter`` — Cohere (``cohere`` >= 5)
* ``mistral_adapter`` — Mistral AI (``mistralai`` >= 1)

Importing this package does NOT import any vendor SDK; modules are
loaded on demand via :class:`AdapterRegistry` or by an explicit
``from layerlens.instrument.adapters.providers.<name> import …`` call
in user code.

The ``vertex`` sub-package is exposed lazily via ``__getattr__`` so
that ``from layerlens.instrument.adapters.providers import VertexAdapter``
also works without forcing every other provider sub-tree to load at
``import providers`` time.
"""

from __future__ import annotations

from typing import Any, List

# Names this package will lazily resolve via ``__getattr__``. Keeping
# the list explicit (rather than blanket-forwarding) means we never
# accidentally import an optional vendor SDK as a side effect of
# attribute access on the package.
_LAZY_ATTRS = {
    "VertexAdapter": (
        "layerlens.instrument.adapters.providers.vertex",
        "VertexAdapter",
    ),
}


def __getattr__(name: str) -> Any:
    """Lazy attribute hook for top-level adapter exports."""
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(
            f"module 'layerlens.instrument.adapters.providers' has no attribute {name!r}"
        )
    module_path, attr_name = target
    import importlib

    module = importlib.import_module(module_path)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))
