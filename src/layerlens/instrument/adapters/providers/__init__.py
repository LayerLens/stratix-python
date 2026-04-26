"""LLM provider adapters for the LayerLens Instrument layer.

Each provider adapter wraps a vendor SDK client to intercept API calls
and emit ``model.invoke``, ``cost.record``, ``tool.call``, and
``policy.violation`` events through the LayerLens telemetry pipeline.

Adapters delivered in this branch:

* ``ollama_adapter`` — Ollama (``ollama >= 0.2``). Local-only; default
  endpoint ``http://localhost:11434``. ``api_cost_usd`` is always
  ``0.0`` (self-hosted); optional ``infra_cost_usd`` from compute
  duration when ``cost_per_second`` is configured.

Sister provider adapters land via M3 fan-out PRs (OpenAI, Anthropic,
Azure OpenAI, AWS Bedrock, Google Vertex, LiteLLM, Cohere, Mistral).

Importing this package does NOT import any vendor SDK; concrete
adapter modules are loaded lazily on attribute access via
:func:`__getattr__`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

if TYPE_CHECKING:
    from layerlens.instrument.adapters.providers.ollama_adapter import OllamaAdapter

__all__: Tuple[str, ...] = ("OllamaAdapter",)


def __getattr__(name: str) -> Any:
    """Lazy attribute access for provider adapter classes.

    Allows ``from layerlens.instrument.adapters.providers import OllamaAdapter``
    without importing the underlying vendor SDK at package-import time.
    The actual import (and any vendor-SDK side effects) is deferred
    until the symbol is first referenced.
    """
    if name == "OllamaAdapter":
        from layerlens.instrument.adapters.providers.ollama_adapter import (
            OllamaAdapter,
        )

        return OllamaAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
