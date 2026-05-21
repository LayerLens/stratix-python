"""Compatibility re-exports for the legacy ``layerlens.adapters.providers`` import path.

LAY-3326 (ADP-070) requires that callers can do::

    from layerlens.adapters.providers import AzureOpenAIAdapter

and call ``connect_client(client)`` / ``health_check()`` on the result. The
canonical implementation lives at :mod:`layerlens.instrument.adapters.providers`
under cleaner names (``*Provider`` with ``.connect()``); this module is a thin
shim that wraps each provider with the legacy API surface so the AC bullets
are verifiable without forking the code.

Note: the wrapper classes are intentionally minimal. ``health_check()`` returns
a small dataclass mirroring ``AdapterHealth`` so this module has no dependency
on any untracked legacy adapter base classes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from dataclasses import dataclass

from layerlens.instrument.adapters.providers.ollama import OllamaProvider
from layerlens.instrument.adapters.providers.openai import OpenAIProvider
from layerlens.instrument.adapters.providers.bedrock import BedrockProvider
from layerlens.instrument.adapters.providers.litellm import LiteLLMProvider

# Canonical providers — single source of truth.
from layerlens.instrument.adapters.providers.anthropic import AnthropicProvider
from layerlens.instrument.adapters.providers.azure_openai import AzureOpenAIProvider
from layerlens.instrument.adapters.providers.google_vertex import GoogleVertexProvider


class AdapterStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class AdapterHealth:
    """Minimal health snapshot returned by ``*Adapter.health_check()``.

    Matches the shape used by the legacy ``layerlens.adapters.base.AdapterHealth``
    pydantic model — ``status`` + ``framework_name`` + ``adapter_version`` are
    the fields callers actually read.
    """

    status: AdapterStatus
    framework_name: str
    adapter_version: str = "1.6.0"
    framework_version: Optional[str] = None
    message: Optional[str] = None


class _LegacyProviderAdapter:
    """Thin wrapper exposing the legacy ``connect_client`` / ``health_check`` API.

    Each subclass binds to a concrete canonical provider class and a stable
    framework name used in the health snapshot.
    """

    _provider_cls: type
    _framework: str

    def __init__(self) -> None:
        self._provider: Any = self._provider_cls()
        self._client: Any = None

    def connect_client(self, client: Any) -> Any:
        """Activate instrumentation against a real SDK client.

        Per LAY-3326 AC bullet: ``LLM calls are traced with token usage,
        latency, and cost`` — that behaviour comes from the underlying
        canonical provider's ``connect``.
        """
        result = self._provider.connect(client)
        self._client = client
        return result

    def disconnect(self) -> None:
        self._provider.disconnect()
        self._client = None

    def health_check(self) -> AdapterHealth:
        status = AdapterStatus.HEALTHY if self._client is not None else AdapterStatus.DISCONNECTED
        return AdapterHealth(
            status=status,
            framework_name=self._framework,
        )

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    @property
    def provider(self) -> Any:
        """Escape hatch — access the underlying canonical provider directly."""
        return self._provider


class OpenAIAdapter(_LegacyProviderAdapter):
    _provider_cls = OpenAIProvider
    _framework = "openai"


class AnthropicAdapter(_LegacyProviderAdapter):
    _provider_cls = AnthropicProvider
    _framework = "anthropic"


class AzureOpenAIAdapter(_LegacyProviderAdapter):
    _provider_cls = AzureOpenAIProvider
    _framework = "azure-openai"


class VertexAIAdapter(_LegacyProviderAdapter):
    _provider_cls = GoogleVertexProvider
    _framework = "google-vertex"


class BedrockAdapter(_LegacyProviderAdapter):
    _provider_cls = BedrockProvider
    _framework = "aws-bedrock"


class OllamaAdapter(_LegacyProviderAdapter):
    _provider_cls = OllamaProvider
    _framework = "ollama"


class LiteLLMAdapter(_LegacyProviderAdapter):
    _provider_cls = LiteLLMProvider
    _framework = "litellm"


__all__ = [
    "AdapterHealth",
    "AdapterStatus",
    "AnthropicAdapter",
    "AzureOpenAIAdapter",
    "BedrockAdapter",
    "LiteLLMAdapter",
    "OllamaAdapter",
    "OpenAIAdapter",
    "VertexAIAdapter",
]
