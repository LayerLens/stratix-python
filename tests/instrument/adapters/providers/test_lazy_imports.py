"""Regression tests for the lazy public API in `providers/__init__.py`.

Mirrors `tests/instrument/adapters/frameworks/test_lazy_imports.py`. Each
`pip install layerlens[providers-*]` extra exists so the default install
stays lean — that contract only holds if importing
`layerlens.instrument.adapters.providers` never eagerly pulls a provider
SDK.
"""

from __future__ import annotations

import sys
import importlib

import pytest

_PROVIDER_SDK_PREFIXES = (
    "openai",
    "anthropic",
    "google",
    "vertexai",
    "boto3",
    "botocore",
    "ollama",
    "litellm",
)


def _purge_provider_sdks() -> None:
    for name in list(sys.modules):
        if name.startswith(_PROVIDER_SDK_PREFIXES):
            del sys.modules[name]
    for name in list(sys.modules):
        if name.startswith("layerlens.instrument.adapters.providers"):
            del sys.modules[name]


def test_providers_package_import_does_not_pull_sdks() -> None:
    """Bare `import layerlens.instrument.adapters.providers` must stay lean."""
    _purge_provider_sdks()
    importlib.import_module("layerlens.instrument.adapters.providers")
    for sdk in _PROVIDER_SDK_PREFIXES:
        assert sdk not in sys.modules, (
            f"providers package import eagerly loaded {sdk!r}; lazy export contract broken"
        )


def test_lazy_getattr_raises_attributeerror_for_unknown_names() -> None:
    _purge_provider_sdks()
    pkg = importlib.import_module("layerlens.instrument.adapters.providers")
    with pytest.raises(AttributeError):
        pkg.ThisProviderDoesNotExist  # noqa: B018


def test_lazy_dir_advertises_all_public_providers() -> None:
    _purge_provider_sdks()
    pkg = importlib.import_module("layerlens.instrument.adapters.providers")
    advertised = set(dir(pkg))
    for expected in (
        "OpenAIProvider",
        "AnthropicProvider",
        "AzureOpenAIProvider",
        "BedrockProvider",
        "GoogleVertexProvider",
        "OllamaProvider",
        "LiteLLMProvider",
    ):
        assert expected in advertised, (
            f"{expected!r} missing from providers.__dir__()"
        )


def test_eager_constants_still_importable() -> None:
    """PRICING + base class + token usage must keep their direct-import contract."""
    _purge_provider_sdks()
    from layerlens.instrument.adapters.providers import (
        PRICING,
        AZURE_PRICING,
        BEDROCK_PRICING,
        MonkeyPatchProvider,
        NormalizedTokenUsage,
        calculate_cost,
    )

    assert PRICING and isinstance(PRICING, dict)
    assert AZURE_PRICING and isinstance(AZURE_PRICING, dict)
    assert BEDROCK_PRICING and isinstance(BEDROCK_PRICING, dict)
    assert callable(calculate_cost)
    assert MonkeyPatchProvider.__name__ == "MonkeyPatchProvider"
    assert NormalizedTokenUsage.__name__ == "NormalizedTokenUsage"
