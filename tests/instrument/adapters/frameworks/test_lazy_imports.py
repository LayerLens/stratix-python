"""Regression tests for the lazy public API in `frameworks/__init__.py`.

Each `pip install layerlens[<framework>]` extra exists so the default install
stays lean. That contract only holds if importing
`layerlens.instrument.adapters.frameworks` never eagerly pulls a framework
SDK. These tests assert that property and that the PEP 562 `__getattr__`
routes to the right submodule when an extra IS installed.
"""

from __future__ import annotations

import importlib
import sys

import pytest


_FRAMEWORK_SDK_PREFIXES = (
    "langgraph",
    "langchain_core",
    "langchain",
    "crewai",
    "autogen",
    "autogen_core",
    "autogen_agentchat",
    "semantic_kernel",
)


def _purge_framework_sdks() -> None:
    """Drop every framework SDK from sys.modules so the next import is fresh."""
    for name in list(sys.modules):
        if name.startswith(_FRAMEWORK_SDK_PREFIXES):
            del sys.modules[name]
    # Also drop our adapter package so its imports re-run.
    for name in list(sys.modules):
        if name.startswith("layerlens.instrument.adapters.frameworks"):
            del sys.modules[name]


def test_frameworks_package_import_does_not_pull_sdks() -> None:
    """Bare `import layerlens.instrument.adapters.frameworks` must stay lean."""
    _purge_framework_sdks()
    importlib.import_module("layerlens.instrument.adapters.frameworks")
    for sdk in _FRAMEWORK_SDK_PREFIXES:
        assert sdk not in sys.modules, (
            f"framework package import eagerly loaded {sdk!r}; lazy export contract broken"
        )


def test_lazy_getattr_raises_attributeerror_for_unknown_names() -> None:
    _purge_framework_sdks()
    pkg = importlib.import_module("layerlens.instrument.adapters.frameworks")
    with pytest.raises(AttributeError):
        pkg.ThisAdapterDoesNotExist  # noqa: B018  - accessing for the side effect


def test_lazy_getattr_resolves_agentforce_without_pulling_other_sdks() -> None:
    """Agentforce only needs httpx (always installed). Exercising its lazy
    export should resolve without pulling any of the heavy framework SDKs."""
    _purge_framework_sdks()
    pkg = importlib.import_module("layerlens.instrument.adapters.frameworks")
    adapter_cls = pkg.AgentforceAdapter  # triggers __getattr__
    assert adapter_cls.__name__ == "AgentforceAdapter"
    for sdk in ("langgraph", "crewai", "autogen", "semantic_kernel"):
        assert sdk not in sys.modules, (
            f"resolving AgentforceAdapter pulled {sdk!r}; lazy exports leaked"
        )


def test_lazy_dir_advertises_all_public_adapters() -> None:
    _purge_framework_sdks()
    pkg = importlib.import_module("layerlens.instrument.adapters.frameworks")
    advertised = set(dir(pkg))
    for expected in (
        "LangChainCallbackHandler",
        "LangGraphCallbackHandler",
        "CrewAIAdapter",
        "AutoGenAdapter",
        "AgentforceAdapter",
        "SemanticKernelAdapter",
    ):
        assert expected in advertised, f"{expected!r} missing from frameworks.__dir__()"
