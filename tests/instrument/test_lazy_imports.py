"""Lazy-import guards for the Instrument layer.

Importing ``layerlens`` (or ``layerlens.instrument``) MUST NOT import
any optional adapter dependency. Adapter modules that wrap heavy
frameworks (langchain, llama-index, crewai, etc.) are loaded by
:class:`AdapterRegistry` only when the user explicitly requests that
framework — never at SDK import time.

This is the single load-bearing guarantee the v1.x stable client SDK
makes about install-and-import surface area. Breaking it would mean
that simply running ``import layerlens`` in a process triggers a 30+MB
of optional package imports, which is a regression.
"""

from __future__ import annotations

import sys
from typing import Set

# Modules that MUST NOT be loaded as a side effect of importing layerlens
# or layerlens.instrument. These are the heavy-framework dependencies of
# the adapter extras.
_FORBIDDEN_PREFIXES: Set[str] = {
    "langchain",
    "langchain_core",
    "langgraph",
    "llama_index",
    "crewai",
    "autogen",
    "pyautogen",
    "semantic_kernel",
    "ag_ui",
    "mcp",
    "smolagents",
    "agno",
    "strands",
    "browser_use",
    "openai",
    "anthropic",
    "boto3",
    "litellm",
    "ollama",
    "google.cloud.aiplatform",
    "pydantic_ai",
    "cohere",
    "mistralai",
}


def _modules_under(prefixes: Set[str]) -> Set[str]:
    """Return loaded module names matching any forbidden prefix."""
    loaded: Set[str] = set()
    for name in list(sys.modules):
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix + "."):
                loaded.add(name)
                break
    return loaded


def test_layerlens_import_does_not_pull_frameworks() -> None:
    """Plain ``import layerlens`` MUST NOT load any framework dep."""
    # Drop forbidden modules first so the test isolates this import.
    for name in list(sys.modules):
        for prefix in _FORBIDDEN_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]

    import layerlens  # noqa: F401

    leaked = _modules_under(_FORBIDDEN_PREFIXES)
    assert not leaked, (
        f"Importing layerlens leaked framework modules: {sorted(leaked)}. "
        "Ensure adapter modules are NOT imported at SDK init time."
    )


def test_instrument_import_does_not_pull_frameworks() -> None:
    """``import layerlens.instrument`` MUST NOT load any framework dep."""
    for name in list(sys.modules):
        for prefix in _FORBIDDEN_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]

    import layerlens.instrument  # noqa: F401
    import layerlens.instrument.adapters  # noqa: F401
    import layerlens.instrument.adapters._base  # noqa: F401

    leaked = _modules_under(_FORBIDDEN_PREFIXES)
    assert not leaked, (
        f"Importing layerlens.instrument leaked framework modules: {sorted(leaked)}. "
        "The instrument package and its _base layer must not import any adapter module."
    )


def test_adapter_packages_importable_without_framework() -> None:
    """The ``frameworks`` and ``providers`` packages must be importable.

    They expose only ``__init__.py`` documentation; concrete adapter
    modules are loaded by :class:`AdapterRegistry` on demand.
    """
    import layerlens.instrument.adapters.protocols  # noqa: F401
    import layerlens.instrument.adapters.providers  # noqa: F401
    import layerlens.instrument.adapters.frameworks  # noqa: F401
