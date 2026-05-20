"""Lazy public API for framework adapters.

Per M2 ticket ACs (LAY-3446..3450) and the M1.C LangChain template, framework
adapter classes are exposed via PEP 562 ``__getattr__`` so importing this
package never eagerly pulls a framework SDK. ``pip install layerlens`` stays
lean by default; ``pip install layerlens[langgraph]`` (etc.) adds the runtime
deps the user actually needs.

Usage::

    from layerlens.instrument.adapters.frameworks import LangGraphCallbackHandler

    handler = LangGraphCallbackHandler(client)

The attribute access triggers a single ``importlib.import_module`` call against
the matching sub-module; subsequent accesses hit Python's module cache.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# Public-name → (sub-module, attribute) mapping. Add new entries when porting
# additional framework adapters.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "LangChainCallbackHandler": ("langchain", "LangChainCallbackHandler"),
    "LangGraphCallbackHandler": ("langgraph", "LangGraphCallbackHandler"),
    "CrewAIAdapter": ("crewai", "CrewAIAdapter"),
    "AutoGenAdapter": ("autogen", "AutoGenAdapter"),
    "AgentforceAdapter": ("agentforce", "AgentforceAdapter"),
    "SemanticKernelAdapter": ("semantic_kernel", "SemanticKernelAdapter"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = importlib.import_module(f".{module_name}", package=__name__)
    return getattr(module, attr)


def __dir__() -> list[str]:
    return sorted(list(_LAZY_EXPORTS.keys()) + list(globals().keys()))


if TYPE_CHECKING:
    # Re-export under TYPE_CHECKING so static analysers see the names without
    # forcing an eager import at runtime.
    from .crewai import CrewAIAdapter as CrewAIAdapter
    from .autogen import AutoGenAdapter as AutoGenAdapter
    from .langchain import LangChainCallbackHandler as LangChainCallbackHandler
    from .langgraph import LangGraphCallbackHandler as LangGraphCallbackHandler
    from .agentforce import AgentforceAdapter as AgentforceAdapter
    from .semantic_kernel import SemanticKernelAdapter as SemanticKernelAdapter


__all__ = list(_LAZY_EXPORTS.keys())
