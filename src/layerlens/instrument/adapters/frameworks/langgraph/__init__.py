"""
STRATIX LangGraph Adapter

Integrates STRATIX tracing with LangGraph agent framework.

Usage:
    from layerlens.instrument.adapters.frameworks.langgraph import (
        LayerLensLangGraphAdapter,
        trace_langgraph_tool,
        wrap_llm_for_langgraph,
    )

    # Create adapter
    adapter = LayerLensLangGraphAdapter(stratix_instance)

    # Wrap your graph
    traced_graph = adapter.wrap_graph(my_graph)

    # Or use decorators for individual components
    @trace_langgraph_tool
    def my_tool(state):
        ...
"""

from __future__ import annotations

from layerlens.instrument.adapters._base.pydantic_compat import PydanticCompat, requires_pydantic

# Round-2 deliberation item 20: LangGraph >=0.2 inherits langchain-core's
# Pydantic v2 requirement; fail fast under v1 with a clear message.
requires_pydantic(PydanticCompat.V2_ONLY)

from layerlens.instrument.adapters.frameworks.langgraph.llm import TracedLLM, wrap_llm_for_langgraph
from layerlens.instrument.adapters.frameworks.langgraph.nodes import NodeTracer, trace_node
from layerlens.instrument.adapters.frameworks.langgraph.state import LangGraphStateAdapter
from layerlens.instrument.adapters.frameworks.langgraph.tools import trace_langgraph_tool
from layerlens.instrument.adapters.frameworks.langgraph.handoff import HandoffDetector, detect_handoff
from layerlens.instrument.adapters.frameworks.langgraph.lifecycle import LayerLensLangGraphAdapter

# Registry lazy-loading convention
ADAPTER_CLASS = LayerLensLangGraphAdapter

__all__ = [
    "LangGraphStateAdapter",
    "LayerLensLangGraphAdapter",
    "trace_node",
    "NodeTracer",
    "trace_langgraph_tool",
    "wrap_llm_for_langgraph",
    "TracedLLM",
    "HandoffDetector",
    "detect_handoff",
    "ADAPTER_CLASS",
]


# ---------------------------------------------------------------------------
# Backward-compat aliases (Round-2 deliberation item 23)
# ---------------------------------------------------------------------------
# Users porting from the ``ateam`` reference implementation imported the
# adapter under the ``STRATIX*`` name. We keep the symbol resolvable but
# raise a ``DeprecationWarning`` on access so callers see the v2.0 removal
# notice in their existing test runs without us breaking import-time
# behaviour. The warning fires via PEP 562 module-level ``__getattr__`` so
# it is emitted at *attribute access* time only — a bare module-level
# assignment would warn at every package import even when the alias is
# never referenced.

import warnings as _warnings
from typing import Any as _Any

_DEPRECATED_ALIASES: dict[str, tuple[str, _Any]] = {
    "STRATIXLangGraphAdapter": ("LayerLensLangGraphAdapter", LayerLensLangGraphAdapter),
}


def __getattr__(name: str) -> _Any:
    """Resolve deprecated ``STRATIX*`` aliases with a ``DeprecationWarning``.

    PEP 562 module-level ``__getattr__`` is invoked for attributes not
    found via normal lookup. Aliases are kept out of ``__all__`` and out
    of static binding so star-imports do not pull them and so the warning
    fires only when a caller explicitly references the old name.
    """
    if name in _DEPRECATED_ALIASES:
        new_name, target = _DEPRECATED_ALIASES[name]
        _warnings.warn(
            f"`{name}` is deprecated and will be removed in layerlens v2.0; "
            f"use `{new_name}` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return target
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
