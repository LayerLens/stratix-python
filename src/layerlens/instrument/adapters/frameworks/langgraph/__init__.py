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


# Backward-compat aliases for users coming from ateam.
STRATIXLangGraphAdapter = LayerLensLangGraphAdapter  # noqa: N816 - backward-compat alias for ateam users
