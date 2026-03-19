"""
STRATIX LangGraph Adapter

Integrates STRATIX tracing with LangGraph agent framework.

Usage:
    from layerlens.instrument.adapters.langgraph import (
        STRATIXLangGraphAdapter,
        trace_langgraph_tool,
        wrap_llm_for_langgraph,
    )

    # Create adapter
    adapter = STRATIXLangGraphAdapter(stratix_instance)

    # Wrap your graph
    traced_graph = adapter.wrap_graph(my_graph)

    # Or use decorators for individual components
    @trace_langgraph_tool
    def my_tool(state):
        ...
"""

from layerlens.instrument.adapters.langgraph.state import LangGraphStateAdapter
from layerlens.instrument.adapters.langgraph.lifecycle import STRATIXLangGraphAdapter
from layerlens.instrument.adapters.langgraph.nodes import trace_node, NodeTracer
from layerlens.instrument.adapters.langgraph.tools import trace_langgraph_tool
from layerlens.instrument.adapters.langgraph.llm import wrap_llm_for_langgraph, TracedLLM
from layerlens.instrument.adapters.langgraph.handoff import HandoffDetector, detect_handoff

# Registry lazy-loading convention
ADAPTER_CLASS = STRATIXLangGraphAdapter

__all__ = [
    "LangGraphStateAdapter",
    "STRATIXLangGraphAdapter",
    "trace_node",
    "NodeTracer",
    "trace_langgraph_tool",
    "wrap_llm_for_langgraph",
    "TracedLLM",
    "HandoffDetector",
    "detect_handoff",
    "ADAPTER_CLASS",
]
