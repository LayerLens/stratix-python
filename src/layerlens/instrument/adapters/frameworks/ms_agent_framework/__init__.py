"""
LayerLens adapter for Microsoft Agent Framework.

Instruments Microsoft Agent Framework (Semantic Kernel Agents) by wrapping
AgentChat.invoke() and AgentGroupChat.invoke() to capture lifecycle events.
"""

from __future__ import annotations

from typing import Any

from layerlens.instrument.adapters.frameworks.ms_agent_framework.lifecycle import MSAgentAdapter

ADAPTER_CLASS = MSAgentAdapter


def instrument_agent(agent: Any, stratix: Any = None, capture_config: dict[str, Any] = None) -> Any:  # type: ignore[assignment]
    """Convenience function to instrument a Microsoft Agent Framework chat."""
    adapter = MSAgentAdapter(stratix=stratix, capture_config=capture_config)
    adapter.connect()
    adapter.instrument_chat(agent)
    return adapter


__all__ = ["MSAgentAdapter", "ADAPTER_CLASS", "instrument_agent"]
