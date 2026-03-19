"""
Stratix adapter for SmolAgents (HuggingFace).

Instruments SmolAgents (CodeAgent, ToolCallingAgent) via wrapper pattern
since the framework has no native callback system.
"""

from layerlens.instrument.adapters.smolagents.lifecycle import SmolAgentsAdapter

ADAPTER_CLASS = SmolAgentsAdapter


def instrument_agent(agent, stratix=None, capture_config=None):
    """Convenience function to instrument a SmolAgents agent."""
    adapter = SmolAgentsAdapter(stratix=stratix, capture_config=capture_config)
    adapter.connect()
    adapter.instrument_agent(agent)
    return adapter


__all__ = ["SmolAgentsAdapter", "ADAPTER_CLASS", "instrument_agent"]
