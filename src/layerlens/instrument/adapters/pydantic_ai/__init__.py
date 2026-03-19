"""
Stratix adapter for PydanticAI.

Instruments PydanticAI agents via OpenTelemetry wrapper (Logfire-compatible)
and Agent wrapper for lifecycle hooks.
"""

from layerlens.instrument.adapters.pydantic_ai.lifecycle import PydanticAIAdapter

ADAPTER_CLASS = PydanticAIAdapter


def instrument_agent(agent, stratix=None, capture_config=None):
    """Convenience function to instrument a PydanticAI agent."""
    adapter = PydanticAIAdapter(stratix=stratix, capture_config=capture_config)
    adapter.connect()
    adapter.instrument_agent(agent)
    return adapter


__all__ = ["PydanticAIAdapter", "ADAPTER_CLASS", "instrument_agent"]
