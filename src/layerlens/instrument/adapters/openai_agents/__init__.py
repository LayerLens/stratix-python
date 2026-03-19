"""
Stratix adapter for OpenAI Agents SDK.

Instruments OpenAI Agents SDK (openai-agents) by registering a custom
TraceProcessor that receives all SDK span events, plus wrapping Runner
for execution lifecycle tracing.
"""

from layerlens.instrument.adapters.openai_agents.lifecycle import OpenAIAgentsAdapter

ADAPTER_CLASS = OpenAIAgentsAdapter


def instrument_runner(runner=None, stratix=None, capture_config=None):
    """Convenience function to instrument OpenAI Agents SDK."""
    adapter = OpenAIAgentsAdapter(stratix=stratix, capture_config=capture_config)
    adapter.connect()
    if runner is not None:
        adapter.instrument_runner(runner)
    return adapter


__all__ = ["OpenAIAgentsAdapter", "ADAPTER_CLASS", "instrument_runner"]
