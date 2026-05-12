"""
LayerLens adapter for OpenAI Agents SDK.

Instruments OpenAI Agents SDK (openai-agents) by registering a custom
TraceProcessor that receives all SDK span events, plus wrapping Runner
for execution lifecycle tracing.
"""

from __future__ import annotations

from typing import Any

from layerlens.instrument.adapters.frameworks.openai_agents.lifecycle import OpenAIAgentsAdapter

ADAPTER_CLASS = OpenAIAgentsAdapter


def instrument_runner(
    runner: Any = None, stratix: Any = None, capture_config: dict[str, Any] | None = None
) -> Any:
    """Convenience function to instrument OpenAI Agents SDK."""
    adapter = OpenAIAgentsAdapter(stratix=stratix, capture_config=capture_config)
    adapter.connect()
    if runner is not None:
        adapter.instrument_runner(runner)
    return adapter


__all__ = ["OpenAIAgentsAdapter", "ADAPTER_CLASS", "instrument_runner"]
