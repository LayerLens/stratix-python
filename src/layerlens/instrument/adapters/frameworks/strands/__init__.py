"""
LayerLens adapter for AWS Strands.

Instruments AWS Strands agents by hooking into the agent callback system
to capture tool calls, model invocations, and conversation state.
"""

from __future__ import annotations

from typing import Any

from layerlens.instrument.adapters.frameworks.strands.lifecycle import StrandsAdapter

ADAPTER_CLASS = StrandsAdapter


def instrument_agent(agent: Any, stratix: Any = None, capture_config: dict[str, Any] = None) -> Any:  # type: ignore[assignment]
    """Convenience function to instrument an AWS Strands agent."""
    adapter = StrandsAdapter(stratix=stratix, capture_config=capture_config)
    adapter.connect()
    adapter.instrument_agent(agent)
    return adapter


__all__ = ["StrandsAdapter", "ADAPTER_CLASS", "instrument_agent"]
