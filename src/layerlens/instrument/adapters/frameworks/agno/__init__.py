"""
LayerLens adapter for Agno.

Instruments Agno agents by wrapping Agent.run() and Agent.arun()
methods to capture lifecycle events across single and multi-agent teams.
"""

from __future__ import annotations

from typing import Any

from layerlens.instrument.adapters.frameworks.agno.lifecycle import AgnoAdapter

ADAPTER_CLASS = AgnoAdapter


def instrument_agent(agent: Any, stratix: Any = None, capture_config: dict[str, Any] = None, org_id: str | None = None) -> Any:  # type: ignore[assignment]
    """Convenience function to instrument an Agno agent."""
    adapter = AgnoAdapter(stratix=stratix, capture_config=capture_config, org_id=org_id)
    adapter.connect()
    adapter.instrument_agent(agent)
    return adapter


__all__ = ["AgnoAdapter", "ADAPTER_CLASS", "instrument_agent"]
