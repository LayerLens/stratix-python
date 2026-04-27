"""
LayerLens adapter for PydanticAI.

Instruments PydanticAI agents via OpenTelemetry wrapper (Logfire-compatible)
and Agent wrapper for lifecycle hooks.
"""

from __future__ import annotations

from typing import Any

from layerlens.instrument.adapters._base.pydantic_compat import PydanticCompat, requires_pydantic

# Round-2 deliberation item 20: pydantic-ai is built on Pydantic v2 only;
# fail fast under v1.
requires_pydantic(PydanticCompat.V2_ONLY)

from layerlens.instrument.adapters.frameworks.pydantic_ai.lifecycle import PydanticAIAdapter

ADAPTER_CLASS = PydanticAIAdapter


def instrument_agent(agent: Any, stratix: Any = None, capture_config: dict[str, Any] = None, org_id: str | None = None) -> Any:  # type: ignore[assignment]
    """Convenience function to instrument a PydanticAI agent."""
    adapter = PydanticAIAdapter(stratix=stratix, capture_config=capture_config, org_id=org_id)
    adapter.connect()
    adapter.instrument_agent(agent)
    return adapter


__all__ = ["PydanticAIAdapter", "ADAPTER_CLASS", "instrument_agent"]
