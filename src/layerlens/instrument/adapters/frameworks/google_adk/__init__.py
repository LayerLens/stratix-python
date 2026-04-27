"""
LayerLens adapter for Google Agent Development Kit (ADK).

Instruments Google ADK agents using the native 6-callback system
(BeforeAgent, AfterAgent, BeforeModel, AfterModel, BeforeTool, AfterTool).
"""

from __future__ import annotations

from typing import Any

from layerlens.instrument.adapters.frameworks.google_adk.lifecycle import GoogleADKAdapter

ADAPTER_CLASS = GoogleADKAdapter


def instrument_agent(agent: Any, stratix: Any = None, capture_config: dict[str, Any] = None, org_id: str | None = None) -> Any:  # type: ignore[assignment]
    """Convenience function to instrument a Google ADK agent."""
    adapter = GoogleADKAdapter(stratix=stratix, capture_config=capture_config, org_id=org_id)
    adapter.connect()
    adapter.instrument_agent(agent)
    return adapter


__all__ = ["GoogleADKAdapter", "ADAPTER_CLASS", "instrument_agent"]
