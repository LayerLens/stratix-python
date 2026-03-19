"""
Stratix adapter for Google Agent Development Kit (ADK).

Instruments Google ADK agents using the native 6-callback system
(BeforeAgent, AfterAgent, BeforeModel, AfterModel, BeforeTool, AfterTool).
"""

from layerlens.instrument.adapters.google_adk.lifecycle import GoogleADKAdapter

ADAPTER_CLASS = GoogleADKAdapter


def instrument_agent(agent, stratix=None, capture_config=None):
    """Convenience function to instrument a Google ADK agent."""
    adapter = GoogleADKAdapter(stratix=stratix, capture_config=capture_config)
    adapter.connect()
    adapter.instrument_agent(agent)
    return adapter


__all__ = ["GoogleADKAdapter", "ADAPTER_CLASS", "instrument_agent"]
