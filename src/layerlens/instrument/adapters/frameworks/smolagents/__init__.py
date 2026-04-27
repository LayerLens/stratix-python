"""LayerLens adapter for SmolAgents (HuggingFace).

Instruments SmolAgents (CodeAgent, ToolCallingAgent) via wrapper pattern
since the framework has no native callback system.
"""

from __future__ import annotations

from typing import Any, Optional

from layerlens.instrument.adapters._base.capture import CaptureConfig
from layerlens.instrument.adapters.frameworks.smolagents.lifecycle import (
    SmolAgentsAdapter,
)

ADAPTER_CLASS = SmolAgentsAdapter


def instrument_agent(
    agent: Any,
    stratix: Any = None,
    capture_config: Optional[CaptureConfig] = None,
    org_id: Optional[str] = None,
) -> SmolAgentsAdapter:
    """Convenience: instrument a SmolAgents agent and return the adapter."""
    adapter = SmolAgentsAdapter(stratix=stratix, capture_config=capture_config, org_id=org_id)
    adapter.connect()
    adapter.instrument_agent(agent)
    return adapter


__all__ = ["ADAPTER_CLASS", "SmolAgentsAdapter", "instrument_agent"]
