"""
LayerLens AutoGen Adapter

Integrates LayerLens tracing with the Microsoft AutoGen framework.

Usage:
    from layerlens.instrument.adapters.frameworks.autogen import (
        AutoGenAdapter,
        instrument_agents,
        GroupChatTracer,
        HumanProxyTracer,
    )

    adapter = AutoGenAdapter(stratix=stratix_instance)
    adapter.connect()
    adapter.connect_agents(agent1, agent2)
"""

from __future__ import annotations

from typing import Any

from layerlens.instrument.adapters.frameworks.autogen.groupchat import GroupChatTracer
from layerlens.instrument.adapters.frameworks.autogen.lifecycle import AutoGenAdapter
from layerlens.instrument.adapters.frameworks.autogen.human_proxy import HumanProxyTracer

# Registry lazy-loading convention
ADAPTER_CLASS = AutoGenAdapter


def instrument_agents(
    *agents: Any, stratix: Any = None, capture_config: dict[str, Any] | None = None
) -> Any:
    """
    Convenience function to instrument AutoGen agents with LayerLens tracing.

    Args:
        *agents: AutoGen ConversableAgent instances
        stratix: LayerLens SDK / client instance to receive emitted events
        capture_config: CaptureConfig to use

    Returns:
        List of instrumented agents
    """
    adapter = AutoGenAdapter(stratix=stratix, capture_config=capture_config)  # type: ignore[arg-type]
    adapter.connect()
    return adapter.connect_agents(*agents)


__all__ = [
    "AutoGenAdapter",
    "GroupChatTracer",
    "HumanProxyTracer",
    "instrument_agents",
    "ADAPTER_CLASS",
]
