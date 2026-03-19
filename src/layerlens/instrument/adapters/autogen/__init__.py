"""
STRATIX AutoGen Adapter

Integrates STRATIX tracing with the Microsoft AutoGen framework.

Usage:
    from layerlens.instrument.adapters.autogen import (
        AutoGenAdapter,
        instrument_agents,
        GroupChatTracer,
        HumanProxyTracer,
    )

    adapter = AutoGenAdapter(stratix=stratix_instance)
    adapter.connect()
    adapter.connect_agents(agent1, agent2)
"""

from layerlens.instrument.adapters.autogen.lifecycle import AutoGenAdapter
from layerlens.instrument.adapters.autogen.groupchat import GroupChatTracer
from layerlens.instrument.adapters.autogen.human_proxy import HumanProxyTracer

# Registry lazy-loading convention
ADAPTER_CLASS = AutoGenAdapter


def instrument_agents(*agents, stratix=None, capture_config=None):
    """
    Convenience function to instrument AutoGen agents with STRATIX tracing.

    Args:
        *agents: AutoGen ConversableAgent instances
        stratix: STRATIX SDK instance
        capture_config: CaptureConfig to use

    Returns:
        List of instrumented agents
    """
    adapter = AutoGenAdapter(stratix=stratix, capture_config=capture_config)
    adapter.connect()
    return adapter.connect_agents(*agents)


__all__ = [
    "AutoGenAdapter",
    "GroupChatTracer",
    "HumanProxyTracer",
    "instrument_agents",
    "ADAPTER_CLASS",
]
