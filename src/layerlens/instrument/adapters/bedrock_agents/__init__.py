"""
Stratix adapter for AWS Bedrock Agents.

Instruments AWS Bedrock Agents via boto3 event hooks and trace
extraction from invoke_agent response streams.
"""

from layerlens.instrument.adapters.bedrock_agents.lifecycle import BedrockAgentsAdapter

ADAPTER_CLASS = BedrockAgentsAdapter


def instrument_client(client, stratix=None, capture_config=None):
    """Convenience function to instrument a Bedrock Agent Runtime client."""
    adapter = BedrockAgentsAdapter(stratix=stratix, capture_config=capture_config)
    adapter.connect()
    adapter.instrument_client(client)
    return adapter


__all__ = ["BedrockAgentsAdapter", "ADAPTER_CLASS", "instrument_client"]
