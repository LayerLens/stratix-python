"""
LayerLens adapter for AWS Bedrock Agents.

Instruments AWS Bedrock Agents via boto3 event hooks and trace
extraction from invoke_agent response streams.
"""

from __future__ import annotations

from typing import Any

from layerlens.instrument.adapters.frameworks.bedrock_agents.lifecycle import BedrockAgentsAdapter

ADAPTER_CLASS = BedrockAgentsAdapter


def instrument_client(client: Any, stratix: Any = None, capture_config: dict[str, Any] | None = None, org_id: str | None = None) -> Any:
    """Convenience function to instrument a Bedrock Agent Runtime client."""
    adapter = BedrockAgentsAdapter(stratix=stratix, capture_config=capture_config, org_id=org_id)
    adapter.connect()
    adapter.instrument_client(client)
    return adapter


__all__ = ["BedrockAgentsAdapter", "ADAPTER_CLASS", "instrument_client"]
