"""
STRATIX AgentForce Trace Import Adapter

Imports Salesforce AgentForce Session Tracing data from Data Cloud (Data 360)
and normalizes it to STRATIX canonical event format.

DMO Objects:
- AIAgentSession
- AIAgentSessionParticipant
- AIAgentInteraction
- AIAgentInteractionStep
- AIAgentInteractionMessage
"""

from layerlens.instrument.adapters.agentforce.adapter import AgentForceAdapter
from layerlens.instrument.adapters.agentforce.auth import (
    NormalizationError,
    SalesforceAuthError,
    SalesforceConnection,
    SalesforceCredentials,
    SalesforceQueryError,
)
from layerlens.instrument.adapters.agentforce.importer import AgentForceImporter, ImportResult
from layerlens.instrument.adapters.agentforce.normalizer import AgentForceNormalizer

__all__ = [
    "AgentForceAdapter",
    "AgentForceImporter",
    "AgentForceNormalizer",
    "ImportResult",
    "NormalizationError",
    "SalesforceAuthError",
    "SalesforceConnection",
    "SalesforceCredentials",
    "SalesforceQueryError",
]
