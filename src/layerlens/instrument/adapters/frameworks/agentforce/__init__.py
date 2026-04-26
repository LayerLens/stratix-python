"""
LayerLens Salesforce Agentforce Adapter.

Full-featured adapter for Salesforce Agentforce agent evaluation:

- Session trace import via Data Cloud SOQL (batch / incremental)
- Agent API REST client (real-time session capture)
- Platform Events subscriber (gRPC Pub/Sub for near-real-time)
- Einstein Trust Layer policy import
- LLM evaluation scenarios (completions, A/B testing, model comparison)

DMO Objects (Data Cloud):

- ``AIAgentSession``
- ``AIAgentSessionParticipant``
- ``AIAgentInteraction``
- ``AIAgentInteractionStep``
- ``AIAgentInteractionMessage``

Install::

    pip install 'layerlens[agentforce]'
"""

from __future__ import annotations

from layerlens.instrument.adapters.frameworks.agentforce.auth import (
    NormalizationError,
    SalesforceAuthError,
    SalesforceConnection,
    SalesforceQueryError,
    SalesforceCredentials,
)
from layerlens.instrument.adapters.frameworks.agentforce.client import AgentApiClient
from layerlens.instrument.adapters.frameworks.agentforce.events import PlatformEventSubscriber
from layerlens.instrument.adapters.frameworks.agentforce.mapper import AgentApiMapper
from layerlens.instrument.adapters.frameworks.agentforce.adapter import AgentForceAdapter
from layerlens.instrument.adapters.frameworks.agentforce.importer import ImportResult, AgentForceImporter
from layerlens.instrument.adapters.frameworks.agentforce.llm_eval import EinsteinEvaluator
from layerlens.instrument.adapters.frameworks.agentforce.normalizer import AgentForceNormalizer
from layerlens.instrument.adapters.frameworks.agentforce.trust_layer import TrustLayerImporter

__all__ = [
    # Core adapter
    "AgentForceAdapter",
    # Auth
    "SalesforceAuthError",
    "SalesforceConnection",
    "SalesforceCredentials",
    "SalesforceQueryError",
    "NormalizationError",
    # Import
    "AgentForceImporter",
    "AgentForceNormalizer",
    "ImportResult",
    # Agent API
    "AgentApiClient",
    "AgentApiMapper",
    # Trust Layer
    "TrustLayerImporter",
    # Platform Events
    "PlatformEventSubscriber",
    # Evaluation
    "EinsteinEvaluator",
]

# Registry lazy-loading convention
ADAPTER_CLASS = AgentForceAdapter
