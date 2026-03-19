"""
STRATIX Core Types and Schemas

This module contains the canonical, normative types for the STRATIX platform
as defined in Step 1: Canonical Event & Trace Schema.
"""

from layerlens.instrument.schema.identity import (
    EvaluationId,
    TrialId,
    TraceId,
    SpanId,
    AgentId,
    SequenceId,
    Timestamps,
    VectorClock,
    IdentityEnvelope,
)
from layerlens.instrument.schema.privacy import (
    PrivacyLevel,
    RedactionMethod,
    PrivacyEnvelope,
)
from layerlens.instrument.schema.attestation import (
    HashScope,
    AttestationEnvelope,
)
from layerlens.instrument.schema.event import STRATIXEvent
from layerlens.instrument.schema.causality import SparseVectorClock

__all__ = [
    # Identity
    "EvaluationId",
    "TrialId",
    "TraceId",
    "SpanId",
    "AgentId",
    "SequenceId",
    "Timestamps",
    "VectorClock",
    "IdentityEnvelope",
    # Privacy
    "PrivacyLevel",
    "RedactionMethod",
    "PrivacyEnvelope",
    # Attestation
    "HashScope",
    "AttestationEnvelope",
    # Event
    "STRATIXEvent",
    # Causality
    "SparseVectorClock",
]
