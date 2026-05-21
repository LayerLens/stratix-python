"""Net-new synthetic trace generation.

Complements :mod:`layerlens.replay` (which produces synthetics from
existing traces). Generation is template-driven with pluggable
providers — an in-process :class:`StochasticProvider` is included for
tests and offline workflows; LLM-backed providers register through
:class:`ProviderRegistry`.
"""

from __future__ import annotations

from .builder import SyntheticDataBuilder
from .providers import (
    ProviderInfo,
    ProviderTier,
    GenerationResult,
    ProviderRegistry,
    SyntheticProvider,
    ProviderCapability,
    StochasticProvider,
)
from .templates import (
    TEMPLATE_LIBRARY,
    TraceCategory,
    TraceTemplate,
    TemplateParameter,
)

__all__ = [
    "GenerationResult",
    "ProviderCapability",
    "ProviderInfo",
    "ProviderRegistry",
    "ProviderTier",
    "StochasticProvider",
    "SyntheticDataBuilder",
    "SyntheticProvider",
    "TEMPLATE_LIBRARY",
    "TemplateParameter",
    "TraceCategory",
    "TraceTemplate",
]
