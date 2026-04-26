"""Compatibility shim for the legacy flat-file Bedrock adapter import path.

The Bedrock adapter was promoted from a single file to a package in the
M3 fan-out. Existing call-sites that imported from
``layerlens.instrument.adapters.providers.bedrock_adapter`` continue to
work via this shim, which simply re-exports the package's public
surface.

New code should import from ``layerlens.instrument.adapters.providers.bedrock``
directly.
"""

from __future__ import annotations

from layerlens.instrument.adapters.providers.bedrock import (
    ADAPTER_CLASS,
    RereadableBody,
    AWSBedrockAdapter,
    STRATIXBedrockAdapter,
    _RereadableBody,
    detect_provider_family,
    _detect_provider_family,
)

__all__ = [
    "ADAPTER_CLASS",
    "AWSBedrockAdapter",
    "RereadableBody",
    "STRATIXBedrockAdapter",
    "_RereadableBody",
    "_detect_provider_family",
    "detect_provider_family",
]
