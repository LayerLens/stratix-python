"""LayerLens AWS Bedrock provider adapter.

Public surface for the M3 Bedrock port::

    from layerlens.instrument.adapters.providers.bedrock import (
        AWSBedrockAdapter,
        detect_provider_family,
    )

Importing this package does NOT import ``boto3`` — the framework
version probe is lazy and the adapter classes themselves only touch the
client object you hand them. This preserves the lazy-import guarantee
verified by ``tests/instrument/test_lazy_imports.py``.

Ported from ``ateam/stratix/sdk/python/adapters/llm_providers/bedrock_adapter.py``.
"""

from __future__ import annotations

from layerlens.instrument.adapters.providers.bedrock.body import (
    RereadableBody,
    _RereadableBody,
)
from layerlens.instrument.adapters.providers.bedrock.family import (
    detect_provider_family,
    _detect_provider_family,
)
from layerlens.instrument.adapters.providers.bedrock.adapter import AWSBedrockAdapter

# Registry lazy-loading convention.
ADAPTER_CLASS = AWSBedrockAdapter

# Backward-compat alias for users coming from ateam where the class was
# spelled with the ``STRATIX`` brand. New code should use the
# LayerLens-branded name.
STRATIXBedrockAdapter = AWSBedrockAdapter  # noqa: N816 - backward-compat alias for ateam users

__all__ = [
    "ADAPTER_CLASS",
    "AWSBedrockAdapter",
    "RereadableBody",
    "STRATIXBedrockAdapter",
    "_RereadableBody",
    "_detect_provider_family",
    "detect_provider_family",
]
