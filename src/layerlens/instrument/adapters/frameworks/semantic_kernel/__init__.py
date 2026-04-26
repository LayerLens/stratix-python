"""LayerLens Semantic Kernel Adapter.

Provides plugin invocation tracing, planner execution tracking, and
memory operation capture for Microsoft Semantic Kernel via the kernel's
native filter API.

Importing this module does NOT import ``semantic-kernel`` itself — that
dependency is only required when the user calls
:meth:`SemanticKernelAdapter.instrument_kernel` against a real kernel.
"""

from __future__ import annotations

from layerlens.instrument.adapters.frameworks.semantic_kernel.filters import (
    LayerLensFunctionFilter,
    LayerLensAutoFunctionFilter,
    LayerLensPromptRenderFilter,
)
from layerlens.instrument.adapters.frameworks.semantic_kernel.metadata import (
    SKMetadataExtractor,
)
from layerlens.instrument.adapters.frameworks.semantic_kernel.lifecycle import (
    StratixMemoryStore,
    SemanticKernelAdapter,
)

# Registry lazy-loading convention.
ADAPTER_CLASS = SemanticKernelAdapter

# Backward-compat aliases for users coming from ateam (``STRATIX*`` →
# ``LayerLens*``). The class objects are identical; only the import name
# changes. Slated for removal in the next major SDK release.
STRATIXFunctionFilter = LayerLensFunctionFilter  # noqa: N816 - backward-compat alias
STRATIXAutoFunctionFilter = LayerLensAutoFunctionFilter  # noqa: N816 - backward-compat alias
STRATIXPromptRenderFilter = LayerLensPromptRenderFilter  # noqa: N816 - backward-compat alias

__all__ = [
    "ADAPTER_CLASS",
    "LayerLensAutoFunctionFilter",
    "LayerLensFunctionFilter",
    "LayerLensPromptRenderFilter",
    "SKMetadataExtractor",
    "STRATIXAutoFunctionFilter",
    "STRATIXFunctionFilter",
    "STRATIXPromptRenderFilter",
    "SemanticKernelAdapter",
    "StratixMemoryStore",
]
