"""LayerLens LiteLLM provider adapter.

Subpackage layout::

    layerlens.instrument.adapters.providers.litellm
        ├── adapter.py    # LiteLLMAdapter — lifecycle + registry surface
        ├── callback.py   # LayerLensLiteLLMCallback — sync + async hooks
        └── routing.py    # detect_provider — model-string → provider name

Public surface (importing this package does NOT import ``litellm`` —
the SDK is loaded only inside :meth:`LiteLLMAdapter.connect`)::

    from layerlens.instrument.adapters.providers.litellm import LiteLLMAdapter

The legacy flat-file import path
``layerlens.instrument.adapters.providers.litellm_adapter`` is still
available alongside this subpackage and re-exports the same symbols for
users who pinned to the M1.B port.
"""

from __future__ import annotations

from layerlens.instrument.adapters.providers.litellm.adapter import LiteLLMAdapter
from layerlens.instrument.adapters.providers.litellm.routing import detect_provider
from layerlens.instrument.adapters.providers.litellm.callback import LayerLensLiteLLMCallback

# Registry lazy-loading convention.
ADAPTER_CLASS = LiteLLMAdapter

# Backward-compat alias for users coming from the ateam codebase where the
# class is named ``STRATIXLiteLLMCallback``. The alias will be removed in
# v2.0; new code should prefer ``LayerLensLiteLLMCallback``.
STRATIXLiteLLMCallback = LayerLensLiteLLMCallback  # noqa: N816 - backward-compat alias

__all__ = [
    "ADAPTER_CLASS",
    "LayerLensLiteLLMCallback",
    "LiteLLMAdapter",
    "STRATIXLiteLLMCallback",
    "detect_provider",
]
