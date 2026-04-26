"""LiteLLM provider adapter — legacy flat-module import path.

The implementation now lives in the
:mod:`layerlens.instrument.adapters.providers.litellm` subpackage so the
adapter can be split across ``adapter.py`` / ``callback.py`` /
``routing.py``. This module is kept as a thin re-export for users who
pinned to the M1.B flat-file path::

    # Both of these import the same class.
    from layerlens.instrument.adapters.providers.litellm_adapter import LiteLLMAdapter
    from layerlens.instrument.adapters.providers.litellm import LiteLLMAdapter
"""

from __future__ import annotations

from layerlens.instrument.adapters.providers.litellm import (
    ADAPTER_CLASS,
    LiteLLMAdapter,
    STRATIXLiteLLMCallback,
    LayerLensLiteLLMCallback,
    detect_provider,
)

__all__ = [
    "ADAPTER_CLASS",
    "LayerLensLiteLLMCallback",
    "LiteLLMAdapter",
    "STRATIXLiteLLMCallback",
    "detect_provider",
]
