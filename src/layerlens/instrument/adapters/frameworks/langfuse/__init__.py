"""
LayerLens Langfuse Adapter

Bidirectional trace sync between LayerLens and Langfuse.

Unlike other adapters that wrap running code in real-time, the Langfuse
adapter is a data import/export pipeline that communicates with a remote
Langfuse HTTP API to pull/push traces in batch.

Usage::

    from layerlens.instrument.adapters.frameworks.langfuse import LangfuseAdapter
    from layerlens.instrument.adapters.frameworks.langfuse.config import LangfuseConfig

    config = LangfuseConfig(
        public_key="pk-...",
        secret_key="sk-...",
    )

    adapter = LangfuseAdapter(stratix=stratix_instance, config=config)
    adapter.connect()

    # Import traces from Langfuse into LayerLens
    result = adapter.import_traces(since=datetime(2024, 1, 1))

    # Export LayerLens traces to Langfuse
    result = adapter.export_traces(events_by_trace={"trace-1": [...]})

Backward compatibility
----------------------

Users coming from ``ateam`` / ``stratix`` can keep importing the old
name ``STRATIXLangfuseAdapter``. Accessing it raises a
:class:`DeprecationWarning` (see PEP 562) and resolves to
:class:`LangfuseAdapter`. The alias will be removed in v2.0.
"""

from __future__ import annotations

import warnings
from typing import Any

from layerlens.instrument.adapters._base.pydantic_compat import PydanticCompat, requires_pydantic

# Round-2 deliberation item 20: ``frameworks/langfuse/config.py`` uses
# ``field_validator`` (v2-only); fail fast under v1 with a clear message
# instead of a confusing ImportError from config.py.
requires_pydantic(PydanticCompat.V2_ONLY)

from layerlens.instrument.adapters.frameworks.langfuse.lifecycle import LangfuseAdapter

# Registry lazy-loading convention
ADAPTER_CLASS = LangfuseAdapter

__all__ = [
    "LangfuseAdapter",
    "STRATIXLangfuseAdapter",
    "ADAPTER_CLASS",
]


def __getattr__(name: str) -> Any:
    """PEP 562 module-level ``__getattr__`` for deprecated aliases.

    Importing :class:`STRATIXLangfuseAdapter` raises a
    :class:`DeprecationWarning` and resolves to :class:`LangfuseAdapter`.
    This preserves backward compatibility with code written against the
    legacy ``stratix.*`` package layout (pre-LayerLens rename) while
    nudging callers toward the new name.
    """
    if name == "STRATIXLangfuseAdapter":
        warnings.warn(
            "STRATIXLangfuseAdapter is a deprecated alias for "
            "LangfuseAdapter and will be removed in v2.0. Import "
            "LangfuseAdapter from "
            "layerlens.instrument.adapters.frameworks.langfuse instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return LangfuseAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
