"""
STRATIX Langfuse Adapter

Bidirectional trace sync between STRATIX and Langfuse.

Unlike other adapters that wrap running code in real-time, the Langfuse
adapter is a data import/export pipeline that communicates with a remote
Langfuse HTTP API to pull/push traces in batch.

Usage:
    from layerlens.instrument.adapters.frameworks.langfuse import LangfuseAdapter
    from layerlens.instrument.adapters.frameworks.langfuse.config import LangfuseConfig

    config = LangfuseConfig(
        public_key="pk-...",
        secret_key="sk-...",
    )

    adapter = LangfuseAdapter(stratix=stratix_instance, config=config)
    adapter.connect()

    # Import traces from Langfuse
    result = adapter.import_traces(since=datetime(2024, 1, 1))

    # Export STRATIX traces to Langfuse
    result = adapter.export_traces(events_by_trace={"trace-1": [...]})
"""

from __future__ import annotations

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
    "ADAPTER_CLASS",
]
