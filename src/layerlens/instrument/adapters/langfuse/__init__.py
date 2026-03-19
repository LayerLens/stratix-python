"""
STRATIX Langfuse Adapter

Bidirectional trace sync between STRATIX and Langfuse.

Unlike other adapters that wrap running code in real-time, the Langfuse
adapter is a data import/export pipeline that communicates with a remote
Langfuse HTTP API to pull/push traces in batch.

Usage:
    from layerlens.instrument.adapters.langfuse import LangfuseAdapter
    from layerlens.instrument.adapters.langfuse.config import LangfuseConfig

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

from layerlens.instrument.adapters.langfuse.lifecycle import LangfuseAdapter

# Registry lazy-loading convention
ADAPTER_CLASS = LangfuseAdapter

__all__ = [
    "LangfuseAdapter",
    "ADAPTER_CLASS",
]
