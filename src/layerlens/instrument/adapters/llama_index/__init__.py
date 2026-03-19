"""
Stratix adapter for LlamaIndex.

Instruments LlamaIndex agents and workflows using the modern
Instrumentation Module (v0.10.20+) with a custom BaseEventHandler.
"""

from layerlens.instrument.adapters.llama_index.lifecycle import LlamaIndexAdapter

ADAPTER_CLASS = LlamaIndexAdapter


def instrument_workflow(workflow=None, stratix=None, capture_config=None):
    """Convenience function to instrument LlamaIndex."""
    adapter = LlamaIndexAdapter(stratix=stratix, capture_config=capture_config)
    adapter.connect()
    if workflow is not None:
        adapter.instrument_workflow(workflow)
    return adapter


__all__ = ["LlamaIndexAdapter", "ADAPTER_CLASS", "instrument_workflow"]
