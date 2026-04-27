"""
LayerLens adapter for LlamaIndex.

Instruments LlamaIndex agents and workflows using the modern
Instrumentation Module (v0.10.20+) with a custom BaseEventHandler.
"""

from __future__ import annotations

from typing import Any

from layerlens.instrument.adapters.frameworks.llama_index.lifecycle import LlamaIndexAdapter

ADAPTER_CLASS = LlamaIndexAdapter


def instrument_workflow(workflow: Any = None, stratix: Any = None, capture_config: dict[str, Any] | None = None, org_id: str | None = None) -> Any:
    """Convenience function to instrument LlamaIndex."""
    adapter = LlamaIndexAdapter(stratix=stratix, capture_config=capture_config, org_id=org_id)
    adapter.connect()
    if workflow is not None:
        adapter.instrument_workflow(workflow)
    return adapter


__all__ = ["LlamaIndexAdapter", "ADAPTER_CLASS", "instrument_workflow"]
