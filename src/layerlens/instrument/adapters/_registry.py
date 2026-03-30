from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ._base import AdapterInfo, BaseAdapter

log: logging.Logger = logging.getLogger(__name__)

_adapters: Dict[str, BaseAdapter] = {}


def register(name: str, adapter: BaseAdapter) -> None:
    """Register an adapter. Disconnects any existing adapter with the same name."""
    existing = _adapters.get(name)
    if existing is not None and existing.is_connected:
        existing.disconnect()
    _adapters[name] = adapter


def unregister(name: str) -> Optional[BaseAdapter]:
    """Remove and disconnect an adapter. Returns the adapter or None."""
    adapter = _adapters.pop(name, None)
    if adapter is not None and adapter.is_connected:
        adapter.disconnect()
    return adapter


def get(name: str) -> Optional[BaseAdapter]:
    """Look up an adapter by name."""
    return _adapters.get(name)


def list_adapters() -> List[AdapterInfo]:
    """Return info for all registered adapters."""
    return [a.adapter_info() for a in _adapters.values()]


def disconnect_all() -> None:
    """Disconnect and remove all adapters."""
    for adapter in _adapters.values():
        try:
            adapter.disconnect()
        except Exception:
            log.warning("Error disconnecting adapter %s", adapter, exc_info=True)
    _adapters.clear()
