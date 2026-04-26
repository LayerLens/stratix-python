"""LayerLens adapter for SmolAgents (HuggingFace).

Instruments SmolAgents (``CodeAgent``, ``ToolCallingAgent``) via the
wrapper pattern since the framework has no native callback system.

Backward compatibility
----------------------

This module previously exported the adapter as ``STRATIXSmolAgentsAdapter``
under the legacy STRATIX brand. The old name remains importable for one
deprecation cycle and emits :class:`DeprecationWarning` on first access::

    # Deprecated — issues a DeprecationWarning. Use SmolAgentsAdapter instead.
    from layerlens.instrument.adapters.frameworks.smolagents import (
        STRATIXSmolAgentsAdapter,
    )

The legacy alias will be removed in a future major release.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

from layerlens.instrument.adapters._base.capture import CaptureConfig
from layerlens.instrument.adapters.frameworks.smolagents.lifecycle import (
    SmolAgentsAdapter,
)

ADAPTER_CLASS = SmolAgentsAdapter


def instrument_agent(
    agent: Any,
    stratix: Any = None,
    capture_config: Optional[CaptureConfig] = None,
) -> SmolAgentsAdapter:
    """Convenience: instrument a SmolAgents agent and return the adapter."""
    adapter = SmolAgentsAdapter(stratix=stratix, capture_config=capture_config)
    adapter.connect()
    adapter.instrument_agent(agent)
    return adapter


__all__ = [
    "ADAPTER_CLASS",
    "SmolAgentsAdapter",
    "STRATIXSmolAgentsAdapter",
    "instrument_agent",
]


# --- PEP 562 deprecation alias ----------------------------------------
# ``STRATIXSmolAgentsAdapter`` is the legacy STRATIX-branded name. We
# expose it via ``__getattr__`` so the warning fires only when callers
# actually reach for the old name — a plain ``from ... import
# SmolAgentsAdapter`` of the new name pays no cost.

_DEPRECATED_ALIASES: Dict[str, str] = {
    "STRATIXSmolAgentsAdapter": "SmolAgentsAdapter",
}


def __getattr__(name: str) -> Any:
    target = _DEPRECATED_ALIASES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"{name} is deprecated; use {target} instead. "
        "The legacy STRATIX-branded alias will be removed in a future major release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return globals()[target]


def __dir__() -> List[str]:
    return sorted(set(__all__) | set(globals()))
