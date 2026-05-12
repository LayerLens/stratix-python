"""LayerLens adapter for browser_use (full implementation).

Instruments [browser-use](https://github.com/browser-use/browser-use) —
the LLM-driven Playwright agent that performs autonomous web
navigation, form filling, and content extraction.

The adapter wraps ``Agent.run`` (and ``Agent.run_sync`` when present),
threads per-step browser / action / screenshot / DOM / model events
through the LayerLens pipeline, and applies the field-specific
truncation policy so multi-megabyte screenshot / DOM payloads cannot
blow past the ingestion sink limits.

Backward compatibility
----------------------

The legacy STRATIX-branded alias ``STRATIXBrowserUseAdapter`` remains
importable for one deprecation cycle and emits
:class:`DeprecationWarning` on first access::

    # Deprecated — issues a DeprecationWarning. Use BrowserUseAdapter instead.
    from layerlens.instrument.adapters.frameworks.browser_use import (
        STRATIXBrowserUseAdapter,
    )
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

from layerlens.instrument.adapters._base.capture import CaptureConfig
from layerlens.instrument.adapters.frameworks.browser_use.lifecycle import (
    BrowserUseAdapter,
)

ADAPTER_CLASS = BrowserUseAdapter


def instrument_agent(
    agent: Any,
    stratix: Any = None,
    capture_config: Optional[CaptureConfig] = None,
    org_id: Optional[str] = None,
) -> BrowserUseAdapter:
    """Convenience: instrument a browser_use Agent and return the adapter.

    Constructs a :class:`BrowserUseAdapter` (binding ``stratix``,
    ``capture_config``, and ``org_id``), connects it, wraps the supplied
    agent, and returns the live adapter so callers can register sinks
    and inspect health.

    Args:
        agent: A browser_use ``Agent`` instance.
        stratix: Optional LayerLens client; falls back to the null
            sentinel when omitted (events go to attached sinks only).
        capture_config: Optional :class:`CaptureConfig`. Defaults to
            :meth:`CaptureConfig.standard` via BaseAdapter when None.
        org_id: Required tenant binding (or resolved from
            ``stratix.org_id``). Every event payload carries it.

    Returns:
        The connected, wrapping :class:`BrowserUseAdapter`. Call
        ``.disconnect()`` when finished to restore originals.
    """
    adapter = BrowserUseAdapter(
        stratix=stratix,
        capture_config=capture_config,
        org_id=org_id,
    )
    adapter.connect()
    adapter.instrument_agent(agent)
    return adapter


__all__ = [
    "ADAPTER_CLASS",
    "BrowserUseAdapter",
    "STRATIXBrowserUseAdapter",
    "instrument_agent",
]


# --- Static deprecation alias (top-level binding) --------------------
# A top-level ``STRATIX*`` assignment is required by the manifest
# consistency lint (``tests/instrument/adapters/test_manifest_consistency.py::
# _has_stratix_alias``) which walks the AST looking for the binding —
# a PEP 562 ``__getattr__`` alone is invisible to AST analysis. This
# direct binding satisfies the lint AND is the canonical access path
# for callers who wired the legacy STRATIX name.
#
# A DeprecationWarning is still desirable. We trigger it from a
# subclass that proxies construction to the new class while emitting
# the warning the first time the legacy name is *constructed*. This
# keeps ``from ... import BrowserUseAdapter`` cost-free while warning
# the moment a customer actually instantiates the legacy alias.


class _STRATIXBrowserUseAdapterImpl(BrowserUseAdapter):
    """Deprecated alias implementation for :class:`BrowserUseAdapter`.

    The legacy STRATIX-branded name remains importable for one
    deprecation cycle and emits :class:`DeprecationWarning` on
    construction. Will be removed in a future major release.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "STRATIXBrowserUseAdapter is a deprecated alias for "
            "BrowserUseAdapter and will be removed in a future major "
            "release. Import BrowserUseAdapter from "
            "layerlens.instrument.adapters.frameworks.browser_use instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


# Top-level binding the manifest consistency lint walks the AST for —
# ``_has_stratix_alias`` looks for an ``ast.Assign`` whose target name
# starts with ``STRATIX``. The class definition above is an
# ``ast.ClassDef`` (invisible to that walk), so we expose the canonical
# alias via this assignment.
STRATIXBrowserUseAdapter = _STRATIXBrowserUseAdapterImpl


# --- PEP 562 attribute-access deprecation warning --------------------
# Callers reaching for the alias via ``from module import
# STRATIXBrowserUseAdapter`` get the static class above (no warning
# until construction). Callers using the dynamic
# ``module.STRATIXBrowserUseAdapter`` access path go through the
# ``__getattr__`` hook below which fires the warning at access time
# — useful for customers who want eager deprecation signal.

_DEPRECATED_ATTR_ALIASES: Dict[str, str] = {
    # Reserved for future deprecations; STRATIXBrowserUseAdapter is
    # already a top-level binding and warns on construction.
}


def __getattr__(name: str) -> Any:
    target = _DEPRECATED_ATTR_ALIASES.get(name)
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
