"""LayerLens adapter for browser_use (M7 placeholder).

The full ``browser_use`` adapter is scheduled for M7 — see the
incubation roadmap. This package exists today as a *minimal*
placeholder so that:

1. The :class:`AdapterRegistry` entry (``browser_use`` → this module)
   resolves without raising ``ModuleNotFoundError`` when adapters are
   enumerated for capability discovery.
2. The field-specific truncation policy wired here in M5 is already
   in place when M7 lands — the placeholder adapter applies
   :data:`DEFAULT_POLICY` from its constructor so any subsequent
   instrumentation work on top of this scaffold inherits the policy
   automatically.

Browser navigation events are uniquely susceptible to unbounded
payloads: a single page-load can capture multi-megabyte base64 PNG
screenshots, full DOM HTML (often >100 KB), and console/network logs.
The cross-pollination audit (§2.4) flags browser_use as CRITICAL for
the truncation policy specifically because of this.

When M7 fleshes out the adapter, the constructor's
``self._truncation_policy = DEFAULT_POLICY`` line will already enforce
correct behaviour for the screenshot / image_data / html / dom fields
defined in
:data:`layerlens.instrument.adapters._base.truncation.DEFAULT_FIELD_CAPS`.
"""

from __future__ import annotations

from layerlens.instrument.adapters.frameworks.browser_use.lifecycle import BrowserUseAdapter

ADAPTER_CLASS = BrowserUseAdapter

__all__ = ["ADAPTER_CLASS", "BrowserUseAdapter"]
