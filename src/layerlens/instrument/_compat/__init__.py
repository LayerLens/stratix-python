"""Instrument-layer compatibility shims.

The :mod:`_compat` package centralises adaptation between the SDK's
public ``layerlens.instrument`` surface and the vendored-from-ateam
canonical types. This module exists so adapter code can write a single
import, e.g.::

    from layerlens.instrument._compat.events import (
        ToolCallEvent,
        ModelInvokeEvent,
        ALL_TYPED_EVENTS,
    )

…without coupling to either the raw vendored snapshots
(``layerlens.instrument._vendored.*``) or the upstream ``stratix.core.events``
package (which is not shipped in the SDK distribution).

See :mod:`layerlens.instrument._compat.events` for the typed-event
foundation introduced in PR `feat/instrument-typed-events-foundation`.
"""

from __future__ import annotations
