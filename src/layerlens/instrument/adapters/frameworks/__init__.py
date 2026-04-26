"""Framework adapters for the LayerLens Instrument layer.

Each framework adapter wraps an agent / chain framework's lifecycle to
intercept agent runs, model invocations, tool calls, state changes, and
handoffs, emitting events through the LayerLens telemetry pipeline.

Adapters available (loaded on demand — importing this package does NOT
import any framework SDK):

* ``agentforce`` — Salesforce Agentforce (auth, client, event mapping)

Usage::

    # Lazy import — does not pull in framework dependencies until used.
    from layerlens.instrument.adapters.frameworks.agentforce import (
        AgentForceAdapter,
        SalesforceCredentials,
    )

The package is intentionally empty so that ``import
layerlens.instrument.adapters.frameworks`` never fails because of an
absent framework SDK. Each per-framework subpackage handles its own
optional dependency surface.
"""

from __future__ import annotations
