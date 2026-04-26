"""Protocol adapters for the LayerLens Instrument layer.

Adapters that conform agents to standardized agent-to-agent and
agent-to-UI protocols, emitting protocol events through the LayerLens
telemetry pipeline.

Adapters available (loaded on demand via :class:`AdapterRegistry`):

* ``a2a`` — Agent-to-Agent protocol (handoff / task delegation)
* ``agui`` — Agent GUI streaming + interactivity
* ``mcp`` — Model Context Protocol (tool calling + resources)
* ``ap2`` — Agent Protocol v2
* ``a2ui`` — Agent-to-UI WebSocket bridge
* ``ucp`` — Universal Connection Protocol (multi-transport)

Plus :mod:`layerlens.instrument.adapters.protocols.certification` —
the certification suite (50+ checks, ``CertificationResult``,
``CheckResult``).
"""

from __future__ import annotations
