"""
Stratix AG-UI (Agent-User Interaction) Protocol Adapter

Instruments AG-UI protocol interactions via ASGI/WSGI middleware
that intercepts the SSE event stream between agent and frontend.
"""

from __future__ import annotations

from layerlens.instrument.adapters.protocols.agui.adapter import AGUIAdapter

ADAPTER_CLASS = AGUIAdapter

__all__ = ["AGUIAdapter", "ADAPTER_CLASS"]
