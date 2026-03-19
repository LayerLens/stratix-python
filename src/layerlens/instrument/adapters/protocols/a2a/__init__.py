"""
Stratix A2A (Agent-to-Agent) Protocol Adapter

Instruments A2A protocol interactions using dual-channel instrumentation:
1. Server-side wrapping: intercepts incoming JSON-RPC requests and SSE streams
2. Client-side wrapping: traces outgoing task submissions and streamed updates

Handles ACP-origin payloads (IBM Agent Communication Protocol, merged into
A2A in August 2025) via the ACPNormalizer.
"""

from layerlens.instrument.adapters.protocols.a2a.adapter import A2AAdapter

ADAPTER_CLASS = A2AAdapter

__all__ = ["A2AAdapter", "ADAPTER_CLASS"]
