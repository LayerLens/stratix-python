"""
STRATIX Protocol Adapter Registry

Protocol adapters for agentic interoperability standards:
- A2A (Agent-to-Agent): Inter-agent task delegation and Agent Cards
- AG-UI (Agent-User Interaction): Agent-to-frontend streaming
- MCP Extensions: Structured outputs, elicitation, async tasks, MCP Apps

Protocol adapters extend BaseProtocolAdapter (which extends BaseAdapter),
inheriting circuit breaker, CaptureConfig gating, and replay serialization.
"""

from layerlens.instrument.adapters.protocols.base import BaseProtocolAdapter

__all__ = ["BaseProtocolAdapter"]
