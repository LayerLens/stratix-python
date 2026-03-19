"""
Stratix MCP Extensions Adapter

Instruments MCP (Model Context Protocol) extensions:
- Elicitation: Server-initiated user input requests
- Structured Tool Outputs: Schema-validated JSON outputs
- Async Tasks: Long-running tool executions
- MCP Apps: Interactive UI components invoked as tools
- OAuth 2.1/OpenID Connect: Auth within MCP sessions
"""

from layerlens.instrument.adapters.protocols.mcp.adapter import MCPExtensionsAdapter

ADAPTER_CLASS = MCPExtensionsAdapter

__all__ = ["MCPExtensionsAdapter", "ADAPTER_CLASS"]
