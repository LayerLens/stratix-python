"""
STRATIX Protocol Exceptions

Typed exception hierarchy for protocol adapter errors.
Maps protocol-native error codes to actionable Stratix exceptions.
"""

from __future__ import annotations


class ProtocolError(Exception):
    """Base exception for all protocol adapter errors."""

    def __init__(
        self,
        message: str,
        protocol: str = "",
        error_code: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.protocol = protocol
        self.error_code = error_code
        self.endpoint = endpoint
        super().__init__(message)


# --- Connection errors ---


class ProtocolConnectionError(ProtocolError):
    """Failed to establish or maintain a protocol connection."""


class ProtocolTimeoutError(ProtocolError):
    """Protocol operation timed out."""


class ProtocolSSEDisconnectError(ProtocolError):
    """SSE stream disconnected unexpectedly."""


# --- Protocol-level errors ---


class ProtocolVersionError(ProtocolError):
    """Protocol version negotiation failed."""


class ProtocolAuthError(ProtocolError):
    """Authentication or authorization failure at the protocol level."""


class ProtocolRateLimitError(ProtocolError):
    """Protocol rate limit exceeded."""


# --- A2A-specific errors ---


class A2ATaskError(ProtocolError):
    """An A2A task reached a failed state."""

    def __init__(
        self,
        message: str,
        task_id: str | None = None,
        error_code: str | None = None,
        **kwargs,
    ) -> None:
        self.task_id = task_id
        kwargs.pop("protocol", None)
        super().__init__(message, protocol="a2a", error_code=error_code, **kwargs)


class A2AAgentCardError(ProtocolError):
    """Failed to discover or parse an A2A Agent Card."""


class ACPNormalizationError(ProtocolError):
    """Failed to normalize ACP-origin payload to A2A format."""


# --- MCP-specific errors ---


class MCPToolError(ProtocolError):
    """An MCP tool call failed at the protocol level."""


class MCPElicitationError(ProtocolError):
    """An MCP elicitation interaction failed."""


class MCPSchemaValidationError(ProtocolError):
    """MCP structured output failed schema validation."""


class MCPAsyncTaskTimeoutError(ProtocolTimeoutError):
    """An MCP async task exceeded its configured timeout."""


# --- AG-UI-specific errors ---


class AGUIStreamError(ProtocolError):
    """AG-UI SSE stream error."""


class AGUIStateDeltaError(ProtocolError):
    """Failed to apply AG-UI state delta (JSON Patch error)."""


# --- Error registry ---

# Maps protocol-native error codes to Stratix exception classes
PROTOCOL_ERROR_REGISTRY: dict[str, type[ProtocolError]] = {
    # A2A JSON-RPC error codes
    "a2a:-32700": ProtocolError,           # Parse error
    "a2a:-32600": ProtocolError,           # Invalid request
    "a2a:-32601": ProtocolError,           # Method not found
    "a2a:-32001": A2ATaskError,            # Task not found
    "a2a:-32002": A2ATaskError,            # Task cancelled
    "a2a:-32003": ProtocolAuthError,       # Authentication required
    # MCP error patterns
    "mcp:tool_not_found": MCPToolError,
    "mcp:schema_validation": MCPSchemaValidationError,
    "mcp:elicitation_timeout": MCPElicitationError,
    "mcp:auth_failed": ProtocolAuthError,
    # AG-UI error patterns
    "agui:stream_error": AGUIStreamError,
    "agui:state_delta_error": AGUIStateDeltaError,
}


def resolve_protocol_error(
    protocol: str,
    error_code: str,
    message: str,
    **kwargs,
) -> ProtocolError:
    """
    Resolve a protocol-native error code to a typed Stratix exception.

    Args:
        protocol: Protocol name (a2a, mcp, agui)
        error_code: Protocol-native error code
        message: Error message

    Returns:
        Typed ProtocolError subclass instance
    """
    key = f"{protocol}:{error_code}"
    exc_cls = PROTOCOL_ERROR_REGISTRY.get(key, ProtocolError)
    return exc_cls(message, protocol=protocol, error_code=error_code, **kwargs)
