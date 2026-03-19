"""
MCP Extensions Adapter — Main adapter class.

Instruments MCP protocol extensions via client-side SDK wrapping.
Monkey-patches MCP client tool call dispatch methods to capture
tool calls, structured outputs, elicitation, and async tasks.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from typing import Any

from layerlens.instrument.adapters._base import (
    AdapterCapability,
    AdapterInfo,
    AdapterStatus,
    ReplayableTrace,
)
from layerlens.instrument.adapters.protocols.base import BaseProtocolAdapter

logger = logging.getLogger(__name__)


class MCPExtensionsAdapter(BaseProtocolAdapter):
    """
    Stratix adapter for MCP (Model Context Protocol) Extensions.

    Instruments MCP client objects by wrapping their tool call dispatch
    methods. Captures structured outputs, elicitation interactions,
    async task lifecycle, and MCP App invocations.
    """

    FRAMEWORK = "mcp_extensions"
    PROTOCOL = "mcp"
    PROTOCOL_VERSION = "1.0.0"
    VERSION = "0.1.0"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._framework_version: str | None = None
        self._originals: dict[str, Any] = {}
        self._async_tasks: dict[str, float] = {}  # task_id → start_time

    # --- Lifecycle ---

    def connect(self) -> None:
        try:
            import mcp  # type: ignore[import-untyped]
            self._framework_version = getattr(mcp, "__version__", "unknown")
        except ImportError:
            self._framework_version = None
            logger.debug("mcp not installed; adapter operates in standalone mode")
        self._connected = True
        self._status = AdapterStatus.HEALTHY

    def disconnect(self) -> None:
        self._originals.clear()
        self._async_tasks.clear()
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED
        self._close_sinks()

    def get_adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            name="MCPExtensionsAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self._framework_version,
            capabilities=[
                AdapterCapability.TRACE_TOOLS,
                AdapterCapability.TRACE_PROTOCOL_EVENTS,
                AdapterCapability.REPLAY,
            ],
            description="Stratix adapter for MCP Extensions",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        return ReplayableTrace(
            adapter_name="MCPExtensionsAdapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[],
            config={"capture_config": self._capture_config.model_dump()},
        )

    def probe_health(self, endpoint: str | None = None) -> dict[str, Any]:
        from layerlens.instrument.adapters.protocols.health import probe_mcp_server
        if endpoint:
            result = probe_mcp_server(endpoint)
            return result.to_dict()
        return {"reachable": self._connected, "latency_ms": 0.0, "protocol_version": self._framework_version}

    # --- Tool call interception ---

    def on_tool_call(
        self,
        tool_name: str,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        error: str | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Record an MCP tool call."""
        from layerlens.instrument.schema.events.l5_tools import ToolCallEvent, IntegrationType

        event = ToolCallEvent.create(
            name=tool_name,
            integration=IntegrationType.SERVICE,
            input_data=input_data,
            output_data=output_data,
            error=error,
            latency_ms=latency_ms,
        )
        self.emit_event(event)

    # --- Structured outputs ---

    def on_structured_output(
        self,
        tool_name: str,
        output: Any,
        schema: dict[str, Any] | None = None,
        validation_passed: bool = True,
        validation_errors: list[str] | None = None,
    ) -> None:
        """Record an MCP structured tool output."""
        from layerlens.instrument.schema.events.protocol import StructuredToolOutputEvent

        schema_str = str(schema or {})
        schema_hash = f"sha256:{hashlib.sha256(schema_str.encode()).hexdigest()}"
        output_hash = f"sha256:{hashlib.sha256(str(output).encode()).hexdigest()}"
        schema_id = None
        if schema and "$id" in schema:
            schema_id = schema["$id"]

        event = StructuredToolOutputEvent.create(
            tool_name=tool_name,
            schema_hash=schema_hash,
            validation_passed=validation_passed,
            output_hash=output_hash,
            schema_id=schema_id,
            validation_errors=validation_errors,
        )
        self.emit_event(event)

    # --- Elicitation ---

    def on_elicitation_request(
        self,
        elicitation_id: str,
        server_name: str,
        schema: dict[str, Any] | None = None,
        title: str | None = None,
    ) -> None:
        """Record an MCP elicitation request."""
        from layerlens.instrument.schema.events.protocol import ElicitationRequestEvent

        schema_str = str(schema or {})
        schema_hash = f"sha256:{hashlib.sha256(schema_str.encode()).hexdigest()}"
        schema_ref = None
        if schema and "$id" in schema:
            schema_ref = schema["$id"]

        event = ElicitationRequestEvent.create(
            elicitation_id=elicitation_id,
            server_name=server_name,
            schema_hash=schema_hash,
            request_title=title,
            schema_ref=schema_ref,
        )
        self.emit_event(event)

    def on_elicitation_response(
        self,
        elicitation_id: str,
        action: str,
        response: Any = None,
        latency_ms: float | None = None,
    ) -> None:
        """Record an MCP elicitation response."""
        from layerlens.instrument.schema.events.protocol import ElicitationResponseEvent

        response_hash = f"sha256:{hashlib.sha256(str(response or '').encode()).hexdigest()}"

        event = ElicitationResponseEvent.create(
            elicitation_id=elicitation_id,
            action=action,
            response_hash=response_hash,
            latency_ms=latency_ms,
        )
        self.emit_event(event)

    # --- Async tasks ---

    def on_async_task(
        self,
        async_task_id: str,
        status: str,
        *,
        originating_span_id: str | None = None,
        progress_pct: float | None = None,
        timeout_ms: int | None = None,
    ) -> None:
        """Record an MCP async task lifecycle event."""
        from layerlens.instrument.schema.events.protocol import AsyncTaskEvent

        elapsed_ms = None
        if status == "created":
            self._async_tasks[async_task_id] = time.monotonic()
        elif async_task_id in self._async_tasks:
            elapsed_ms = (time.monotonic() - self._async_tasks[async_task_id]) * 1000
            if status in ("completed", "failed", "timeout"):
                self._async_tasks.pop(async_task_id, None)

        event = AsyncTaskEvent.create(
            async_task_id=async_task_id,
            status=status,
            protocol="mcp",
            originating_tool_call_span_id=originating_span_id,
            progress_pct=progress_pct,
            timeout_ms=timeout_ms,
            elapsed_ms=elapsed_ms,
        )
        self.emit_event(event)

    # --- MCP Apps ---

    def on_mcp_app_invocation(
        self,
        app_id: str,
        component_type: str,
        interaction_result: str,
        parameters: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Record an MCP App invocation."""
        from layerlens.instrument.schema.events.protocol import McpAppInvocationEvent

        params_hash = f"sha256:{hashlib.sha256(str(parameters or {}).encode()).hexdigest()}"
        result_hash = None
        if result is not None:
            result_hash = f"sha256:{hashlib.sha256(str(result).encode()).hexdigest()}"

        event = McpAppInvocationEvent.create(
            app_id=app_id,
            component_type=component_type,
            interaction_result=interaction_result,
            parameters_hash=params_hash,
            result_hash=result_hash,
        )
        self.emit_event(event)

    # --- OAuth 2.1 auth events ---

    def on_auth_event(
        self,
        auth_type: str,
        success: bool,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record an MCP OAuth/OIDC auth event as environment.config."""
        from layerlens.instrument.schema.events.l4_environment import EnvironmentConfigEvent, EnvironmentType

        event = EnvironmentConfigEvent.create(
            env_type=EnvironmentType.CLOUD,
            attributes={
                "auth_event": auth_type,
                "auth_success": success,
                **(details or {}),
            },
        )
        self.emit_event(event)
