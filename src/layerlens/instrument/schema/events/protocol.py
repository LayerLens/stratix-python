"""
STRATIX Protocol Events — Schema v1.2.0

Nine new event types for agentic protocol standards:

Protocol Discovery (L6a):
- protocol.agent_card: A2A Agent Card discovery and registration

Protocol Streams (L6b):
- protocol.stream.event: AG-UI/A2A streaming event

Protocol Lifecycle (L6c):
- protocol.task.submitted: A2A task submitted (cross-cutting, always enabled)
- protocol.task.completed: A2A task completed (cross-cutting, always enabled)
- protocol.async_task: MCP/A2A async task lifecycle (cross-cutting, always enabled)

Tool-Layer Protocol Events (L5a):
- protocol.elicitation.request: MCP Elicitation server-initiated user input
- protocol.elicitation.response: MCP Elicitation user response
- protocol.tool.structured_output: MCP structured tool output
- protocol.mcp_app.invocation: MCP App interactive UI component
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class SkillInfo(BaseModel):
    """A skill declared in an A2A Agent Card."""
    id: str = Field(description="Skill identifier")
    name: str = Field(description="Human-readable skill name")
    description: str | None = Field(default=None, description="Skill description")
    tags: list[str] = Field(default_factory=list, description="Skill tags")
    examples: list[str] = Field(default_factory=list, description="Example inputs")


class AgentCardInfo(BaseModel):
    """Parsed content of an A2A Agent Card."""
    agent_id: str = Field(description="Matches identity envelope agent_id")
    name: str = Field(description="Human-readable agent name from the card")
    description: str | None = Field(default=None, description="Agent description")
    url: str = Field(description="Base URL of the A2A endpoint")
    version: str = Field(description="Protocol version declared in the card")
    capabilities: dict[str, Any] = Field(
        default_factory=dict,
        description="Capability flags (streaming, pushNotifications, etc.)",
    )
    skills: list[SkillInfo] = Field(default_factory=list, description="Declared skills")
    auth_scheme: str | None = Field(
        default=None,
        description="Authentication scheme: none | bearer | oauth2 | apiKey",
    )
    source: str = Field(
        default="discovery",
        description="How the card was obtained: discovery | registration | refresh",
    )


# ---------------------------------------------------------------------------
# L6a — Protocol Discovery
# ---------------------------------------------------------------------------


class AgentCardEvent(BaseModel):
    """
    L6a: Emitted when an A2A Agent Card is discovered or registered.

    Captures the full capability advertisement of an A2A-compliant agent.
    """
    event_type: str = Field(
        default="protocol.agent_card",
        description="Event type identifier",
    )
    layer: str = Field(default="L6a", description="Layer identifier")
    card: AgentCardInfo = Field(description="Parsed Agent Card content")

    @classmethod
    def create(
        cls,
        agent_id: str,
        name: str,
        url: str,
        version: str,
        *,
        description: str | None = None,
        capabilities: dict[str, Any] | None = None,
        skills: list[SkillInfo] | None = None,
        auth_scheme: str | None = None,
        source: str = "discovery",
    ) -> AgentCardEvent:
        return cls(
            card=AgentCardInfo(
                agent_id=agent_id,
                name=name,
                description=description,
                url=url,
                version=version,
                capabilities=capabilities or {},
                skills=skills or [],
                auth_scheme=auth_scheme,
                source=source,
            )
        )


# ---------------------------------------------------------------------------
# L6c — Protocol Lifecycle (cross-cutting, always enabled)
# ---------------------------------------------------------------------------


class TaskSubmittedEvent(BaseModel):
    """
    Cross-cutting: Emitted when an A2A task is submitted.

    Always enabled — task lifecycle events are infrastructure signals.
    """
    event_type: str = Field(
        default="protocol.task.submitted",
        description="Event type identifier",
    )
    task_id: str = Field(description="A2A task identifier")
    task_type: str | None = Field(
        default=None, description="Semantic task type (from skill definition)",
    )
    submitter_agent_id: str | None = Field(
        default=None, description="Agent submitting the task",
    )
    receiver_agent_url: str = Field(
        description="A2A endpoint that received the task",
    )
    protocol_origin: str = Field(
        default="a2a", description="Protocol origin: a2a | acp",
    )
    message_role: str = Field(
        default="user", description="Message role: user | agent",
    )

    @classmethod
    def create(
        cls,
        task_id: str,
        receiver_agent_url: str,
        *,
        task_type: str | None = None,
        submitter_agent_id: str | None = None,
        protocol_origin: str = "a2a",
        message_role: str = "user",
    ) -> TaskSubmittedEvent:
        return cls(
            task_id=task_id,
            task_type=task_type,
            submitter_agent_id=submitter_agent_id,
            receiver_agent_url=receiver_agent_url,
            protocol_origin=protocol_origin,
            message_role=message_role,
        )


class TaskCompletedEvent(BaseModel):
    """
    Cross-cutting: Emitted when an A2A task reaches a terminal state.
    """
    event_type: str = Field(
        default="protocol.task.completed",
        description="Event type identifier",
    )
    task_id: str = Field(description="A2A task identifier")
    final_status: str = Field(
        description="Terminal status: completed | failed | cancelled",
    )
    artifact_count: int = Field(default=0, description="Number of artifacts returned")
    artifact_hashes: list[str] = Field(
        default_factory=list, description="sha256:<hex> per artifact",
    )
    error_code: str | None = Field(default=None, description="A2A error code if failed")
    error_message: str | None = Field(default=None, description="Error message if failed")
    duration_ms: float | None = Field(
        default=None, description="Wall time from submitted to completed",
    )

    @classmethod
    def create(
        cls,
        task_id: str,
        final_status: str,
        *,
        artifact_count: int = 0,
        artifact_hashes: list[str] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        duration_ms: float | None = None,
    ) -> TaskCompletedEvent:
        return cls(
            task_id=task_id,
            final_status=final_status,
            artifact_count=artifact_count,
            artifact_hashes=artifact_hashes or [],
            error_code=error_code,
            error_message=error_message,
            duration_ms=duration_ms,
        )


class AsyncTaskEvent(BaseModel):
    """
    Cross-cutting: Emitted for MCP/A2A async task lifecycle transitions.

    Always enabled — async task tracking is critical infrastructure.
    """
    event_type: str = Field(
        default="protocol.async_task",
        description="Event type identifier",
    )
    async_task_id: str = Field(description="Async task identifier")
    originating_tool_call_span_id: str | None = Field(
        default=None, description="Links to the originating tool.call span",
    )
    status: str = Field(
        description="Status: created | running | completed | failed | timeout",
    )
    protocol: str = Field(description="Protocol: mcp | a2a")
    progress_pct: float | None = Field(
        default=None, description="0.0-100.0 progress if reported",
    )
    timeout_ms: int | None = Field(default=None, description="Configured timeout")
    elapsed_ms: float | None = Field(default=None, description="Time since creation")

    @classmethod
    def create(
        cls,
        async_task_id: str,
        status: str,
        protocol: str,
        *,
        originating_tool_call_span_id: str | None = None,
        progress_pct: float | None = None,
        timeout_ms: int | None = None,
        elapsed_ms: float | None = None,
    ) -> AsyncTaskEvent:
        return cls(
            async_task_id=async_task_id,
            status=status,
            protocol=protocol,
            originating_tool_call_span_id=originating_tool_call_span_id,
            progress_pct=progress_pct,
            timeout_ms=timeout_ms,
            elapsed_ms=elapsed_ms,
        )


# ---------------------------------------------------------------------------
# L6b — Protocol Streams
# ---------------------------------------------------------------------------


class ProtocolStreamEvent(BaseModel):
    """
    L6b: Emitted for each event in an SSE protocol stream.

    High-frequency: gated by CaptureConfig.l6b_protocol_streams.
    """
    event_type: str = Field(
        default="protocol.stream.event",
        description="Event type identifier",
    )
    layer: str = Field(default="L6b", description="Layer identifier")
    protocol: str = Field(description="Protocol: agui | a2a")
    agui_event_type: str | None = Field(
        default=None, description="AG-UI event type (e.g. TEXT_MESSAGE_CONTENT)",
    )
    sequence_in_stream: int = Field(
        description="Position within the SSE stream",
    )
    payload_summary: str | None = Field(
        default=None, description="Truncated payload for low-verbosity capture",
    )
    payload_hash: str = Field(description="sha256 of full payload")

    @classmethod
    def create(
        cls,
        protocol: str,
        sequence_in_stream: int,
        payload_hash: str,
        *,
        agui_event_type: str | None = None,
        payload_summary: str | None = None,
    ) -> ProtocolStreamEvent:
        return cls(
            protocol=protocol,
            agui_event_type=agui_event_type,
            sequence_in_stream=sequence_in_stream,
            payload_summary=payload_summary,
            payload_hash=payload_hash,
        )


# ---------------------------------------------------------------------------
# L5a — MCP Extension Events (tool layer)
# ---------------------------------------------------------------------------


class ElicitationRequestEvent(BaseModel):
    """
    L5a: Emitted when an MCP server initiates a user input request.
    """
    event_type: str = Field(
        default="protocol.elicitation.request",
        description="Event type identifier",
    )
    layer: str = Field(default="L5a", description="Layer identifier")
    elicitation_id: str = Field(description="Unique elicitation identifier")
    server_name: str = Field(description="MCP server that issued the request")
    request_title: str | None = Field(
        default=None, description="Human-readable request title",
    )
    schema_ref: str | None = Field(
        default=None, description="JSON Schema $id for the requested input",
    )
    schema_hash: str = Field(description="sha256 of the request schema")

    @classmethod
    def create(
        cls,
        elicitation_id: str,
        server_name: str,
        schema_hash: str,
        *,
        request_title: str | None = None,
        schema_ref: str | None = None,
    ) -> ElicitationRequestEvent:
        return cls(
            elicitation_id=elicitation_id,
            server_name=server_name,
            request_title=request_title,
            schema_ref=schema_ref,
            schema_hash=schema_hash,
        )


class ElicitationResponseEvent(BaseModel):
    """
    L5a: Emitted when a user responds to an MCP elicitation request.
    """
    event_type: str = Field(
        default="protocol.elicitation.response",
        description="Event type identifier",
    )
    layer: str = Field(default="L5a", description="Layer identifier")
    elicitation_id: str = Field(description="Links to protocol.elicitation.request")
    action: str = Field(description="User action: submit | cancel")
    response_hash: str = Field(
        description="sha256 of the user's response (never cleartext)",
    )
    latency_ms: float | None = Field(
        default=None, description="Time from request to response",
    )

    @classmethod
    def create(
        cls,
        elicitation_id: str,
        action: str,
        response_hash: str,
        *,
        latency_ms: float | None = None,
    ) -> ElicitationResponseEvent:
        return cls(
            elicitation_id=elicitation_id,
            action=action,
            response_hash=response_hash,
            latency_ms=latency_ms,
        )


class StructuredToolOutputEvent(BaseModel):
    """
    L5a: Emitted when an MCP tool returns a structured output.

    Extends tool.call — both events are emitted for structured MCP tool calls.
    """
    event_type: str = Field(
        default="protocol.tool.structured_output",
        description="Event type identifier",
    )
    layer: str = Field(default="L5a", description="Layer identifier")
    tool_name: str = Field(description="MCP tool name")
    schema_id: str | None = Field(
        default=None, description="JSON Schema $id reference",
    )
    schema_hash: str = Field(description="sha256 of the output schema")
    validation_passed: bool = Field(
        description="Whether output validated against schema",
    )
    validation_errors: list[str] = Field(
        default_factory=list, description="Schema validation error messages",
    )
    output_hash: str = Field(description="sha256 of the structured output value")

    @classmethod
    def create(
        cls,
        tool_name: str,
        schema_hash: str,
        validation_passed: bool,
        output_hash: str,
        *,
        schema_id: str | None = None,
        validation_errors: list[str] | None = None,
    ) -> StructuredToolOutputEvent:
        return cls(
            tool_name=tool_name,
            schema_id=schema_id,
            schema_hash=schema_hash,
            validation_passed=validation_passed,
            validation_errors=validation_errors or [],
            output_hash=output_hash,
        )


class McpAppInvocationEvent(BaseModel):
    """
    L5a: Emitted when an MCP App (interactive UI component) is invoked.
    """
    event_type: str = Field(
        default="protocol.mcp_app.invocation",
        description="Event type identifier",
    )
    layer: str = Field(default="L5a", description="Layer identifier")
    app_id: str = Field(description="MCP App identifier")
    component_type: str = Field(
        description="Component type: form | confirmation | picker | custom",
    )
    interaction_result: str = Field(
        description="Result: submitted | cancelled | timeout",
    )
    parameters_hash: str = Field(description="sha256 of invocation parameters")
    result_hash: str | None = Field(
        default=None, description="sha256 of user interaction result",
    )

    @classmethod
    def create(
        cls,
        app_id: str,
        component_type: str,
        interaction_result: str,
        parameters_hash: str,
        *,
        result_hash: str | None = None,
    ) -> McpAppInvocationEvent:
        return cls(
            app_id=app_id,
            component_type=component_type,
            interaction_result=interaction_result,
            parameters_hash=parameters_hash,
            result_hash=result_hash,
        )
