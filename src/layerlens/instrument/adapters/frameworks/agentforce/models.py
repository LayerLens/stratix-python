"""
Pydantic models for Salesforce Agentforce data structures.

Provides type-safe representations of:
- Salesforce DMO records (AIAgentSession, AIAgentInteractionStep, etc.)
- Agent API request/response payloads
- Platform Event payloads
- Trust Layer configuration
- LLM evaluation inputs/outputs
"""

from __future__ import annotations

from enum import Enum  # Python 3.11+ has StrEnum; using `(str, Enum)` for 3.9/3.10 compat.
from typing import Any, Optional

from pydantic import Field, BaseModel

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AgentChannelType(str, Enum):
    """Agentforce session channel types."""

    WEB = "Web"
    MESSAGING = "Messaging"
    VOICE = "Voice"
    SLACK = "Slack"
    API = "Api"


class SessionEndType(str, Enum):
    """How an Agentforce session ended."""

    COMPLETED = "Completed"
    ESCALATED = "Escalated"
    TIMED_OUT = "TimedOut"
    ERROR = "Error"
    ABANDONED = "Abandoned"


class StepType(str, Enum):
    """Agentforce interaction step types from DMO."""

    USER_INPUT = "UserInputStep"
    LLM_EXECUTION = "LLMExecutionStep"
    FUNCTION = "FunctionStep"
    ACTION_INVOCATION = "ActionInvocationStep"


class ParticipantType(str, Enum):
    """Participant roles in an Agentforce session."""

    AI = "ai"
    HUMAN = "human"


class AuthFlow(str, Enum):
    """Supported Salesforce authentication flows."""

    JWT_BEARER = "jwt_bearer"
    CLIENT_CREDENTIALS = "client_credentials"
    NAMED_CREDENTIAL = "named_credential"


class CaptureMode(str, Enum):
    """Agentforce capture modes."""

    POLLING = "polling"
    REALTIME = "realtime"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# DMO Record Models
# ---------------------------------------------------------------------------


class AgentSession(BaseModel):
    """AIAgentSession DMO record."""

    id: str = Field(alias="Id", description="Salesforce record ID")
    start_timestamp: Optional[str] = Field(
        default=None,
        alias="StartTimestamp",
        description="ISO 8601 session start time",
    )
    end_timestamp: Optional[str] = Field(
        default=None,
        alias="EndTimestamp",
        description="ISO 8601 session end time",
    )
    channel_type: Optional[str] = Field(
        default=None,
        alias="AiAgentChannelTypeId",
        description="Session channel (Web, Messaging, Voice, etc.)",
    )
    session_end_type: Optional[str] = Field(
        default=None,
        alias="AiAgentSessionEndType",
        description="How the session ended",
    )
    voice_call_id: Optional[str] = Field(default=None, alias="VoiceCallId")
    messaging_session_id: Optional[str] = Field(default=None, alias="MessagingSessionId")
    previous_session_id: Optional[str] = Field(default=None, alias="PreviousSessionId")

    model_config = {"populate_by_name": True}


class AgentParticipant(BaseModel):
    """AIAgentSessionParticipant DMO record."""

    id: str = Field(alias="Id")
    session_id: str = Field(alias="AiAgentSessionId")
    agent_type: Optional[str] = Field(default=None, alias="AiAgentTypeId")
    agent_api_name: Optional[str] = Field(default=None, alias="AiAgentApiName")
    agent_version: Optional[str] = Field(default=None, alias="AiAgentVersionApiName")
    participant_id: Optional[str] = Field(default=None, alias="ParticipantId")
    role: Optional[str] = Field(default=None, alias="AiAgentSessionParticipantRoleId")

    model_config = {"populate_by_name": True}


class AgentInteraction(BaseModel):
    """AIAgentInteraction DMO record."""

    id: str = Field(alias="Id")
    session_id: str = Field(alias="AiAgentSessionId")
    interaction_type: Optional[str] = Field(default=None, alias="AiAgentInteractionTypeId")
    telemetry_trace_id: Optional[str] = Field(default=None, alias="TelemetryTraceId")
    telemetry_span_id: Optional[str] = Field(default=None, alias="TelemetryTraceSpanId")
    topic_api_name: Optional[str] = Field(default=None, alias="TopicApiName")
    attribute_text: Optional[str] = Field(default=None, alias="AttributeText")
    prev_interaction_id: Optional[str] = Field(default=None, alias="PrevInteractionId")

    model_config = {"populate_by_name": True}


class AgentInteractionStep(BaseModel):
    """AIAgentInteractionStep DMO record."""

    id: str = Field(alias="Id")
    interaction_id: str = Field(alias="AiAgentInteractionId")
    step_type: Optional[str] = Field(default=None, alias="AiAgentInteractionStepTypeId")
    input_value: Optional[str] = Field(default=None, alias="InputValueText")
    output_value: Optional[str] = Field(default=None, alias="OutputValueText")
    error_message: Optional[str] = Field(default=None, alias="ErrorMessageText")
    generation_id: Optional[str] = Field(default=None, alias="GenerationId")
    gateway_request_id: Optional[str] = Field(default=None, alias="GenAiGatewayRequestId")
    gateway_response_id: Optional[str] = Field(default=None, alias="GenAiGatewayResponseId")
    name: Optional[str] = Field(default=None, alias="Name")
    telemetry_span_id: Optional[str] = Field(default=None, alias="TelemetryTraceSpanId")
    start_timestamp: Optional[str] = Field(default=None, alias="StartTimestamp")
    end_timestamp: Optional[str] = Field(default=None, alias="EndTimestamp")

    model_config = {"populate_by_name": True}


class AgentInteractionMessage(BaseModel):
    """AIAgentInteractionMessage DMO record."""

    id: str = Field(alias="Id")
    interaction_id: str = Field(alias="AiAgentInteractionId")
    message_type: Optional[str] = Field(default=None, alias="AiAgentInteractionMessageTypeId")
    content_text: Optional[str] = Field(default=None, alias="ContentText")
    content_type: Optional[str] = Field(default=None, alias="AiAgentInteractionMsgContentTypeId")
    sent_timestamp: Optional[str] = Field(default=None, alias="MessageSentTimestamp")
    parent_message_id: Optional[str] = Field(default=None, alias="ParentMessageId")

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Agent API Models
# ---------------------------------------------------------------------------


class AgentApiMessage(BaseModel):
    """A message in an Agent API session."""

    id: Optional[str] = Field(default=None, description="Message ID")
    role: str = Field(description="Message role (user, agent, system)")
    content: str = Field(description="Message content text")
    timestamp: Optional[str] = Field(default=None, description="ISO 8601 timestamp")
    topic: Optional[str] = Field(default=None, description="Classified topic name")
    actions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Actions taken by the agent",
    )
    guardrail_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Trust Layer guardrail check results",
    )


class AgentApiSession(BaseModel):
    """Represents an Agent API session."""

    session_id: str = Field(description="Salesforce session ID")
    agent_name: Optional[str] = Field(default=None, description="Agentforce agent name")
    status: str = Field(default="active", description="Session status")
    messages: list[AgentApiMessage] = Field(
        default_factory=list,
        description="Session messages in order",
    )
    created_at: Optional[str] = Field(default=None, description="Session creation timestamp")
    ended_at: Optional[str] = Field(default=None, description="Session end timestamp")


# ---------------------------------------------------------------------------
# Trust Layer Models
# ---------------------------------------------------------------------------


class TrustLayerGuardrail(BaseModel):
    """Einstein Trust Layer guardrail configuration."""

    name: str = Field(description="Guardrail name")
    type: str = Field(description="Guardrail type (toxicity, pii, custom)")
    enabled: bool = Field(default=True, description="Whether the guardrail is active")
    threshold: Optional[float] = Field(
        default=None,
        description="Detection threshold (0.0-1.0)",
    )
    action: str = Field(
        default="block",
        description="Action on violation (block, warn, log)",
    )


class TrustLayerConfig(BaseModel):
    """Complete Einstein Trust Layer configuration."""

    guardrails: list[TrustLayerGuardrail] = Field(
        default_factory=list,
        description="Configured guardrails",
    )
    data_masking_enabled: bool = Field(
        default=False,
        description="Whether PII/PCI masking is active",
    )
    zero_data_retention: bool = Field(
        default=True,
        description="Whether zero data retention is enabled for LLM calls",
    )
    audit_trail_enabled: bool = Field(
        default=True,
        description="Whether audit trail logging is active",
    )


# ---------------------------------------------------------------------------
# Platform Event Models
# ---------------------------------------------------------------------------


class AgentSessionEvent(BaseModel):
    """Platform Event payload for AgentSession__e."""

    session_id: str = Field(description="Agentforce session ID")
    agent_name: Optional[str] = Field(default=None, description="Agent name")
    topic_name: Optional[str] = Field(default=None, description="Classified topic")
    actions_taken: Optional[str] = Field(
        default=None,
        description="JSON-encoded actions list",
    )
    response_text: Optional[str] = Field(
        default=None,
        description="Agent response text",
    )
    trust_layer_flags: Optional[str] = Field(
        default=None,
        description="JSON-encoded Trust Layer results",
    )
    replay_id: Optional[str] = Field(
        default=None,
        description="Platform Event replay ID for redelivery",
    )


# ---------------------------------------------------------------------------
# Evaluation Models
# ---------------------------------------------------------------------------


class EvaluationRequest(BaseModel):
    """Request to evaluate Agentforce sessions."""

    session_ids: list[str] = Field(description="Salesforce session IDs to evaluate")
    graders: list[str] = Field(
        default_factory=lambda: ["relevance", "faithfulness"],
        description="Grader names to run",
    )
    include_ground_truth: bool = Field(
        default=False,
        description="Whether to fetch CRM outcome ground truth",
    )
    ground_truth_query: Optional[str] = Field(
        default=None,
        description="SOQL query for ground truth data",
    )


class EvaluationResult(BaseModel):
    """Result of evaluating Agentforce sessions."""

    session_id: str = Field(description="Evaluated session ID")
    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Grader name -> score mapping",
    )
    composite_score: Optional[float] = Field(
        default=None,
        description="Weighted composite quality score",
    )
    ground_truth: dict[str, Any] = Field(
        default_factory=dict,
        description="CRM outcome data if fetched",
    )
    errors: list[str] = Field(default_factory=list, description="Evaluation errors")
