"""
AgentForce DMO to STRATIX Event Normalizer

Maps AgentForce Data Model Objects to STRATIX canonical event types:
- AIAgentSession → agent.lifecycle (start/end)
- AIAgentSessionParticipant → agent.identity
- AIAgentInteraction → agent.input / agent.output
- AIAgentInteractionStep (UserInputStep) → agent.input (L1)
- AIAgentInteractionStep (LLMExecutionStep) → model.invoke (L3)
- AIAgentInteractionStep (FunctionStep / ActionInvocationStep) → tool.call (L5)
- AIAgentInteractionMessage (Input) → agent.input
- AIAgentInteractionMessage (Output) → agent.output
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Step type to STRATIX event type mapping
_STEP_TYPE_MAP = {
    "UserInputStep": "agent.input",
    "LLMExecutionStep": "model.invoke",
    "FunctionStep": "tool.call",
    "ActionInvocationStep": "tool.call",
}


class AgentForceNormalizer:
    """Normalize AgentForce DMO records to STRATIX events."""

    def normalize_session(
        self,
        session: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Normalize an AIAgentSession to agent.lifecycle start/end events."""
        events = []

        sf_meta = {
            "sf.session.id": session.get("Id"),
            "sf.session.channel": session.get("AiAgentChannelTypeId"),
            "sf.session.end_type": session.get("AiAgentSessionEndType"),
        }

        # Start event
        events.append({
            "event_type": "agent.lifecycle",
            "payload": {
                "lifecycle_action": "start",
                "session_id": session.get("Id"),
                "channel_type": session.get("AiAgentChannelTypeId"),
                "previous_session_id": session.get("PreviousSessionId"),
                "voice_call_id": session.get("VoiceCallId"),
                "messaging_session_id": session.get("MessagingSessionId"),
            },
            "metadata": sf_meta,
            "timestamp": session.get("StartTimestamp"),
        })

        # End event (if session has ended)
        end_ts = session.get("EndTimestamp")
        if end_ts:
            events.append({
                "event_type": "agent.lifecycle",
                "payload": {
                    "lifecycle_action": "end",
                    "session_id": session.get("Id"),
                    "session_end_type": session.get("AiAgentSessionEndType"),
                    "channel_type": session.get("AiAgentChannelTypeId"),
                },
                "metadata": sf_meta,
                "timestamp": end_ts,
            })

        return events

    def normalize_participant(
        self,
        participant: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize an AIAgentSessionParticipant to agent identity metadata."""
        agent_type = participant.get("AiAgentTypeId", "")
        is_human = agent_type == "Employee"

        return {
            "event_type": "agent.identity",
            "payload": {
                "participant_type": "human" if is_human else "ai",
                "agent_type": agent_type,
                "agent_api_name": participant.get("AiAgentApiName"),
                "agent_version": participant.get("AiAgentVersionApiName"),
                "participant_id": participant.get("ParticipantId"),
                "role": participant.get("AiAgentSessionParticipantRoleId"),
                "session_id": participant.get("AiAgentSessionId"),
            },
        }

    def normalize_interaction(
        self,
        interaction: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize an AIAgentInteraction to a trace span."""
        # Parse AttributeText as JSON if present
        attr_text = interaction.get("AttributeText")
        attributes = {}
        if attr_text:
            try:
                attributes = json.loads(attr_text)
            except (json.JSONDecodeError, TypeError):
                attributes = {"raw": attr_text}

        return {
            "event_type": "agent.interaction",
            "identity": {
                "trace_id": interaction.get("TelemetryTraceId"),
                "span_id": interaction.get("TelemetryTraceSpanId"),
            },
            "payload": {
                "interaction_id": interaction.get("Id"),
                "interaction_type": interaction.get("AiAgentInteractionTypeId"),
                "topic": interaction.get("TopicApiName"),
                "attributes": attributes,
                "prev_interaction_id": interaction.get("PrevInteractionId"),
                "session_id": interaction.get("AiAgentSessionId"),
            },
            "metadata": {
                "sf.topic.name": interaction.get("TopicApiName"),
                "sf.session.id": interaction.get("AiAgentSessionId"),
            },
        }

    def normalize_step(
        self,
        step: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize an AIAgentInteractionStep to the appropriate STRATIX event."""
        step_type = step.get("AiAgentInteractionStepTypeId", "")
        event_type = _STEP_TYPE_MAP.get(step_type, "tool.call")

        base: dict[str, Any] = {
            "event_type": event_type,
            "identity": {
                "span_id": step.get("TelemetryTraceSpanId"),
            },
        }

        # Salesforce metadata passthrough
        base["metadata"] = {
            "sf.step.name": step.get("Name"),
            "sf.step.id": step.get("Id"),
            "sf.generation.id": step.get("GenerationId"),
        }

        # Extract timing if available
        start_ts = step.get("StartTimestamp")
        end_ts = step.get("EndTimestamp")
        if start_ts:
            base["timestamp"] = start_ts
        if start_ts and end_ts:
            try:
                start_dt = datetime.fromisoformat(str(start_ts).replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(str(end_ts).replace("Z", "+00:00"))
                base["duration_ms"] = (end_dt - start_dt).total_seconds() * 1000
            except (ValueError, TypeError):
                pass

        if event_type == "model.invoke":
            base["payload"] = {
                "model": {
                    "provider": "salesforce",
                    "name": step.get("Name", "unknown"),
                    "version": "unavailable",
                    "parameters": {},
                },
                "input_messages": [{"role": "user", "content": step.get("InputValueText", "")}],
                "output_message": {"role": "assistant", "content": step.get("OutputValueText", "")},
                "error": step.get("ErrorMessageText"),
                "metadata": {
                    "generation_id": step.get("GenerationId"),
                    "gateway_request_id": step.get("GenAiGatewayRequestId"),
                    "gateway_response_id": step.get("GenAiGatewayResponseId"),
                },
            }

        elif event_type == "tool.call":
            input_text = step.get("InputValueText", "")
            output_text = step.get("OutputValueText")

            base["payload"] = {
                "tool": {
                    "name": step.get("Name", "unknown"),
                    "version": "unavailable",
                    "integration": "salesforce_agentforce",
                },
                "input": _try_parse_json(input_text),
                "output": _try_parse_json(output_text) if output_text else None,
                "error": step.get("ErrorMessageText"),
            }

        else:  # agent.input
            base["payload"] = {
                "content": {
                    "role": "human",
                    "message": step.get("InputValueText", ""),
                },
            }

        return base

    def normalize_message(
        self,
        message: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize an AIAgentInteractionMessage to agent.input or agent.output."""
        msg_type = message.get("AiAgentInteractionMessageTypeId", "")
        event_type = "agent.output" if msg_type == "Output" else "agent.input"
        role = "agent" if msg_type == "Output" else "human"

        return {
            "event_type": event_type,
            "payload": {
                "content": {
                    "role": role,
                    "message": message.get("ContentText", ""),
                    "metadata": {
                        "content_type": message.get("AiAgentInteractionMsgContentTypeId"),
                        "parent_message_id": message.get("ParentMessageId"),
                    },
                },
            },
            "timestamp": message.get("MessageSentTimestamp"),
        }


def _try_parse_json(text: str) -> dict[str, Any]:
    """Try to parse text as JSON, falling back to raw string wrapper."""
    if not text:
        return {}
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {"raw": text}
    except (json.JSONDecodeError, TypeError):
        return {"raw": text}
