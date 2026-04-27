"""
Agent API Session to LayerLens Trace Event Mapper

Maps Agent API session data (from ``client.py``) to LayerLens canonical
event types. This is distinct from ``normalizer.py`` which handles
Data Cloud DMO records from SOQL queries.

Mapping:
- Session creation    -> agent.state.change (trace_start)
- User message        -> agent.input (L1)
- Agent response      -> agent.output (L1)
- Topic classification-> environment.config (L4a)
- Action invocation   -> tool.call (L5a)
- Guardrail check     -> policy.violation (Cross)
- Escalation          -> agent.handoff (Cross)
- Session end         -> agent.state.change (trace_end)
"""

from __future__ import annotations

import time
import logging
from typing import Any

from layerlens.instrument.adapters.frameworks.agentforce.models import (
    AgentApiMessage,
    AgentApiSession,
)

logger = logging.getLogger(__name__)


class AgentApiMapper:
    """
    Maps Agent API sessions to LayerLens trace events.

    Each public method returns a list of event dicts compatible with
    ``BaseAdapter.emit_dict_event(event_type, payload)``.
    """

    def map_session(self, session: AgentApiSession) -> list[dict[str, Any]]:
        """
        Map a complete Agent API session to a sequence of LayerLens events.

        Args:
            session: Complete AgentApiSession with messages.

        Returns:
            Ordered list of ``{event_type, payload}`` dicts.
        """
        events: list[dict[str, Any]] = []

        # Session start
        events.append(self.map_session_start(session))

        # Process each message
        seen_topics: set[str] = set()
        for msg in session.messages:
            if msg.role == "user":
                events.append(self.map_user_message(msg, session.session_id))
            elif msg.role == "agent":
                events.append(self.map_agent_response(msg, session.session_id))

                # Topic classification (emit once per topic)
                if msg.topic and msg.topic not in seen_topics:
                    events.append(
                        self.map_topic_classification(
                            msg.topic,
                            session.agent_name or "unknown",
                            session.session_id,
                        )
                    )
                    seen_topics.add(msg.topic)

                # Action invocations
                for action in msg.actions:
                    events.append(
                        self.map_action_invocation(
                            action,
                            session.session_id,
                        )
                    )

                # Guardrail checks
                for gr in msg.guardrail_results:
                    events.append(
                        self.map_guardrail_check(
                            gr,
                            session.session_id,
                        )
                    )

        # Session end
        events.append(self.map_session_end(session))

        return events

    def map_session_start(self, session: AgentApiSession) -> dict[str, Any]:
        """Map session creation to agent.state.change (trace_start)."""
        return {
            "event_type": "agent.state.change",
            "payload": {
                "framework": "salesforce_agentforce",
                "event_subtype": "trace_start",
                "session_id": session.session_id,
                "agent_name": session.agent_name,
                "timestamp_ns": _ts_to_ns(session.created_at),
            },
        }

    def map_session_end(self, session: AgentApiSession) -> dict[str, Any]:
        """Map session end to agent.state.change (trace_end)."""
        start_ns = _ts_to_ns(session.created_at)
        end_ns = _ts_to_ns(session.ended_at)
        duration_ns = end_ns - start_ns if start_ns and end_ns else 0

        return {
            "event_type": "agent.state.change",
            "payload": {
                "framework": "salesforce_agentforce",
                "event_subtype": "trace_end",
                "session_id": session.session_id,
                "agent_name": session.agent_name,
                "duration_ns": duration_ns,
                "message_count": len(session.messages),
            },
        }

    @staticmethod
    def map_user_message(
        msg: AgentApiMessage,
        session_id: str,
    ) -> dict[str, Any]:
        """Map a user message to agent.input (L1)."""
        return {
            "event_type": "agent.input",
            "payload": {
                "framework": "salesforce_agentforce",
                "session_id": session_id,
                "content": {
                    "role": "human",
                    "message": msg.content,
                },
                "timestamp_ns": _ts_to_ns(msg.timestamp),
            },
        }

    @staticmethod
    def map_agent_response(
        msg: AgentApiMessage,
        session_id: str,
    ) -> dict[str, Any]:
        """Map an agent response to agent.output (L1)."""
        return {
            "event_type": "agent.output",
            "payload": {
                "framework": "salesforce_agentforce",
                "session_id": session_id,
                "content": {
                    "role": "agent",
                    "message": msg.content,
                },
                "timestamp_ns": _ts_to_ns(msg.timestamp),
            },
        }

    @staticmethod
    def map_topic_classification(
        topic: str,
        agent_name: str,
        session_id: str,
    ) -> dict[str, Any]:
        """Map topic classification to environment.config (L4a)."""
        return {
            "event_type": "environment.config",
            "payload": {
                "framework": "salesforce_agentforce",
                "session_id": session_id,
                "agent_name": agent_name,
                "topic": topic,
                "config_type": "topic_classification",
            },
        }

    @staticmethod
    def map_action_invocation(
        action: dict[str, Any],
        session_id: str,
    ) -> dict[str, Any]:
        """Map an Agentforce action to tool.call (L5a)."""
        return {
            "event_type": "tool.call",
            "payload": {
                "framework": "salesforce_agentforce",
                "session_id": session_id,
                "tool_name": action.get("name", "unknown"),
                "tool_input": action.get("parameters", {}),
                "tool_output": action.get("result"),
                "tool_type": "salesforce_action",
            },
        }

    @staticmethod
    def map_guardrail_check(
        guardrail: dict[str, Any],
        session_id: str,
    ) -> dict[str, Any]:
        """Map a guardrail check to policy.violation (Cross-cutting)."""
        return {
            "event_type": "policy.violation",
            "payload": {
                "framework": "salesforce_agentforce",
                "session_id": session_id,
                "guardrail_name": guardrail.get("name", "unknown"),
                "triggered": guardrail.get("triggered", False),
                "message": guardrail.get("message"),
                "source": "einstein_trust_layer",
            },
        }

    @staticmethod
    def map_escalation(
        session_id: str,
        from_agent: str,
        to_agent: str = "human",
        reason: str = "escalation",
    ) -> dict[str, Any]:
        """Map an escalation to agent.handoff (Cross-cutting)."""
        return {
            "event_type": "agent.handoff",
            "payload": {
                "from_agent": from_agent,
                "to_agent": to_agent,
                "reason": reason,
                "framework": "salesforce_agentforce",
                "session_id": session_id,
            },
        }


def _ts_to_ns(ts: str | None) -> int:
    """Convert an ISO 8601 timestamp string to nanoseconds since epoch."""
    if not ts:
        return time.time_ns()
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1_000_000_000)
    except (ValueError, TypeError):
        return time.time_ns()
