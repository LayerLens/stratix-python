"""
Salesforce Agent API REST Client

Provides a typed client for the Salesforce Agent API:
- Session creation and lifecycle management
- Synchronous and streaming message exchange
- Response parsing with action and guardrail extraction

Reference: https://developer.salesforce.com/docs/ai/agentforce/guide/agent-api.html
"""

from __future__ import annotations

import json
import time
import logging
from typing import Any
from collections.abc import Generator

from layerlens.instrument.adapters.frameworks.agentforce.auth import (
    SalesforceConnection,
    SalesforceQueryError,
)
from layerlens.instrument.adapters.frameworks.agentforce.models import (
    AgentApiMessage,
    AgentApiSession,
)

logger = logging.getLogger(__name__)

# Agent API path prefix
_AGENT_API_PREFIX = "/services/data/{version}/agent"

# Default timeout for Agent API calls (seconds)
_API_TIMEOUT = 30

# Maximum response text length to capture (prevent memory bloat)
_MAX_RESPONSE_LENGTH = 50_000


class AgentApiClient:
    """
    REST client for the Salesforce Agent API.

    Wraps session creation, message exchange, and response parsing.
    All methods use the authenticated ``SalesforceConnection`` for
    token management and retry logic.

    Usage:
        client = AgentApiClient(connection=connection)
        session = client.create_session(agent_name="Service_Agent")
        response = client.send_message(session.session_id, "How do I reset my password?")
        client.end_session(session.session_id)
    """

    def __init__(
        self,
        connection: SalesforceConnection,
        api_timeout: int = _API_TIMEOUT,
    ) -> None:
        self._connection = connection
        self._api_timeout = api_timeout
        self._base_url = ""

    @property
    def base_url(self) -> str:
        """Build the Agent API base URL from the connection."""
        if not self._base_url:
            instance = self._connection.instance_url
            version = self._connection.api_version
            self._base_url = f"{instance}{_AGENT_API_PREFIX.format(version=version)}"
        return self._base_url

    def create_session(
        self,
        agent_name: str,
        context: dict[str, Any] | None = None,
    ) -> AgentApiSession:
        """
        Create a new Agent API session.

        Args:
            agent_name: Name of the Agentforce agent to connect to.
            context: Optional context variables for the session.

        Returns:
            AgentApiSession with the session ID and initial state.

        Raises:
            SalesforceQueryError: If the API call fails.
        """
        import requests  # type: ignore[import-untyped,unused-ignore]

        if not agent_name or not agent_name.strip():
            raise ValueError("agent_name must be a non-empty string")

        if self._connection.credentials.is_expired:
            self._connection.authenticate()

        url = f"{self.base_url}/sessions"
        headers = {
            "Authorization": f"Bearer {self._connection.credentials.access_token}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {"agentName": agent_name}
        if context:
            body["context"] = context

        try:
            response = requests.post(
                url,
                headers=headers,
                json=body,
                timeout=self._api_timeout,
            )
            response.raise_for_status()
            data = response.json()

            return AgentApiSession(
                session_id=data.get("sessionId", ""),
                agent_name=agent_name,
                status="active",
                created_at=data.get("createdAt"),
            )
        except requests.exceptions.RequestException as e:
            raise SalesforceQueryError(
                f"Failed to create Agent API session: {e}",
                status_code=getattr(getattr(e, "response", None), "status_code", None),
            ) from e

    def send_message(
        self,
        session_id: str,
        message: str,
        stream: bool = False,
    ) -> AgentApiMessage | Generator[str, None, None]:
        """
        Send a message to an active Agent API session.

        Args:
            session_id: The session ID from ``create_session()``.
            message: User message text to send.
            stream: If True, return a generator of streaming response chunks.

        Returns:
            AgentApiMessage with the agent response, or a generator if streaming.

        Raises:
            SalesforceQueryError: If the API call fails.
        """
        if not session_id or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if not message or not message.strip():
            raise ValueError("message must be a non-empty string")

        import requests

        if self._connection.credentials.is_expired:
            self._connection.authenticate()

        url = f"{self.base_url}/sessions/{session_id}/messages"
        headers = {
            "Authorization": f"Bearer {self._connection.credentials.access_token}",
            "Content-Type": "application/json",
        }
        if stream:
            headers["Accept"] = "text/event-stream"

        body = {"message": {"text": message}}

        try:
            response = requests.post(
                url,
                headers=headers,
                json=body,
                timeout=self._api_timeout,
                stream=stream,
            )
            response.raise_for_status()

            if stream:
                return self._stream_response(response)

            return self._parse_message_response(response.json())

        except requests.exceptions.RequestException as e:
            raise SalesforceQueryError(
                f"Failed to send Agent API message: {e}",
                status_code=getattr(getattr(e, "response", None), "status_code", None),
            ) from e

    def end_session(self, session_id: str) -> None:
        """
        End an active Agent API session.

        Args:
            session_id: The session ID to end.

        Raises:
            SalesforceQueryError: If the API call fails.
        """
        if not session_id or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")

        import requests

        if self._connection.credentials.is_expired:
            self._connection.authenticate()

        url = f"{self.base_url}/sessions/{session_id}"
        headers = {
            "Authorization": f"Bearer {self._connection.credentials.access_token}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.delete(
                url,
                headers=headers,
                timeout=self._api_timeout,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise SalesforceQueryError(
                f"Failed to end Agent API session: {e}",
                status_code=getattr(getattr(e, "response", None), "status_code", None),
            ) from e

    def capture_session(
        self,
        agent_name: str,
        messages: list[str],
        context: dict[str, Any] | None = None,
    ) -> AgentApiSession:
        """
        Convenience method: create session, send all messages, end session.

        Returns an ``AgentApiSession`` with all messages and responses.

        Args:
            agent_name: Agentforce agent name.
            messages: List of user messages to send sequentially.
            context: Optional session context.

        Returns:
            Complete AgentApiSession with all exchanged messages.
        """
        session = self.create_session(agent_name, context)
        all_messages: list[AgentApiMessage] = []

        for msg_text in messages:
            # Record user message
            all_messages.append(AgentApiMessage(role="user", content=msg_text))

            # Send and capture response
            response = self.send_message(session.session_id, msg_text)
            if isinstance(response, AgentApiMessage):
                all_messages.append(response)

        self.end_session(session.session_id)

        session.messages = all_messages
        session.status = "ended"
        session.ended_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return session

    # --- Internal helpers ---

    @staticmethod
    def _parse_message_response(data: dict[str, Any]) -> AgentApiMessage:
        """Parse a synchronous Agent API message response."""
        messages = data.get("messages", [])
        if not messages:
            return AgentApiMessage(
                role="agent",
                content=data.get("text", ""),
                timestamp=data.get("timestamp"),
            )

        # Take the last agent message
        last = messages[-1]
        actions = []
        guardrails = []

        # Extract actions if present
        for action in data.get("actions", []):
            actions.append(
                {
                    "name": action.get("name", "unknown"),
                    "parameters": action.get("parameters", {}),
                    "result": action.get("result"),
                }
            )

        # Extract guardrail results if present
        for gr in data.get("guardrailResults", []):
            guardrails.append(
                {
                    "name": gr.get("name", "unknown"),
                    "triggered": gr.get("triggered", False),
                    "message": gr.get("message"),
                }
            )

        return AgentApiMessage(
            id=last.get("id"),
            role="agent",
            content=str(last.get("text", ""))[:_MAX_RESPONSE_LENGTH],
            timestamp=last.get("timestamp"),
            topic=data.get("topic"),
            actions=actions,
            guardrail_results=guardrails,
        )

    @staticmethod
    def _stream_response(response: Any) -> Generator[str, None, None]:
        """Parse a streaming Agent API response (SSE format)."""
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data_str)
                        text = chunk.get("text", "")
                        if text:
                            yield text
                    except json.JSONDecodeError:
                        logger.debug("Failed to parse SSE chunk: %s", data_str[:100])
        finally:
            response.close()
