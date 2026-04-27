"""Pydantic models for the Stratix Assistant SDK resource.

Mirrors the OpenAPI + AsyncAPI contracts published by atlas-app at
DOCS/api/assistant-{open,async}api.yaml. When the server-side spec
changes, update the schemas here in lockstep.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict


class AssistantToolCall(BaseModel):
    """A single tool the assistant invoked during a turn."""

    model_config = ConfigDict(protected_namespaces=())

    name: str
    args: str  # JSON-encoded argument string
    result: str


class AssistantConversation(BaseModel):
    """A conversation between the user and the assistant."""

    model_config = ConfigDict(protected_namespaces=())

    id: str
    organization_id: str
    project_id: str
    user_id: str
    title: str
    summary: Optional[str] = None
    summary_through_message_id: Optional[str] = None
    summary_updated_at: Optional[str] = None
    created_at: str
    updated_at: str


class AssistantMessage(BaseModel):
    """A single message in a conversation (user or assistant)."""

    model_config = ConfigDict(protected_namespaces=())

    id: str
    conversation_id: str
    organization_id: str
    role: Literal["user", "assistant"]
    content: str
    tool_calls: List[AssistantToolCall] = []
    created_at: str


class AssistantConversationList(BaseModel):
    """List response shape for GET .../assistant/conversations."""

    model_config = ConfigDict(protected_namespaces=())

    conversations: List[AssistantConversation]
    total: int


class AssistantMessageList(BaseModel):
    """List response shape for GET .../assistant/conversations/{id}/messages."""

    model_config = ConfigDict(protected_namespaces=())

    messages: List[AssistantMessage]
    total: int


# ── SSE event payloads (mirrors assistant-asyncapi.yaml) ──────────────────


class AssistantTokenUsage(BaseModel):
    """LLM token counts reported on the terminal `done` event."""

    model_config = ConfigDict(protected_namespaces=())

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AssistantStreamEvent(BaseModel):
    """One Server-Sent Event from the assistant chat stream.

    The `event` field names the channel (`token`, `tool_call`,
    `tool_result`, `done`, `moderation_refused`, `error`); `data` is
    the parsed JSON payload (shape varies by event). See
    DOCS/api/assistant-asyncapi.yaml for per-event schemas.

    Convenience accessors:
      - `is_terminal()` — true for done / moderation_refused / error.
      - `text()` — for `token` and `done` events, returns the content
        fragment / full content; empty string otherwise.
    """

    model_config = ConfigDict(protected_namespaces=())

    event: Literal[
        "token",
        "tool_call",
        "tool_result",
        "done",
        "moderation_refused",
        "error",
    ]
    data: Dict[str, Any]

    def is_terminal(self) -> bool:
        return self.event in {"done", "moderation_refused", "error"}

    def text(self) -> str:
        """Return text content if this is a token or done event, else empty string."""
        if self.event == "token":
            return str(self.data.get("content", ""))
        if self.event == "done":
            return str(self.data.get("content", ""))
        return ""
