"""Stratix Assistant SDK resource handler.

Mirrors the REST surface defined in atlas-app's
`DOCS/api/assistant-openapi.yaml` and the SSE event channel in
`DOCS/api/assistant-asyncapi.yaml`. The chat() method returns an
iterator of AssistantStreamEvent objects parsed from the SSE response.

Default-deny notice: the assistant is gated by a per-tier
`AssistantSDKEnabled` flag and a per-org daily token cap that
defaults to 0. Calling `chat()` for an unenabled organization will
raise `PermissionDeniedError` (403); calling it when the cap is 0
or exhausted will raise `RateLimitError` (429). Contact LayerLens
support to request access for your organization.
"""

from __future__ import annotations

import json
from typing import AsyncIterator, Iterator, Literal, Optional, cast

import httpx

from ...models import (
    AssistantConversation,
    AssistantConversationList,
    AssistantMessageList,
    AssistantStreamEvent,
)
from ..._constants import DEFAULT_TIMEOUT
from ..._resource import AsyncAPIResource, SyncAPIResource


def _unwrap(resp: object) -> object:
    """Unwrap {"status": ..., "data": ...} envelope if present."""
    if isinstance(resp, dict) and "data" in resp and "status" in resp:
        return resp["data"]
    return resp


def _parse_sse_block(block: str) -> Optional[AssistantStreamEvent]:
    """Parse one `event: ...\\ndata: ...` SSE block into an event object.

    Returns None for malformed / unrecognized blocks rather than raising
    — the stream contract specifies forward-compat: clients should
    skip events they don't understand and look for the terminal one.
    """
    event_name: Optional[str] = None
    data_line: Optional[str] = None
    for line in block.splitlines():
        if line.startswith("event: "):
            event_name = line[len("event: ") :].strip()
        elif line.startswith("data: "):
            data_line = line[len("data: ") :]
    if event_name is None or data_line is None:
        return None
    try:
        payload = json.loads(data_line)
    except json.JSONDecodeError:
        return None
    valid_events = {
        "token",
        "tool_call",
        "tool_result",
        "done",
        "moderation_refused",
        "error",
    }
    if event_name not in valid_events:
        # Forward-compat: unknown event types are skipped.
        return None
    # Narrow str → Literal after the membership check above. mypy can't
    # follow the in-set narrowing through a variable, so the cast is
    # required at the boundary; runtime correctness is enforced by the
    # check above.
    typed_event = cast(
        Literal["token", "tool_call", "tool_result", "done", "moderation_refused", "error"],
        event_name,
    )
    return AssistantStreamEvent(event=typed_event, data=payload)


class Assistant(SyncAPIResource):
    """Synchronous Stratix Assistant client.

    Usage:
        client = Stratix(api_key="...")
        conv = client.assistant.create_conversation(title="My session")
        for event in client.assistant.chat(conv.id, "What traces do I have?"):
            if event.event == "token":
                print(event.text(), end="", flush=True)
            elif event.is_terminal():
                break
    """

    def _base_url(self) -> str:
        return (
            f"/organizations/{self._client.organization_id}"
            f"/projects/{self._client.project_id}/assistant"
        )

    def list_conversations(
        self,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> AssistantConversationList:
        resp = self._get(
            f"{self._base_url()}/conversations",
            timeout=timeout,
            cast_to=dict,
        )
        data = _unwrap(resp)
        if isinstance(data, dict):
            return AssistantConversationList(**data)
        return AssistantConversationList(conversations=[], total=0)

    def create_conversation(
        self,
        *,
        title: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> AssistantConversation:
        body: dict[str, object] = {}
        if title is not None:
            body["title"] = title
        resp = self._post(
            f"{self._base_url()}/conversations",
            body=body,
            timeout=timeout,
            cast_to=dict,
        )
        data = _unwrap(resp)
        if not isinstance(data, dict):
            raise ValueError(f"unexpected create_conversation response shape: {type(data).__name__}")
        return AssistantConversation(**data)

    def get_conversation(
        self,
        conversation_id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> AssistantConversation:
        resp = self._get(
            f"{self._base_url()}/conversations/{conversation_id}",
            timeout=timeout,
            cast_to=dict,
        )
        data = _unwrap(resp)
        if not isinstance(data, dict):
            raise ValueError(f"unexpected get_conversation response shape: {type(data).__name__}")
        return AssistantConversation(**data)

    def rename_conversation(
        self,
        conversation_id: str,
        *,
        title: str,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> AssistantConversation:
        resp = self._patch(
            f"{self._base_url()}/conversations/{conversation_id}",
            body={"title": title},
            timeout=timeout,
            cast_to=dict,
        )
        data = _unwrap(resp)
        if not isinstance(data, dict):
            raise ValueError(f"unexpected rename_conversation response shape: {type(data).__name__}")
        return AssistantConversation(**data)

    def delete_conversation(
        self,
        conversation_id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> None:
        self._delete(
            f"{self._base_url()}/conversations/{conversation_id}",
            timeout=timeout,
            cast_to=dict,
        )

    def list_messages(
        self,
        conversation_id: str,
        *,
        limit: Optional[int] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> AssistantMessageList:
        params: dict[str, object] = {}
        if limit is not None:
            params["limit"] = limit
        resp = self._get(
            f"{self._base_url()}/conversations/{conversation_id}/messages",
            params=params if params else None,
            timeout=timeout,
            cast_to=dict,
        )
        data = _unwrap(resp)
        if isinstance(data, dict):
            return AssistantMessageList(**data)
        return AssistantMessageList(messages=[], total=0)

    def chat(
        self,
        conversation_id: str,
        content: str,
        *,
        timeout: float | httpx.Timeout | None = httpx.Timeout(300.0, connect=10.0),
    ) -> Iterator[AssistantStreamEvent]:
        """Send a message and yield SSE events as they arrive.

        Iterates until a terminal event (`done`, `moderation_refused`,
        `error`) or the stream closes. Caller decides how to react —
        typically: render `token` events in real time and break on the
        first terminal event.

        Raises:
            httpx.HTTPStatusError on 4xx/5xx (e.g. 403 if SDK access
            isn't enabled, 429 if the daily token budget is exhausted).
        """
        url = f"{self._client.base_url}{self._base_url()}/conversations/{conversation_id}/chat".replace(
            "//", "/"
        ).replace(":/", "://")
        headers = {
            **self._client.default_headers,
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=timeout) as http:
            with http.stream(
                "POST",
                url,
                headers=headers,
                json={"content": content},
            ) as response:
                response.raise_for_status()
                buffer: list[str] = []
                for line in response.iter_lines():
                    if line == "":
                        # Blank line terminates the SSE block.
                        if buffer:
                            event = _parse_sse_block("\n".join(buffer))
                            buffer.clear()
                            if event is not None:
                                yield event
                                if event.is_terminal():
                                    return
                    else:
                        buffer.append(line)
                # Flush any trailing block (server should always end on a blank line, defensively handle).
                if buffer:
                    event = _parse_sse_block("\n".join(buffer))
                    if event is not None:
                        yield event


class AsyncAssistant(AsyncAPIResource):
    """Asynchronous Stratix Assistant client. Surface mirrors the sync version."""

    def _base_url(self) -> str:
        return (
            f"/organizations/{self._client.organization_id}"
            f"/projects/{self._client.project_id}/assistant"
        )

    async def list_conversations(
        self,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> AssistantConversationList:
        resp = await self._get(
            f"{self._base_url()}/conversations",
            timeout=timeout,
            cast_to=dict,
        )
        data = _unwrap(resp)
        if isinstance(data, dict):
            return AssistantConversationList(**data)
        return AssistantConversationList(conversations=[], total=0)

    async def create_conversation(
        self,
        *,
        title: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> AssistantConversation:
        body: dict[str, object] = {}
        if title is not None:
            body["title"] = title
        resp = await self._post(
            f"{self._base_url()}/conversations",
            body=body,
            timeout=timeout,
            cast_to=dict,
        )
        data = _unwrap(resp)
        if not isinstance(data, dict):
            raise ValueError(f"unexpected create_conversation response shape: {type(data).__name__}")
        return AssistantConversation(**data)

    async def get_conversation(
        self,
        conversation_id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> AssistantConversation:
        resp = await self._get(
            f"{self._base_url()}/conversations/{conversation_id}",
            timeout=timeout,
            cast_to=dict,
        )
        data = _unwrap(resp)
        if not isinstance(data, dict):
            raise ValueError(f"unexpected get_conversation response shape: {type(data).__name__}")
        return AssistantConversation(**data)

    async def rename_conversation(
        self,
        conversation_id: str,
        *,
        title: str,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> AssistantConversation:
        resp = await self._patch(
            f"{self._base_url()}/conversations/{conversation_id}",
            body={"title": title},
            timeout=timeout,
            cast_to=dict,
        )
        data = _unwrap(resp)
        if not isinstance(data, dict):
            raise ValueError(f"unexpected rename_conversation response shape: {type(data).__name__}")
        return AssistantConversation(**data)

    async def delete_conversation(
        self,
        conversation_id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> None:
        await self._delete(
            f"{self._base_url()}/conversations/{conversation_id}",
            timeout=timeout,
            cast_to=dict,
        )

    async def list_messages(
        self,
        conversation_id: str,
        *,
        limit: Optional[int] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> AssistantMessageList:
        params: dict[str, object] = {}
        if limit is not None:
            params["limit"] = limit
        resp = await self._get(
            f"{self._base_url()}/conversations/{conversation_id}/messages",
            params=params if params else None,
            timeout=timeout,
            cast_to=dict,
        )
        data = _unwrap(resp)
        if isinstance(data, dict):
            return AssistantMessageList(**data)
        return AssistantMessageList(messages=[], total=0)

    async def chat(
        self,
        conversation_id: str,
        content: str,
        *,
        timeout: float | httpx.Timeout | None = httpx.Timeout(300.0, connect=10.0),
    ) -> AsyncIterator[AssistantStreamEvent]:
        """Async version of chat(). Yields SSE events as they arrive."""
        url = f"{self._client.base_url}{self._base_url()}/conversations/{conversation_id}/chat".replace(
            "//", "/"
        ).replace(":/", "://")
        headers = {
            **self._client.default_headers,
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=timeout) as http:
            async with http.stream(
                "POST",
                url,
                headers=headers,
                json={"content": content},
            ) as response:
                response.raise_for_status()
                buffer: list[str] = []
                async for line in response.aiter_lines():
                    if line == "":
                        if buffer:
                            event = _parse_sse_block("\n".join(buffer))
                            buffer.clear()
                            if event is not None:
                                yield event
                                if event.is_terminal():
                                    return
                    else:
                        buffer.append(line)
                if buffer:
                    event = _parse_sse_block("\n".join(buffer))
                    if event is not None:
                        yield event
