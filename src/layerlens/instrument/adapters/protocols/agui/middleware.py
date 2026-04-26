"""
AG-UI ASGI/WSGI Middleware

Intercepts the SSE event stream between agent and frontend without
modifying either side. Each AG-UI event is translated to a Stratix
event before being forwarded unchanged.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from collections.abc import Callable

logger = logging.getLogger(__name__)


class AGUIASGIMiddleware:
    """
    ASGI middleware that intercepts AG-UI SSE streams.

    Wraps an ASGI application, detecting SSE responses and passing
    each event through the AG-UI adapter for tracing before forwarding
    to the client.

    Usage::

        app = AGUIASGIMiddleware(app, adapter=agui_adapter)
    """

    def __init__(self, app: Any, adapter: Any) -> None:
        self._app = app
        self._adapter = adapter

    async def __call__(
        self, scope: dict[str, Any], receive: Callable[..., Any], send: Callable[..., Any]
    ) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        is_sse = False

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal is_sse

            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                content_type = headers.get(b"content-type", b"").decode("utf-8", errors="replace")
                if "text/event-stream" in content_type:
                    is_sse = True

            if message["type"] == "http.response.body" and is_sse:
                body = message.get("body", b"")
                if body:
                    self._process_sse_chunk(body)

            await send(message)

        await self._app(scope, receive, send_wrapper)

    def _process_sse_chunk(self, chunk: bytes) -> None:
        """Parse SSE chunk and forward events to the adapter."""
        try:
            text = chunk.decode("utf-8", errors="replace")
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        continue
                    try:
                        data = json.loads(data_str)
                        event_type = data.get("type", data.get("event", ""))
                        if event_type:
                            self._adapter.on_agui_event(event_type, data)
                    except json.JSONDecodeError:
                        pass
        except Exception as exc:
            logger.debug("Failed to process SSE chunk: %s", exc)


class AGUIWSGIMiddleware:
    """
    WSGI middleware that intercepts AG-UI SSE streams.

    For non-async frameworks (Flask, Django WSGI, etc.).

    Usage::

        app = AGUIWSGIMiddleware(app, adapter=agui_adapter)
    """

    def __init__(self, app: Any, adapter: Any) -> None:
        self._app = app
        self._adapter = adapter

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> Any:
        response_started = False
        is_sse = False

        def custom_start_response(
            status: str, headers: list[Any], exc_info: Any = None
        ) -> Callable[..., Any]:
            nonlocal response_started, is_sse
            response_started = True
            for name, value in headers:
                if name.lower() == "content-type" and "text/event-stream" in value:
                    is_sse = True
                    break
            return start_response(status, headers, exc_info)  # type: ignore[no-any-return]

        result = self._app(environ, custom_start_response)

        if is_sse:
            return self._wrap_sse_response(result)
        return result

    def _wrap_sse_response(self, response: Any) -> Any:
        """Wrap SSE response iterator, processing each chunk."""
        for chunk in response:
            if isinstance(chunk, bytes):
                self._process_chunk(chunk)
            yield chunk

    def _process_chunk(self, chunk: bytes) -> None:
        """Parse SSE chunk and forward to adapter."""
        try:
            text = chunk.decode("utf-8", errors="replace")
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        continue
                    try:
                        data = json.loads(data_str)
                        event_type = data.get("type", data.get("event", ""))
                        if event_type:
                            self._adapter.on_agui_event(event_type, data)
                    except json.JSONDecodeError:
                        pass
        except Exception as exc:
            logger.debug("Failed to process SSE chunk: %s", exc)
