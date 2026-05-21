"""ASGI / WSGI middleware that intercepts AG-UI SSE streams.

Wraps an application and inspects outbound ``text/event-stream`` bodies,
routing each decoded event through an :class:`AGUIProtocolAdapter`
without modifying the response.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict
from collections.abc import Callable

from ...._events import PROTOCOL_STREAM_EVENT
from .event_mapper import map_agui_to_stratix

log = logging.getLogger(__name__)


def _emit_event(adapter: Any, event_type: str, data: Dict[str, Any]) -> None:
    """Forward a decoded AG-UI event to the adapter's emit pipeline."""
    mapping = map_agui_to_stratix(event_type)
    adapter.emit(
        mapping.get("stratix_event", PROTOCOL_STREAM_EVENT),
        {
            "protocol": "agui",
            "agui_event": event_type,
            "category": mapping.get("category", "unknown"),
            "data": data,
        },
    )


def _process_sse_chunk(adapter: Any, chunk: bytes) -> None:
    if not chunk:
        return
    try:
        text = chunk.decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - decode failure
        log.debug("AG-UI middleware: decode failed: %s", exc)
        return
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            continue
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        event_type = data.get("type") or data.get("event") or ""
        if event_type:
            _emit_event(adapter, str(event_type), data)


class AGUIASGIMiddleware:
    """ASGI middleware intercepting AG-UI SSE responses.

    Usage::

        app = AGUIASGIMiddleware(app, adapter=agui_adapter)
    """

    def __init__(self, app: Any, adapter: Any) -> None:
        self._app = app
        self._adapter = adapter

    async def __call__(
        self,
        scope: Dict[str, Any],
        receive: Callable[..., Any],
        send: Callable[..., Any],
    ) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return

        is_sse = False

        async def send_wrapper(message: Dict[str, Any]) -> None:
            nonlocal is_sse
            if message.get("type") == "http.response.start":
                for name, value in message.get("headers", []) or []:
                    if name.lower() == b"content-type" and b"text/event-stream" in value:
                        is_sse = True
                        break
            elif message.get("type") == "http.response.body" and is_sse:
                body = message.get("body", b"") or b""
                if body:
                    _process_sse_chunk(self._adapter, body)
            await send(message)

        await self._app(scope, receive, send_wrapper)


class AGUIWSGIMiddleware:
    """WSGI middleware intercepting AG-UI SSE responses.

    Usage::

        app = AGUIWSGIMiddleware(app, adapter=agui_adapter)
    """

    def __init__(self, app: Any, adapter: Any) -> None:
        self._app = app
        self._adapter = adapter

    def __call__(
        self,
        environ: Dict[str, Any],
        start_response: Callable[..., Any],
    ) -> Any:
        # Flag is set by ``custom_start_response``; for generator-style WSGI
        # apps that only invoke ``start_response`` on first iteration, we
        # always return a wrapper and consult the flag per-chunk.
        is_sse = [False]

        def custom_start_response(
            status: str,
            headers: list,
            exc_info: Any = None,
        ) -> Callable[..., Any]:
            for name, value in headers:
                if name.lower() == "content-type" and "text/event-stream" in value:
                    is_sse[0] = True
                    break
            return start_response(status, headers, exc_info)

        result = self._app(environ, custom_start_response)
        return self._wrap_response(result, is_sse)

    def _wrap_response(self, response: Any, is_sse: list) -> Any:
        for chunk in response:
            if is_sse[0] and isinstance(chunk, (bytes, bytearray)):
                _process_sse_chunk(self._adapter, bytes(chunk))
            yield chunk
