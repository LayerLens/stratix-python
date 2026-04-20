from __future__ import annotations

import json
import asyncio
from unittest.mock import MagicMock

from layerlens.instrument.adapters.protocols.agui.middleware import (
    AGUIASGIMiddleware,
    AGUIWSGIMiddleware,
    _process_sse_chunk,
)


class TestProcessSSEChunk:
    def test_routes_events_through_adapter(self):
        adapter = MagicMock()
        chunk = b'data: {"type": "TEXT_MESSAGE_START", "payload": {"id": 1}}\n\ndata: {"type": "TEXT_MESSAGE_END"}\n\n'
        _process_sse_chunk(adapter, chunk)
        assert adapter.emit.call_count == 2

    def test_empty_chunk_noop(self):
        adapter = MagicMock()
        _process_sse_chunk(adapter, b"")
        assert adapter.emit.call_count == 0

    def test_ignores_done_sentinel(self):
        adapter = MagicMock()
        _process_sse_chunk(adapter, b"data: [DONE]\n\n")
        assert adapter.emit.call_count == 0

    def test_ignores_invalid_json(self):
        adapter = MagicMock()
        _process_sse_chunk(adapter, b"data: {not-json\n\n")
        assert adapter.emit.call_count == 0

    def test_uses_event_field_fallback(self):
        adapter = MagicMock()
        _process_sse_chunk(adapter, b'data: {"event": "TEXT_MESSAGE_CONTENT", "text": "hi"}\n\n')
        assert adapter.emit.call_count == 1
        payload = adapter.emit.call_args.args[1]
        assert payload["agui_event"] == "TEXT_MESSAGE_CONTENT"

    def test_skips_events_without_type(self):
        adapter = MagicMock()
        _process_sse_chunk(adapter, b'data: {"payload": {}}\n\n')
        assert adapter.emit.call_count == 0


class TestASGIMiddleware:
    def test_non_http_passthrough(self):
        adapter = MagicMock()
        inner = MagicMock()

        async def app(scope, receive, send):
            inner(scope, receive, send)

        middleware = AGUIASGIMiddleware(app, adapter)
        asyncio.run(middleware({"type": "lifespan"}, MagicMock(), MagicMock()))
        inner.assert_called_once()
        assert adapter.emit.call_count == 0

    def test_captures_sse_body(self):
        adapter = MagicMock()
        sent = []

        async def fake_send(msg):
            sent.append(msg)

        async def app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'data: {"type": "RUN_STARTED"}\n\n',
                }
            )

        middleware = AGUIASGIMiddleware(app, adapter)
        asyncio.run(middleware({"type": "http"}, MagicMock(), fake_send))
        assert adapter.emit.call_count == 1
        payload = adapter.emit.call_args.args[1]
        assert payload["agui_event"] == "RUN_STARTED"
        # Original messages still flow to the real send.
        assert [m["type"] for m in sent] == ["http.response.start", "http.response.body"]

    def test_non_sse_response_not_processed(self):
        adapter = MagicMock()

        async def fake_send(msg):
            pass

        async def app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": json.dumps({"ok": 1}).encode()})

        middleware = AGUIASGIMiddleware(app, adapter)
        asyncio.run(middleware({"type": "http"}, MagicMock(), fake_send))
        assert adapter.emit.call_count == 0


class TestWSGIMiddleware:
    def test_sse_chunks_routed_through_adapter(self):
        adapter = MagicMock()

        def app(environ, start_response):
            start_response(
                "200 OK",
                [("Content-Type", "text/event-stream")],
            )
            yield b'data: {"type": "RUN_STARTED"}\n\n'
            yield b'data: {"type": "RUN_FINISHED"}\n\n'

        middleware = AGUIWSGIMiddleware(app, adapter)
        chunks = list(middleware({}, lambda *_: None))
        assert len(chunks) == 2
        assert adapter.emit.call_count == 2

    def test_non_sse_passthrough(self):
        adapter = MagicMock()

        def app(environ, start_response):
            start_response("200 OK", [("Content-Type", "application/json")])
            return [b'{"ok": 1}']

        middleware = AGUIWSGIMiddleware(app, adapter)
        chunks = list(middleware({}, lambda *_: None))
        # Non-SSE: chunks yielded verbatim, no emit.
        assert chunks == [b'{"ok": 1}']
        assert adapter.emit.call_count == 0
