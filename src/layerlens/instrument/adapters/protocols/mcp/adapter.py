"""MCP (Model Context Protocol) adapter.

Wraps an MCP ``ClientSession`` (or any object exposing ``call_tool`` /
``list_tools``) to capture tool-call lifecycle, structured outputs,
elicitation requests, and long-running async task tracking.
"""

from __future__ import annotations

import time
import uuid
import logging
from typing import Any, Dict, Callable

from ...._events import (
    MCP_TOOL_CALL,
    MCP_ASYNC_TASK,
    MCP_ELICITATION,
    MCP_STRUCTURED_OUTPUT,
)
from .elicitation import ElicitationTracker
from .._base_protocol import BaseProtocolAdapter
from .structured_output import (
    compute_output_hash,
    compute_schema_hash,
    validate_structured_output,
)
from .async_task_tracker import AsyncTaskTracker

log = logging.getLogger(__name__)


class MCPProtocolAdapter(BaseProtocolAdapter):
    """Instrument MCP client sessions.

    Patches (if present on the provided target):
    - ``call_tool(name, arguments)`` — emits ``mcp.tool.call`` + optional ``mcp.structured_output``
    - ``list_tools()`` — discovery telemetry
    - ``elicit(...)`` — emits ``mcp.elicitation``
    """

    PROTOCOL = "mcp"
    PROTOCOL_VERSION = "1.0.0"

    def __init__(self) -> None:
        super().__init__()
        self._async_tasks = AsyncTaskTracker()
        self._elicitations = ElicitationTracker()

    def connect(self, target: Any = None, **kwargs: Any) -> Any:  # noqa: ARG002
        self._client = target

        if hasattr(target, "call_tool"):
            orig = target.call_tool
            self._originals["call_tool"] = orig
            target.call_tool = self._wrap_call_tool(orig)

        if hasattr(target, "list_tools"):
            orig = target.list_tools
            self._originals["list_tools"] = orig
            target.list_tools = self._wrap_list_tools(orig)

        if hasattr(target, "elicit"):
            orig = target.elicit
            self._originals["elicit"] = orig
            target.elicit = self._wrap_elicit(orig)

        return target

    # --- wrappers ---

    def _wrap_call_tool(self, original: Callable[..., Any]) -> Callable[..., Any]:
        # Split by original signature: sync callers must not get a coroutine back.
        def _before(name: str, _arguments: Any) -> tuple[str, float]:
            parent = uuid.uuid4().hex[:16]
            start = time.time()
            self._emit_async_task_start(name, parent)
            return parent, start

        def _on_error(name: str, arguments: Any, parent: str, start: float, exc: Exception) -> None:
            self.emit(
                MCP_TOOL_CALL,
                {
                    "tool_name": name,
                    "arguments": arguments,
                    "error": str(exc),
                    "latency_ms": (time.time() - start) * 1000,
                },
                parent_span_id=parent,
            )
            self._emit_async_task_end(name, parent, error=str(exc))

        def _after(name: str, arguments: Any, parent: str, start: float, result: Any) -> None:
            latency_ms = (time.time() - start) * 1000
            self.emit(
                MCP_TOOL_CALL,
                {
                    "tool_name": name,
                    "arguments": arguments,
                    "result": _summarize(result),
                    "latency_ms": latency_ms,
                },
                parent_span_id=parent,
            )
            structured = _extract_structured_output(result)
            if structured is not None:
                schema = _extract_output_schema(result)
                payload: Dict[str, Any] = {
                    "tool_name": name,
                    "output_hash": compute_output_hash(structured),
                    "validation_passed": True,
                }
                if schema is not None:
                    payload["schema_hash"] = compute_schema_hash(schema)
                    ok, errors = validate_structured_output(structured, schema)
                    payload["validation_passed"] = ok
                    if errors:
                        payload["validation_errors"] = errors
                self.emit(MCP_STRUCTURED_OUTPUT, payload, parent_span_id=parent)
            self._emit_async_task_end(name, parent)

        if _is_awaitable(original):

            async def wrapped_async(name: str, arguments: Any = None, **kwargs: Any) -> Any:
                parent, start = _before(name, arguments)
                try:
                    result = await original(name, arguments, **kwargs)
                except Exception as exc:
                    _on_error(name, arguments, parent, start, exc)
                    raise
                _after(name, arguments, parent, start, result)
                return result

            return wrapped_async

        def wrapped_sync(name: str, arguments: Any = None, **kwargs: Any) -> Any:
            parent, start = _before(name, arguments)
            try:
                result = original(name, arguments, **kwargs)
            except Exception as exc:
                _on_error(name, arguments, parent, start, exc)
                raise
            _after(name, arguments, parent, start, result)
            return result

        return wrapped_sync

    def _wrap_list_tools(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def _emit(result: Any) -> None:
            tools = getattr(result, "tools", None) or (result if isinstance(result, list) else [])
            self.emit(
                "mcp.tools.listed",
                {
                    "tool_count": len(tools),
                    "tool_names": [getattr(t, "name", t) for t in tools[:50]],
                },
            )

        if _is_awaitable(original):

            async def wrapped_async(*args: Any, **kwargs: Any) -> Any:
                result = await original(*args, **kwargs)
                _emit(result)
                return result

            return wrapped_async

        def wrapped_sync(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            _emit(result)
            return result

        return wrapped_sync

    def _wrap_elicit(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def _before(args: tuple, kwargs: dict) -> tuple[str, str, Any, Any]:
            schema = kwargs.get("schema") or (args[1] if len(args) >= 2 else None)
            title = kwargs.get("title") or (args[0] if args else None)
            server_name = kwargs.get("server_name") or self.PROTOCOL
            parent = uuid.uuid4().hex[:16]
            eid = self._elicitations.start_request(server_name, schema, title)
            self.emit(
                MCP_ELICITATION,
                {
                    "elicitation_id": eid,
                    "title": title,
                    "schema_hash": ElicitationTracker.hash_schema(schema),
                    "phase": "request",
                },
                parent_span_id=parent,
            )
            return parent, eid, title, schema

        def _after(parent: str, eid: str, title: Any, result: Any) -> None:
            latency_ms = self._elicitations.complete_response(eid, action="submit", response=result)
            self.emit(
                MCP_ELICITATION,
                {
                    "elicitation_id": eid,
                    "title": title,
                    "phase": "response",
                    "response_hash": ElicitationTracker.hash_response(result),
                    "latency_ms": latency_ms,
                },
                parent_span_id=parent,
            )

        if _is_awaitable(original):

            async def wrapped_async(*args: Any, **kwargs: Any) -> Any:
                parent, eid, title, _schema = _before(args, kwargs)
                try:
                    result = await original(*args, **kwargs)
                except Exception:
                    self._elicitations.complete_response(eid, action="error")
                    raise
                _after(parent, eid, title, result)
                return result

            return wrapped_async

        def wrapped_sync(*args: Any, **kwargs: Any) -> Any:
            parent, eid, title, _schema = _before(args, kwargs)
            try:
                result = original(*args, **kwargs)
            except Exception:
                self._elicitations.complete_response(eid, action="error")
                raise
            _after(parent, eid, title, result)
            return result

        return wrapped_sync

    # --- async task lifecycle ---

    def _emit_async_task_start(self, name: str, parent_span_id: str) -> None:
        self._async_tasks.create(parent_span_id, originating_span_id=parent_span_id)
        payload = self._async_tasks.update(parent_span_id, status="running") or {
            "async_task_id": parent_span_id,
            "status": "running",
        }
        self.emit(
            MCP_ASYNC_TASK,
            {"tool_name": name, "phase": "start", **payload},
            parent_span_id=parent_span_id,
        )

    def _emit_async_task_end(self, name: str, parent_span_id: str, *, error: str | None = None) -> None:
        status = "failed" if error else "completed"
        payload = self._async_tasks.update(parent_span_id, status=status) or {
            "async_task_id": parent_span_id,
            "status": status,
        }
        payload["tool_name"] = name
        payload["phase"] = "end"
        if error:
            payload["error"] = error
        self.emit(MCP_ASYNC_TASK, payload, parent_span_id=parent_span_id)


def _is_awaitable(fn: Any) -> bool:
    import inspect

    return inspect.iscoroutinefunction(fn)


def _extract_structured_output(result: Any) -> Any:
    if result is None:
        return None
    for attr in ("structured_content", "structuredContent"):
        val = getattr(result, attr, None)
        if val is not None:
            return val
    if isinstance(result, dict):
        for key in ("structured_content", "structuredContent"):
            if key in result:
                return result[key]
    return None


def _extract_output_schema(result: Any) -> Any:
    """Best-effort lookup of a JSON Schema attached to a tool result."""
    if result is None:
        return None
    for attr in ("output_schema", "outputSchema"):
        val = getattr(result, attr, None)
        if val is not None:
            return val
    if isinstance(result, dict):
        for key in ("output_schema", "outputSchema"):
            if key in result:
                return result[key]
    return None


def _summarize(result: Any) -> Any:
    """Avoid dumping large tool results into telemetry — summarize shape."""
    if result is None:
        return None
    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")
    if isinstance(content, list):
        return {"content_items": len(content)}
    if isinstance(result, (str, int, float, bool)):
        return result
    return {"type": type(result).__name__}


def instrument_mcp(client: Any) -> MCPProtocolAdapter:
    from ..._registry import get, register

    existing = get("mcp")
    if existing is not None:
        existing.disconnect()
    adapter = MCPProtocolAdapter()
    adapter.connect(client)
    register("mcp", adapter)
    return adapter


def uninstrument_mcp() -> None:
    from ..._registry import unregister

    unregister("mcp")
