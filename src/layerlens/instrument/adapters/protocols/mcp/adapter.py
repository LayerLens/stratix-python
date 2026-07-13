"""MCP (Model Context Protocol) adapter — real ``mcp`` SDK 1.27 (spec 2025-11-25).

Instruments a real MCP :class:`mcp.client.session.ClientSession` (the agent's
side) and/or a FastMCP server context, capturing the four load-bearing surfaces:

* **Tool calls** (``ClientSession.call_tool`` → ``mcp.tool.call`` +
  ``mcp.structured_output``) and **tool discovery** (``list_tools``).
* **Elicitation** — server-initiated consent prompts. On a real ``ClientSession``
  these do NOT arrive via a method: the server's ``elicitation/create`` request is
  dispatched to the user-supplied ``_elicitation_callback`` instance attribute
  (``mcp/client/session.py:577``). We wrap that callback (and also the FastMCP
  server ``elicit`` method, where THAT is the real surface). We read the REAL
  ``mcp.types.ElicitResult.action`` (``accept`` / ``decline`` / ``cancel``,
  ``types.py:1898``) and emit it — a decline/cancel is consent-faithful: it is
  distinguishable from an accept and carries NO hash of a refused payload (D1).
  The prompt ``message`` (``ElicitRequestFormParams.message`` / URL params) is
  CONTENT, stripped under ``capture_content=False`` (D2). Form vs URL mode is
  distinguished — URL mode is the sensitive credential/OAuth/payment flow whose
  ``ElicitResult.content`` is absent (D6).
* **Sampling** (``sampling/createMessage``) — the server asks the CLIENT's LLM to
  generate, nested inside an MCP feature. This is the agentic, money-burning path.
  It also arrives via a callback (``_sampling_callback``, ``session.py:567``), not
  a method. We emit ``mcp.sampling`` AND a ``cost.record`` carrying the model
  (``CreateMessageResult.model``) + token counts so the central price-on-emit
  chokepoint (``_collector.emit`` → ``price_cost_record``) bills it. The wire
  carries NO token usage on a sampling result, so token counts are ESTIMATED from
  the request/response text (chars/4) and flagged ``tokens_estimated=True`` (D3).

We do not re-implement the protocol; we wrap user-supplied callables on the real
session object. ``connect()`` attaches against a real ``ClientSession`` (or a
FastMCP server / any duck-typed target) and ``disconnect()`` restores everything.
"""

from __future__ import annotations

import time
import uuid
import inspect
import logging
from typing import Any, Dict, Callable, Optional

from ...._events import (
    AGENT_ERROR,
    COST_RECORD,
    MCP_SAMPLING,
    MCP_TOOL_CALL,
    MCP_ASYNC_TASK,
    MCP_ELICITATION,
    MCP_TOOLS_LISTED,
    MCP_SERVER_CONNECTED,
    MCP_STRUCTURED_OUTPUT,
)
from ...._context import _current_span_id
from .elicitation import ElicitationTracker
from .._base_protocol import BaseProtocolAdapter
from .structured_output import (
    compute_output_hash,
    compute_schema_hash,
    validate_structured_output,
)
from .async_task_tracker import AsyncTaskTracker

log = logging.getLogger(__name__)

# Fail-CLOSED structured-output sentinel (D4): when no output schema is available
# to validate against, we MUST NOT assert validation passed. "unknown" is a
# distinct, non-True value so a malformed/unschema'd structured output is never
# reported as validated-OK by default.
_VALIDATION_UNKNOWN = "unknown"


class MCPProtocolAdapter(BaseProtocolAdapter):
    """Instrument MCP client sessions / FastMCP server contexts."""

    PROTOCOL = "mcp"
    # The real MCP spec is date-stamped, not semver. We pin the spec version of
    # the installed lib (mcp 1.27 ⇒ 2025-11-25) so telemetry records the real
    # negotiated protocol generation, not a made-up "1.0.0".
    PROTOCOL_VERSION = "2025-11-25"

    def __init__(self, *, capture_config: Any = None) -> None:
        super().__init__(capture_config=capture_config)
        self._async_tasks = AsyncTaskTracker()
        self._elicitations = ElicitationTracker()
        # The server's declared name, captured at the initialize() handshake and
        # fed to the ElicitationTracker (S14/F7). None until a handshake is seen.
        self._server_name: str | None = None

    def connect(self, target: Any = None, **kwargs: Any) -> Any:  # noqa: ARG002
        """Attach to a real MCP surface.

        On a ``ClientSession`` the server-initiated surfaces (elicitation,
        sampling) live as the ``_elicitation_callback`` / ``_sampling_callback``
        INSTANCE ATTRIBUTES (async callables), not methods — so we wrap those.
        ``call_tool`` / ``list_tools`` ARE methods. On a FastMCP server context
        the real surface is the ``elicit`` method, which we also support.
        """
        self._client = target

        if hasattr(target, "call_tool"):
            orig = target.call_tool
            self._originals["call_tool"] = orig
            target.call_tool = self._wrap_call_tool(orig)

        if hasattr(target, "list_tools"):
            orig = target.list_tools
            self._originals["list_tools"] = orig
            target.list_tools = self._wrap_list_tools(orig)

        # Capture the server identity at the initialize() handshake (S14/F7).
        if hasattr(target, "initialize"):
            orig = target.initialize
            self._originals["initialize"] = orig
            target.initialize = self._wrap_initialize(orig)

        # Real ClientSession server-initiated callbacks (the live surface, D5).
        if hasattr(target, "_elicitation_callback"):
            orig = target._elicitation_callback
            self._originals["_elicitation_callback"] = orig
            target._elicitation_callback = self._wrap_elicitation_callback(orig)

        if hasattr(target, "_sampling_callback"):
            orig = target._sampling_callback
            self._originals["_sampling_callback"] = orig
            target._sampling_callback = self._wrap_sampling_callback(orig)

        # FastMCP / server-side elicit METHOD (where THAT is the real surface).
        if hasattr(target, "elicit"):
            orig = target.elicit
            self._originals["elicit"] = orig
            target.elicit = self._wrap_elicit(orig)

        return target

    # ── tool calls ──────────────────────────────────────────────────────────

    def _wrap_call_tool(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def _before(name: str, _arguments: Any) -> tuple[str, float]:
            parent = _current_span_id.get() or uuid.uuid4().hex[:16]
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
                    "error_type": type(exc).__name__,
                    "status": "error",
                    "latency_ms": (time.time() - start) * 1000,
                },
                parent_span_id=parent,
            )
            # Terminal tool failure -> agent.error so the trace's derived status
            # is error, not completed — the mcp.tool.call status is read by no
            # engine (S12/F4).
            self.emit(
                AGENT_ERROR,
                {
                    "tool_name": name,
                    "source": "mcp",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
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
            self._emit_structured_output(name, result, parent)
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

    def _emit_structured_output(self, name: str, result: Any, parent: str) -> None:
        """Validate ``CallToolResult.structuredContent`` against the TOOL's
        ``outputSchema`` (threaded from ``list_tools``) — FAIL CLOSED (D4).

        ``outputSchema`` lives on the ``Tool`` definition, NOT on
        ``CallToolResult`` (``types.py:1322`` vs ``:1363``). The old code looked
        it up on the result, so the schema was ~always absent and
        ``validation_passed`` was hardcoded ``True`` (fail-OPEN). We instead read
        the schema from the real ``ClientSession._tool_output_schemas`` cache and,
        when no schema is available, report ``validation_passed="unknown"`` — never
        an unearned ``True``.
        """
        structured = _extract_structured_output(result)
        if structured is None:
            return
        schema = self._lookup_output_schema(name, result)
        payload: Dict[str, Any] = {
            "tool_name": name,
            "output_hash": compute_output_hash(structured),
        }
        if schema is not None:
            payload["schema_hash"] = compute_schema_hash(schema)
            ok, errors = validate_structured_output(structured, schema)
            payload["validation_passed"] = ok
            if errors:
                payload["validation_errors"] = errors
        else:
            # No schema to validate against → cannot assert success. Fail closed.
            payload["validation_passed"] = _VALIDATION_UNKNOWN
        self.emit(MCP_STRUCTURED_OUTPUT, payload, parent_span_id=parent)

    def _lookup_output_schema(self, name: str, result: Any) -> Any:
        """Resolve the Tool.outputSchema for ``name``.

        Preference order: (1) the real ClientSession's ``_tool_output_schemas``
        cache (populated by ``list_tools``, ``session.py:537``); (2) a schema
        threaded onto the result by an integrator (best-effort back-compat).
        NEVER falls back to "no schema ⇒ valid".
        """
        cache = getattr(self._client, "_tool_output_schemas", None)
        if isinstance(cache, dict) and name in cache:
            return cache.get(name)  # may legitimately be None (tool has no schema)
        return _extract_output_schema(result)

    # ── tool discovery ──────────────────────────────────────────────────────

    def _wrap_list_tools(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def _emit(result: Any) -> None:
            tools = getattr(result, "tools", None) or (result if isinstance(result, list) else [])
            self.emit(
                MCP_TOOLS_LISTED,
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

    # ── server handshake ──────────────────────────────────────────────────────

    def _wrap_initialize(self, original: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap ``ClientSession.initialize`` to capture the server identity from
        the negotiated ``InitializeResult`` (S14/F7)."""
        if _is_awaitable(original):

            async def wrapped_async(*args: Any, **kwargs: Any) -> Any:
                result = await original(*args, **kwargs)
                self._emit_server_connected(result)
                return result

            return wrapped_async

        def wrapped_sync(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            self._emit_server_connected(result)
            return result

        return wrapped_sync

    def _emit_server_connected(self, result: Any) -> None:
        """Emit ``mcp.server.connected`` from an ``InitializeResult``. Reads only
        what the server actually declared (serverInfo.name/version +
        protocolVersion); emits nothing when none are present (honest blank). The
        server name also feeds the ElicitationTracker — a server is not an agent,
        so it is never promoted to agent identity."""
        info = _attr(result, "serverInfo")
        server_name = _attr(info, "name")
        server_version = _attr(info, "version")
        protocol_version = _attr(result, "protocolVersion")
        if server_name:
            self._server_name = str(server_name)
        payload: Dict[str, Any] = {}
        if server_name:
            payload["server_name"] = str(server_name)
        if server_version:
            payload["server_version"] = str(server_version)
        if protocol_version:
            payload["protocol_version"] = str(protocol_version)
        if not payload:
            return
        self.emit(MCP_SERVER_CONNECTED, payload)

    # ── elicitation: real ClientSession callback surface (D5) ─────────────────

    def _wrap_elicitation_callback(self, original: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a real ``ClientSession._elicitation_callback``.

        The callback is ``async (context, params) -> ElicitResult | ErrorData``
        where ``params`` is an ``ElicitRequest{Form,URL}Params`` (the prompt) and
        the return is the user's real ``ElicitResult`` (the consent decision).
        """

        async def wrapped(context: Any, params: Any, *args: Any, **kwargs: Any) -> Any:
            parent, eid = self._elicit_request(params)
            try:
                result = await original(context, params, *args, **kwargs)
            except Exception:
                self._elicit_error(parent, eid, params)
                raise
            self._elicit_response(parent, eid, params, result)
            return result

        return wrapped

    def _wrap_elicit(self, original: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a FastMCP/server ``elicit(message, schema)`` method.

        Its result is a server-side ``ElicitationResult`` (Accepted/Declined/
        Cancelled) whose ``.action`` is also accept/decline/cancel
        (``server/elicitation.py``). The first positional / ``message`` kwarg is
        the prompt; ``schema`` / ``requestedSchema`` is the requested schema.
        """

        def _request(args: tuple, kwargs: dict) -> tuple[str, str, Any]:
            message = kwargs.get("message") or (args[0] if args else None)
            schema = kwargs.get("requestedSchema") or kwargs.get("schema") or (args[1] if len(args) >= 2 else None)
            params = _ElicitMethodParams(message=message, schema=schema)
            parent, eid = self._elicit_request(params)
            return parent, eid, params

        if _is_awaitable(original):

            async def wrapped_async(*args: Any, **kwargs: Any) -> Any:
                parent, eid, params = _request(args, kwargs)
                try:
                    result = await original(*args, **kwargs)
                except Exception:
                    self._elicit_error(parent, eid, params)
                    raise
                self._elicit_response(parent, eid, params, result)
                return result

            return wrapped_async

        def wrapped_sync(*args: Any, **kwargs: Any) -> Any:
            parent, eid, params = _request(args, kwargs)
            try:
                result = original(*args, **kwargs)
            except Exception:
                self._elicit_error(parent, eid, params)
                raise
            self._elicit_response(parent, eid, params, result)
            return result

        return wrapped_sync

    def _elicit_request(self, params: Any) -> tuple[str, str]:
        message = _attr(params, "message")
        schema = _request_schema(params)
        mode = _elicit_mode(params)
        parent = _current_span_id.get() or uuid.uuid4().hex[:16]
        # Feed the real server name captured at initialize() when known; fall back
        # to the protocol label only if the handshake wasn't observed (S14/F7).
        server_name = self._server_name or self.PROTOCOL
        eid = _attr(params, "elicitationId") or self._elicitations.start_request(server_name, schema, message)
        # an elicitationId from URL params is the server's opaque id; still track latency
        if not self._elicitations.is_active(eid):
            self._elicitations.start_request(server_name, schema, message, elicitation_id=eid)
        self.emit(
            MCP_ELICITATION,
            {
                "elicitation_id": eid,
                "phase": "request",
                "mode": mode,
                "schema_hash": ElicitationTracker.hash_schema(schema),
                # content (stripped under no-content): the user-facing prompt.
                "message": message,
                # back-compat: the old field name carried the same string.
                "title": message,
            },
            parent_span_id=parent,
        )
        return parent, eid

    def _elicit_response(self, parent: str, eid: str, params: Any, result: Any) -> None:
        action = ElicitationTracker.normalize_action(_attr(result, "action"))
        latency_ms = self._elicitations.complete_response(eid, action=action, response=result)
        mode = _elicit_mode(params)
        payload: Dict[str, Any] = {
            "elicitation_id": eid,
            "phase": "response",
            "mode": mode,
            # the consent CATEGORY — survives redaction so a refusal is auditable.
            "action": action,
            "latency_ms": latency_ms,
        }
        # ONLY an accepted FORM-mode reply carries submitted content. A decline /
        # cancel, and ANY url-mode reply, has ElicitResult.content == None — we
        # hash NOTHING (never a refused payload). content_hash is itself
        # content-derived, so it is stripped under capture_content=False.
        if action == "accept" and mode != "url":
            content_hash = ElicitationTracker.hash_content(_attr(result, "content"))
            if content_hash is not None:
                payload["content_hash"] = content_hash
        self.emit(MCP_ELICITATION, payload, parent_span_id=parent)

    def _elicit_error(self, parent: str, eid: str, params: Any) -> None:
        self._elicitations.complete_response(eid, action="unknown")
        self.emit(
            MCP_ELICITATION,
            {
                "elicitation_id": eid,
                "phase": "response",
                "mode": _elicit_mode(params),
                "action": "unknown",
                "status": "error",
            },
            parent_span_id=parent,
        )

    # ── sampling: server-initiated nested LLM round-trip + cost (D3) ──────────

    def _wrap_sampling_callback(self, original: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a real ``ClientSession._sampling_callback``.

        ``async (context, params) -> CreateMessageResult{,WithTools} | ErrorData``
        where ``params`` is ``CreateMessageRequestParams`` (the messages the
        server wants sampled) and the result carries the model + sampled content.
        """

        async def wrapped(context: Any, params: Any, *args: Any, **kwargs: Any) -> Any:
            parent = _current_span_id.get() or uuid.uuid4().hex[:16]
            start = time.time()
            try:
                result = await original(context, params, *args, **kwargs)
            except Exception as exc:
                self.emit(
                    MCP_SAMPLING,
                    {
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "latency_ms": (time.time() - start) * 1000,
                    },
                    parent_span_id=parent,
                )
                raise
            self._emit_sampling(params, result, parent, (time.time() - start) * 1000)
            return result

        return wrapped

    def _emit_sampling(self, params: Any, result: Any, parent: str, latency_ms: float) -> None:
        """Emit ``mcp.sampling`` + a ``cost.record`` for a sampling round-trip.

        The wire carries NO token usage on a sampling result (verified: grep of
        the installed types.py shows ``CreateMessageResult`` = role/content/model/
        stopReason only). So we ESTIMATE tokens from the request prompt + the
        sampled completion text (~chars/4, the standard rough heuristic) and flag
        ``tokens_estimated=True`` so a consumer knows the count is derived, not
        metered. The ``cost.record`` carries the real ``model`` — the central
        price-on-emit chokepoint (``_collector.emit`` → ``price_cost_record``)
        fills ``cost_usd`` from model + tokens. An ``ErrorData`` result (not a
        message) is recorded without a cost (nothing was sampled).
        """
        model = _attr(result, "model")
        if not isinstance(model, str):
            # An ErrorData (no model) — record the round-trip, no cost.
            self.emit(
                MCP_SAMPLING,
                {"status": "error", "latency_ms": latency_ms},
                parent_span_id=parent,
            )
            return

        prompt_chars = _messages_chars(_attr(params, "messages")) + len(str(_attr(params, "systemPrompt") or ""))
        completion_chars = _content_chars(_attr(result, "content"))
        prompt_tokens = _chars_to_tokens(prompt_chars)
        completion_tokens = _chars_to_tokens(completion_chars)
        stop_reason = _attr(result, "stopReason")

        self.emit(
            MCP_SAMPLING,
            {
                "status": "completed",
                "model": model,
                "stop_reason": stop_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tokens_estimated": True,
                "latency_ms": latency_ms,
            },
            parent_span_id=parent,
        )
        # Paired cost.record — model + token counts; the collector prices it.
        self.emit(
            COST_RECORD,
            {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "tokens_estimated": True,
                "source": "mcp.sampling",
            },
            parent_span_id=parent,
        )

    # ── async task lifecycle ──────────────────────────────────────────────────

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


class _ElicitMethodParams:
    """Tiny carrier for the FastMCP ``elicit(message, schema)`` method args so the
    request/response helpers see the same ``message`` / ``schema`` shape as the
    real ``ElicitRequestParams``. (Method elicit is always form mode.)"""

    mode = "form"

    def __init__(self, message: Any, schema: Any) -> None:
        self.message = message
        self.schema = schema
        self.requestedSchema = schema


def _is_awaitable(fn: Any) -> bool:
    """True iff *fn* is a coroutine function. The real surfaces we wrap are all
    plain async functions / bound methods (the ClientSession callbacks and the
    FastMCP ``elicit`` method), so ``iscoroutinefunction`` resolves them; a sync
    duck-typed target falls through to the sync wrapper."""
    return inspect.iscoroutinefunction(fn)


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Duck-typed attribute/key read (pydantic model OR dict)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _request_schema(params: Any) -> Optional[dict]:
    """The requested JSON Schema for an elicitation, or None.

    Form params carry ``requestedSchema``; URL params carry NONE. We accept only a
    real dict — never a pydantic model's deprecated ``.schema`` bound method (URL
    params would otherwise surface that and break hashing/serialization)."""
    for name in ("requestedSchema", "schema"):
        val = _attr(params, name)
        if isinstance(val, dict):
            return val
    return None


def _elicit_mode(params: Any) -> str:
    """Distinguish form vs URL mode (D6). URL mode is the sensitive
    credential/OAuth/PAYMENT flow whose ElicitResult.content is absent."""
    mode = _attr(params, "mode")
    if mode in ("form", "url"):
        return mode
    # URL params carry a `url`; form params do not.
    if _attr(params, "url") is not None:
        return "url"
    return "form"


def _chars_to_tokens(chars: int) -> int:
    """Rough token estimate: ~4 chars/token (the standard heuristic). At least 1
    token for any non-empty text so a tiny prompt is never billed as zero."""
    if chars <= 0:
        return 0
    return max(1, chars // 4)


def _content_chars(content: Any) -> int:
    """Sum text length over a sampling content block or a list of blocks."""
    if content is None:
        return 0
    if isinstance(content, list):
        return sum(_content_chars(c) for c in content)
    text = _attr(content, "text")
    if isinstance(text, str):
        return len(text)
    if isinstance(content, str):
        return len(content)
    # image/audio blocks: count the base64 payload length as a proxy.
    data = _attr(content, "data")
    return len(data) if isinstance(data, str) else 0


def _messages_chars(messages: Any) -> int:
    """Sum text length over the request's SamplingMessage list."""
    if not messages:
        return 0
    if not isinstance(messages, (list, tuple)):
        messages = [messages]
    return sum(_content_chars(_attr(m, "content")) for m in messages)


def _extract_structured_output(result: Any) -> Any:
    if result is None:
        return None
    for attr in ("structuredContent", "structured_content"):
        val = getattr(result, attr, None)
        if val is not None:
            return val
    if isinstance(result, dict):
        for key in ("structuredContent", "structured_content"):
            if key in result:
                return result[key]
    return None


def _extract_output_schema(result: Any) -> Any:
    """Best-effort lookup of a JSON Schema threaded onto a result by an
    integrator. NOTE: a real ``CallToolResult`` does NOT carry ``outputSchema``
    (it lives on the ``Tool`` definition) — so this returns None on the real
    wire shape, and the caller falls back to ``validation_passed="unknown"``
    (fail closed), never to True."""
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
