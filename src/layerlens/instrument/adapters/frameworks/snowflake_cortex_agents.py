"""Snowflake Cortex Agents adapter.

`Cortex Agents <https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-agents>`_
are Snowflake's governed, server-side data agents (GA Nov 2025). They are
invoked through the ``agent:run`` REST API, which streams its response back as
server-sent events (SSE) — text deltas, tool-use / tool-result blocks, Cortex
Analyst (text-to-SQL) deltas, thinking deltas, and a final ``response`` event
carrying token usage. There is no in-process Python SDK to hook the way boto3
exposes an event system, so this adapter observes the SSE stream itself.

Two ways to use it:

Instrumented invoke — the adapter makes the ``agent:run`` call for you::

    import httpx
    from layerlens.instrument.adapters.frameworks import SnowflakeCortexAgentsAdapter

    adapter = SnowflakeCortexAgentsAdapter(layerlens_client)
    adapter.connect(
        account_url="https://ACCOUNT.snowflakecomputing.com",
        auth_token=os.environ["SNOWFLAKE_TOKEN"],   # PAT / OAuth / key-pair JWT
        agent="MY_DB.MY_SCHEMA.MY_AGENT",           # omit to use /api/v2/cortex/agent:run
    )
    final = adapter.run("What were Q3 sales by region?")
    adapter.disconnect()

Bring-your-own call — you already POSTed to ``agent:run`` and hold the SSE
stream (from ``httpx``, ``requests``, or the Snowflake connector's REST
transport); hand the raw lines (or parsed ``(event, data)`` tuples) to the
adapter::

    with httpx.stream("POST", url, headers=headers, json=body) as resp:
        final = adapter.ingest_stream(resp.iter_lines(), request=body)

Both paths turn the stream into a single trace: an ``agent.input`` root span,
a ``tool.call`` per tool use (with the generated SQL / retrieval results for
Cortex Analyst and Cortex Search), ``model.invoke`` + ``cost.record`` from the
final usage block, and an ``agent.output`` carrying the assistant text.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple, Iterable, Iterator, Optional

from ._utils import truncate, safe_serialize
from ._base_framework import FrameworkAdapter
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

try:
    import httpx  # pyright: ignore[reportMissingImports]

    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False


# SSE event names emitted by the agent:run API. See:
# https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-agents-run
_EVT_TEXT_DELTA = "response.text.delta"
_EVT_THINKING_DELTA = "response.thinking.delta"
_EVT_TOOL_USE = "response.tool_use"
_EVT_TOOL_RESULT = "response.tool_result"
_EVT_ANALYST_DELTA = "response.tool_result.analyst.delta"
_EVT_STATUS = "response.status"
_EVT_RESPONSE = "response"
_EVT_METADATA = "metadata"
_EVT_ERROR = "error"

_STATELESS_PATH = "/api/v2/cortex/agent:run"


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _iter_sse(lines: Iterable[str]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Parse a raw SSE line stream into ``(event_name, data_dict)`` pairs.

    Groups lines into events on blank-line boundaries, concatenating any
    multi-line ``data:`` payloads (per the SSE spec) before JSON-decoding.
    Frames whose data is ``[DONE]`` or not valid JSON are skipped. Accepts
    ``bytes`` or ``str`` lines with or without a trailing newline.
    """
    event = ""
    data_parts: List[str] = []

    def _flush() -> Optional[Tuple[str, Dict[str, Any]]]:
        nonlocal event, data_parts
        raw = "\n".join(data_parts).strip()
        name, data_parts = event, []
        event = ""
        if not raw or raw == "[DONE]":
            return None
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return None
        if not isinstance(parsed, dict):
            parsed = {"value": parsed}
        return (name or _EVT_RESPONSE, parsed)

    for line in lines:
        if isinstance(line, (bytes, bytearray)):
            line = line.decode("utf-8", "replace")
        line = line.rstrip("\r\n")
        if line == "":
            frame = _flush()
            if frame is not None:
                yield frame
            continue
        if line.startswith(":"):  # SSE comment / keep-alive
            continue
        if line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_parts.append(line[len("data:") :].lstrip(" "))

    frame = _flush()
    if frame is not None:
        yield frame


def _normalize_events(
    events: Iterable[Any],
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Accept either raw SSE lines or pre-parsed ``(event, data)`` tuples.

    A caller who already framed the stream can pass ``(name, dict)`` pairs
    directly; a caller with the raw HTTP body passes line strings. We sniff
    the first item and route accordingly so both are one code path downstream.
    """
    it = iter(events)
    try:
        first = next(it)
    except StopIteration:
        return

    def _chain() -> Iterator[Any]:
        yield first
        yield from it

    if isinstance(first, tuple) and len(first) == 2 and isinstance(first[1], dict):
        for name, data in _chain():  # type: ignore[misc]
            yield str(name), data
    else:
        yield from _iter_sse(_chain())


def _last_user_text(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Pull the text of the most recent user message from a messages array."""
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            joined = "".join(parts).strip()
            if joined:
                return joined
    return None


def _normalize_messages(messages: Any) -> List[Dict[str, Any]]:
    """Coerce a plain string or single message into the agent:run array shape."""
    if isinstance(messages, str):
        return [{"role": "user", "content": [{"type": "text", "text": messages}]}]
    if isinstance(messages, dict):
        return [messages]
    return list(messages or [])


class SnowflakeCortexAgentsAdapter(FrameworkAdapter):
    """Snowflake Cortex Agents adapter over the ``agent:run`` SSE stream.

    Each ``run`` / ``ingest_stream`` call is one trace via ``_begin_run`` /
    ``_end_run``. The stream is single-threaded and synchronous, so the
    ContextVar-based run isolation in the base class applies cleanly.

    ``connect`` accepts either a ready ``httpx.Client`` as ``target`` or an
    ``account_url`` + ``auth_token`` from which the adapter builds its own
    client. Snowflake auth (PAT, OAuth, or key-pair JWT) is the caller's
    responsibility — we only attach the token as a bearer credential.
    """

    name = "snowflake_cortex_agents"
    package = "httpx"

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        super().__init__(client, capture_config)
        self._http: Optional[Any] = None
        self._owns_http = False
        self._account_url: str = ""
        self._auth_token: str = ""
        self._agent: Optional[str] = None
        self._default_model: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:
        self._check_dependency(_HAS_HTTPX)

        self._account_url = str(kwargs.get("account_url", "")).rstrip("/")
        self._auth_token = str(kwargs.get("auth_token", "") or "")
        self._agent = kwargs.get("agent")
        self._default_model = kwargs.get("model")

        if target is not None and hasattr(target, "stream"):
            # A user-supplied httpx.Client (or compatible transport).
            self._http = target
            self._owns_http = False
        elif self._account_url:
            self._http = httpx.Client(timeout=kwargs.get("timeout", 120.0))
            self._owns_http = True
        else:
            # ingest-only mode: no transport, caller brings the stream.
            self._http = None
            self._owns_http = False

        if self._account_url:
            self._metadata["account_url"] = self._account_url
        if self._agent:
            self._metadata["agent"] = self._agent

    def _on_disconnect(self) -> None:
        if self._http is not None and self._owns_http:
            try:
                self._http.close()
            except Exception:
                log.debug("layerlens: error closing snowflake http client", exc_info=True)
        self._http = None
        self._owns_http = False

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(
        self,
        messages: Any,
        *,
        agent: Optional[str] = None,
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        thread_id: Optional[int] = None,
        parent_message_id: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        """Invoke the agent and return the aggregated final response.

        POSTs to ``agent:run`` with ``stream: true``, parses the SSE stream
        into a trace, and returns the ``response`` event's payload (or an
        aggregated fallback if the stream ends without one).
        """
        if not self._connected:
            raise RuntimeError("Adapter is not connected — call connect() first")
        if self._http is None:
            raise RuntimeError(
                "run() needs an HTTP transport. Pass account_url/auth_token to "
                "connect(), or use ingest_stream() with your own SSE stream."
            )
        if not self._auth_token:
            raise ValueError("auth_token is required to call the Cortex agent:run API")

        norm_messages = _normalize_messages(messages)
        body: Dict[str, Any] = {"messages": norm_messages, "stream": True}
        resolved_model = model or self._default_model
        if resolved_model:
            body["models"] = {"orchestration": resolved_model}
        if tools:
            body["tools"] = tools
        if thread_id is not None:
            body["thread_id"] = thread_id
        if parent_message_id is not None:
            body["parent_message_id"] = parent_message_id
        if extra_body:
            body.update(extra_body)

        url = self._build_url(agent or self._agent)
        headers = {
            "Authorization": f"Bearer {self._auth_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        stream = self._post_sse(url, headers, body, timeout)
        return self._ingest(stream, request=body)

    def ingest_stream(
        self,
        events: Iterable[Any],
        *,
        request: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Parse an SSE stream you already have into a trace.

        ``events`` is either raw SSE line strings/bytes or pre-parsed
        ``(event_name, data_dict)`` tuples. ``request`` is the request body
        you sent (used to seed ``agent.input`` with the user's question).
        """
        if not self._connected:
            raise RuntimeError("Adapter is not connected — call connect() first")
        return self._ingest(events, request=request or {})

    # ------------------------------------------------------------------
    # Transport (isolated so tests can inject an SSE stream)
    # ------------------------------------------------------------------

    def _build_url(self, agent: Optional[str]) -> str:
        base = self._account_url
        if agent:
            parts = agent.split(".")
            if len(parts) == 3:
                db, schema, name = parts
                return f"{base}/api/v2/databases/{db}/schemas/{schema}/agents/{name}:run"
            log.warning(
                "layerlens: agent %r is not a 'db.schema.name' reference; "
                "falling back to the stateless agent:run endpoint",
                agent,
            )
        return f"{base}{_STATELESS_PATH}"

    def _post_sse(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        timeout: float,
    ) -> Iterator[str]:
        """Yield raw SSE lines from the agent:run response.

        Overridden in tests. The default streams via the connected
        ``httpx.Client`` and raises on a non-2xx status.
        """
        with self._http.stream("POST", url, headers=headers, json=body, timeout=timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                yield line

    # ------------------------------------------------------------------
    # Stream ingestion → events
    # ------------------------------------------------------------------

    def _ingest(self, events: Iterable[Any], *, request: Dict[str, Any]) -> Dict[str, Any]:
        self._begin_run()
        self._start_timer("run")
        # Per-run accumulators.
        assistant_text: List[str] = []
        thinking_text: List[str] = []
        tool_spans: Dict[str, Dict[str, Any]] = {}
        final_response: Dict[str, Any] = {}
        last_status: Optional[str] = None
        run_meta: Dict[str, Any] = {}

        try:
            root = self._get_root_span()
            request_messages = _normalize_messages(request.get("messages", []))
            in_payload = self._payload(
                agent=self._agent,
                thread_id=request.get("thread_id"),
            )
            models = request.get("models")
            if isinstance(models, dict) and models.get("orchestration"):
                in_payload["model"] = models["orchestration"]
            self._set_if_capturing(in_payload, "input", _last_user_text(request_messages))
            self._emit(
                "agent.input",
                in_payload,
                span_id=root,
                parent_span_id=None,
                span_name="cortex.agent_run",
            )

            for name, data in _normalize_events(events):
                try:
                    if name == _EVT_TEXT_DELTA:
                        assistant_text.append(str(data.get("text", "")))
                    elif name == _EVT_THINKING_DELTA:
                        thinking_text.append(str(data.get("text", "")))
                    elif name == _EVT_TOOL_USE:
                        self._on_tool_use(data, tool_spans)
                    elif name == _EVT_ANALYST_DELTA:
                        self._on_analyst_delta(data, tool_spans)
                    elif name == _EVT_TOOL_RESULT:
                        self._on_tool_result(data, tool_spans)
                    elif name == _EVT_STATUS:
                        last_status = data.get("status") or last_status
                    elif name == _EVT_METADATA:
                        for key in ("message_id", "run_id"):
                            if data.get(key) is not None:
                                run_meta[key] = data[key]
                    elif name == _EVT_ERROR:
                        self._on_error(data)
                    elif name == _EVT_RESPONSE:
                        final_response = data
                except Exception:
                    log.warning("layerlens: error handling %s event", name, exc_info=True)

            # Flush any tool uses that never received a matching result.
            for span in tool_spans.values():
                if not span.get("emitted"):
                    self._emit_tool_call(span, status="incomplete")

            self._emit_usage(final_response)

            latency_ms = self._stop_timer("run")
            out_payload = self._payload(agent=self._agent)
            if latency_ms is not None:
                out_payload["latency_ms"] = latency_ms
            if last_status:
                out_payload["status"] = last_status
            out_payload.update(run_meta)
            warnings = final_response.get("warnings") if isinstance(final_response, dict) else None
            if warnings:
                out_payload["warnings"] = safe_serialize(warnings)
            self._set_if_capturing(out_payload, "output", "".join(assistant_text).strip() or None)
            self._set_if_capturing(out_payload, "reasoning", truncate("".join(thinking_text).strip() or None, 4000))
            self._emit(
                "agent.output",
                out_payload,
                span_id=root,
                parent_span_id=None,
                span_name="cortex.agent_run",
            )
        finally:
            self._end_run()

        return final_response

    # ------------------------------------------------------------------
    # Tool events
    # ------------------------------------------------------------------

    def _on_tool_use(self, data: Dict[str, Any], tool_spans: Dict[str, Dict[str, Any]]) -> None:
        tool_use_id = str(data.get("tool_use_id") or self._new_span_id())
        span = tool_spans.setdefault(tool_use_id, {"span_id": self._new_span_id()})
        span["name"] = data.get("name") or span.get("name") or "cortex_tool"
        span["tool_type"] = data.get("type") or span.get("tool_type")
        if data.get("input") is not None:
            span["input"] = data.get("input")

    def _on_analyst_delta(self, data: Dict[str, Any], tool_spans: Dict[str, Dict[str, Any]]) -> None:
        # Cortex Analyst (text-to-SQL) streams its SQL, explanation, and result
        # set incrementally. Accumulate onto the owning tool span so the emitted
        # tool.call carries the generated SQL and how many rows it returned.
        tool_use_id = str(data.get("tool_use_id") or "analyst")
        span = tool_spans.setdefault(tool_use_id, {"span_id": self._new_span_id()})
        span.setdefault("name", data.get("tool_name") or "cortex_analyst")
        span.setdefault("tool_type", data.get("tool_type") or "cortex_analyst_text_to_sql")
        analyst = span.setdefault("analyst", {"sql": [], "sql_explanation": []})
        delta = data.get("delta")
        if isinstance(delta, dict):
            if delta.get("sql"):
                analyst["sql"].append(str(delta["sql"]))
            if delta.get("sql_explanation"):
                analyst["sql_explanation"].append(str(delta["sql_explanation"]))
            if delta.get("query_id"):
                analyst["query_id"] = delta["query_id"]
            if delta.get("result_set") is not None:
                analyst["result_set"] = delta["result_set"]

    def _on_tool_result(self, data: Dict[str, Any], tool_spans: Dict[str, Dict[str, Any]]) -> None:
        tool_use_id = str(data.get("tool_use_id") or self._new_span_id())
        span = tool_spans.setdefault(tool_use_id, {"span_id": self._new_span_id()})
        if data.get("name"):
            span["name"] = data["name"]
        # A tool_result's ``type`` is the content-block kind ("tool_results"),
        # not the tool kind — the meaningful tool_type comes from the tool_use.
        span.setdefault("tool_type", data.get("type"))
        span["output"] = data.get("content")
        self._emit_tool_call(span, status=str(data.get("status", "success")))

    def _emit_tool_call(self, span: Dict[str, Any], *, status: str) -> None:
        payload = self._payload(
            tool_name=span.get("name", "cortex_tool"),
            tool_type=span.get("tool_type"),
            status=status,
        )
        analyst = span.get("analyst")
        if isinstance(analyst, dict):
            sql = "".join(analyst.get("sql", [])).strip()
            explanation = "".join(analyst.get("sql_explanation", [])).strip()
            if analyst.get("query_id"):
                payload["query_id"] = analyst["query_id"]
            result_set = analyst.get("result_set")
            # Row count is safe to surface unconditionally; raw rows are gated.
            rows = _row_count(result_set)
            if rows is not None:
                payload["num_rows"] = rows
            if explanation:
                self._set_if_capturing(payload, "sql_explanation", truncate(explanation, 4000))
            self._set_if_capturing(payload, "sql", truncate(sql, 4000) if sql else None)
            self._set_if_capturing(payload, "result_set", safe_serialize(result_set))
        self._set_if_capturing(payload, "input", safe_serialize(span.get("input")))
        self._set_if_capturing(payload, "output", safe_serialize(span.get("output")))
        self._emit(
            "tool.call",
            payload,
            span_id=span.get("span_id"),
            span_name=f"cortex.{span.get('name', 'tool')}",
        )
        span["emitted"] = True

    # ------------------------------------------------------------------
    # Usage / cost and errors
    # ------------------------------------------------------------------

    def _emit_usage(self, final_response: Dict[str, Any]) -> None:
        if not isinstance(final_response, dict):
            return
        metadata = final_response.get("metadata")
        if not isinstance(metadata, dict):
            return
        usage = metadata.get("usage")
        if not isinstance(usage, dict):
            return
        consumed = usage.get("tokens_consumed")
        if not isinstance(consumed, list):
            return
        for entry in consumed:
            if not isinstance(entry, dict):
                continue
            tokens = self._normalize_tokens(entry)
            if not tokens:
                continue
            model = entry.get("model") or entry.get("model_name")
            span_id = self._new_span_id()
            invoke_payload = self._payload(provider="snowflake_cortex", **tokens)
            if model:
                invoke_payload["model"] = model
            self._emit("model.invoke", invoke_payload, span_id=span_id, span_name="cortex.model")

            cost_payload = self._payload(**tokens)
            if model:
                cost_payload["model"] = model
            self._emit("cost.record", cost_payload, span_id=span_id)

    def _on_error(self, data: Dict[str, Any]) -> None:
        payload = self._payload(
            code=data.get("code"),
            request_id=data.get("request_id"),
        )
        self._set_if_capturing(payload, "message", truncate(data.get("message"), 2000))
        self._emit("agent.error", payload, span_name="cortex.error")


def _row_count(result_set: Any) -> Optional[int]:
    """Best-effort row count from a Cortex Analyst result_set (SQL API shape)."""
    if not isinstance(result_set, dict):
        return None
    data = result_set.get("data")
    if isinstance(data, list):
        return len(data)
    meta = result_set.get("resultSetMetaData")
    if isinstance(meta, dict) and isinstance(meta.get("numRows"), int):
        return meta["numRows"]
    return None
