from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional

from ._utils import safe_serialize
from ..._collector import TraceCollector
from ._base_framework import FrameworkAdapter
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

_HAS_LLAMAINDEX = False
try:
    from llama_index.core.instrumentation import (
        get_dispatcher as _get_dispatcher,  # pyright: ignore[reportMissingImports]
    )
    from llama_index.core.instrumentation.span import BaseSpan as _BaseSpan  # pyright: ignore[reportMissingImports]
    from llama_index.core.instrumentation.span_handlers import (
        BaseSpanHandler as _BaseSpanHandler,  # pyright: ignore[reportMissingImports]
    )
    from llama_index.core.instrumentation.event_handlers import (
        BaseEventHandler as _BaseEventHandler,  # pyright: ignore[reportMissingImports]
    )

    _HAS_LLAMAINDEX = True
except ImportError:
    _BaseSpan = None  # type: ignore[assignment,misc]
    _BaseSpanHandler = None  # type: ignore[assignment,misc]
    _BaseEventHandler = None  # type: ignore[assignment,misc]


class LlamaIndexAdapter(FrameworkAdapter):
    """LlamaIndex adapter using the instrumentation API (llama-index-core >= 0.10.41).

    Registers a span handler and event handler on the root dispatcher.
    Manages per-root-span collectors so concurrent queries each get
    their own trace.

    Usage::

        adapter = LlamaIndexAdapter(client)
        adapter.connect()
        response = index.as_query_engine().query("hello")
        adapter.disconnect()
    """

    name = "llamaindex"
    package = "llama-index-core"

    _EVENT_DISPATCH = {
        "LLMChatStartEvent": "_on_llm_chat_start",
        "LLMChatEndEvent": "_on_llm_chat_end",
        "LLMCompletionStartEvent": "_on_llm_completion_start",
        "LLMCompletionEndEvent": "_on_llm_completion_end",
        "AgentToolCallEvent": "_on_tool_call",
        "RetrievalStartEvent": "_on_retrieval_start",
        "RetrievalEndEvent": "_on_retrieval_end",
        "EmbeddingStartEvent": "_on_embedding_start",
        "EmbeddingEndEvent": "_on_embedding_end",
        "QueryStartEvent": "_on_query_start",
        "QueryEndEvent": "_on_query_end",
        "AgentRunStepStartEvent": "_on_agent_step_start",
        "AgentRunStepEndEvent": "_on_agent_step_end",
        "ExceptionEvent": "_on_exception",
        "ReRankStartEvent": "_on_rerank_start",
        "ReRankEndEvent": "_on_rerank_end",
    }

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        super().__init__(client, capture_config)
        self._span_handler: Optional[Any] = None
        self._event_handler: Optional[Any] = None
        # Per-root-span collectors (concurrent query support)
        self._collectors: Dict[str, TraceCollector] = {}
        self._open_spans: Dict[str, Any] = {}  # span_id → BaseSpan
        self._timestamps: Dict[str, float] = {}
        self._llm_start_times: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:
        self._check_dependency(_HAS_LLAMAINDEX)
        dispatcher = _get_dispatcher()
        self._span_handler = _make_span_handler(self)
        self._event_handler = _make_event_handler(self)
        dispatcher.add_span_handler(self._span_handler)
        dispatcher.add_event_handler(self._event_handler)

    def _on_disconnect(self) -> None:
        try:
            dispatcher = _get_dispatcher()
            if self._event_handler in dispatcher.event_handlers:
                dispatcher.event_handlers.remove(self._event_handler)
            if self._span_handler in dispatcher.span_handlers:
                dispatcher.span_handlers.remove(self._span_handler)
        except Exception:
            log.warning("layerlens: error removing LlamaIndex handlers", exc_info=True)
        self._flush_all()
        self._event_handler = None
        self._span_handler = None

    # ------------------------------------------------------------------
    # Collector + span management
    # ------------------------------------------------------------------

    def _fire(
        self,
        event_type: str,
        payload: Dict[str, Any],
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        span_name: Optional[str] = None,
    ) -> None:
        """Emit directly to the collector that owns this span."""
        collector = self._collector_for(span_id)
        if collector is None:
            return
        sid = _trunc(span_id) if span_id else self._new_span_id()
        parent = _trunc(parent_span_id) if parent_span_id else None
        if parent is None and span_id:
            raw_parent = self._parent_of(span_id)
            parent = _trunc(raw_parent) if raw_parent else None
        collector.emit(event_type, payload, span_id=sid, parent_span_id=parent, span_name=span_name)

    def _collector_for(self, span_id: Optional[str]) -> Optional[TraceCollector]:
        """Walk up the span tree to find the owning collector."""
        if span_id is None:
            return None
        with self._lock:
            current = span_id
            while current is not None:
                if current in self._collectors:
                    return self._collectors[current]
                span = self._open_spans.get(current)
                current = span.parent_id if span is not None else None
            # Fallback: any active collector
            if self._collectors:
                return next(iter(self._collectors.values()))
        return None

    def _parent_of(self, span_id: Optional[str]) -> Optional[str]:
        if span_id is None:
            return None
        with self._lock:
            span = self._open_spans.get(span_id)
            return span.parent_id if span is not None else None

    def _flush_all(self) -> None:
        with self._lock:
            collectors = list(self._collectors.values())
            self._collectors.clear()
            self._open_spans.clear()
            self._timestamps.clear()
            self._llm_start_times.clear()
        for c in collectors:
            try:
                c.flush()
            except Exception:
                log.warning("layerlens: error flushing LlamaIndex collector", exc_info=True)

    # ------------------------------------------------------------------
    # Span lifecycle (called by the thin span handler)
    # ------------------------------------------------------------------

    def _on_span_enter(self, id_: str, parent_span_id: Optional[str]) -> Any:
        with self._lock:
            span = _BaseSpan(id_=id_, parent_id=parent_span_id)
            self._open_spans[id_] = span
            self._timestamps[id_] = time.time()
            if parent_span_id is None or parent_span_id not in self._open_spans:
                self._collectors[id_] = TraceCollector(self._client, self._config)
            return span

    def _on_span_exit(self, id_: str) -> Any:
        with self._lock:
            span = self._open_spans.get(id_)
            self._timestamps.pop(id_, None)
            collector = self._collectors.pop(id_, None)
        if collector is not None:
            collector.flush()
        return span

    def _on_span_drop(self, id_: str) -> Any:
        return self._on_span_exit(id_)  # same cleanup

    # ------------------------------------------------------------------
    # Event dispatch (called by the thin event handler)
    # ------------------------------------------------------------------

    def _handle_event(self, event: Any) -> None:
        try:
            handler_name = self._EVENT_DISPATCH.get(type(event).__name__)
            if handler_name is not None:
                getattr(self, handler_name)(event)
        except Exception:
            log.warning("layerlens: error in LlamaIndex event handler", exc_info=True)

    # ------------------------------------------------------------------
    # LLM Chat
    # ------------------------------------------------------------------

    def _on_llm_chat_start(self, event: Any) -> None:
        span_id = getattr(event, "span_id", None)
        if span_id:
            self._llm_start_times[span_id] = time.time()

    def _on_llm_chat_end(self, event: Any) -> None:
        span_id = getattr(event, "span_id", None)
        response = getattr(event, "response", None)

        payload = self._payload()
        model = _model_from_response(response)
        if model:
            payload["model"] = model

        tokens = self._normalize_tokens(_usage_from_response(response))
        payload.update(tokens)

        start = self._llm_start_times.pop(span_id, None) if span_id else None
        if start is not None:
            payload["latency_ms"] = (time.time() - start) * 1000

        if self._config.capture_content:
            messages = getattr(event, "messages", None)
            if messages:
                payload["messages"] = _serialize_messages(messages)
            if response:
                output = _chat_output(response)
                if output:
                    payload["output_message"] = output

        self._fire("model.invoke", payload, span_id=span_id)

        if tokens:
            cost = self._payload()
            if model:
                cost["model"] = model
            cost.update(tokens)
            self._fire("cost.record", cost, span_id=span_id)

    # ------------------------------------------------------------------
    # LLM Completion
    # ------------------------------------------------------------------

    def _on_llm_completion_start(self, event: Any) -> None:
        span_id = getattr(event, "span_id", None)
        if span_id:
            self._llm_start_times[span_id] = time.time()

    def _on_llm_completion_end(self, event: Any) -> None:
        span_id = getattr(event, "span_id", None)
        response = getattr(event, "response", None)

        payload = self._payload()
        model = _model_from_response(response)
        if model:
            payload["model"] = model

        tokens = self._normalize_tokens(_usage_from_response(response))
        payload.update(tokens)

        start = self._llm_start_times.pop(span_id, None) if span_id else None
        if start is not None:
            payload["latency_ms"] = (time.time() - start) * 1000

        if self._config.capture_content:
            prompt = getattr(event, "prompt", None)
            if prompt:
                payload["messages"] = [{"role": "user", "content": str(prompt)}]
            if response:
                text = getattr(response, "text", None)
                if text:
                    payload["output_message"] = str(text)

        self._fire("model.invoke", payload, span_id=span_id)

        if tokens:
            cost = self._payload()
            if model:
                cost["model"] = model
            cost.update(tokens)
            self._fire("cost.record", cost, span_id=span_id)

    # ------------------------------------------------------------------
    # Tool calls
    # ------------------------------------------------------------------

    def _on_tool_call(self, event: Any) -> None:
        span_id = getattr(event, "span_id", None)
        tool = getattr(event, "tool", None)
        tool_name = getattr(tool, "name", None) or "unknown" if tool else "unknown"

        payload = self._payload(tool_name=tool_name)
        if self._config.capture_content:
            args = getattr(event, "arguments", None)
            if args is not None:
                payload["input"] = str(args)
            if tool:
                desc = getattr(tool, "description", None)
                if desc:
                    payload["tool_description"] = str(desc)

        self._fire("tool.call", payload, span_id=span_id)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _on_retrieval_start(self, event: Any) -> None:
        span_id = getattr(event, "span_id", None)
        payload = self._payload(tool_name="retrieval")
        if self._config.capture_content:
            query = getattr(event, "str_or_query_bundle", None)
            if query is not None:
                payload["input"] = str(query)
        self._fire("tool.call", payload, span_id=span_id, span_name="retrieval")

    def _on_retrieval_end(self, event: Any) -> None:
        span_id = getattr(event, "span_id", None)
        nodes = getattr(event, "nodes", None)
        payload = self._payload(tool_name="retrieval")
        if nodes is not None:
            payload["num_results"] = len(nodes)
            if self._config.capture_content:
                payload["output"] = _serialize_nodes(nodes)
        self._fire("tool.result", payload, span_id=span_id, span_name="retrieval")

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def _on_embedding_start(self, event: Any) -> None:
        span_id = getattr(event, "span_id", None)
        payload = self._payload(embedding=True)
        model = _model_from_dict(getattr(event, "model_dict", None))
        if model:
            payload["model"] = model
        self._fire("model.invoke", payload, span_id=span_id, span_name="embedding")

    def _on_embedding_end(self, event: Any) -> None:
        span_id = getattr(event, "span_id", None)
        chunks = getattr(event, "chunks", None)
        embeddings = getattr(event, "embeddings", None)
        payload = self._payload(embedding=True)
        if chunks is not None:
            payload["num_chunks"] = len(chunks)
        if embeddings is not None:
            payload["num_embeddings"] = len(embeddings)
            if embeddings:
                payload["embedding_dim"] = len(embeddings[0])
        self._fire("model.invoke", payload, span_id=span_id, span_name="embedding")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _on_query_start(self, event: Any) -> None:
        span_id = getattr(event, "span_id", None)
        payload = self._payload()
        if self._config.capture_content:
            query = getattr(event, "query", None)
            if query is not None:
                payload["input"] = str(query)
        self._fire("agent.input", payload, span_id=span_id, span_name="query")

    def _on_query_end(self, event: Any) -> None:
        span_id = getattr(event, "span_id", None)
        payload = self._payload(status="ok")
        if self._config.capture_content:
            response = getattr(event, "response", None)
            if response is not None:
                payload["output"] = str(response)
        self._fire("agent.output", payload, span_id=span_id, span_name="query")

    # ------------------------------------------------------------------
    # Agent steps
    # ------------------------------------------------------------------

    def _on_agent_step_start(self, event: Any) -> None:
        span_id = getattr(event, "span_id", None)
        payload = self._payload()
        task_id = getattr(event, "task_id", None)
        if task_id is not None:
            payload["task_id"] = str(task_id)
        if self._config.capture_content:
            step_input = getattr(event, "input", None)
            if step_input is not None:
                payload["input"] = safe_serialize(step_input)
        self._fire("agent.input", payload, span_id=span_id, span_name="agent_step")

    def _on_agent_step_end(self, event: Any) -> None:
        span_id = getattr(event, "span_id", None)
        payload = self._payload(status="ok")
        if self._config.capture_content:
            output = getattr(event, "step_output", None)
            if output is not None:
                payload["output"] = safe_serialize(output)
        self._fire("agent.output", payload, span_id=span_id, span_name="agent_step")

    # ------------------------------------------------------------------
    # Rerank
    # ------------------------------------------------------------------

    def _on_rerank_start(self, event: Any) -> None:
        span_id = getattr(event, "span_id", None)
        payload = self._payload(tool_name="rerank")
        model_name = getattr(event, "model_name", None)
        if model_name:
            payload["model"] = str(model_name)
        top_n = getattr(event, "top_n", None)
        if top_n is not None:
            payload["top_n"] = top_n
        self._fire("tool.call", payload, span_id=span_id, span_name="rerank")

    def _on_rerank_end(self, event: Any) -> None:
        span_id = getattr(event, "span_id", None)
        payload = self._payload(tool_name="rerank")
        nodes = getattr(event, "nodes", None)
        if nodes is not None:
            payload["num_results"] = len(nodes)
        self._fire("tool.result", payload, span_id=span_id, span_name="rerank")

    # ------------------------------------------------------------------
    # Exceptions
    # ------------------------------------------------------------------

    def _on_exception(self, event: Any) -> None:
        span_id = getattr(event, "span_id", None)
        exc = getattr(event, "exception", None)
        payload = self._payload(
            error=str(exc) if exc else "unknown error",
            error_type=type(exc).__name__ if isinstance(exc, BaseException) else "Exception",
        )
        self._fire("agent.error", payload, span_id=span_id)


# ======================================================================
# Thin handler classes (delegate everything to the adapter)
# ======================================================================


def _make_span_handler(adapter: LlamaIndexAdapter) -> Any:
    """Create a LlamaIndex-compatible span handler that delegates to the adapter."""
    if not _HAS_LLAMAINDEX:
        raise ImportError("llama-index-core is required")

    class _SpanHandler(_BaseSpanHandler[_BaseSpan]):  # type: ignore[type-arg]
        model_config = {"arbitrary_types_allowed": True}

        def new_span(
            self,
            id_: str,
            bound_args: Any,
            instance: Any = None,
            parent_span_id: Any = None,
            tags: Any = None,
            **kw: Any,
        ) -> Any:
            return adapter._on_span_enter(id_, parent_span_id)

        def prepare_to_exit_span(
            self, id_: str, bound_args: Any, instance: Any = None, result: Any = None, **kw: Any
        ) -> Any:
            return adapter._on_span_exit(id_)

        def prepare_to_drop_span(
            self, id_: str, bound_args: Any, instance: Any = None, err: Any = None, **kw: Any
        ) -> Any:
            return adapter._on_span_drop(id_)

    handler = _SpanHandler()
    handler.open_spans = adapter._open_spans
    return handler


def _make_event_handler(adapter: LlamaIndexAdapter) -> Any:
    """Create a LlamaIndex-compatible event handler that delegates to the adapter."""
    if not _HAS_LLAMAINDEX:
        raise ImportError("llama-index-core is required")

    class _EventHandler(_BaseEventHandler):  # type: ignore[misc]
        model_config = {"arbitrary_types_allowed": True}

        @classmethod
        def class_name(cls) -> str:
            return "LayerLensEventHandler"

        def handle(self, event: Any, **kw: Any) -> None:
            adapter._handle_event(event)

    return _EventHandler()


# ======================================================================
# Module-level helpers
# ======================================================================


def _trunc(span_id: str | None) -> str | None:
    """LlamaIndex span IDs are long (ClassName.method-uuid4) — truncate to 16 chars."""
    if span_id is None:
        return None
    if "-" in span_id:
        parts = span_id.rsplit("-", 1)
        if len(parts) == 2 and len(parts[1]) >= 16:
            return parts[1][:16]
    return span_id[:16] if len(span_id) > 16 else span_id


def _model_from_response(response: Any) -> str | None:
    """Extract model name from ChatResponse / CompletionResponse."""
    if response is None:
        return None
    raw = getattr(response, "raw", None)
    if isinstance(raw, dict):
        model = raw.get("model")
        if model:
            return str(model)
    if raw is not None:
        model = getattr(raw, "model", None)
        if model:
            return str(model)
    return None


def _model_from_dict(model_dict: dict | None) -> str | None:
    """Extract model name from model_dict on start events."""
    if not model_dict:
        return None
    for key in ("model", "model_name", "model_id"):
        val = model_dict.get(key)
        if val:
            return str(val)
    return None


def _usage_from_response(response: Any) -> Any:
    """Unwrap the usage object from a response to pass to ``_normalize_tokens``."""
    if response is None:
        return None
    raw = getattr(response, "raw", None)
    if raw is not None:
        usage = raw.get("usage") if isinstance(raw, dict) else getattr(raw, "usage", None)
        if usage is not None:
            return usage
    additional = getattr(response, "additional_kwargs", None)
    if isinstance(additional, dict):
        return additional.get("usage")
    return None


def _chat_output(response: Any) -> str | None:
    """Extract output text from a ChatResponse."""
    if response is None:
        return None
    message = getattr(response, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content:
            return str(content)
    return None


def _serialize_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """Serialize ChatMessage list for payload."""
    result = []
    for msg in messages:
        if hasattr(msg, "model_dump"):
            try:
                result.append(msg.model_dump())
                continue
            except Exception:
                pass
        entry: Dict[str, Any] = {}
        role = getattr(msg, "role", None)
        if role is not None:
            entry["role"] = str(role)
        content = getattr(msg, "content", None)
        if content is not None:
            entry["content"] = str(content)
        result.append(entry)
    return result


def _serialize_nodes(nodes: List[Any]) -> List[Dict[str, Any]]:
    """Serialize retrieval nodes (truncated to 10)."""
    result = []
    for node in nodes[:10]:
        entry: Dict[str, Any] = {}
        score = getattr(node, "score", None)
        if score is not None:
            entry["score"] = score
        node_obj = getattr(node, "node", None) or node
        text = getattr(node_obj, "text", None) or getattr(node_obj, "get_content", lambda: None)()
        if text:
            entry["text"] = str(text)[:500]
        node_id = getattr(node_obj, "node_id", None) or getattr(node_obj, "id_", None)
        if node_id:
            entry["node_id"] = str(node_id)
        result.append(entry)
    return result
