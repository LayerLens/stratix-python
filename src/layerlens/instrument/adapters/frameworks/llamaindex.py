from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional

from ._utils import safe_serialize
from ..._context import _current_collector
from ..._identity import _s, honest_agent_type
from ..._collector import TraceCollector
from ._base_framework import FrameworkAdapter
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

_HAS_LLAMAINDEX = False
try:
    from llama_index.core.instrumentation import (
        get_dispatcher as _get_dispatcher,  # pyright: ignore[reportMissingImports]
    )
    from llama_index.core.instrumentation.span import (
        BaseSpan as _BaseSpan,
    )  # pyright: ignore[reportMissingImports]
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
        # Honest-graph state (AgentWorkflow multi-agent path), keyed by the
        # root span id that owns the trace collector:
        #   _current_agent  — the developer-declared name of the agent whose
        #                     turn is active, stamped onto model.invoke so its
        #                     graph node is not orphaned.
        #   _handoffs_seen  — deduped (from_agent, to_agent) edges already
        #                     emitted for the run.
        self._current_agent: Dict[str, str] = {}
        self._handoffs_seen: Dict[str, set] = {}

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
        """Emit to the caller-bound collector when present, else the span's own collector.

        Adapters must honor a collector bound on ``_current_collector`` (e.g. by
        ``instrument()`` / ``samples/adapters/_shared.capture_events``); only fall
        back to the per-span collector for the standalone (no-bound) path.
        """
        collector = _current_collector.get() or self._collector_for(span_id)
        if collector is None:
            return
        sid = _trunc(span_id) if span_id else self._new_span_id()
        parent = _trunc(parent_span_id) if parent_span_id else None
        if parent is None and span_id:
            raw_parent = self._parent_of(span_id)
            parent = _trunc(raw_parent) if raw_parent else None
        if event_type == "cost.record" and payload.get("cost_usd") is None:
            self._price_cost_record(payload)
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

    def _root_span_of(self, span_id: Optional[str]) -> Optional[str]:
        """The collector-owning (root) span id above ``span_id``, or None."""
        if span_id is None:
            return None
        with self._lock:
            current: Optional[str] = span_id
            while current is not None:
                if current in self._collectors:
                    return current
                span = self._open_spans.get(current)
                current = span.parent_id if span is not None else None
        return None

    # ------------------------------------------------------------------
    # Honest graph contract — AgentWorkflow (multi-agent) path
    # ------------------------------------------------------------------
    # AgentWorkflow carries the developer-declared agent name only inside its
    # workflow event stream: the structured ``AgentInput``/``AgentOutput``
    # (``current_agent_name``) and the built-in ``handoff`` ``ToolCall``
    # (``tool_kwargs["to_agent"]``). None of these are instrumentation events —
    # they reach the adapter as the step function's ``ev`` argument on each
    # workflow-step span. We read them off ``bound_args`` to emit a
    # producer-honest agent_name + a real agent.handoff, guarded by the shared
    # honesty filter so the framework's unnamed default ("Agent") never becomes
    # a fabricated node.

    def _stamp_current_agent(self, payload: Dict[str, Any], span_id: Optional[str]) -> None:
        """Stamp the run's active AgentWorkflow agent onto a payload (e.g. an
        in-turn model.invoke) so its graph node links to the acting agent. A
        no-op for the pure RAG path, where no agent turn is ever active."""
        root = self._root_span_of(span_id)
        if root is None:
            return
        with self._lock:
            name = self._current_agent.get(root)
        if name:
            payload["agent_name"] = name

    def _handle_workflow_step(self, span_id: str, bound_args: Any) -> None:
        """Surface honest agent identity / handoff from an AgentWorkflow step."""
        if bound_args is None:
            return
        args = getattr(bound_args, "arguments", None)
        if not isinstance(args, dict):
            return
        ev = args.get("ev")
        if ev is None:
            return
        ev_type = type(ev).__name__
        if ev_type == "AgentInput":
            self._on_workflow_agent_turn(ev, span_id, "agent.input")
        elif ev_type == "AgentOutput":
            self._on_workflow_agent_turn(ev, span_id, "agent.output")
        elif ev_type == "ToolCall":
            self._on_workflow_handoff(ev, span_id)

    def _on_workflow_agent_turn(self, ev: Any, span_id: str, event_type: str) -> None:
        """Emit agent.input/agent.output for a workflow agent turn, stamping the
        honest ``current_agent_name`` (and tracking it as the run's active agent
        so an in-turn model.invoke inherits it). Emits the event even when the
        name is a generic default, but WITHOUT a fabricated agent_name."""
        # ateam parity (#3): prefer the honest type; fall back to the raw
        # current_agent_name VERBATIM (sanitized) so an AgentWorkflow whose agent
        # uses the generic default ("Agent") still renders like ateam. Handoff
        # endpoints below stay on honest_agent_type.
        name = honest_agent_type(getattr(ev, "current_agent_name", None)) or _s(getattr(ev, "current_agent_name", None))
        payload = self._payload()
        if name:
            payload["agent_name"] = name
            root = self._root_span_of(span_id)
            if root is not None:
                with self._lock:
                    self._current_agent[root] = name
        if self._config.capture_content:
            if event_type == "agent.input":
                messages = getattr(ev, "input", None)
                if messages:
                    payload["input"] = safe_serialize(_serialize_messages(messages))
            else:
                response = getattr(ev, "response", None)
                content = getattr(response, "content", None) if response is not None else None
                if content:
                    payload["output"] = str(content)
        self._fire(event_type, payload, span_id=span_id)

    def _on_workflow_handoff(self, ev: Any, span_id: str) -> None:
        """Emit a deduped agent.handoff for the built-in AgentWorkflow handoff
        tool. ``from_agent`` is the run's active agent; ``to_agent`` is the
        transfer target. Both endpoints must be producer-honest."""
        if getattr(ev, "tool_name", None) != "handoff":
            return
        kwargs = getattr(ev, "tool_kwargs", None)
        if not isinstance(kwargs, dict):
            return
        to_agent = honest_agent_type(kwargs.get("to_agent"))
        if not to_agent:
            return
        root = self._root_span_of(span_id)
        with self._lock:
            from_agent = self._current_agent.get(root) if root is not None else None
        if not from_agent:
            return
        edge = (from_agent, to_agent)
        if root is not None:
            with self._lock:
                seen = self._handoffs_seen.setdefault(root, set())
                if edge in seen:
                    return
                seen.add(edge)
        self._fire(
            "agent.handoff",
            self._payload(from_agent=from_agent, to_agent=to_agent),
            span_id=span_id,
        )

    def _flush_all(self) -> None:
        with self._lock:
            collectors = list(self._collectors.values())
            self._collectors.clear()
            self._open_spans.clear()
            self._timestamps.clear()
            self._llm_start_times.clear()
            self._current_agent.clear()
            self._handoffs_seen.clear()
        for c in collectors:
            try:
                c.flush()
            except Exception:
                log.warning("layerlens: error flushing LlamaIndex collector", exc_info=True)

    # ------------------------------------------------------------------
    # Span lifecycle (called by the thin span handler)
    # ------------------------------------------------------------------

    def _on_span_enter(self, id_: str, parent_span_id: Optional[str], bound_args: Any = None) -> Any:
        with self._lock:
            span = _BaseSpan(id_=id_, parent_id=parent_span_id)
            self._open_spans[id_] = span
            self._timestamps[id_] = time.time()
            if parent_span_id is None or parent_span_id not in self._open_spans:
                self._collectors[id_] = TraceCollector(self._client, self._config)
        # Inspect the AgentWorkflow step event (if any) for honest agent
        # identity + handoffs. Done AFTER releasing the lock and registering the
        # span so ``_fire``/``_collector_for`` can resolve the owning collector.
        self._handle_workflow_step(id_, bound_args)
        return span

    def _on_span_exit(self, id_: str) -> Any:
        with self._lock:
            span = self._open_spans.get(id_)
            self._timestamps.pop(id_, None)
            collector = self._collectors.pop(id_, None)
            if collector is not None:
                self._current_agent.pop(id_, None)
                self._handoffs_seen.pop(id_, None)
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

        resp_id = _response_id_from_response(response)
        if resp_id:
            payload["response_id"] = resp_id

        tokens = self._normalize_tokens(_usage_from_response(response))
        payload.update(tokens)

        start = self._llm_start_times.pop(span_id, None) if span_id else None
        if start is not None:
            payload["latency_ms"] = (time.time() - start) * 1000

        self._stamp_current_agent(payload, span_id)

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

        self._stamp_current_agent(payload, span_id)

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
        # When L3 model metadata is suppressed, skip the costly embedding serialization
        # — bulk ingestion runs fire thousands of these events and the collector
        # would drop them anyway.
        if not self._config.l3_model_metadata:
            return
        span_id = getattr(event, "span_id", None)
        payload = self._payload(embedding=True)
        model = _model_from_dict(getattr(event, "model_dict", None))
        if model:
            payload["model"] = model
        self._fire("model.invoke", payload, span_id=span_id, span_name="embedding")

    def _on_embedding_end(self, event: Any) -> None:
        if not self._config.l3_model_metadata:
            return
        span_id = getattr(event, "span_id", None)
        chunks = getattr(event, "chunks", None)
        embeddings = getattr(event, "embeddings", None)
        payload = self._payload(embedding=True)
        if chunks is not None:
            payload["num_chunks"] = len(chunks)
            # Chunking metrics: surface total/avg length so slow-retrieval diagnosis
            # can correlate chunk size against downstream latency.
            total_len = 0
            nonempty = 0
            for c in chunks:
                try:
                    s = str(c)
                except Exception:
                    continue
                if s:
                    total_len += len(s)
                    nonempty += 1
            if nonempty:
                payload["chunk_chars_total"] = total_len
                payload["chunk_chars_avg"] = total_len // nonempty
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
            error_type=(type(exc).__name__ if isinstance(exc, BaseException) else "Exception"),
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
            return adapter._on_span_enter(id_, parent_span_id, bound_args=bound_args)

        def prepare_to_exit_span(
            self,
            id_: str,
            bound_args: Any,
            instance: Any = None,
            result: Any = None,
            **kw: Any,
        ) -> Any:
            return adapter._on_span_exit(id_)

        def prepare_to_drop_span(
            self,
            id_: str,
            bound_args: Any,
            instance: Any = None,
            err: Any = None,
            **kw: Any,
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


def _response_id_from_response(response: Any) -> str | None:
    """Extract the provider's response id from ChatResponse / CompletionResponse."""
    if response is None:
        return None
    raw = getattr(response, "raw", None)
    if raw is None:
        return None
    if isinstance(raw, dict):
        resp_id = raw.get("id")
    else:
        resp_id = getattr(raw, "id", None)
    if resp_id:
        return str(resp_id)
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
