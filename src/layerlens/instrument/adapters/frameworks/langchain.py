from __future__ import annotations

import functools
from uuid import UUID
from typing import Any, Dict, List, Optional, Sequence

from ._base_framework import FrameworkAdapter
from ..._capture_config import CaptureConfig


def _auto_flush(fn):  # type: ignore[type-arg]
    """Decorator: after the callback returns, flush if this was the outermost run."""
    @functools.wraps(fn)
    def wrapper(self, *args, run_id, **kwargs):  # type: ignore[no-untyped-def]
        fn(self, *args, run_id=run_id, **kwargs)
        run = self._get_run()
        if run is not None:
            if str(run_id) == run.data.get("root_run_id"):
                self._end_run()
        elif str(run_id) == self._root_run_id and self._collector is not None:
            self._flush_collector()
            self._root_run_id = None
    return wrapper


try:
    from langchain_core.callbacks import BaseCallbackHandler  # pyright: ignore[reportAssignmentType]
except ImportError:

    class BaseCallbackHandler:  # type: ignore[no-redef]
        def __init_subclass__(cls, **kwargs: Any) -> None:
            raise ImportError(
                "The 'langchain-core' package is required for LangChain instrumentation. "
                "Install it with: pip install layerlens[langchain]"
            )


class LangChainCallbackHandler(BaseCallbackHandler, FrameworkAdapter):
    name = "langchain"

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        BaseCallbackHandler.__init__(self)
        FrameworkAdapter.__init__(self, client, capture_config=capture_config)
        # Pending LLM runs: run_id -> {name, messages, parent_run_id}
        self._pending_llm: Dict[str, Dict[str, Any]] = {}
        # Context tokens for span propagation: run_id -> token from _push_context
        self._run_contexts: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Chain callbacks
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None:
            run = self._begin_run()
            run.data["root_run_id"] = str(run_id)
        serialized = serialized or {}
        name = serialized.get("name") or serialized.get("id", ["unknown"])[-1]
        payload = self._payload(name=name)
        self._set_if_capturing(payload, "input", inputs)
        self._emit("agent.input", payload, run_id=run_id, parent_run_id=parent_run_id)

    @_auto_flush
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        payload = self._payload(status="ok")
        self._set_if_capturing(payload, "output", outputs)
        self._emit("agent.output", payload, run_id=run_id)

    @_auto_flush
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._emit("agent.error", self._payload(error=str(error), status="error"), run_id=run_id)

    # ------------------------------------------------------------------
    # LLM callbacks — merged into single model.invoke on end
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: Optional[Dict[str, Any]],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        serialized = serialized or {}
        name = serialized.get("name") or serialized.get("id", ["unknown"])[-1]
        self._start_timer(str(run_id))
        pending: Dict[str, Any] = {
            "name": name,
            "parent_run_id": parent_run_id,
        }
        self._set_if_capturing(pending, "messages", prompts)
        self._pending_llm[str(run_id)] = pending
        span_id, _ = self._span_id_for(run_id)
        self._run_contexts[str(run_id)] = self._push_context(span_id)

    def on_chat_model_start(
        self,
        serialized: Optional[Dict[str, Any]],
        messages: List[List[Any]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        serialized = serialized or {}
        name = serialized.get("name") or serialized.get("id", ["unknown"])[-1]
        self._start_timer(str(run_id))
        pending: Dict[str, Any] = {
            "name": name,
            "parent_run_id": parent_run_id,
        }
        self._set_if_capturing(
            pending, "messages",
            [[_serialize_lc_message(m) for m in batch] for batch in messages],
        )
        self._pending_llm[str(run_id)] = pending
        span_id, _ = self._span_id_for(run_id)
        self._run_contexts[str(run_id)] = self._push_context(span_id)

    @_auto_flush
    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._pop_context(self._run_contexts.pop(str(run_id), None))
        pending = self._pending_llm.pop(str(run_id), {})

        # Extract response data
        output = None
        try:
            generations = response.generations
            if generations and generations[0]:
                output = generations[0][0].text
        except (AttributeError, IndexError):
            pass

        try:
            llm_output = response.llm_output or {}
        except AttributeError:
            llm_output = {}

        model_name = llm_output.get("model_name")

        # Build single merged model.invoke event
        payload = self._payload()
        if pending.get("name"):
            payload["name"] = pending["name"]
        if model_name:
            payload["model"] = model_name
        self._set_if_capturing(payload, "messages", pending.get("messages"))
        self._set_if_capturing(payload, "output_message", output)

        # Latency
        latency_ms = self._stop_timer(str(run_id))
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms

        # Tokens
        usage = llm_output.get("token_usage") or llm_output.get("usage_metadata")
        tokens = self._normalize_tokens(usage)
        payload.update(tokens)

        self._emit(
            "model.invoke", payload,
            run_id=run_id, parent_run_id=pending.get("parent_run_id"),
        )

        # Separate cost.record if we have token data
        if tokens:
            cost_payload = self._payload()
            if model_name:
                cost_payload["model"] = model_name
            cost_payload.update(tokens)
            self._emit("cost.record", cost_payload, run_id=run_id, parent_run_id=pending.get("parent_run_id"))

    @_auto_flush
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._pop_context(self._run_contexts.pop(str(run_id), None))
        pending = self._pending_llm.pop(str(run_id), {})

        payload = self._payload(error=str(error))
        if pending.get("name"):
            payload["name"] = pending["name"]
        latency_ms = self._stop_timer(str(run_id))
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        self._emit("model.invoke", payload, run_id=run_id, parent_run_id=pending.get("parent_run_id"))

        self._emit("agent.error", self._payload(error=str(error), status="error"), run_id=run_id)

    # ------------------------------------------------------------------
    # Tool callbacks
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: Optional[Dict[str, Any]],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name", "tool")
        payload = self._payload(name=name)
        self._set_if_capturing(payload, "input", input_str)
        self._emit("tool.call", payload, run_id=run_id, parent_run_id=parent_run_id)

    @_auto_flush
    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        payload = self._payload()
        self._set_if_capturing(payload, "output", output)
        self._emit("tool.result", payload, run_id=run_id)

    @_auto_flush
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._emit("agent.error", self._payload(error=str(error), status="error"), run_id=run_id)

    # ------------------------------------------------------------------
    # Retriever callbacks
    # ------------------------------------------------------------------

    def on_retriever_start(
        self,
        serialized: Optional[Dict[str, Any]],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name", "retriever")
        payload = self._payload(name=name)
        self._set_if_capturing(payload, "input", query)
        self._emit("tool.call", payload, run_id=run_id, parent_run_id=parent_run_id)

    @_auto_flush
    def on_retriever_end(
        self,
        documents: Sequence[Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        payload = self._payload()
        self._set_if_capturing(
            payload, "output",
            [_serialize_lc_document(d) for d in documents],
        )
        self._emit("tool.result", payload, run_id=run_id)

    @_auto_flush
    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._emit("agent.error", self._payload(error=str(error), status="error"), run_id=run_id)

    # ------------------------------------------------------------------
    # Agent callbacks
    # ------------------------------------------------------------------

    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        payload = self._payload(tool=getattr(action, "tool", "unknown"))
        self._set_if_capturing(payload, "tool_input", getattr(action, "tool_input", None))
        self._set_if_capturing(payload, "log", getattr(action, "log", None) or None)
        self._emit("agent.input", payload, run_id=run_id, parent_run_id=parent_run_id)

    @_auto_flush
    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        payload = self._payload(status="ok")
        self._set_if_capturing(payload, "output", getattr(finish, "return_values", None))
        self._set_if_capturing(payload, "log", getattr(finish, "log", None) or None)
        self._emit("agent.output", payload, run_id=run_id, parent_run_id=parent_run_id)

    # ------------------------------------------------------------------
    # No-ops (required by base)
    # ------------------------------------------------------------------

    def on_text(self, text: str, **kwargs: Any) -> None:
        pass


def _serialize_lc_message(msg: Any) -> Any:
    try:
        return {"type": msg.type, "content": msg.content}
    except AttributeError:
        return str(msg)


def _serialize_lc_document(doc: Any) -> Any:
    try:
        return {"page_content": doc.page_content, "metadata": doc.metadata}
    except AttributeError:
        return str(doc)
