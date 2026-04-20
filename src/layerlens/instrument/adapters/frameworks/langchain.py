from __future__ import annotations

import time
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
        if run is not None and str(run_id) == run.data.get("root_run_id"):
            self._end_run()

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
        # Pending LLM runs: run_id -> {name, messages, parent_run_id, tokens_accum, first_token_at_ns}
        self._pending_llm: Dict[str, Dict[str, Any]] = {}

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
        self._emit("agent.output", payload, run_id=run_id, parent_run_id=parent_run_id)

    @_auto_flush
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._emit(
            "agent.error", self._payload(error=str(error), status="error"), run_id=run_id, parent_run_id=parent_run_id
        )

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
            pending,
            "messages",
            [[_serialize_lc_message(m) for m in batch] for batch in messages],
        )
        self._pending_llm[str(run_id)] = pending

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Accumulate streaming tokens; captures time-to-first-token per run."""
        pending = self._pending_llm.get(str(run_id))
        if pending is None:
            return
        if pending.get("first_token_at_ns") is None:
            pending["first_token_at_ns"] = time.time_ns()
        pending["tokens_accum"] = (pending.get("tokens_accum") or 0) + 1
        if self._config.capture_content:
            pending["streamed_text"] = (pending.get("streamed_text") or "") + (token or "")

    @_auto_flush
    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        pending = self._pending_llm.pop(str(run_id), {})

        # Extract response data
        output = None
        finish_reason = None
        tool_calls: list[dict[str, Any]] = []
        try:
            generations = response.generations
            if generations and generations[0]:
                gen0 = generations[0][0]
                raw_output = getattr(gen0, "text", None)
                output = raw_output if isinstance(raw_output, str) else None
                gen_info = getattr(gen0, "generation_info", None)
                if not isinstance(gen_info, dict):
                    gen_info = {}
                fr = gen_info.get("finish_reason")
                finish_reason = fr if isinstance(fr, str) else None
                # Extract tool_calls from message additional_kwargs (chat models)
                msg = getattr(gen0, "message", None)
                if msg is not None:
                    extra = getattr(msg, "additional_kwargs", None)
                    if not isinstance(extra, dict):
                        extra = {}
                    raw_calls = extra.get("tool_calls") or getattr(msg, "tool_calls", None) or []
                    if not isinstance(raw_calls, (list, tuple)):
                        raw_calls = []
                    for tc in raw_calls:
                        if isinstance(tc, dict):
                            fn = tc.get("function") or {}
                            tool_calls.append(
                                {
                                    "id": tc.get("id"),
                                    "tool_name": fn.get("name") or tc.get("name"),
                                    "arguments": fn.get("arguments") or tc.get("args"),
                                }
                            )
        except (AttributeError, IndexError):
            pass

        try:
            llm_output = response.llm_output
        except AttributeError:
            llm_output = None
        if not isinstance(llm_output, dict):
            llm_output = {}

        raw_model = llm_output.get("model_name")
        model_name = raw_model if isinstance(raw_model, str) else None

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

        # Streaming metrics — time-to-first-token + chunk count
        first_tok = pending.get("first_token_at_ns")
        if first_tok is not None:
            payload["streaming"] = True
            payload["streamed_chunks"] = pending.get("tokens_accum", 0)

        if finish_reason is not None:
            payload["finish_reason"] = finish_reason

        # Tokens
        usage = llm_output.get("token_usage") or llm_output.get("usage_metadata")
        tokens = self._normalize_tokens(usage)
        payload.update(tokens)

        self._emit(
            "model.invoke",
            payload,
            run_id=run_id,
            parent_run_id=pending.get("parent_run_id"),
        )

        # Emit tool.call events for any tool calls the model requested
        for tc in tool_calls:
            tc_payload = self._payload(**tc)
            if model_name:
                tc_payload["model"] = model_name
            self._emit("tool.call", tc_payload, run_id=run_id, parent_run_id=pending.get("parent_run_id"))

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
        pending = self._pending_llm.pop(str(run_id), {})

        payload = self._payload(error=str(error))
        if pending.get("name"):
            payload["name"] = pending["name"]
        latency_ms = self._stop_timer(str(run_id))
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        self._emit("model.invoke", payload, run_id=run_id, parent_run_id=pending.get("parent_run_id"))

        self._emit(
            "agent.error",
            self._payload(error=str(error), status="error"),
            run_id=run_id,
            parent_run_id=pending.get("parent_run_id"),
        )

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
        self._emit("tool.result", payload, run_id=run_id, parent_run_id=parent_run_id)

    @_auto_flush
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._emit(
            "agent.error", self._payload(error=str(error), status="error"), run_id=run_id, parent_run_id=parent_run_id
        )

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
            payload,
            "output",
            [_serialize_lc_document(d) for d in documents],
        )
        self._emit("tool.result", payload, run_id=run_id, parent_run_id=parent_run_id)

    @_auto_flush
    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._emit(
            "agent.error", self._payload(error=str(error), status="error"), run_id=run_id, parent_run_id=parent_run_id
        )

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
