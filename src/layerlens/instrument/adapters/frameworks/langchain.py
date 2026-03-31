from __future__ import annotations

from uuid import UUID
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from ._base_framework import FrameworkTracer

if TYPE_CHECKING:
    from ..._capture_config import CaptureConfig

try:
    from langchain_core.callbacks import BaseCallbackHandler  # pyright: ignore[reportAssignmentType]
except ImportError:

    class BaseCallbackHandler:  # type: ignore[no-redef]
        def __init_subclass__(cls, **kwargs: Any) -> None:
            raise ImportError(
                "The 'langchain-core' package is required for LangChain instrumentation. "
                "Install it with: pip install layerlens[langchain]"
            )


class LangChainCallbackHandler(BaseCallbackHandler, FrameworkTracer):
    _adapter_name: str = "langchain"

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        BaseCallbackHandler.__init__(self)
        FrameworkTracer.__init__(self, client, capture_config=capture_config)

    # -- Chain --

    def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        serialized = serialized or {}
        name = serialized.get("name") or serialized.get("id", ["unknown"])[-1]
        self._emit("agent.input", {"name": name, "input": inputs}, run_id, parent_run_id)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._emit("agent.output", {"output": outputs, "status": "ok"}, run_id)
        self._maybe_flush(run_id)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._emit("agent.error", {"error": str(error), "status": "error"}, run_id)
        self._maybe_flush(run_id)

    # -- LLM --

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
        self._emit("model.invoke", {"name": name, "messages": prompts}, run_id, parent_run_id)

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
        self._emit(
            "model.invoke",
            {"name": name, "messages": [[_serialize_lc_message(m) for m in batch] for batch in messages]},
            run_id,
            parent_run_id,
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
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
        if model_name or output:
            self._emit(
                "model.invoke",
                {"model": model_name, "output_message": output},
                run_id,
                parent_run_id,
            )

        usage = llm_output.get("token_usage", {})
        if usage:
            self._emit("cost.record", usage, run_id, parent_run_id)

        self._maybe_flush(run_id)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._emit("agent.error", {"error": str(error), "status": "error"}, run_id)
        self._maybe_flush(run_id)

    # -- Tool --

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
        self._emit("tool.call", {"name": name, "input": input_str}, run_id, parent_run_id)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._emit("tool.result", {"output": output}, run_id)
        self._maybe_flush(run_id)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._emit("agent.error", {"error": str(error), "status": "error"}, run_id)
        self._maybe_flush(run_id)

    # -- Retriever --

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
        self._emit("tool.call", {"name": name, "input": query}, run_id, parent_run_id)

    def on_retriever_end(
        self,
        documents: Sequence[Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        output = [_serialize_lc_document(d) for d in documents]
        self._emit("tool.result", {"output": output}, run_id)
        self._maybe_flush(run_id)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._emit("agent.error", {"error": str(error), "status": "error"}, run_id)
        self._maybe_flush(run_id)

    # -- Text (required by base) --

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
