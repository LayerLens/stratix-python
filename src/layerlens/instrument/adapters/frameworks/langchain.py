from __future__ import annotations

from uuid import UUID
from typing import Any, Dict, List, Optional, Sequence

from ._base_framework import FrameworkTracer

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
    def __init__(self, client: Any) -> None:
        BaseCallbackHandler.__init__(self)
        FrameworkTracer.__init__(self, client)

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
        self._get_or_create_span(run_id, parent_run_id, name=name, kind="chain", input=inputs)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._finish_span(run_id, output=outputs)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._finish_span(run_id, error=str(error))

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
        self._get_or_create_span(run_id, parent_run_id, name=name, kind="llm", input=prompts)

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
        input_data = [[_serialize_lc_message(m) for m in batch] for batch in messages]
        self._get_or_create_span(run_id, parent_run_id, name=name, kind="llm", input=input_data)

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

        s = self._spans.get(str(run_id))
        if s is not None:
            try:
                llm_output = response.llm_output
                if llm_output:
                    if "token_usage" in llm_output:
                        s.metadata["usage"] = llm_output["token_usage"]
                    if "model_name" in llm_output:
                        s.metadata["model"] = llm_output["model_name"]
            except AttributeError:
                pass

        self._finish_span(run_id, output=output)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._finish_span(run_id, error=str(error))

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
        self._get_or_create_span(run_id, parent_run_id, name=name, kind="tool", input=input_str)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._finish_span(run_id, output=output)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._finish_span(run_id, error=str(error))

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
        self._get_or_create_span(run_id, parent_run_id, name=name, kind="retriever", input=query)

    def on_retriever_end(
        self,
        documents: Sequence[Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        output = [_serialize_lc_document(d) for d in documents]
        self._finish_span(run_id, output=output)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._finish_span(run_id, error=str(error))

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
