from __future__ import annotations

import time
import functools
from uuid import UUID
from typing import Any, Dict, List, Optional, Sequence

from ..._identity import _API_METHOD_RE, _is_generic
from ._base_framework import FrameworkAdapter
from ..._capture_config import CaptureConfig
from ._langchain_memory import TracedMemory, MemoryMutationTracker, wrap_memory
from ....attestation._hash import canonical_json

__all__ = [
    "LangChainCallbackHandler",
    "MemoryMutationTracker",
    "TracedMemory",
    "wrap_memory",
]


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
    # fmt: off
    from langchain_core.messages import BaseMessage
    from langchain_core.callbacks import BaseCallbackHandler  # pyright: ignore[reportAssignmentType]
    # fmt: on
except ImportError:
    BaseMessage = ()  # type: ignore[assignment] # isinstance(x, ()) is always False

    class BaseCallbackHandler:  # type: ignore[no-redef]
        def __init_subclass__(cls, **kwargs: Any) -> None:
            raise ImportError(
                "The 'langchain-core' package is required for LangChain instrumentation. "
                "Install it with: pip install layerlens[langchain]"
            )


# Recursion guard for pathological/cyclic state. Graph state is shallow in
# practice; this only bounds adversarial inputs.
_MAX_JSONABLE_DEPTH = 25


def _to_jsonable(obj: Any, _depth: int = 0) -> Any:
    """Convert framework state into a JSON-safe structure for trace payloads.

    LangGraph threads the graph state — typically full of LangChain message
    objects — through the chain callbacks. Those objects are not JSON
    serializable, so embedding them raw would make the attestation hash chain
    raise and silently drop the event (``agent.input`` / ``agent.node.enter`` /
    ``agent.handoff``). We serialize messages the same way the ``model.invoke``
    path does (:func:`_serialize_lc_message`) and recurse through dicts/lists.

    Values that already serialize are returned **unchanged**, so attestation
    hashes are identical for the common, already-clean case.
    """
    if _depth > _MAX_JSONABLE_DEPTH:
        return repr(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, BaseMessage):
        data: Dict[str, Any] = {
            "type": getattr(obj, "type", None),
            "content": _to_jsonable(getattr(obj, "content", None), _depth + 1),
        }
        tool_calls = getattr(obj, "tool_calls", None)
        if tool_calls:
            data["tool_calls"] = _to_jsonable(tool_calls, _depth + 1)
        return data
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v, _depth + 1) for v in obj]
    # Leaf object: keep it if the attestation serializer already handles it
    # (datetime / Enum / dataclass / to_dict), otherwise fall back to a string
    # so we never drop the event.
    try:
        canonical_json(obj)
        return obj
    except TypeError:
        return str(obj)


# LangChain composition primitives + legacy chain/agent classes + LCEL component
# class defaults. When a runnable is invoked WITHOUT a developer-declared
# ``run_name``, LangChain passes the runnable's ``get_name()`` (a class default
# like ``RunnableSequence``, ``AgentExecutor``, or the class of the prompt/parser
# component running as a chain step — ``ChatPromptTemplate``, ``StrOutputParser``)
# as the ``name`` kwarg to ``on_chain_start``. That is plumbing, never a
# producer-declared agent identity — surfacing it in the Agent column is a
# fabrication, so it must stay honestly blank. All of LangChain's composition
# wrappers live under the ``Runnable*`` prefix (handled separately); this is the
# precise denylist of the remaining legacy chain/agent + LCEL component class
# defaults. Model component class defaults (``ChatOpenAI`` ...) are rejected by
# the shared ``_is_generic`` guard, and models run through the on_(chat_)model
# path rather than on_chain_start, so they never reach this list as an agent
# identity anyway.
_LANGCHAIN_CLASS_DEFAULTS = frozenset(
    {
        # legacy chain / agent class defaults
        "agentexecutor",
        "llmchain",
        "conversationchain",
        "conversationalretrievalchain",
        "retrievalqa",
        "retrievalqawithsourceschain",
        "stuffdocumentschain",
        "mapreducedocumentschain",
        "refinedocumentschain",
        "sequentialchain",
        "simplesequentialchain",
        "transformchain",
        # LCEL prompt-template component class defaults
        "prompttemplate",
        "chatprompttemplate",
        "humanmessageprompttemplate",
        "systemmessageprompttemplate",
        "aimessageprompttemplate",
        "chatmessageprompttemplate",
        "messagesplaceholder",
        "fewshotprompttemplate",
        "fewshotchatmessageprompttemplate",
        "pipelineprompttemplate",
        "imageprompttemplate",
        # LCEL output-parser component class defaults
        "stroutputparser",
        "jsonoutputparser",
        "simplejsonoutputparser",
        "pydanticoutputparser",
        "listoutputparser",
        "commaseparatedlistoutputparser",
        "markdownlistoutputparser",
        "numberedlistoutputparser",
        "xmloutputparser",
        "structuredoutputparser",
        "booleanoutputparser",
        "datetimeoutputparser",
        "enumoutputparser",
        "regexparser",
        "outputfixingparser",
        "retryoutputparser",
        "jsonoutputkeytoolsparser",
        "pydantictoolsparser",
    }
)


def _honest_run_name(raw: Any) -> Optional[str]:
    """Return a developer-DECLARED LCEL run name honest for the Agent column, else None.

    LangChain surfaces the ``run_name`` a developer set via
    ``.with_config(run_name=...)`` as the ``name`` kwarg on ``on_chain_start``.
    When no run_name was declared, the same kwarg carries the runnable's
    ``get_name()`` class default (``RunnableSequence``, ``RunnableParallel<a,b>``,
    ``AgentExecutor``, ...). We surface ONLY a genuine, distinctive developer name
    and reject every class default / generic placeholder / dotted API-method label,
    reusing the shared identity guard so this adapter never fabricates a node.
    A run genuinely without a declared name stays honestly blank.
    """
    if not isinstance(raw, str):
        return None
    name = raw.strip()
    if not name:
        return None
    # ``RunnableParallel<first,second>`` -> ``RunnableParallel`` before matching.
    base = name.split("<", 1)[0].strip()
    low = base.lower()
    if low.startswith("runnable") or low in _LANGCHAIN_CLASS_DEFAULTS:
        return None
    if _is_generic(name) or _API_METHOD_RE.match(name.lower()):
        return None
    return name


class LangChainCallbackHandler(BaseCallbackHandler, FrameworkAdapter):
    name = "langchain"

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        BaseCallbackHandler.__init__(self)
        FrameworkAdapter.__init__(self, client, capture_config=capture_config)
        # Pending LLM runs: run_id -> {name, messages, parent_run_id, tokens_accum, first_token_at_ns}
        self._pending_llm: Dict[str, Dict[str, Any]] = {}
        # run_id -> producer-declared honest agent_name for a chain run. Populated
        # from the LCEL run_name in on_chain_start and inherited by sub-chains /
        # child model+tool runs so they attribute to the same honest node. Cleared
        # in on_chain_end / on_chain_error.
        self._chain_agent_names: Dict[str, str] = {}

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
        # Honest node identity: a developer-declared LCEL run_name (the ``name``
        # kwarg) — never a class default. Inherit the enclosing named chain's
        # identity for sub-chains so nested runs attribute to the same node.
        agent_name = _honest_run_name(kwargs.get("name"))
        if agent_name is None and parent_run_id is not None:
            agent_name = self._chain_agent_names.get(str(parent_run_id))
        if agent_name is not None:
            self._chain_agent_names[str(run_id)] = agent_name
            payload["agent_name"] = agent_name
        self._set_if_capturing(payload, "input", _to_jsonable(inputs))
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
        agent_name = self._chain_agent_names.pop(str(run_id), None)
        if agent_name is not None:
            payload["agent_name"] = agent_name
        self._set_if_capturing(payload, "output", _to_jsonable(outputs))
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
        payload = self._payload(error=str(error), error_type=type(error).__name__, status="error")
        agent_name = self._chain_agent_names.pop(str(run_id), None)
        if agent_name is not None:
            payload["agent_name"] = agent_name
        self._emit(
            "agent.error",
            payload,
            run_id=run_id,
            parent_run_id=parent_run_id,
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
            "agent_name": self._chain_agent_names.get(str(parent_run_id)) if parent_run_id else None,
            "start_ns": time.time_ns(),
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
            "agent_name": self._chain_agent_names.get(str(parent_run_id)) if parent_run_id else None,
            "start_ns": time.time_ns(),
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
        if pending.get("agent_name"):
            payload["agent_name"] = pending["agent_name"]
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
            start_ns = pending.get("start_ns")
            if first_tok and start_ns:
                payload["ttft_ms"] = (first_tok - start_ns) / 1e6

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
            self._emit(
                "tool.call",
                tc_payload,
                run_id=run_id,
                parent_run_id=pending.get("parent_run_id"),
            )

        # Separate cost.record if we have token data
        if tokens:
            cost_payload = self._payload()
            if model_name:
                cost_payload["model"] = model_name
            cost_payload.update(tokens)
            self._emit(
                "cost.record",
                cost_payload,
                run_id=run_id,
                parent_run_id=pending.get("parent_run_id"),
            )

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
        if pending.get("agent_name"):
            payload["agent_name"] = pending["agent_name"]
        latency_ms = self._stop_timer(str(run_id))
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        self._emit(
            "model.invoke",
            payload,
            run_id=run_id,
            parent_run_id=pending.get("parent_run_id"),
        )

        self._emit(
            "agent.error",
            self._payload(error=str(error), error_type=type(error).__name__, status="error"),
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
        self._set_if_capturing(payload, "input", _to_jsonable(input_str))
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
            "agent.error",
            self._payload(error=str(error), error_type=type(error).__name__, status="error"),
            run_id=run_id,
            parent_run_id=parent_run_id,
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
            "agent.error",
            self._payload(error=str(error), error_type=type(error).__name__, status="error"),
            run_id=run_id,
            parent_run_id=parent_run_id,
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
        self._set_if_capturing(payload, "tool_input", _to_jsonable(getattr(action, "tool_input", None)))
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
        self._set_if_capturing(payload, "output", _to_jsonable(getattr(finish, "return_values", None)))
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
