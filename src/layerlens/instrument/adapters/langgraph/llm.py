"""
STRATIX LangGraph LLM Wrapper

Wraps LLM calls to emit model.invoke (L3) events.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    from layerlens.instrument.adapters._base import BaseAdapter

logger = logging.getLogger(__name__)


MessageT = TypeVar("MessageT")


@dataclass
class LLMInvocation:
    """Tracks a single LLM invocation."""
    model: str
    provider: str
    start_time_ns: int
    end_time_ns: int | None = None
    input_messages: list[Any] | None = None
    output_message: Any | None = None
    token_usage: dict[str, int] | None = None
    error: str | None = None


class TracedLLM:
    """
    Wrapper around an LLM that emits model.invoke events.

    Compatible with LangChain/LangGraph chat models.

    Usage:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4")
        traced_llm = TracedLLM(llm, stratix_instance=stratix)

        # Use as normal
        response = traced_llm.invoke(messages)
    """

    def __init__(
        self,
        llm: Any,
        stratix_instance: Any = None,
        adapter: BaseAdapter | None = None,
        model_name: str | None = None,
        provider: str | None = None,
    ):
        """
        Initialize the traced LLM.

        Args:
            llm: The underlying LLM instance
            stratix_instance: STRATIX SDK instance (legacy)
            adapter: BaseAdapter instance (new-style)
            model_name: Model name override (auto-detected if not provided)
            provider: Provider name override (auto-detected if not provided)
        """
        self._llm = llm
        self._stratix = stratix_instance
        self._adapter = adapter
        self._model_name = model_name or self._detect_model_name()
        self._provider = provider or self._detect_provider()
        self._invocations: list[LLMInvocation] = []

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        """
        Invoke the LLM with tracing.

        Args:
            messages: Input messages
            **kwargs: Additional arguments

        Returns:
            LLM response
        """
        invocation = LLMInvocation(
            model=self._model_name,
            provider=self._provider,
            start_time_ns=time.time_ns(),
            input_messages=self._serialize_messages(messages),
        )
        self._invocations.append(invocation)

        try:
            response = self._llm.invoke(messages, **kwargs)
            invocation.end_time_ns = time.time_ns()
            invocation.output_message = self._serialize_response(response)
            invocation.token_usage = self._extract_token_usage(response)
            self._emit_model_invoke(invocation)
            return response

        except Exception as e:
            invocation.end_time_ns = time.time_ns()
            invocation.error = str(e)
            self._emit_model_invoke(invocation)
            raise

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        """
        Async invoke the LLM with tracing.

        Args:
            messages: Input messages
            **kwargs: Additional arguments

        Returns:
            LLM response
        """
        invocation = LLMInvocation(
            model=self._model_name,
            provider=self._provider,
            start_time_ns=time.time_ns(),
            input_messages=self._serialize_messages(messages),
        )
        self._invocations.append(invocation)

        try:
            response = await self._llm.ainvoke(messages, **kwargs)
            invocation.end_time_ns = time.time_ns()
            invocation.output_message = self._serialize_response(response)
            invocation.token_usage = self._extract_token_usage(response)
            self._emit_model_invoke(invocation)
            return response

        except Exception as e:
            invocation.end_time_ns = time.time_ns()
            invocation.error = str(e)
            self._emit_model_invoke(invocation)
            raise

    def stream(self, messages: Any, **kwargs: Any) -> Any:
        """
        Stream the LLM response with tracing.

        Note: For streaming, we emit the event after the stream is consumed.

        Args:
            messages: Input messages
            **kwargs: Additional arguments

        Yields:
            Response chunks
        """
        invocation = LLMInvocation(
            model=self._model_name,
            provider=self._provider,
            start_time_ns=time.time_ns(),
            input_messages=self._serialize_messages(messages),
        )
        self._invocations.append(invocation)

        try:
            chunks = []
            for chunk in self._llm.stream(messages, **kwargs):
                chunks.append(chunk)
                yield chunk

            invocation.end_time_ns = time.time_ns()
            invocation.output_message = self._combine_chunks(chunks)
            self._emit_model_invoke(invocation)

        except Exception as e:
            invocation.end_time_ns = time.time_ns()
            invocation.error = str(e)
            self._emit_model_invoke(invocation)
            raise

    def _detect_model_name(self) -> str:
        """Auto-detect model name from LLM instance."""
        # Try common attribute names
        for attr in ["model_name", "model", "_model_name", "model_id"]:
            if hasattr(self._llm, attr):
                value = getattr(self._llm, attr)
                if value:
                    return str(value)
        return "unknown"

    def _detect_provider(self) -> str:
        """Auto-detect provider from LLM instance."""
        class_name = self._llm.__class__.__name__.lower()

        if "openai" in class_name:
            return "openai"
        elif "anthropic" in class_name or "claude" in class_name:
            return "anthropic"
        elif "google" in class_name or "gemini" in class_name:
            return "google"
        elif "cohere" in class_name:
            return "cohere"
        elif "huggingface" in class_name:
            return "huggingface"

        return "unknown"

    def _serialize_messages(self, messages: Any) -> list[dict[str, Any]]:
        """Serialize input messages."""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        result = []
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict):
                    result.append(msg)
                elif hasattr(msg, "content") and hasattr(msg, "type"):
                    result.append({
                        "role": getattr(msg, "type", "unknown"),
                        "content": str(msg.content),
                    })
                else:
                    result.append({"content": str(msg)})

        return result

    def _serialize_response(self, response: Any) -> dict[str, Any]:
        """Serialize LLM response."""
        if isinstance(response, str):
            return {"content": response}

        if hasattr(response, "content"):
            result = {"content": str(response.content)}
            if hasattr(response, "type"):
                result["role"] = response.type
            return result

        return {"content": str(response)}

    def _extract_token_usage(self, response: Any) -> dict[str, int] | None:
        """Extract token usage from response."""
        # Try response_metadata (LangChain style)
        if hasattr(response, "response_metadata"):
            metadata = response.response_metadata
            if isinstance(metadata, dict) and "usage" in metadata:
                return metadata["usage"]

        # Try usage_metadata
        if hasattr(response, "usage_metadata"):
            return response.usage_metadata

        return None

    def _combine_chunks(self, chunks: list[Any]) -> dict[str, Any]:
        """Combine streaming chunks into single response."""
        content_parts = []
        for chunk in chunks:
            if hasattr(chunk, "content"):
                content_parts.append(str(chunk.content))
            elif isinstance(chunk, str):
                content_parts.append(chunk)

        return {"content": "".join(content_parts)}

    def _emit_model_invoke(self, invocation: LLMInvocation) -> None:
        """Emit model.invoke event via adapter (preferred) or legacy path."""
        duration_ns = (invocation.end_time_ns or 0) - invocation.start_time_ns

        # New-style: route through adapter.emit_event
        if self._adapter is not None:
            try:
                from layerlens.instrument.schema.events import ModelInvokeEvent
                typed_payload = ModelInvokeEvent.create(
                    model_name=invocation.model,
                    provider=invocation.provider,
                    input_messages=invocation.input_messages or [],
                    output_message=invocation.output_message,
                    token_usage=invocation.token_usage,
                    duration_ns=duration_ns,
                    error=invocation.error,
                )
                self._adapter.emit_event(typed_payload)
                return
            except Exception:
                logger.debug("Typed event emission failed, falling back to legacy", exc_info=True)

        # Legacy fallback
        if self._stratix and hasattr(self._stratix, "emit"):
            self._stratix.emit("model.invoke", {
                "model": invocation.model,
                "provider": invocation.provider,
                "input_messages": invocation.input_messages,
                "output_message": invocation.output_message,
                "token_usage": invocation.token_usage,
                "duration_ns": duration_ns,
                "error": invocation.error,
            })

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to underlying LLM."""
        return getattr(self._llm, name)


def wrap_llm_for_langgraph(
    llm: Any,
    stratix_instance: Any = None,
    adapter: BaseAdapter | None = None,
    model_name: str | None = None,
    provider: str | None = None,
) -> TracedLLM:
    """
    Wrap an LLM for use in LangGraph with STRATIX tracing.

    Usage:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4")
        traced_llm = wrap_llm_for_langgraph(llm, stratix_instance=stratix)

    Args:
        llm: LLM instance to wrap
        stratix_instance: STRATIX SDK instance
        adapter: BaseAdapter instance (new-style)
        model_name: Model name override
        provider: Provider name override

    Returns:
        TracedLLM wrapper
    """
    return TracedLLM(
        llm=llm,
        stratix_instance=stratix_instance,
        adapter=adapter,
        model_name=model_name,
        provider=provider,
    )


class LLMCallNode:
    """
    A LangGraph node that wraps an LLM call with tracing.

    Usage:
        llm_node = LLMCallNode(
            llm=ChatOpenAI(),
            stratix_instance=stratix,
            messages_key="messages",
        )

        graph.add_node("llm", llm_node)
    """

    def __init__(
        self,
        llm: Any,
        stratix_instance: Any = None,
        adapter: BaseAdapter | None = None,
        messages_key: str = "messages",
        response_key: str = "messages",
    ):
        """
        Initialize the LLM call node.

        Args:
            llm: LLM instance
            stratix_instance: STRATIX SDK instance
            adapter: BaseAdapter instance (new-style)
            messages_key: Key in state containing messages
            response_key: Key in state to add response to
        """
        self._traced_llm = TracedLLM(llm, stratix_instance, adapter=adapter)
        self._messages_key = messages_key
        self._response_key = response_key

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the LLM node.

        Args:
            state: LangGraph state

        Returns:
            Updated state with LLM response
        """
        messages = state.get(self._messages_key, [])
        response = self._traced_llm.invoke(messages)

        # Return state update
        return {self._response_key: [response]}

    async def __acall__(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Async execute the LLM node.

        Args:
            state: LangGraph state

        Returns:
            Updated state with LLM response
        """
        messages = state.get(self._messages_key, [])
        response = await self._traced_llm.ainvoke(messages)

        return {self._response_key: [response]}
