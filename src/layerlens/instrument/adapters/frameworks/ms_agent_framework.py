"""Microsoft Agent Framework adapter (semantic-kernel agents).

Wraps :class:`semantic_kernel.agents.AgentChat` (single-agent) and
``AgentGroupChat`` (multi-agent) — both expose an async-generator
``invoke()`` (and optionally ``invoke_stream()``) that yields
``ChatMessageContent`` objects. We bracket each invocation with
``_begin_run`` / ``_end_run`` and process each yielded message for:

* ``agent.handoff`` — when the message's ``agent_name`` differs from the
  previous one we saw (group-chat turn transitions).
* ``tool.call`` / ``tool.result`` — function-call / function-result
  items in the message's ``items`` list.
* ``model.invoke`` + ``cost.record`` — derived from
  ``message.metadata["model"]`` and ``message.metadata["usage"]``.

A one-time ``environment.config`` event is emitted per chat instance
on first instrument with agents / plugins / strategy metadata.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ._utils import safe_serialize
from ._handoff import HandoffDetector
from ._base_framework import FrameworkAdapter
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

try:
    import semantic_kernel  # pyright: ignore[reportMissingImports]  # noqa: F401

    _HAS_SK_AGENTS = True
except (ImportError, TypeError):
    _HAS_SK_AGENTS = False


class MSAgentFrameworkAdapter(FrameworkAdapter):
    """Layerlens adapter for Microsoft Agent Framework (semantic-kernel agents).

    Usage::

        adapter = MSAgentFrameworkAdapter(client)
        adapter.connect()
        adapter.instrument_chat(my_chat)
        # ... run chat.invoke() ...
        adapter.disconnect()
    """

    name = "ms_agent_framework"
    package = "semantic-kernel"

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        super().__init__(client, capture_config)
        # id(chat) -> dict[method_name -> original callable]
        self._originals: Dict[int, Dict[str, Any]] = {}
        self._wrapped_chats: List[Any] = []
        self._seen_chats: set[int] = set()
        self._handoff_detector = HandoffDetector()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:
        self._check_dependency(_HAS_SK_AGENTS)
        if target is not None:
            self.instrument_chat(target)

    def _on_disconnect(self) -> None:
        for chat in self._wrapped_chats:
            self._unwrap_chat(chat)
        self._wrapped_chats.clear()
        self._originals.clear()
        self._seen_chats.clear()
        self._handoff_detector.reset()

    def _unwrap_chat(self, chat: Any) -> None:
        chat_id = id(chat)
        originals = self._originals.get(chat_id, {})
        for method_name, original in originals.items():
            try:
                setattr(chat, method_name, original)
            except Exception:
                log.debug("layerlens.ms_agent_framework: could not unwrap %s", method_name, exc_info=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def instrument_chat(self, chat: Any) -> Any:
        """Wrap ``chat.invoke`` (and ``invoke_stream`` if present).

        The first time the wrapped ``invoke`` runs we emit a one-shot
        ``environment.config`` event with the chat metadata. We can't
        emit it here because there's no active collector outside a run.
        """
        chat_id = id(chat)
        if chat_id in self._originals:
            return chat

        originals: Dict[str, Any] = {}
        if hasattr(chat, "invoke"):
            originals["invoke"] = chat.invoke
            chat.invoke = self._traced_invoke(chat, chat.invoke)
        if hasattr(chat, "invoke_stream"):
            originals["invoke_stream"] = chat.invoke_stream
            chat.invoke_stream = self._traced_invoke(chat, chat.invoke_stream)

        self._originals[chat_id] = originals
        self._wrapped_chats.append(chat)
        return chat

    # ------------------------------------------------------------------
    # Wrapping
    # ------------------------------------------------------------------

    def _traced_invoke(self, chat: Any, original: Any) -> Any:
        """Build an async-generator wrapper around ``chat.invoke``."""
        adapter = self

        async def wrapper(*args: Any, **kwargs: Any):
            chat_name = getattr(chat, "name", None) or type(chat).__name__
            # AgentChat lets the agent be passed via `agent=` kwarg or as the
            # first positional; AgentGroupChat doesn't — the active agent is
            # whichever one yields next. Fall back to `chat.agent.name` so
            # single-agent chats trace under the agent's name instead of the
            # chat's name.
            agent = kwargs.get("agent") or (args[0] if args else None) or getattr(chat, "agent", None)
            agent_name = (getattr(agent, "name", None) if agent else None) or chat_name
            input_data = kwargs.get("input") or kwargs.get("message")

            adapter._begin_run()
            adapter._handoff_detector.reset()
            adapter._handoff_detector.set_current_agent(agent_name)
            adapter._start_timer("run")

            # One-shot environment.config per chat instance (now that we
            # have an active collector inside _begin_run).
            adapter._maybe_emit_chat_config(chat)

            input_payload = adapter._payload(agent_name=agent_name, chat_name=chat_name)
            adapter._set_if_capturing(input_payload, "input", safe_serialize(input_data))
            adapter._emit("agent.input", input_payload)

            error: Optional[BaseException] = None
            last_message: Any = None
            try:
                async for message in original(*args, **kwargs):
                    last_message = message
                    adapter._process_message(message, agent_name)
                    yield message
            except BaseException as exc:
                error = exc
                raise
            finally:
                latency_ms = adapter._stop_timer("run")
                output_payload = adapter._payload(agent_name=agent_name, chat_name=chat_name)
                if latency_ms is not None:
                    output_payload["latency_ms"] = latency_ms
                if error is not None:
                    output_payload["error"] = str(error)
                    adapter._set_if_capturing(output_payload, "output", safe_serialize(last_message))
                    adapter._emit("agent.error", output_payload)
                else:
                    adapter._set_if_capturing(output_payload, "output", safe_serialize(last_message))
                    adapter._emit("agent.output", output_payload)
                adapter._end_run()

        wrapper._layerlens_original = original  # type: ignore[attr-defined]
        return wrapper

    # ------------------------------------------------------------------
    # Message processing — tool calls, model invocations, handoffs
    # ------------------------------------------------------------------

    def _process_message(self, message: Any, current_agent: str) -> None:
        """Extract handoff / tool / model events from one chat message."""
        try:
            msg_agent = getattr(message, "agent_name", None) or getattr(message, "name", None)
            if msg_agent and msg_agent != current_agent:
                # Group-chat turn transition.
                self._handoff_detector.detect(
                    msg_agent,
                    context={"prev_agent": current_agent, "message": safe_serialize(message)},
                    reason="group_chat_turn",
                )

            for item in getattr(message, "items", None) or []:
                self._process_message_item(item)

            metadata = getattr(message, "metadata", None)
            if isinstance(metadata, dict):
                self._emit_model_metadata(metadata)
        except Exception:
            log.debug("layerlens.ms_agent_framework: error processing message", exc_info=True)

    def _process_message_item(self, item: Any) -> None:
        item_type = type(item).__name__
        tool_name = getattr(item, "name", None) or getattr(item, "function_name", None) or "unknown"
        if "FunctionCall" in item_type or "ToolCall" in item_type:
            payload = self._payload(tool_name=tool_name)
            self._set_if_capturing(payload, "input", safe_serialize(getattr(item, "arguments", None)))
            self._emit("tool.call", payload)
        elif "FunctionResult" in item_type or "ToolResult" in item_type:
            payload = self._payload(tool_name=tool_name)
            self._set_if_capturing(payload, "output", safe_serialize(getattr(item, "result", None)))
            self._emit("tool.result", payload)

    def _emit_model_metadata(self, metadata: Dict[str, Any]) -> None:
        model = metadata.get("model") or metadata.get("model_id")
        if model:
            self._emit(
                "model.invoke",
                self._payload(model=str(model), provider=_detect_provider(str(model))),
            )
        usage = metadata.get("usage")
        if usage is not None:
            tokens = self._normalize_tokens(usage)
            if tokens:
                payload = self._payload(model=str(model) if model else None)
                payload.update(tokens)
                self._emit("cost.record", payload)

    # ------------------------------------------------------------------
    # First-encounter chat config
    # ------------------------------------------------------------------

    def _maybe_emit_chat_config(self, chat: Any) -> None:
        cid = id(chat)
        if cid in self._seen_chats:
            return
        self._seen_chats.add(cid)

        chat_name = getattr(chat, "name", None) or type(chat).__name__
        payload = self._payload(chat_name=chat_name, chat_type=type(chat).__name__)

        agents = getattr(chat, "agents", None)
        if agents:
            payload["agents"] = [getattr(a, "name", str(a)) for a in agents]

        agent = getattr(chat, "agent", None)
        if agent is not None:
            payload["agent_name"] = getattr(agent, "name", str(agent))
            instructions = getattr(agent, "instructions", None)
            if instructions and self._config.capture_content:
                payload["instructions"] = str(instructions)[:500]
            kernel = getattr(agent, "kernel", None)
            plugins = getattr(kernel, "plugins", None) if kernel is not None else None
            if plugins:
                if isinstance(plugins, dict):
                    payload["plugins"] = list(plugins.keys())
                else:
                    payload["plugins"] = [str(p) for p in plugins]

        sel = getattr(chat, "selection_strategy", None)
        if sel is not None:
            payload["selection_strategy"] = type(sel).__name__
        term = getattr(chat, "termination_strategy", None)
        if term is not None:
            payload["termination_strategy"] = type(term).__name__

        self._emit("environment.config", payload)


_PROVIDER_PATTERNS = (
    (("gpt", "o1", "o3", "o4"), "openai"),
    (("claude",), "anthropic"),
    (("gemini",), "google"),
    (("mistral", "mixtral"), "mistral"),
    (("phi",), "microsoft"),
    (("llama",), "meta"),
)


def _detect_provider(model: Optional[str]) -> Optional[str]:
    if not model:
        return None
    low = model.lower()
    for tokens, provider in _PROVIDER_PATTERNS:
        if any(t in low for t in tokens):
            return provider
    # MS Agent Framework most commonly fronts Azure OpenAI.
    return "azure_openai"
