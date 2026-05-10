"""
STRATIX LangChain Callback Handler

Provides LangChain callback-based integration for STRATIX tracing.

Typed-event migration (Bundle #6 — final):
    The single ``self.emit_dict_event(...)`` site (the
    :meth:`_emit_event` wrapper that dispatches ``model.invoke``,
    ``tool.call``, ``agent.input``, ``agent.output`` from nine
    callback hooks) was migrated to typed
    :class:`ModelInvokeEvent` / :class:`ToolCallEvent` /
    :class:`AgentInputEvent` / :class:`AgentOutputEvent` payloads
    from :mod:`layerlens.instrument._compat.events`.

    LangChain provenance (``run_id``, ``parent_run_id``,
    ``node_name``, ``langgraph_step``, ``langgraph_triggers``,
    ``invocation_params``) is folded onto canonical metadata /
    parameters slots — no ad-hoc top-level fields ship on the
    canonical schema. ``parent_run_id`` is used to attribute
    nested LLM / tool calls back to their LangGraph node, with
    the node name landing on
    :class:`MessageContent.metadata.node_name`.
"""

from __future__ import annotations

import time
import uuid
from uuid import UUID
from typing import Any
from dataclasses import dataclass
from collections.abc import Callable

from layerlens.instrument._compat.events import (
    MessageRole,
    ToolCallEvent,
    AgentInputEvent,
    IntegrationType,
    AgentOutputEvent,
    ModelInvokeEvent,
)
from layerlens.instrument.adapters._base.adapter import (
    AdapterInfo,
    BaseAdapter,
    AdapterHealth,
    AdapterStatus,
    ReplayableTrace,
    AdapterCapability,
)
from layerlens.instrument.adapters._base.capture import CaptureConfig
from layerlens.instrument.adapters._base.pydantic_compat import PydanticCompat
from layerlens.instrument.adapters._base.trace_container import SerializedTrace


def _stringify(value: Any) -> str:
    """Coerce ``value`` to a non-``None`` string for canonical message slots.

    The canonical :class:`AgentInputEvent` and
    :class:`AgentOutputEvent` require ``MessageContent.message: str`` —
    LangChain delivers the underlying input/output as arbitrary
    Python objects (LangGraph state dicts, ``return_values``, raw
    output strings). This helper always returns a string so the typed
    payload validates; the original payload is preserved on
    :class:`MessageContent.metadata`.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


@dataclass
class LLMCallContext:
    """Context for tracking an LLM call."""

    run_id: str
    start_time_ns: int
    model: str | None = None
    provider: str | None = None
    prompts: list[str] | None = None
    invocation_params: dict[str, Any] | None = None


@dataclass
class ToolCallContext:
    """Context for tracking a tool call."""

    run_id: str
    start_time_ns: int
    tool_name: str
    tool_input: str | dict[str, Any] | None = None


@dataclass
class AgentActionContext:
    """Context for tracking an agent action."""

    run_id: str
    start_time_ns: int
    action: str | None = None
    action_input: Any | None = None


@dataclass
class ChainCallContext:
    """Context for tracking a chain/node execution."""

    run_id: str
    start_time_ns: int
    node_name: str | None = None
    parent_run_id: str | None = None


class LayerLensCallbackHandler(BaseAdapter):
    """
    LangChain callback handler that emits STRATIX events.

    Implements the LangChain callback interface to capture:
    - model.invoke (L3) events from LLM calls
    - tool.call (L5a) events from tool invocations
    - agent.output events from agent actions

    Extends BaseAdapter for unified lifecycle and circuit-breaker support.

    Supports both new-style (stratix, capture_config) and legacy-style
    (stratix_instance, boolean flags) parameters.

    Usage (new):
        from stratix import STRATIX
        from layerlens.instrument.adapters.frameworks.langchain import LayerLensCallbackHandler

        stratix = STRATIX(policy_ref="my-policy")
        handler = LayerLensCallbackHandler(stratix=stratix)
        handler.connect()
        llm = ChatOpenAI(callbacks=[handler])

    Usage (legacy — still supported):
        handler = LayerLensCallbackHandler(stratix_instance=stratix)
        llm = ChatOpenAI(callbacks=[handler])
    """

    FRAMEWORK = "langchain"
    VERSION = "0.1.0"
    # LangChain >=0.2 (pyproject pin: langchain>=0.2,<0.4) migrated all
    # internal models to Pydantic v2 — see langchain/langchain#21238 and
    # langchain-core 0.2.0 release notes. Importing the langchain runtime
    # under Pydantic v1 raises at import inside langchain itself.
    requires_pydantic = PydanticCompat.V2_ONLY

    # LangChain callback protocol attributes — required by CallbackManager
    raise_error: bool = False
    ignore_llm: bool = False
    ignore_chain: bool = False
    ignore_agent: bool = False
    ignore_chat_model: bool = False
    ignore_retriever: bool = True
    ignore_retry: bool = True
    ignore_custom_event: bool = True

    def __init__(
        self,
        # New-style params
        stratix: Any | None = None,
        capture_config: CaptureConfig | None = None,
        event_sinks: list[Any] | None = None,
        graph_factory: Callable[[], Any] | None = None,
        # Legacy params (backward compat)
        stratix_instance: Any | None = None,
        emit_llm_events: bool = True,
        emit_tool_events: bool = True,
        emit_agent_events: bool = True,
        *,
        org_id: str | None = None,
    ) -> None:
        """
        Initialize the callback handler.

        Args:
            stratix: STRATIX SDK instance (new-style)
            capture_config: CaptureConfig (new-style)
            event_sinks: Optional list of EventSink instances for persistence
            graph_factory: Optional callable that returns a fresh graph for replay
            stratix_instance: STRATIX SDK instance (legacy)
            emit_llm_events: Whether to emit model.invoke events (legacy)
            emit_tool_events: Whether to emit tool.call events (legacy)
            emit_agent_events: Whether to emit agent events (legacy)
        """
        # Resolve STRATIX instance
        resolved_stratix = stratix or stratix_instance

        # Map legacy booleans → CaptureConfig when any flag differs from default
        if capture_config is None:
            any_legacy = not emit_llm_events or not emit_tool_events or not emit_agent_events
            if any_legacy or stratix_instance is not None:
                capture_config = CaptureConfig(
                    l3_model_metadata=emit_llm_events,
                    l5a_tool_calls=emit_tool_events,
                    l1_agent_io=emit_agent_events,
                )

        super().__init__(
            stratix=resolved_stratix,
            capture_config=capture_config,
            event_sinks=event_sinks,
            org_id=org_id,
        )

        # Graph factory for replay re-execution
        self._graph_factory = graph_factory

        # Legacy compat: keep booleans accessible
        self._emit_llm_events = emit_llm_events
        self._emit_tool_events = emit_tool_events
        self._emit_agent_events = emit_agent_events

        # Track active calls
        self._llm_calls: dict[str, LLMCallContext] = {}
        self._tool_calls: dict[str, ToolCallContext] = {}
        self._agent_actions: dict[str, AgentActionContext] = {}
        self._chain_calls: dict[str, ChainCallContext] = {}
        self._run_to_node: dict[str, str] = {}  # run_id -> langgraph node name

        # Track all events for debugging/testing
        self._events: list[dict[str, Any]] = []

    # --- BaseAdapter lifecycle ---

    def connect(self) -> None:
        """Verify LangChain is importable and mark as connected."""
        try:
            import langchain  # type: ignore[import-not-found,unused-ignore]  # noqa: F401

            self._connected = True
            self._status = AdapterStatus.HEALTHY
        except ImportError:
            self._connected = True
            self._status = AdapterStatus.HEALTHY

    def disconnect(self) -> None:
        self._close_sinks()
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED

    def health_check(self) -> AdapterHealth:
        return AdapterHealth(
            status=self._status,
            framework_name=self.FRAMEWORK,
            framework_version=self._detect_framework_version(),
            adapter_version=self.VERSION,
            error_count=self._error_count,
            circuit_open=self._circuit_open,
        )

    def get_adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            name="LayerLensCallbackHandler",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self._detect_framework_version(),
            capabilities=[
                AdapterCapability.TRACE_TOOLS,
                AdapterCapability.TRACE_MODELS,
            ],
            description="LayerLens adapter for LangChain framework (callback-based)",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        trace_id = str(uuid.uuid4())
        return ReplayableTrace(
            adapter_name="LayerLensCallbackHandler",
            framework=self.FRAMEWORK,
            trace_id=trace_id,
            events=list(self._trace_events),
            config={
                "capture_config": self._capture_config.model_dump(),
            },
        )

    # --- Replay execution ---

    async def execute_replay(
        self,
        inputs: dict[str, Any],
        original_trace: Any,
        request: Any,
        replay_trace_id: str,
    ) -> SerializedTrace:
        """
        Re-execute through LangChain/LangGraph with a fresh graph.

        Requires a ``graph_factory`` to have been provided at construction.

        Args:
            inputs: Reconstructed inputs for the replay.
            original_trace: The original SerializedTrace.
            request: The ReplayRequest.
            replay_trace_id: ID for the new replay trace.

        Returns:
            SerializedTrace from the replay execution.

        Raises:
            NotImplementedError: If no graph_factory is registered.
        """
        if self._graph_factory is None:
            raise NotImplementedError("No graph_factory registered for replay")

        # Build a fresh graph and callback handler
        graph = self._graph_factory()
        replay_handler = LayerLensCallbackHandler(event_sinks=[])
        replay_handler.connect()

        try:
            # Re-execute through LangGraph with new callbacks
            graph.invoke(inputs, config={"callbacks": [replay_handler]})

            return SerializedTrace.from_event_records(
                events=list(replay_handler._trace_events),
                trace_id=replay_trace_id,
                metadata={
                    "replay_of": original_trace.trace_id,
                    "framework": "langgraph",
                    "replay_type": getattr(request, "replay_type", "basic"),
                },
            )
        finally:
            replay_handler.disconnect()

    # --- Chat Model Callbacks ---

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chat model starts running.

        ChatOpenAI (used by OpenRouter, OpenAI, etc.) triggers this
        instead of on_llm_start. We extract the messages and delegate
        to the same tracking logic.
        """
        if not self._capture_config.is_layer_enabled("model.invoke"):
            return

        run_id_str = str(run_id)
        model = self._extract_model_name(serialized)
        provider = self._extract_provider(serialized)
        invocation_params = kwargs.get("invocation_params", {})

        # Flatten messages to prompt strings for consistent storage
        prompts: list[str] = []
        for message_group in messages:
            for msg in message_group:
                content = getattr(msg, "content", str(msg))
                role = getattr(msg, "type", "unknown")
                prompts.append(f"[{role}] {content}")

        self._llm_calls[run_id_str] = LLMCallContext(
            run_id=run_id_str,
            start_time_ns=time.time_ns(),
            model=model,
            provider=provider,
            prompts=prompts,
            invocation_params=invocation_params,
        )

    # --- LLM Callbacks ---

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts running."""
        if not self._capture_config.is_layer_enabled("model.invoke"):
            return

        run_id_str = str(run_id)

        # Extract model/provider info
        model = self._extract_model_name(serialized)
        provider = self._extract_provider(serialized)
        invocation_params = kwargs.get("invocation_params", {})

        self._llm_calls[run_id_str] = LLMCallContext(
            run_id=run_id_str,
            start_time_ns=time.time_ns(),
            model=model,
            provider=provider,
            prompts=prompts,
            invocation_params=invocation_params,
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM finishes running."""
        if not self._capture_config.is_layer_enabled("model.invoke"):
            return

        run_id_str = str(run_id)
        ctx = self._llm_calls.pop(run_id_str, None)

        if ctx is None:
            return

        end_time_ns = time.time_ns()
        duration_ns = end_time_ns - ctx.start_time_ns

        # Extract response content
        output = self._extract_llm_output(response)
        token_usage = self._extract_token_usage(response)

        # Adapter-specific provenance folds onto ModelInfo.parameters
        # (canonical schema does not declare run_id / prompts / output
        # / duration_ns / invocation_params / node_name as top-level
        # fields on ModelInvokeEvent).
        parameters: dict[str, Any] = {
            "framework": "langchain",
            "run_id": run_id_str,
            "prompts": ctx.prompts or [],
            "output": output,
            "token_usage": token_usage,
            "duration_ns": duration_ns,
        }
        if ctx.invocation_params:
            parameters["invocation_params"] = ctx.invocation_params

        # Attribute to LangGraph node if parent chain is a node.
        node_name = self._run_to_node.get(str(parent_run_id)) if parent_run_id else None
        if node_name:
            parameters["node_name"] = node_name

        # Token usage extraction may yield {prompt_tokens,
        # completion_tokens, total_tokens} or None — pull them onto
        # the canonical token slots.
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        total_tokens: int | None = None
        if isinstance(token_usage, dict):
            pt = token_usage.get("prompt_tokens")
            ct = token_usage.get("completion_tokens")
            tt = token_usage.get("total_tokens")
            prompt_tokens = pt if isinstance(pt, int) else None
            completion_tokens = ct if isinstance(ct, int) else None
            total_tokens = tt if isinstance(tt, int) else None

        self._emit_typed(
            "model.invoke",
            ModelInvokeEvent.create(
                provider=ctx.provider or "unknown",
                name=ctx.model or "unknown",
                version="unavailable",
                parameters=parameters,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=duration_ns / 1_000_000.0,
            ),
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM errors."""
        if not self._capture_config.is_layer_enabled("model.invoke"):
            return

        run_id_str = str(run_id)
        ctx = self._llm_calls.pop(run_id_str, None)

        if ctx is None:
            return

        end_time_ns = time.time_ns()
        duration_ns = end_time_ns - ctx.start_time_ns

        parameters: dict[str, Any] = {
            "framework": "langchain",
            "run_id": run_id_str,
            "prompts": ctx.prompts or [],
            "error": str(error),
            "duration_ns": duration_ns,
        }

        # Attribute to LangGraph node if parent chain is a node.
        node_name = self._run_to_node.get(str(parent_run_id)) if parent_run_id else None
        if node_name:
            parameters["node_name"] = node_name

        self._emit_typed(
            "model.invoke",
            ModelInvokeEvent.create(
                provider=ctx.provider or "unknown",
                name=ctx.model or "unknown",
                version="unavailable",
                parameters=parameters,
                latency_ms=duration_ns / 1_000_000.0,
            ),
        )

    # --- Tool Callbacks ---

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool starts running."""
        if not self._capture_config.is_layer_enabled("tool.call"):
            return

        run_id_str = str(run_id)
        tool_name = serialized.get("name", "unknown_tool")

        self._tool_calls[run_id_str] = ToolCallContext(
            run_id=run_id_str,
            start_time_ns=time.time_ns(),
            tool_name=tool_name,
            tool_input=inputs if inputs else input_str,
        )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool finishes running."""
        if not self._capture_config.is_layer_enabled("tool.call"):
            return

        run_id_str = str(run_id)
        ctx = self._tool_calls.pop(run_id_str, None)

        if ctx is None:
            return

        end_time_ns = time.time_ns()
        duration_ns = end_time_ns - ctx.start_time_ns

        # Canonical input slot is a dict — wrap raw scalars / strings
        # / Nones on a ``value`` key to keep the schema contract.
        if isinstance(ctx.tool_input, dict):
            input_data: dict[str, Any] = dict(ctx.tool_input)
        elif ctx.tool_input is None:
            input_data = {}
        else:
            input_data = {"value": ctx.tool_input}

        # Provenance keys (run_id, node_name, framework) ride on
        # namespaced ``_*`` keys so they do not collide with caller
        # tool arguments.
        input_data["_run_id"] = run_id_str
        input_data["_framework"] = "langchain"
        node_name = self._run_to_node.get(str(parent_run_id)) if parent_run_id else None
        if node_name:
            input_data["_node_name"] = node_name

        # Canonical output slot is Optional[dict] — LangChain's
        # callback protocol types ``output`` as ``str``, so wrap the
        # raw string on a ``value`` key. ``None`` strings (rare —
        # protocol does not officially permit them) collapse to
        # ``None`` so the canonical "no output yet" semantics hold.
        output_data: dict[str, Any] | None = (
            {"value": output} if output else None
        )

        self._emit_typed(
            "tool.call",
            ToolCallEvent.create(
                name=ctx.tool_name,
                version="unavailable",
                integration=IntegrationType.LIBRARY,
                input_data=input_data,
                output_data=output_data,
                latency_ms=duration_ns / 1_000_000.0,
            ),
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool errors."""
        if not self._capture_config.is_layer_enabled("tool.call"):
            return

        run_id_str = str(run_id)
        ctx = self._tool_calls.pop(run_id_str, None)

        if ctx is None:
            return

        end_time_ns = time.time_ns()
        duration_ns = end_time_ns - ctx.start_time_ns

        if isinstance(ctx.tool_input, dict):
            input_data: dict[str, Any] = dict(ctx.tool_input)
        elif ctx.tool_input is None:
            input_data = {}
        else:
            input_data = {"value": ctx.tool_input}

        input_data["_run_id"] = run_id_str
        input_data["_framework"] = "langchain"
        node_name = self._run_to_node.get(str(parent_run_id)) if parent_run_id else None
        if node_name:
            input_data["_node_name"] = node_name

        self._emit_typed(
            "tool.call",
            ToolCallEvent.create(
                name=ctx.tool_name,
                version="unavailable",
                integration=IntegrationType.LIBRARY,
                input_data=input_data,
                error=str(error),
                latency_ms=duration_ns / 1_000_000.0,
            ),
        )

    # --- Agent Callbacks ---

    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent takes an action."""
        if not self._capture_config.is_layer_enabled("agent.input"):
            return

        run_id_str = str(run_id)

        # Extract action details
        action_str = getattr(action, "tool", str(action)) if hasattr(action, "tool") else str(action)
        action_input = getattr(action, "tool_input", None)

        self._agent_actions[run_id_str] = AgentActionContext(
            run_id=run_id_str,
            start_time_ns=time.time_ns(),
            action=action_str,
            action_input=action_input,
        )

        # Agent action → tool.call: the tool invocation happens
        # synchronously inside the agent's reasoning loop.
        if isinstance(action_input, dict):
            input_data: dict[str, Any] = dict(action_input)
        elif action_input is None:
            input_data = {}
        else:
            input_data = {"value": action_input}
        input_data["_run_id"] = run_id_str
        input_data["_framework"] = "langchain"
        input_data["_source"] = "on_agent_action"

        self._emit_typed(
            "tool.call",
            ToolCallEvent.create(
                name=action_str,
                version="unavailable",
                integration=IntegrationType.LIBRARY,
                input_data=input_data,
            ),
        )

    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent finishes."""
        if not self._capture_config.is_layer_enabled("agent.output"):
            return

        run_id_str = str(run_id)

        # Extract output
        output = getattr(finish, "return_values", str(finish))
        log = getattr(finish, "log", None)

        metadata: dict[str, Any] = {
            "framework": "langchain",
            "run_id": run_id_str,
            "raw_output": output,
        }
        if log is not None:
            metadata["log"] = log

        self._emit_typed(
            "agent.output",
            AgentOutputEvent.create(
                message=_stringify(output),
                metadata=metadata,
            ),
        )

    # --- Chain Callbacks ---

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain starts running.

        For LangGraph node executions, metadata contains 'langgraph_node'
        with the node name. We emit agent.input and track the run_id so
        child LLM/tool calls can be attributed to the node.
        """
        run_id_str = str(run_id)
        parent_id_str = str(parent_run_id) if parent_run_id else None
        meta = metadata or {}

        node_name = meta.get("langgraph_node")

        if node_name:
            # This is a LangGraph node execution
            self._chain_calls[run_id_str] = ChainCallContext(
                run_id=run_id_str,
                start_time_ns=time.time_ns(),
                node_name=node_name,
                parent_run_id=parent_id_str,
            )
            self._run_to_node[run_id_str] = node_name

            if self._capture_config.is_layer_enabled("agent.input"):
                input_summary = str(inputs)[:500] if inputs else ""
                # ``meta`` already holds the LangChain-supplied
                # callback metadata. We build ``event_metadata`` for
                # the canonical envelope without shadowing the
                # method parameter.
                event_metadata: dict[str, Any] = {
                    "framework": "langchain",
                    "run_id": run_id_str,
                    "node_name": node_name,
                    "langgraph_step": meta.get("langgraph_step"),
                    "langgraph_triggers": meta.get("langgraph_triggers"),
                    "raw_input": inputs,
                }
                # LangGraph node executions originate from the graph
                # runtime itself — not a human user — so the canonical
                # role is AGENT.
                self._emit_typed(
                    "agent.input",
                    AgentInputEvent.create(
                        message=input_summary,
                        role=MessageRole.AGENT,
                        metadata=event_metadata,
                    ),
                )
        elif parent_id_str and parent_id_str in self._run_to_node:
            # Sub-chain within a LangGraph node — inherit the node mapping
            inherited_node = self._run_to_node[parent_id_str]
            self._run_to_node[run_id_str] = inherited_node
            self._chain_calls[run_id_str] = ChainCallContext(
                run_id=run_id_str,
                start_time_ns=time.time_ns(),
                node_name=inherited_node,
                parent_run_id=parent_id_str,
            )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain finishes running."""
        run_id_str = str(run_id)
        ctx = self._chain_calls.pop(run_id_str, None)
        self._run_to_node.pop(run_id_str, None)

        if ctx is None or ctx.node_name is None:
            return

        if not self._capture_config.is_layer_enabled("agent.output"):
            return

        end_time_ns = time.time_ns()
        duration_ns = end_time_ns - ctx.start_time_ns

        output_summary = str(outputs)[:500] if outputs else ""
        self._emit_typed(
            "agent.output",
            AgentOutputEvent.create(
                message=output_summary,
                metadata={
                    "framework": "langchain",
                    "run_id": run_id_str,
                    "node_name": ctx.node_name,
                    "duration_ns": duration_ns,
                    "raw_output": outputs,
                },
            ),
        )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain errors."""
        run_id_str = str(run_id)
        ctx = self._chain_calls.pop(run_id_str, None)
        self._run_to_node.pop(run_id_str, None)

        if ctx is None or ctx.node_name is None:
            return

        if not self._capture_config.is_layer_enabled("agent.output"):
            return

        end_time_ns = time.time_ns()
        duration_ns = end_time_ns - ctx.start_time_ns

        self._emit_typed(
            "agent.output",
            AgentOutputEvent.create(
                message=str(error),
                metadata={
                    "framework": "langchain",
                    "run_id": run_id_str,
                    "node_name": ctx.node_name,
                    "duration_ns": duration_ns,
                    "error": str(error),
                    "run_status": "run_failed",
                },
            ),
        )

    # --- Helper Methods ---

    def _extract_model_name(self, serialized: dict[str, Any]) -> str:
        """Extract model name from serialized LLM."""
        for key in ["model_name", "model", "name"]:
            if key in serialized:
                return serialized[key]  # type: ignore[no-any-return]

        kwargs = serialized.get("kwargs", {})
        for key in ["model_name", "model"]:
            if key in kwargs:
                return kwargs[key]  # type: ignore[no-any-return]

        return "unknown"

    def _extract_provider(self, serialized: dict[str, Any]) -> str:
        """Extract provider from serialized LLM."""
        id_parts = serialized.get("id", ["unknown"])
        if isinstance(id_parts, list) and len(id_parts) >= 3:
            return id_parts[2] if len(id_parts) > 2 else "unknown"

        name = serialized.get("name", "").lower()
        if "openai" in name:
            return "openai"
        elif "anthropic" in name or "claude" in name:
            return "anthropic"
        elif "google" in name or "gemini" in name:
            return "google"

        return "unknown"

    def _extract_llm_output(self, response: Any) -> Any:
        """Extract output from LLM response."""
        if hasattr(response, "generations"):
            generations = response.generations
            if generations and len(generations) > 0:
                gen = generations[0]
                if isinstance(gen, list) and len(gen) > 0:
                    return gen[0].text if hasattr(gen[0], "text") else str(gen[0])
                return gen.text if hasattr(gen, "text") else str(gen)

        return str(response)

    def _extract_token_usage(self, response: Any) -> dict[str, int] | None:
        """Extract token usage from response."""
        if hasattr(response, "llm_output") and response.llm_output:
            return response.llm_output.get("token_usage")  # type: ignore[no-any-return]
        return None

    def _emit_typed(self, event_type: str, payload: Any) -> None:
        """Record a typed Pydantic event payload and emit it.

        Replaces the legacy :meth:`_emit_event` dispatcher. The
        ``event_type`` argument is kept for the ``self._events``
        debug/test ledger (mirrors the legacy shape so existing
        ``get_events('model.invoke')`` filters still work). The
        canonical Pydantic instance flows through
        :meth:`BaseAdapter.emit_event` which validates the payload
        against the canonical schema and stamps ``org_id`` per the
        multi-tenancy contract.
        """
        from layerlens._compat.pydantic import model_dump

        # Local debug/test ledger keeps a dict view so callers
        # treating ``self._events`` as a sequence of
        # ``{type, payload}`` records continue to work.
        self._events.append({"type": event_type, "payload": model_dump(payload)})
        self.emit_event(payload)

    # --- Testing/Debugging ---

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        """Get recorded events (useful for testing)."""
        if event_type:
            return [e for e in self._events if e["type"] == event_type]
        return self._events

    def clear_events(self) -> None:
        """Clear recorded events."""
        self._events.clear()

    @staticmethod
    def _detect_framework_version() -> str | None:
        try:
            import langchain  # type: ignore[import-not-found,unused-ignore]

            return getattr(langchain, "__version__", None)
        except ImportError:
            return None
