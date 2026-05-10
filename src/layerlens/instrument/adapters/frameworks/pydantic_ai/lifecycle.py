"""
PydanticAI adapter lifecycle.

Instrumentation strategy: OTel wrapper (Logfire-compatible) + Agent wrapper
  Agent.run() start    → agent.input (L1)
  Agent.run() end      → agent.output (L1)
  ModelRequestNode     → model.invoke (L3)
  CallToolsNode        → tool.call (L5a)
  AgentRun transitions → agent.state.change (Cross)

Typed-event status (post PR #129 migration, bundle 5):

* Every emission flows through :meth:`BaseAdapter.emit_event` with a
  canonical Pydantic payload imported from
  :mod:`layerlens.instrument._compat.events`.
* PydanticAI-specific provenance (``framework``, ``agent_name``,
  ``timestamp_ns``, ``duration_ns``, ``tools``, ``result_type``,
  ``system_prompt``, ``model``) is carried in the canonical model's
  metadata / attributes / parameters / input slots.
* The ad-hoc ``agent.state.change`` ``run_complete`` / ``run_failed``
  marker emitted by :meth:`on_run_end` does NOT satisfy the canonical
  :class:`AgentStateChangeEvent` ``before_hash`` / ``after_hash``
  contract (the run boundary has no real state mutation to hash).
  Following the PR #151 ms_agent_framework precedent, the marker is
  carried as ``run_status`` on :class:`AgentOutputEvent.metadata`,
  preserving the cross-cutting completion signal without violating
  the canonical schema.
* PydanticAI tools execute in-process (via the ``@agent.tool``
  decorator), so the L5a integration is :class:`IntegrationType.LIBRARY`.
* The handoff context hash is generated via SHA-256 over the context
  string (or the empty string when no context is available) so the
  canonical :class:`AgentHandoffEvent.handoff_context_hash` validator
  always passes.
"""

from __future__ import annotations

import time
import uuid
import hashlib
import logging
import threading
from typing import Any

from layerlens.instrument._compat.events import (
    MessageRole,
    ToolCallEvent,
    AgentInputEvent,
    CostRecordEvent,
    EnvironmentType,
    IntegrationType,
    AgentOutputEvent,
    ModelInvokeEvent,
    AgentHandoffEvent,
    EnvironmentConfigEvent,
)
from layerlens.instrument.adapters._base.adapter import (
    AdapterInfo,
    BaseAdapter,
    AdapterHealth,
    AdapterStatus,
    ReplayableTrace,
    AdapterCapability,
)
from layerlens.instrument.adapters._base.pydantic_compat import PydanticCompat

logger = logging.getLogger(__name__)


def _stringify(value: Any) -> str:
    """Return a string view of ``value`` suitable for the canonical
    :class:`MessageContent.message` field.

    The canonical schema requires :class:`AgentInputEvent` and
    :class:`AgentOutputEvent` to carry a ``message: str``. PydanticAI
    delivers user prompts and run results as arbitrary Python objects
    (strings, Pydantic ``BaseModel`` instances when ``result_type`` is
    set, ``None``); this helper converts each to a (possibly empty)
    string so the typed event always validates. The original payload
    is preserved on :class:`MessageContent.metadata.raw_input` /
    ``raw_output``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_to_dict(value: Any) -> dict[str, Any]:
    """Coerce ``value`` into a dict suitable for the canonical
    :class:`ToolCallEvent` ``input`` / ``output`` slots.

    The canonical schema requires ``input: dict[str, Any]`` and
    ``output: dict[str, Any] | None``. PydanticAI tool returns can be
    arbitrary Python values (Pydantic models, scalars, dicts). This
    helper wraps non-dict values in ``{"value": ...}`` so the
    canonical slot is always satisfied.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return {"value": value}


def _sha256_of(value: str) -> str:
    """Return a canonical ``sha256:<hex64>`` hash string for ``value``.

    The canonical schema's :class:`AgentHandoffEvent` requires
    ``handoff_context_hash`` to start with ``sha256:`` and have a
    64-character hex tail. Centralising the format here ensures every
    emit site uses the same wire format — including the empty-string
    fallback used when PydanticAI has no handoff context to hash.
    """
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


class PydanticAIAdapter(BaseAdapter):
    """LayerLens adapter for PydanticAI."""

    FRAMEWORK = "pydantic_ai"
    VERSION = "0.1.0"
    # Pydantic-AI is built on Pydantic v2 from day one — see
    # pydantic-ai's own pyproject which requires ``pydantic>=2.7``.
    # There is no v1 path; the framework cannot be installed alongside v1.
    requires_pydantic = PydanticCompat.V2_ONLY

    # Per-adapter ``extra="allow"`` decision: pydantic_ai targets the
    # canonical 13-event taxonomy exclusively. Unknown event types must
    # be rejected by the base adapter's typed-event validator, so this
    # stays ``False``. See ``docs/adapters/typed-events.md`` for the
    # opt-in policy.
    ALLOW_UNREGISTERED_EVENTS: bool = False

    def __init__(
        self,
        stratix: Any | None = None,
        capture_config: Any | None = None,
        stratix_instance: Any | None = None,
        *,
        org_id: str | None = None,
    ) -> None:
        resolved = stratix or stratix_instance
        super().__init__(stratix=resolved, capture_config=capture_config, org_id=org_id)
        self._originals: dict[int, dict[str, Any]] = {}  # id(agent) -> {method: original}
        self._wrapped_agents: list[Any] = []  # strong refs for disconnect unwrap
        self._adapter_lock = threading.Lock()
        self._seen_agents: set[str] = set()
        self._framework_version: str | None = None
        self._run_starts: dict[int, int] = {}  # thread_id -> start_ns

    def connect(self) -> None:
        try:
            import pydantic_ai  # type: ignore[import-not-found,unused-ignore]

            self._framework_version = getattr(pydantic_ai, "__version__", "unknown")
        except ImportError:
            logger.debug("pydantic-ai not installed")
        self._connected = True
        self._status = AdapterStatus.HEALTHY

    def disconnect(self) -> None:
        for agent in self._wrapped_agents:
            self._unwrap_agent(agent)
        self._wrapped_agents.clear()
        self._originals.clear()
        self._seen_agents.clear()
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED

    def _unwrap_agent(self, agent: Any) -> None:
        """Restore original methods on a wrapped agent."""
        agent_id = id(agent)
        originals = self._originals.get(agent_id)
        if not originals:
            return
        for method_name, original in originals.items():
            try:
                setattr(agent, method_name, original)
            except Exception:
                logger.debug("Could not unwrap %s.%s", agent_id, method_name, exc_info=True)

    def health_check(self) -> AdapterHealth:
        return AdapterHealth(
            status=self._status,
            framework_name=self.FRAMEWORK,
            framework_version=self._framework_version,
            adapter_version=self.VERSION,
            error_count=self._error_count,
            circuit_open=self._circuit_open,
        )

    def get_adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            name="PydanticAIAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self._framework_version,
            capabilities=[
                AdapterCapability.TRACE_TOOLS,
                AdapterCapability.TRACE_MODELS,
                AdapterCapability.TRACE_STATE,
            ],
            description="LayerLens adapter for PydanticAI",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        return ReplayableTrace(
            adapter_name="PydanticAIAdapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[],
            config={"capture_config": self._capture_config.model_dump()},
        )

    # --- Framework Integration ---

    def instrument_agent(self, agent: Any) -> Any:
        """Wrap PydanticAI agent.run() methods to capture lifecycle events."""
        agent_id = id(agent)
        if agent_id in self._originals:
            return agent
        originals: dict[str, Any] = {}
        # Wrap run()
        if hasattr(agent, "run"):
            originals["run"] = agent.run
            agent.run = self._create_traced_run(agent, agent.run)
        # Wrap run_sync()
        if hasattr(agent, "run_sync"):
            originals["run_sync"] = agent.run_sync
            agent.run_sync = self._create_traced_run_sync(agent, agent.run_sync)
        self._originals[agent_id] = originals
        self._wrapped_agents.append(agent)
        agent_name = getattr(agent, "name", None) or str(type(agent).__name__)
        self._emit_agent_config(agent_name, agent)
        return agent

    def _create_traced_run(self, agent: Any, original_run: Any) -> Any:
        adapter = self

        async def traced_run(*args: Any, **kwargs: Any) -> Any:
            agent_name = getattr(agent, "name", None) or "pydantic_ai_agent"
            user_prompt = args[0] if args else kwargs.get("user_prompt")
            adapter.on_run_start(agent_name=agent_name, input_data=user_prompt)
            error: Exception | None = None
            result = None
            try:
                result = await original_run(*args, **kwargs)
            except Exception as exc:
                error = exc
                raise
            finally:
                output = None
                if result is not None:
                    output = getattr(result, "data", result)
                adapter.on_run_end(agent_name=agent_name, output=output, error=error)
                adapter._extract_run_usage(result)
            return result

        traced_run._layerlens_original = original_run  # type: ignore[attr-defined]
        return traced_run

    def _create_traced_run_sync(self, agent: Any, original_run_sync: Any) -> Any:
        adapter = self

        def traced_run_sync(*args: Any, **kwargs: Any) -> Any:
            agent_name = getattr(agent, "name", None) or "pydantic_ai_agent"
            user_prompt = args[0] if args else kwargs.get("user_prompt")
            adapter.on_run_start(agent_name=agent_name, input_data=user_prompt)
            error: Exception | None = None
            result = None
            try:
                result = original_run_sync(*args, **kwargs)
            except Exception as exc:
                error = exc
                raise
            finally:
                output = None
                if result is not None:
                    output = getattr(result, "data", result)
                adapter.on_run_end(agent_name=agent_name, output=output, error=error)
                adapter._extract_run_usage(result)
            return result

        traced_run_sync._layerlens_original = original_run_sync  # type: ignore[attr-defined]
        return traced_run_sync

    def _extract_run_usage(self, result: Any) -> None:
        """Extract usage info from PydanticAI RunResult.

        Emits typed :class:`CostRecordEvent`, :class:`ModelInvokeEvent`,
        and :class:`ToolCallEvent` payloads depending on what the
        ``RunResult`` exposes. PydanticAI tool returns are in-process
        Python callables (``@agent.tool``), so the integration is
        :class:`IntegrationType.LIBRARY`.
        """
        if result is None:
            return
        try:
            usage = getattr(result, "usage", None) or getattr(result, "_usage", None)
            if usage:
                prompt_tokens = getattr(usage, "request_tokens", None)
                completion_tokens = getattr(usage, "response_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)
                self.emit_event(
                    CostRecordEvent.create(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        tokens=total_tokens,
                    )
                )
            # Extract model invocation details
            all_messages = getattr(result, "all_messages", None) or []
            for msg in all_messages:
                msg_kind = getattr(msg, "kind", None)
                if msg_kind == "response":
                    model_raw = getattr(result, "model_name", None)
                    model_name = str(model_raw) if model_raw else "unknown"
                    provider = self._detect_provider(model_name) or "unknown"
                    self.emit_event(
                        ModelInvokeEvent.create(
                            provider=provider,
                            name=model_name,
                            version="unavailable",
                            parameters={"framework": "pydantic_ai"},
                        )
                    )
                elif msg_kind == "tool-return":
                    tool_name_raw = getattr(msg, "tool_name", None)
                    tool_name = str(tool_name_raw) if tool_name_raw else "unknown"
                    serialised_output = self._safe_serialize(
                        getattr(msg, "content", None)
                    )
                    output_data: dict[str, Any] | None = (
                        _coerce_to_dict(serialised_output)
                        if serialised_output is not None
                        else None
                    )
                    self.emit_event(
                        ToolCallEvent.create(
                            name=tool_name,
                            version="unavailable",
                            integration=IntegrationType.LIBRARY,
                            input_data={"framework": "pydantic_ai"},
                            output_data=output_data,
                        )
                    )
        except Exception:
            logger.debug("Could not extract run usage", exc_info=True)

    # --- Lifecycle Hooks ---

    def on_run_start(self, agent_name: str | None = None, input_data: Any = None) -> None:
        """Emit a typed :class:`AgentInputEvent` when an agent run starts."""
        if not self._connected:
            return
        try:
            tid = threading.get_ident()
            start_ns = time.time_ns()
            with self._adapter_lock:
                self._run_starts[tid] = start_ns
            raw_input = self._safe_serialize(input_data)
            self.emit_event(
                AgentInputEvent.create(
                    message=_stringify(raw_input),
                    role=MessageRole.HUMAN,
                    metadata={
                        "framework": "pydantic_ai",
                        "agent_name": agent_name,
                        "timestamp_ns": start_ns,
                        "raw_input": raw_input,
                    },
                )
            )
        except Exception:
            logger.warning("Error in on_run_start", exc_info=True)

    def on_run_end(
        self,
        agent_name: str | None = None,
        output: Any = None,
        error: Exception | None = None,
    ) -> None:
        """Emit a typed :class:`AgentOutputEvent` when an agent run ends.

        The previous adapter implementation also emitted a separate
        ``agent.state.change`` payload carrying only an
        ``event_subtype`` marker (``run_complete`` / ``run_failed``).
        That payload did not satisfy the canonical
        :class:`AgentStateChangeEvent` ``before_hash`` / ``after_hash``
        contract — the run boundary has no real state mutation to
        hash. The post-migration mapping (matching the PR #151
        ms_agent_framework precedent) carries the same signal as
        ``run_status`` on :class:`MessageContent.metadata`, preserving
        the cross-cutting completion marker without violating the
        canonical schema.
        """
        if not self._connected:
            return
        try:
            tid = threading.get_ident()
            end_ns = time.time_ns()
            with self._adapter_lock:
                start_ns = self._run_starts.pop(tid, 0)
            duration_ns = end_ns - start_ns if start_ns else 0
            raw_output = self._safe_serialize(output)
            metadata: dict[str, Any] = {
                "framework": "pydantic_ai",
                "agent_name": agent_name,
                "duration_ns": duration_ns,
                "raw_output": raw_output,
                "run_status": "run_complete" if not error else "run_failed",
            }
            if error:
                metadata["error"] = str(error)
            self.emit_event(
                AgentOutputEvent.create(
                    message=_stringify(raw_output),
                    metadata=metadata,
                )
            )
        except Exception:
            logger.warning("Error in on_run_end", exc_info=True)

    def on_tool_use(
        self,
        tool_name: str,
        tool_input: Any = None,
        tool_output: Any = None,
        error: Exception | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Emit a typed :class:`ToolCallEvent` for a tool invocation.

        PydanticAI tools are in-process Python callables decorated
        with ``@agent.tool`` — this maps to
        :class:`IntegrationType.LIBRARY`. Tool versions are not
        surfaced by the framework, so ``version`` falls back to
        ``"unavailable"`` per the canonical NORMATIVE rule.
        """
        if not self._connected:
            return
        try:
            serialised_input = self._safe_serialize(tool_input)
            serialised_output = self._safe_serialize(tool_output)
            input_data = _coerce_to_dict(serialised_input)
            input_data.setdefault("framework", "pydantic_ai")
            output_data: dict[str, Any] | None = (
                _coerce_to_dict(serialised_output) if serialised_output is not None else None
            )
            self.emit_event(
                ToolCallEvent.create(
                    name=tool_name,
                    version="unavailable",
                    integration=IntegrationType.LIBRARY,
                    input_data=input_data,
                    output_data=output_data,
                    error=str(error) if error else None,
                    latency_ms=latency_ms,
                )
            )
        except Exception:
            logger.warning("Error in on_tool_use", exc_info=True)

    def on_llm_call(
        self,
        provider: str | None = None,
        model: str | None = None,
        tokens_prompt: int | None = None,
        tokens_completion: int | None = None,
        latency_ms: float | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> None:
        """Emit a typed :class:`ModelInvokeEvent` for an LLM call."""
        if not self._connected:
            return
        try:
            model_name = model or "unknown"
            resolved_provider = provider or self._detect_provider(model_name) or "unknown"
            input_messages: list[dict[str, str]] | None = None
            if self._capture_config.capture_content and messages:
                input_messages = messages
            self.emit_event(
                ModelInvokeEvent.create(
                    provider=resolved_provider,
                    name=model_name,
                    version="unavailable",
                    parameters={"framework": "pydantic_ai"},
                    prompt_tokens=tokens_prompt,
                    completion_tokens=tokens_completion,
                    latency_ms=latency_ms,
                    input_messages=input_messages,
                )
            )
        except Exception:
            logger.warning("Error in on_llm_call", exc_info=True)

    def on_handoff(self, from_agent: str, to_agent: str, context: Any = None) -> None:
        """Emit a typed :class:`AgentHandoffEvent` for an agent handoff."""
        if not self._connected:
            return
        try:
            context_str = str(context) if context else ""
            self.emit_event(
                AgentHandoffEvent.create(
                    from_agent=from_agent,
                    to_agent=to_agent,
                    handoff_context_hash=_sha256_of(context_str),
                )
            )
        except Exception:
            logger.warning("Error in on_handoff", exc_info=True)

    # --- Helpers ---

    def _detect_provider(self, model: str | None) -> str | None:
        if not model:
            return None
        model_lower = model.lower()
        if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
            return "openai"
        if "claude" in model_lower:
            return "anthropic"
        if "gemini" in model_lower:
            return "google"
        if "mistral" in model_lower or "mixtral" in model_lower:
            return "mistral"
        return None

    def _emit_agent_config(self, agent_name: str, agent: Any) -> None:
        """Emit a typed :class:`EnvironmentConfigEvent` per agent.

        Idempotent per agent — only the first call for a given agent
        name actually emits. PydanticAI agents run in a ``simulated``
        environment by default (the host application is responsible
        for emitting the real ``cloud`` / ``on_prem`` environment
        record). PydanticAI-specific provenance (``framework``,
        ``agent_name``, ``model``, ``tools``, ``system_prompt``,
        ``result_type``) lives on
        :attr:`EnvironmentInfo.attributes`.
        """
        with self._adapter_lock:
            if agent_name in self._seen_agents:
                return
            self._seen_agents.add(agent_name)
        attributes: dict[str, Any] = {
            "framework": "pydantic_ai",
            "agent_name": agent_name,
        }
        model = getattr(agent, "model", None)
        if model:
            attributes["model"] = str(model)
        system_prompt = getattr(agent, "system_prompt", None)
        if system_prompt and self._capture_config.capture_content:
            attributes["system_prompt"] = str(system_prompt)[:500]
        tools = getattr(agent, "_function_tools", None) or getattr(agent, "tools", None)
        if tools:
            if isinstance(tools, dict):
                attributes["tools"] = list(tools.keys())
            else:
                attributes["tools"] = [getattr(t, "name", str(t)) for t in tools]
        result_type = getattr(agent, "result_type", None)
        if result_type:
            attributes["result_type"] = str(result_type)
        self.emit_event(
            EnvironmentConfigEvent.create(
                env_type=EnvironmentType.SIMULATED,
                attributes=attributes,
            )
        )

    def _safe_serialize(self, value: Any) -> Any:
        try:
            if value is None:
                return None
            if hasattr(value, "model_dump"):
                return value.model_dump()
            if hasattr(value, "dict"):
                return value.dict()
            if isinstance(value, dict):
                return dict(value)
            if isinstance(value, (str, int, float, bool)):
                return value
            return str(value)
        except Exception:
            return str(value)
