"""SmolAgents adapter lifecycle.

Instrumentation strategy: agent wrapper + lifecycle hooks (no native callbacks).

* ``Agent.run()`` start → :class:`AgentInputEvent`
* ``Agent.run()`` end   → :class:`AgentOutputEvent`
* Model call            → :class:`ModelInvokeEvent`
* Tool execution        → :class:`ToolCallEvent`
* Code execution        → :class:`ToolCallEvent` (``name="code_execution"``,
  ``integration=SCRIPT``) — the previous adapter emitted an ad-hoc
  ``agent.code`` event type that is NOT in the canonical 13-event
  taxonomy
* Manager → managed     → :class:`AgentHandoffEvent`

Typed-event status (post PR #129 migration, bundle 1):

* Every emission flows through :meth:`BaseAdapter.emit_event` with a
  canonical Pydantic payload imported from
  :mod:`layerlens.instrument._compat.events`.
* SmolAgents-specific provenance (``framework``, ``agent_name``,
  ``timestamp_ns``, ``agent_type``, ``raw_input``, ``raw_output``,
  ``logs``) is carried in the canonical model's metadata / attributes
  / parameters / input slots — the canonical schema does not expose
  these as top-level fields.
* The previous handoff payload carried ``context_hash`` as a bare
  hex string and ``context_preview`` as a top-level field; the
  canonical :class:`AgentHandoffEvent.handoff_context_hash` validator
  requires the ``sha256:<hex64>`` prefix. Helper
  :func:`_sha256_of` produces the conformant format.

Ported from ``ateam/stratix/sdk/python/adapters/smolagents/lifecycle.py``.
"""

from __future__ import annotations

import time
import uuid
import hashlib
import logging
import threading
from typing import Any, Set, Dict, List, Optional

from layerlens.instrument._compat.events import (
    MessageRole,
    ToolCallEvent,
    AgentInputEvent,
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
    :class:`AgentOutputEvent` to carry a ``message: str``. SmolAgents
    callbacks deliver inputs/outputs as arbitrary Python objects (the
    task string at run-start, ``RunResult`` at run-end, ``None`` on
    error); this helper converts each to a (possibly empty) string so
    the typed event always validates. The original payload is
    preserved on :class:`MessageContent.metadata.raw_input` /
    ``raw_output``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # SmolAgents RunResult.model_dump() commonly produces
        # ``{"output": ..., "logs": [...], ...}``; surface the output
        # slot when present.
        out = value.get("output") or value.get("content") or value.get("message")
        if isinstance(out, str):
            return out
    return str(value)


def _sha256_of(value: str) -> str:
    """Return a canonical ``sha256:<hex64>`` hash string for ``value``.

    The canonical schema's :class:`AgentHandoffEvent` requires
    ``handoff_context_hash`` to start with ``sha256:`` and have a
    64-character hex tail (see
    ``ateam/stratix/core/events/cross_cutting.py``). Centralising
    the format here ensures every emit site uses the same wire
    format — including the empty-string fallback used when the
    SmolAgents manager has no explicit context.
    """
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


class SmolAgentsAdapter(BaseAdapter):
    """LayerLens adapter for SmolAgents (HuggingFace)."""

    FRAMEWORK = "smolagents"
    VERSION = "0.1.0"
    # The only Pydantic touch in the adapter is
    # ``from layerlens._compat.pydantic import model_dump`` (the
    # v1/v2 shim itself). SmolAgents 1.x uses Pydantic internally
    # but the adapter only wraps ``Agent.run()`` and emits typed
    # events via :mod:`layerlens.instrument._compat.events`, never
    # touching framework Pydantic models directly.
    requires_pydantic = PydanticCompat.V1_OR_V2

    # Per-adapter ``extra="allow"`` decision: SmolAgents targets the
    # canonical 13-event taxonomy exclusively. Unknown event types
    # must be rejected by the base adapter's typed-event validator,
    # so this stays ``False``. The previous ``agent.code`` event
    # emitted on CodeAgent execution is migrated to
    # :class:`ToolCallEvent` (``name="code_execution"``,
    # ``integration=SCRIPT``) — see :meth:`_emit_code_execution`.
    # See ``docs/adapters/typed-events.md`` for the opt-in policy.
    ALLOW_UNREGISTERED_EVENTS: bool = False

    def __init__(
        self,
        stratix: Any = None,
        capture_config: Any = None,
        stratix_instance: Any = None,
        *,
        org_id: Optional[str] = None,
    ) -> None:
        resolved = stratix or stratix_instance
        super().__init__(stratix=resolved, capture_config=capture_config, org_id=org_id)
        self._originals: Dict[int, Dict[str, Any]] = {}
        self._adapter_lock = threading.Lock()
        self._seen_agents: Set[str] = set()
        self._framework_version: Optional[str] = None
        self._run_starts: Dict[int, int] = {}
        self._wrapped_agents: List[Any] = []

    def connect(self) -> None:
        try:
            import smolagents  # type: ignore[import-not-found,unused-ignore]

            version = getattr(smolagents, "__version__", "unknown")
            self._framework_version = str(version) if version is not None else "unknown"
        except ImportError:
            logger.debug("smolagents not installed")
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
            name="SmolAgentsAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self._framework_version,
            capabilities=[
                AdapterCapability.TRACE_TOOLS,
                AdapterCapability.TRACE_MODELS,
                AdapterCapability.TRACE_STATE,
                AdapterCapability.TRACE_HANDOFFS,
            ],
            description="LayerLens adapter for SmolAgents (HuggingFace)",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        from layerlens._compat.pydantic import model_dump

        return ReplayableTrace(
            adapter_name="SmolAgentsAdapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[],
            config={"capture_config": model_dump(self._capture_config)},
        )

    # --- Framework integration ---

    def instrument_agent(self, agent: Any) -> Any:
        """Wrap a SmolAgents agent's ``run()`` method."""
        agent_id = id(agent)
        if agent_id in self._originals:
            return agent
        originals: Dict[str, Any] = {}
        if hasattr(agent, "run"):
            originals["run"] = agent.run
            agent.run = self._create_traced_run(agent, agent.run)
        self._originals[agent_id] = originals
        self._wrapped_agents.append(agent)
        agent_name = self._get_agent_name(agent)
        agent_type = type(agent).__name__
        self._emit_agent_config(agent_name, agent, agent_type)
        managed = getattr(agent, "managed_agents", None)
        if managed:
            if isinstance(managed, dict):
                for _name, managed_agent in managed.items():
                    self.instrument_agent(managed_agent)
            elif isinstance(managed, list):
                for managed_agent in managed:
                    self.instrument_agent(managed_agent)
        return agent

    def _create_traced_run(self, agent: Any, original_run: Any) -> Any:
        adapter = self

        def traced_run(*args: Any, **kwargs: Any) -> Any:
            agent_name = adapter._get_agent_name(agent)
            task = args[0] if args else kwargs.get("task")
            adapter.on_run_start(agent_name=agent_name, input_data=task)
            error: Optional[Exception] = None
            result: Any = None
            try:
                result = original_run(*args, **kwargs)
            except Exception as exc:
                error = exc
                raise
            finally:
                adapter.on_run_end(agent_name=agent_name, output=result, error=error)
                agent_type = type(agent).__name__
                if agent_type == "CodeAgent" and result is not None:
                    adapter._emit_code_execution(agent_name, result)
            return result

        traced_run._layerlens_original = original_run  # type: ignore[attr-defined]
        return traced_run

    def _unwrap_agent(self, agent: Any) -> None:
        agent_id = id(agent)
        originals = self._originals.get(agent_id)
        if not originals:
            return
        for method_name, original in originals.items():
            try:
                setattr(agent, method_name, original)
            except Exception:
                logger.debug("Could not unwrap %s", method_name, exc_info=True)

    # --- Lifecycle hooks ---

    def on_run_start(
        self,
        agent_name: Optional[str] = None,
        input_data: Any = None,
    ) -> None:
        """Emit a typed :class:`AgentInputEvent` when an agent run starts.

        SmolAgents-specific provenance (``framework``, ``agent_name``,
        ``timestamp_ns``, ``raw_input``) lives on
        :class:`MessageContent.metadata`.
        """
        if not self._connected:
            return
        try:
            tid = threading.get_ident()
            start_ns = time.time_ns()
            with self._adapter_lock:
                self._run_starts[tid] = start_ns
            serialised_input = self._safe_serialize(input_data)
            self.emit_event(
                AgentInputEvent.create(
                    message=_stringify(serialised_input),
                    role=MessageRole.HUMAN,
                    metadata={
                        "framework": "smolagents",
                        "agent_name": agent_name,
                        "timestamp_ns": start_ns,
                        "raw_input": serialised_input,
                    },
                )
            )
        except Exception:
            logger.warning("Error in on_run_start", exc_info=True)

    def on_run_end(
        self,
        agent_name: Optional[str] = None,
        output: Any = None,
        error: Optional[Exception] = None,
    ) -> None:
        """Emit a typed :class:`AgentOutputEvent` when an agent run ends.

        Termination metadata (``duration_ns``, ``framework``,
        ``agent_name``, ``raw_output``, ``error``, ``run_status``) is
        carried on :class:`MessageContent.metadata` — the canonical
        :class:`AgentOutputEvent` has no top-level slot for these
        SmolAgents-specific signals.
        """
        if not self._connected:
            return
        try:
            tid = threading.get_ident()
            end_ns = time.time_ns()
            with self._adapter_lock:
                start_ns = self._run_starts.pop(tid, 0)
            duration_ns = end_ns - start_ns if start_ns else 0
            serialised_output = self._safe_serialize(output)
            metadata: Dict[str, Any] = {
                "framework": "smolagents",
                "agent_name": agent_name,
                "duration_ns": duration_ns,
                "raw_output": serialised_output,
                "run_status": "run_failed" if error else "run_complete",
            }
            if error:
                metadata["error"] = str(error)
            self.emit_event(
                AgentOutputEvent.create(
                    message=_stringify(serialised_output),
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
        error: Optional[Exception] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Emit a typed :class:`ToolCallEvent` for a tool invocation.

        Scalar tool inputs/outputs are wrapped in ``{"value": ...}``
        so the canonical ``input`` / ``output`` dict slots are
        satisfied. SmolAgents does not surface tool versions, so
        ``version`` falls back to ``"unavailable"`` per the
        canonical schema's NORMATIVE rule.
        """
        if not self._connected:
            return
        try:
            serialised_input = self._safe_serialize(tool_input)
            serialised_output = self._safe_serialize(tool_output)
            input_data: Dict[str, Any]
            if isinstance(serialised_input, dict):
                input_data = dict(serialised_input)
            elif serialised_input is None:
                input_data = {}
            else:
                input_data = {"value": serialised_input}
            output_data: Optional[Dict[str, Any]]
            if isinstance(serialised_output, dict):
                output_data = dict(serialised_output)
            elif serialised_output is None:
                output_data = None
            else:
                output_data = {"value": serialised_output}
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
        provider: Optional[str] = None,
        model: Optional[str] = None,
        tokens_prompt: Optional[int] = None,
        tokens_completion: Optional[int] = None,
        latency_ms: Optional[float] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """Emit a typed :class:`ModelInvokeEvent` for an LLM call.

        Provider / model identifiers fall back to ``"unknown"`` when
        not supplied so the canonical schema validators are
        satisfied. Token counts use the canonical ``prompt_tokens`` /
        ``completion_tokens`` slots.
        """
        if not self._connected:
            return
        try:
            input_messages: Optional[List[Dict[str, str]]] = None
            if self._capture_config.capture_content and messages:
                input_messages = messages
            self.emit_event(
                ModelInvokeEvent.create(
                    provider=provider or "unknown",
                    name=model or "unknown",
                    version="unavailable",
                    parameters={"framework": "smolagents"},
                    prompt_tokens=tokens_prompt,
                    completion_tokens=tokens_completion,
                    latency_ms=latency_ms,
                    input_messages=input_messages,
                )
            )
        except Exception:
            logger.warning("Error in on_llm_call", exc_info=True)

    def on_handoff(
        self,
        from_agent: str,
        to_agent: str,
        context: Any = None,
    ) -> None:
        """Emit a typed :class:`AgentHandoffEvent` for manager-to-managed
        delegation.

        The canonical schema requires ``handoff_context_hash`` to be a
        ``sha256:<hex64>`` string — empty contexts are still hashed
        (over the empty string) so the wire format is uniform. The
        previous adapter implementation emitted ``None`` when context
        was missing, which violated the canonical
        :class:`AgentHandoffEvent.handoff_context_hash` validator.
        """
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

    def _get_agent_name(self, agent: Any) -> str:
        return getattr(agent, "name", None) or type(agent).__name__

    def _emit_agent_config(
        self,
        agent_name: str,
        agent: Any,
        agent_type: str,
    ) -> None:
        """Emit a typed :class:`EnvironmentConfigEvent` per agent.

        Idempotent per agent name — only the first call for a given
        agent actually emits. SmolAgents' runtime is treated as a
        ``simulated`` environment by default; the real production
        environment (``cloud`` / ``on_prem``) is the responsibility
        of the host application's environment.config emission, not
        this framework adapter (mirrors the agno reference pattern).
        """
        with self._adapter_lock:
            if agent_name in self._seen_agents:
                return
            self._seen_agents.add(agent_name)
        attributes: Dict[str, Any] = {
            "framework": "smolagents",
            "agent_name": agent_name,
            "agent_type": agent_type,
        }
        tools = getattr(agent, "tools", None)
        if tools:
            if isinstance(tools, dict):
                attributes["tools"] = list(tools.keys())
            else:
                attributes["tools"] = [getattr(t, "name", str(t)) for t in tools]
        model = getattr(agent, "model", None)
        if model:
            attributes["model"] = str(model)
        managed = getattr(agent, "managed_agents", None)
        if managed:
            if isinstance(managed, dict):
                attributes["managed_agents"] = list(managed.keys())
            elif isinstance(managed, list):
                attributes["managed_agents"] = [getattr(a, "name", str(a)) for a in managed]
        system_prompt = getattr(agent, "system_prompt", None)
        if system_prompt and self._capture_config.capture_content:
            attributes["system_prompt"] = str(system_prompt)[:500]
        self.emit_event(
            EnvironmentConfigEvent.create(
                env_type=EnvironmentType.SIMULATED,
                attributes=attributes,
            )
        )

    def _emit_code_execution(self, agent_name: str, result: Any) -> None:
        """Emit a typed :class:`ToolCallEvent` for CodeAgent execution.

        The previous adapter implementation emitted an ad-hoc
        ``agent.code`` event type that is NOT in the canonical
        13-event taxonomy. The typed migration maps the
        code-execution boundary onto :class:`ToolCallEvent` with
        ``name="code_execution"`` and
        ``integration=IntegrationType.SCRIPT`` (CodeAgent compiles
        and runs Python code as an inline script, not a library
        call). Framework provenance (``framework``, ``agent_name``,
        ``event_subtype``, ``logs``, ``raw_output``) is carried on
        the canonical input/output dicts.
        """
        try:
            logs = getattr(result, "logs", None) or getattr(result, "inner_messages", None)
            serialised_result = self._safe_serialize(result)
            serialised_logs = self._safe_serialize(logs)

            input_data: Dict[str, Any] = {
                "framework": "smolagents",
                "agent_name": agent_name,
                "event_subtype": "code_execution",
            }
            output_data: Dict[str, Any] = {
                "raw_output": serialised_result,
                "logs": serialised_logs,
            }

            self.emit_event(
                ToolCallEvent.create(
                    name="code_execution",
                    version="unavailable",
                    integration=IntegrationType.SCRIPT,
                    input_data=input_data,
                    output_data=output_data,
                )
            )
        except Exception:
            logger.debug("Could not emit code execution event", exc_info=True)

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


# Registry lazy-loading convention.
ADAPTER_CLASS = SmolAgentsAdapter
