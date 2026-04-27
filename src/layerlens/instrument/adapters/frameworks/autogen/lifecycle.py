"""
STRATIX AutoGen Lifecycle Hooks

Provides the main AutoGenAdapter class with monkey-patch-based instrumentation
for AutoGen ConversableAgent instances.

Typed-event status (post PR #129 migration, bundle 1):

* Every emission flows through :meth:`BaseAdapter.emit_event` with a
  canonical Pydantic payload imported from
  :mod:`layerlens.instrument._compat.events`.
* AutoGen-specific provenance (``framework``, ``agent``, ``message_seq``,
  ``message_preview``) is carried in the canonical model's
  metadata / attributes / parameters slots — the canonical schema does
  not expose these as top-level fields.
* The agent.state.change "message_received" marker emitted by
  :meth:`on_receive` does not satisfy the canonical
  :class:`AgentStateChangeEvent` ``before_hash`` / ``after_hash``
  contract (the receive boundary has no real state mutation to hash).
  It is mapped onto :class:`AgentInputEvent` with ``role=AGENT`` so the
  cross-agent receive boundary is still emitted, with the framework
  metadata preserved on :class:`MessageContent.metadata`.
* The handoff context hash is generated via SHA-256 over the message
  preview (or the empty string when no message is available) so the
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
    EnvironmentType,
    IntegrationType,
    AgentOutputEvent,
    ModelInvokeEvent,
    AgentHandoffEvent,
    ToolEnvironmentEvent,
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
from layerlens.instrument.adapters._base.capture import CaptureConfig
from layerlens.instrument.adapters._base.pydantic_compat import PydanticCompat
from layerlens.instrument.adapters.frameworks.autogen.metadata import AutoGenAgentMetadataExtractor

logger = logging.getLogger(__name__)


def _stringify(value: Any) -> str:
    """Return a string view of ``value`` suitable for the canonical
    :class:`MessageContent.message` field.

    The canonical schema requires :class:`AgentInputEvent` and
    :class:`AgentOutputEvent` to carry a ``message: str``. AutoGen
    callbacks deliver the underlying input/output as arbitrary Python
    objects (dicts with ``content`` keys, model responses, ``None``);
    this helper converts each to a (possibly empty) string so the
    typed event always validates. The original payload is preserved
    on :class:`MessageContent.metadata.raw_input` /
    ``raw_output``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # AutoGen messages are typically ``{"role": ..., "content": ...}``;
        # surface the content slot when present.
        content = value.get("content")
        if isinstance(content, str):
            return content
    return str(value)


def _sha256_of(value: str) -> str:
    """Return a canonical ``sha256:<hex64>`` hash string for ``value``.

    The canonical schema's :class:`AgentHandoffEvent` requires
    ``handoff_context_hash`` to start with ``sha256:`` and have a
    64-character hex tail (see
    ``ateam/stratix/core/events/cross_cutting.py``). Centralising the
    format here ensures every emit site uses the same wire format —
    including the empty-string fallback used when AutoGen has no
    message context to hash.
    """
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


class AutoGenAdapter(BaseAdapter):
    """
    Main adapter for integrating STRATIX with Microsoft AutoGen.

    Uses monkey-patching to intercept ConversableAgent methods (send, receive,
    generate_reply, execute_code_blocks) and emit STRATIX telemetry events.

    Supports both new-style (stratix, capture_config) and legacy (stratix_instance)
    constructor parameters.

    Usage:
        adapter = AutoGenAdapter(stratix=stratix_instance)
        adapter.connect()
        adapter.connect_agents(agent1, agent2)
        agent1.initiate_chat(agent2, message="Hello")
    """

    FRAMEWORK = "autogen"
    VERSION = "0.1.0"
    # The adapter source files import nothing from ``pydantic`` directly
    # (verified by grep across ``frameworks/autogen/``). pyautogen 0.2.x
    # supports both Pydantic majors; the adapter only monkey-patches
    # ConversableAgent methods and emits typed events through the
    # canonical schema (PR #129), never touching the framework's
    # Pydantic models.
    requires_pydantic = PydanticCompat.V1_OR_V2

    # Per-adapter ``extra="allow"`` decision: AutoGen targets the
    # canonical 13-event taxonomy exclusively. Unknown event types must
    # be rejected by the base adapter's typed-event validator, so this
    # stays ``False``. See ``docs/adapters/typed-events.md`` for the
    # opt-in policy.
    ALLOW_UNREGISTERED_EVENTS: bool = False

    def __init__(
        self,
        stratix: Any | None = None,
        capture_config: CaptureConfig | None = None,
        # Legacy param
        stratix_instance: Any | None = None,
        memory_service: Any | None = None,
        *,
        org_id: str | None = None,
    ) -> None:
        resolved_stratix = stratix or stratix_instance
        super().__init__(stratix=resolved_stratix, capture_config=capture_config, org_id=org_id)

        self._metadata_extractor = AutoGenAgentMetadataExtractor()
        self._adapter_lock = threading.Lock()
        self._seen_agents: set[str] = set()
        self._wrapped_agents: list[Any] = []
        self._originals: dict[int, dict[str, Any]] = {}  # agent id -> original methods
        self._message_seq: int = 0
        self._conversation_start_ns: int = 0
        self._framework_version: str | None = None
        self._memory_service = memory_service

    # --- BaseAdapter lifecycle ---

    def connect(self) -> None:
        """Verify AutoGen is importable and mark as connected."""
        try:
            import autogen  # type: ignore[import-not-found,unused-ignore]  # noqa: F401

            version = getattr(autogen, "__version__", "unknown")
            logger.debug("AutoGen %s detected", version)
        except ImportError:
            logger.debug("AutoGen not installed; adapter usable in mock/test mode")
        self._framework_version = self._detect_framework_version()
        self._connected = True
        self._status = AdapterStatus.HEALTHY

    def disconnect(self) -> None:
        """Unwrap agents and disconnect."""
        for agent in self._wrapped_agents:
            self._unwrap_agent(agent)
        self._wrapped_agents.clear()
        self._originals.clear()
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
            name="AutoGenAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self._framework_version,
            capabilities=[
                AdapterCapability.TRACE_TOOLS,
                AdapterCapability.TRACE_MODELS,
                AdapterCapability.TRACE_STATE,
                AdapterCapability.TRACE_HANDOFFS,
            ],
            description="LayerLens adapter for Microsoft AutoGen framework",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        return ReplayableTrace(
            adapter_name="AutoGenAdapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[],
            config={
                "capture_config": self._capture_config.model_dump(),
            },
        )

    # --- Agent wrapping ---

    def connect_agents(self, *agents: Any) -> list[Any]:
        """
        Monkey-patch AutoGen agents with STRATIX tracing.

        Wraps send, receive, generate_reply, and execute_code_blocks methods.
        Stores originals for unwrap on disconnect.

        Emits environment.config (L4a) on first encounter per agent.

        Args:
            *agents: AutoGen ConversableAgent instances

        Returns:
            List of wrapped agents (same objects, modified in-place)
        """
        from layerlens.instrument.adapters.frameworks.autogen.wrappers import (
            create_traced_send,
            create_traced_receive,
            create_traced_execute_code,
            create_traced_generate_reply,
        )

        result = []
        for agent in agents:
            agent_id = id(agent)
            if agent_id in self._originals:
                result.append(agent)
                continue

            originals: dict[str, Any] = {}

            # Wrap send
            if hasattr(agent, "send"):
                originals["send"] = agent.send
                agent.send = create_traced_send(self, agent, agent.send)

            # Wrap receive
            if hasattr(agent, "receive"):
                originals["receive"] = agent.receive
                agent.receive = create_traced_receive(self, agent, agent.receive)

            # Wrap generate_reply
            if hasattr(agent, "generate_reply"):
                originals["generate_reply"] = agent.generate_reply
                agent.generate_reply = create_traced_generate_reply(self, agent, agent.generate_reply)

            # Wrap execute_code_blocks
            if hasattr(agent, "execute_code_blocks"):
                originals["execute_code_blocks"] = agent.execute_code_blocks
                agent.execute_code_blocks = create_traced_execute_code(self, agent, agent.execute_code_blocks)

            self._originals[agent_id] = originals
            self._wrapped_agents.append(agent)

            # Emit agent config on first encounter
            self._emit_agent_config(agent)

            result.append(agent)

        return result

    def _unwrap_agent(self, agent: Any) -> None:
        """Restore original methods on an agent."""
        agent_id = id(agent)
        originals = self._originals.get(agent_id)
        if not originals:
            return
        for method_name, original in originals.items():
            try:
                setattr(agent, method_name, original)
            except Exception:
                logger.debug("Could not unwrap %s on agent", method_name, exc_info=True)

    # --- Lifecycle hooks (called by wrappers) ---

    def on_send(
        self,
        sender: Any,
        message: Any,
        recipient: Any,
    ) -> None:
        """Emit a typed :class:`AgentHandoffEvent` for an agent send.

        AutoGen's ``send`` boundary is the cleanest signal for agent
        delegation in conversational multi-agent flows. The canonical
        :class:`AgentHandoffEvent` requires a ``handoff_context_hash``
        in the ``sha256:<hex64>`` format — we hash the message preview
        (or the empty string when no message is supplied) so the wire
        format is always conformant.

        AutoGen-specific provenance (``framework``, ``message_seq``,
        ``message_preview``) lives on
        :attr:`AgentHandoffEvent.context_privacy_level` is left at its
        default; the message_seq + message_preview are tracked through
        the replay buffer's per-event metadata via the recording stratix
        — they are not part of the canonical handoff schema.
        """
        with self._adapter_lock:
            self._message_seq += 1
            msg_seq = self._message_seq  # noqa: F841 — reserved for future replay metadata
        sender_name = getattr(sender, "name", str(sender))
        recipient_name = getattr(recipient, "name", str(recipient))
        message_preview = self._truncate(self._message_content(message))

        self.emit_event(
            AgentHandoffEvent.create(
                from_agent=sender_name,
                to_agent=recipient_name,
                handoff_context_hash=_sha256_of(message_preview),
            )
        )

    def on_receive(
        self,
        receiver: Any,
        message: Any,
        sender: Any,
    ) -> None:
        """Emit a typed :class:`AgentInputEvent` for an agent receive.

        The previous adapter implementation emitted an ad-hoc
        ``agent.state.change`` payload carrying only an
        ``event_subtype`` marker. That payload did not satisfy the
        canonical :class:`AgentStateChangeEvent` schema's
        ``before_hash`` / ``after_hash`` requirement (the receive
        boundary has no real state mutation to hash).

        Receiving a message is logically an inbound to the receiving
        agent, so the canonical mapping is :class:`AgentInputEvent`
        with ``role=AGENT`` (the message arrives from another agent,
        not a human). Framework provenance (the receiving agent's
        name, the sender's name, the original
        ``event_subtype="message_received"`` marker) lives on
        :class:`MessageContent.metadata`.
        """
        receiver_name = getattr(receiver, "name", str(receiver))
        sender_name = getattr(sender, "name", str(sender)) if sender else None
        message_preview = self._truncate(self._message_content(message))

        self.emit_event(
            AgentInputEvent.create(
                message=message_preview,
                role=MessageRole.AGENT,
                metadata={
                    "framework": "autogen",
                    "agent": receiver_name,
                    "event_subtype": "message_received",
                    "from_agent": sender_name,
                },
            )
        )

    def on_generate_reply(
        self,
        agent: Any,
        messages: Any = None,
        reply: Any = None,
        latency_ms: float | None = None,
    ) -> None:
        """Emit a typed :class:`ModelInvokeEvent` for reply generation.

        AutoGen does not expose model versions at the
        ``ConversableAgent.llm_config`` level, so ``version`` falls
        back to ``"unavailable"`` per the canonical schema's NORMATIVE
        rule. The token usage extracted from the reply object lives
        in the canonical ``prompt_tokens`` / ``completion_tokens``
        slots; framework provenance (``framework``, ``agent``,
        ``reply_preview``) is carried on
        :attr:`ModelInfo.parameters`.

        Provider detection is best-effort — AutoGen models are
        identified by name (e.g. ``gpt-5``); the canonical schema
        requires both ``provider`` and ``name``, so a heuristic is
        applied to derive the provider from the model identifier.
        Unknown identifiers fall back to ``provider="unknown"``.
        """
        agent_name = getattr(agent, "name", str(agent))
        model_name = self._extract_model_name(agent) or "unknown"
        provider = self._detect_provider(model_name) or "unknown"

        parameters: dict[str, Any] = {
            "framework": "autogen",
            "agent": agent_name,
            "reply_preview": self._truncate(self._message_content(reply)),
        }

        # Extract token counts if available
        token_usage = self._extract_token_usage_from_reply(reply)
        prompt_tokens = token_usage.get("tokens_prompt") if token_usage else None
        completion_tokens = token_usage.get("tokens_completion") if token_usage else None

        # Include messages for Prompt Lab extraction (gated by capture_content)
        input_messages: list[dict[str, str]] | None = None
        if self._capture_config.capture_content and messages:
            normalized: list[dict[str, str]] = []
            # Prepend system message from agent config
            sys_msg = self._extract_system_message(agent)
            if sys_msg:
                normalized.append({"role": "system", "content": self._truncate(sys_msg, 10_000)})
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        normalized.append(
                            {
                                "role": str(msg["role"]),
                                "content": str(msg["content"])[:10_000],
                            }
                        )
                    elif isinstance(msg, str):
                        normalized.append({"role": "user", "content": msg[:10_000]})
            if normalized:
                input_messages = normalized

        self.emit_event(
            ModelInvokeEvent.create(
                provider=provider,
                name=model_name,
                version="unavailable",
                parameters=parameters,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                input_messages=input_messages,
            )
        )

    def on_execute_code(
        self,
        agent: Any,
        code_blocks: Any = None,
        result: Any = None,
        latency_ms: float | None = None,
    ) -> None:
        """Emit typed :class:`ToolCallEvent` + :class:`ToolEnvironmentEvent`.

        AutoGen's code-execution boundary is modelled as a tool call
        with ``name="code_execution"`` and
        ``integration=IntegrationType.SCRIPT`` (the closest canonical
        match — AutoGen executes the generated code as an inline
        script, not as a library call). Framework provenance
        (``framework``, ``agent``, ``code_blocks_count``,
        ``result_preview``) is carried on the canonical input/output
        dicts and on
        :attr:`ToolEnvironmentInfo.config`.
        """
        agent_name = getattr(agent, "name", str(agent))
        code_blocks_count = len(code_blocks) if code_blocks else 0

        # tool.call for the code execution. The canonical ``input``
        # slot carries the framework-specific block count + agent
        # binding; ``output`` carries the result preview when
        # available.
        input_data: dict[str, Any] = {
            "framework": "autogen",
            "agent": agent_name,
            "code_blocks_count": code_blocks_count,
        }
        output_data: dict[str, Any] | None = None
        if result is not None:
            output_data = {"result_preview": self._truncate(str(result))}

        self.emit_event(
            ToolCallEvent.create(
                name="code_execution",
                version="unavailable",
                integration=IntegrationType.SCRIPT,
                input_data=input_data,
                output_data=output_data,
                latency_ms=latency_ms,
            )
        )

        # tool.environment for execution environment details. The
        # canonical schema does not declare a top-level "agent" or
        # "execution_type" field; both move into
        # :attr:`ToolEnvironmentInfo.config`.
        self.emit_event(
            ToolEnvironmentEvent.create(
                config={
                    "framework": "autogen",
                    "agent": agent_name,
                    "execution_type": "code_block",
                    "code_blocks_count": code_blocks_count,
                },
            )
        )

    def on_conversation_start(
        self,
        initiator: Any,
        message: Any,
    ) -> None:
        """Emit a typed :class:`AgentInputEvent` for conversation start.

        AutoGen-specific provenance (``framework``, ``initiator``,
        ``timestamp_ns``, ``raw_input``) lives on
        :class:`MessageContent.metadata`. The canonical ``message``
        field carries a string view of the inbound message.
        """
        with self._adapter_lock:
            self._conversation_start_ns = time.time_ns()
        initiator_name = getattr(initiator, "name", str(initiator))
        serialised_message = self._safe_serialize(message)

        self.emit_event(
            AgentInputEvent.create(
                message=_stringify(serialised_message),
                role=MessageRole.HUMAN,
                metadata={
                    "framework": "autogen",
                    "initiator": initiator_name,
                    "timestamp_ns": self._conversation_start_ns,
                    "raw_input": serialised_message,
                },
            )
        )

    def on_conversation_end(
        self,
        final_message: Any = None,
        termination_reason: str | None = None,
    ) -> None:
        """Emit a typed :class:`AgentOutputEvent` for conversation end.

        Termination metadata (``termination_reason``, ``duration_ns``,
        ``framework``, ``raw_output``) is carried on
        :class:`MessageContent.metadata` — the canonical
        :class:`AgentOutputEvent` has no top-level slot for these
        AutoGen-specific signals.
        """
        end_ns = time.time_ns()
        duration_ns = end_ns - self._conversation_start_ns if self._conversation_start_ns else 0
        serialised_final = self._safe_serialize(final_message)

        self.emit_event(
            AgentOutputEvent.create(
                message=_stringify(serialised_final),
                metadata={
                    "framework": "autogen",
                    "termination_reason": termination_reason,
                    "duration_ns": duration_ns,
                    "raw_output": serialised_final,
                },
            )
        )

    # --- Memory integration ---

    def on_message(
        self,
        agent_id: str,
        message: Any,
    ) -> None:
        """Store an agent message as episodic memory.

        Only active when ``memory_service`` is provided. Failures are
        logged and swallowed to avoid disrupting normal operation.

        Args:
            agent_id: Identifier of the agent that sent/received the message.
            message: The message content to store.
        """
        if self._memory_service is None:
            return

        try:
            from layerlens.instrument._vendored.memory_models import MemoryEntry

            content = self._message_content(message)
            timestamp = int(time.time())
            entry = MemoryEntry(
                org_id=getattr(self._stratix, "org_id", ""),
                agent_id=agent_id,
                memory_type="episodic",
                key=f"message_{timestamp}",
                content=content[:4000],
                importance=0.4,
                metadata={"source": "autogen_adapter"},
            )
            self._memory_service.store(entry)
        except Exception:
            logger.debug(
                "Failed to store episodic memory for agent %s",
                agent_id,
                exc_info=True,
            )

    def on_conversation_end_memory(
        self,
        agent_id: str,
        summary: str,
    ) -> None:
        """Consolidate a conversation summary into semantic memory.

        Only active when ``memory_service`` is provided. Failures are
        logged and swallowed.

        Args:
            agent_id: Agent whose conversation to consolidate.
            summary: High-level summary of the conversation.
        """
        if self._memory_service is None:
            return

        try:
            from layerlens.instrument._vendored.memory_models import MemoryEntry

            timestamp = int(time.time())
            entry = MemoryEntry(
                org_id=getattr(self._stratix, "org_id", ""),
                agent_id=agent_id,
                memory_type="semantic",
                key=f"conversation_summary_{timestamp}",
                content=summary[:4000],
                importance=0.7,
                metadata={"source": "autogen_adapter", "type": "conversation_consolidation"},
            )
            self._memory_service.store(entry)
        except Exception:
            logger.debug(
                "Failed to store conversation summary for agent %s",
                agent_id,
                exc_info=True,
            )

    # --- Agent config emission ---

    def _emit_agent_config(self, agent: Any) -> None:
        """Emit a typed :class:`EnvironmentConfigEvent` per agent.

        Idempotent per agent — only the first call for a given
        agent name actually emits. AutoGen's runtime is treated as
        a ``simulated`` environment by default; the real production
        environment (``cloud`` / ``on_prem``) is the responsibility
        of the host application's environment.config emission, not
        this framework adapter (mirrors the agno reference pattern).
        """
        name = getattr(agent, "name", None) or str(agent)
        with self._adapter_lock:
            if name in self._seen_agents:
                return
            self._seen_agents.add(name)

        metadata = self._metadata_extractor.extract(agent)

        attributes: dict[str, Any] = {
            "framework": "autogen",
            **metadata,
        }
        self.emit_event(
            EnvironmentConfigEvent.create(
                env_type=EnvironmentType.SIMULATED,
                attributes=attributes,
            )
        )

    # --- Internal helpers ---

    def _safe_serialize(self, value: Any) -> Any:
        """Safely serialize a value for events."""
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

    def _message_content(self, message: Any) -> str:
        """Extract string content from a message."""
        if message is None:
            return ""
        if isinstance(message, str):
            return message
        if isinstance(message, dict):
            return str(message.get("content", message))
        return str(message)

    def _truncate(self, text: str, max_len: int = 500) -> str:
        """Truncate text to max_len."""
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    def _extract_system_message(self, agent: Any) -> str | None:
        """Extract system message from agent config."""
        try:
            # AutoGen 0.2.x: agent.system_message
            sys_msg = getattr(agent, "system_message", None)
            if sys_msg:
                return str(sys_msg)
            # AutoGen 0.4+/agentchat: agent._system_messages
            sys_msgs = getattr(agent, "_system_messages", None)
            if sys_msgs and isinstance(sys_msgs, list) and sys_msgs:
                first = sys_msgs[0]
                content = getattr(first, "content", None) or str(first)
                return str(content)
        except Exception:
            pass
        return None

    def _detect_provider(self, model: str | None) -> str | None:
        """Detect the LLM provider from a model identifier.

        AutoGen does not surface ``provider`` directly on
        ``llm_config``; callers pass identifiers like ``"gpt-5"``,
        ``"claude-opus-4"``, etc. The canonical
        :class:`ModelInvokeEvent` requires both ``provider`` and
        ``name``, so this heuristic derives the provider from
        well-known model name prefixes. Mirrors the agno reference
        implementation. Unknown identifiers return ``None`` and the
        caller falls back to ``provider="unknown"``.
        """
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
        if "llama" in model_lower:
            return "meta"
        if "command" in model_lower:
            return "cohere"
        return None

    def _extract_model_name(self, agent: Any) -> str | None:
        """Extract model name from agent's llm_config."""
        try:
            llm_config = getattr(agent, "llm_config", None)
            if not llm_config or not isinstance(llm_config, dict):
                return None
            if "model" in llm_config:
                return llm_config["model"]  # type: ignore[no-any-return]
            config_list = llm_config.get("config_list", [])
            if config_list and isinstance(config_list[0], dict):
                return config_list[0].get("model")
        except Exception:
            pass
        return None

    def _extract_token_usage_from_reply(self, reply: Any) -> dict[str, Any] | None:
        """Extract token usage from a reply if available."""
        if reply is None:
            return None
        try:
            usage = getattr(reply, "usage", None)
            if usage:
                if isinstance(usage, dict):
                    return {
                        "tokens_prompt": usage.get("prompt_tokens"),
                        "tokens_completion": usage.get("completion_tokens"),
                    }
                return {
                    "tokens_prompt": getattr(usage, "prompt_tokens", None),
                    "tokens_completion": getattr(usage, "completion_tokens", None),
                }
        except Exception:
            pass
        return None

    @staticmethod
    def _detect_framework_version() -> str | None:
        try:
            import autogen  # type: ignore[import-not-found,unused-ignore]

            return getattr(autogen, "__version__", None)
        except ImportError:
            return None
