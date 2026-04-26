"""
STRATIX AutoGen Lifecycle Hooks

Provides the main AutoGenAdapter class with monkey-patch-based instrumentation
for AutoGen ConversableAgent instances.
"""

from __future__ import annotations

import time
import uuid
import logging
import threading
from typing import Any

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
    # ConversableAgent methods and emits dict events, never touching the
    # framework's Pydantic models.
    requires_pydantic = PydanticCompat.V1_OR_V2

    def __init__(
        self,
        stratix: Any | None = None,
        capture_config: CaptureConfig | None = None,
        # Legacy param
        stratix_instance: Any | None = None,
        memory_service: Any | None = None,
    ) -> None:
        resolved_stratix = stratix or stratix_instance
        super().__init__(stratix=resolved_stratix, capture_config=capture_config)

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
                agent.generate_reply = create_traced_generate_reply(
                    self, agent, agent.generate_reply
                )

            # Wrap execute_code_blocks
            if hasattr(agent, "execute_code_blocks"):
                originals["execute_code_blocks"] = agent.execute_code_blocks
                agent.execute_code_blocks = create_traced_execute_code(
                    self, agent, agent.execute_code_blocks
                )

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
        """
        Handle agent send.

        Emits agent.handoff (cross-cutting).
        """
        with self._adapter_lock:
            self._message_seq += 1
            msg_seq = self._message_seq
        sender_name = getattr(sender, "name", str(sender))
        recipient_name = getattr(recipient, "name", str(recipient))

        self.emit_dict_event(
            "agent.handoff",
            {
                "framework": "autogen",
                "from_agent": sender_name,
                "to_agent": recipient_name,
                "message_preview": self._truncate(self._message_content(message)),
                "message_seq": msg_seq,
            },
        )

    def on_receive(
        self,
        receiver: Any,
        message: Any,
        sender: Any,
    ) -> None:
        """
        Handle agent receive.

        Emits agent.state.change (cross-cutting).
        """
        receiver_name = getattr(receiver, "name", str(receiver))
        sender_name = getattr(sender, "name", str(sender)) if sender else None

        self.emit_dict_event(
            "agent.state.change",
            {
                "framework": "autogen",
                "agent": receiver_name,
                "event_subtype": "message_received",
                "from_agent": sender_name,
                "message_preview": self._truncate(self._message_content(message)),
            },
        )

    def on_generate_reply(
        self,
        agent: Any,
        messages: Any = None,
        reply: Any = None,
        latency_ms: float | None = None,
    ) -> None:
        """
        Handle reply generation.

        Emits model.invoke (L3).
        """
        agent_name = getattr(agent, "name", str(agent))
        model = self._extract_model_name(agent)

        payload: dict[str, Any] = {
            "framework": "autogen",
            "agent": agent_name,
            "model": model,
            "reply_preview": self._truncate(self._message_content(reply)),
        }
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms

        # Extract token counts if available
        token_usage = self._extract_token_usage_from_reply(reply)
        if token_usage:
            payload.update(token_usage)

        # Include messages for Prompt Lab extraction (gated by capture_content)
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
                payload["messages"] = normalized

        self.emit_dict_event("model.invoke", payload)

    def on_execute_code(
        self,
        agent: Any,
        code_blocks: Any = None,
        result: Any = None,
        latency_ms: float | None = None,
    ) -> None:
        """
        Handle code execution.

        Emits tool.call (L5a) and tool.environment (L5c).
        """
        agent_name = getattr(agent, "name", str(agent))

        # tool.call for the code execution
        self.emit_dict_event(
            "tool.call",
            {
                "framework": "autogen",
                "tool_name": "code_execution",
                "agent": agent_name,
                "code_blocks_count": len(code_blocks) if code_blocks else 0,
                "result_preview": self._truncate(str(result)) if result else None,
                "latency_ms": latency_ms,
            },
        )

        # tool.environment for execution environment details
        self.emit_dict_event(
            "tool.environment",
            {
                "framework": "autogen",
                "agent": agent_name,
                "execution_type": "code_block",
                "code_blocks_count": len(code_blocks) if code_blocks else 0,
            },
        )

    def on_conversation_start(
        self,
        initiator: Any,
        message: Any,
    ) -> None:
        """
        Handle conversation start.

        Emits agent.input (L1).
        """
        with self._adapter_lock:
            self._conversation_start_ns = time.time_ns()
        initiator_name = getattr(initiator, "name", str(initiator))

        self.emit_dict_event(
            "agent.input",
            {
                "framework": "autogen",
                "initiator": initiator_name,
                "message": self._safe_serialize(message),
                "timestamp_ns": self._conversation_start_ns,
            },
        )

    def on_conversation_end(
        self,
        final_message: Any = None,
        termination_reason: str | None = None,
    ) -> None:
        """
        Handle conversation end.

        Emits agent.output (L1).
        """
        end_ns = time.time_ns()
        duration_ns = end_ns - self._conversation_start_ns if self._conversation_start_ns else 0

        self.emit_dict_event(
            "agent.output",
            {
                "framework": "autogen",
                "final_message": self._safe_serialize(final_message),
                "termination_reason": termination_reason,
                "duration_ns": duration_ns,
            },
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
        """Emit environment.config for an agent on first encounter."""
        name = getattr(agent, "name", None) or str(agent)
        with self._adapter_lock:
            if name in self._seen_agents:
                return
            self._seen_agents.add(name)

        metadata = self._metadata_extractor.extract(agent)

        self.emit_dict_event(
            "environment.config",
            {
                "framework": "autogen",
                **metadata,
            },
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
