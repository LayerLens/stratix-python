"""
A2A Protocol Adapter — Main adapter class.

Instruments A2A protocol interactions at both server and client sides.
Captures Agent Card discovery, task lifecycle, SSE streams, and multi-agent
delegation chains.
"""

from __future__ import annotations

import time
import uuid
import hashlib
import logging
from typing import Any

from layerlens.instrument.adapters._base.adapter import (
    AdapterInfo,
    AdapterStatus,
    ReplayableTrace,
    AdapterCapability,
)
from layerlens.instrument.adapters.protocols.base import BaseProtocolAdapter
from layerlens.instrument.adapters.protocols.a2a.acp_normalizer import ACPNormalizer
from layerlens.instrument.adapters.protocols.a2a.task_lifecycle import TaskStateMachine

logger = logging.getLogger(__name__)


class A2AAdapter(BaseProtocolAdapter):
    """
    LayerLens adapter for the A2A (Agent-to-Agent) protocol.

    Provides dual-channel instrumentation:
    - ``serve()`` wraps server-side A2A handlers
    - ``client()`` returns a traced A2A client wrapper
    """

    FRAMEWORK = "a2a"
    PROTOCOL = "a2a"
    PROTOCOL_VERSION = "0.2.1"
    VERSION = "0.1.0"

    def __init__(self, memory_service: Any | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._framework_version: str | None = None
        self._agent_cards: dict[str, dict[str, Any]] = {}
        self._task_machines: dict[str, TaskStateMachine] = {}
        self._acp_normalizer = ACPNormalizer()
        self._task_start_times: dict[str, float] = {}
        self._memory_service = memory_service

    # --- Lifecycle ---

    def connect(self) -> None:
        try:
            import a2a  # type: ignore[import-not-found,unused-ignore]

            self._framework_version = getattr(a2a, "__version__", "unknown")
        except ImportError:
            self._framework_version = None
            logger.debug("a2a-sdk not installed; adapter operates in standalone mode")
        self._connected = True
        self._status = AdapterStatus.HEALTHY

    def disconnect(self) -> None:
        self._agent_cards.clear()
        self._task_machines.clear()
        self._task_start_times.clear()
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED
        self._close_sinks()

    def get_adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            name="A2AAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self._framework_version,
            capabilities=[
                AdapterCapability.TRACE_TOOLS,
                AdapterCapability.TRACE_STATE,
                AdapterCapability.TRACE_HANDOFFS,
                AdapterCapability.TRACE_PROTOCOL_EVENTS,
                AdapterCapability.STREAMING,
                AdapterCapability.REPLAY,
            ],
            description="LayerLens adapter for the A2A (Agent-to-Agent) protocol",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        return ReplayableTrace(
            adapter_name="A2AAdapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[],
            config={
                "capture_config": self._capture_config.model_dump(),
                "agent_cards": dict(self._agent_cards.items()),
            },
        )

    def probe_health(self, endpoint: str | None = None) -> dict[str, Any]:
        from layerlens.instrument.adapters.protocols.health import probe_a2a_agent_card

        if endpoint:
            result = probe_a2a_agent_card(endpoint)
            return result.to_dict()
        return {
            "reachable": self._connected,
            "latency_ms": 0.0,
            "protocol_version": self._framework_version,
        }

    # --- Agent Card handling ---

    def register_agent_card(self, card_data: dict[str, Any], source: str = "discovery") -> None:
        """
        Register an A2A Agent Card and emit a protocol.agent_card event.

        Args:
            card_data: Parsed Agent Card JSON
            source: How the card was obtained (discovery | registration | refresh)
        """
        from layerlens.instrument._vendored.events_protocol import (
            SkillInfo,
            AgentCardEvent,
        )

        agent_id = card_data.get("name", card_data.get("id", "unknown"))
        url = card_data.get("url", "")
        version = card_data.get("protocolVersion", card_data.get("version", "unknown"))

        skills = []
        for s in card_data.get("skills", []):
            skills.append(
                SkillInfo(
                    id=s.get("id", ""),
                    name=s.get("name", ""),
                    description=s.get("description"),
                    tags=s.get("tags", []),
                    examples=s.get("examples", []),
                )
            )

        self._agent_cards[agent_id] = card_data

        event = AgentCardEvent.create(
            agent_id=agent_id,
            name=card_data.get("name", "unknown"),
            url=url,
            version=version,
            description=card_data.get("description"),
            capabilities=card_data.get("capabilities", {}),
            skills=skills,
            auth_scheme=card_data.get("authScheme")
            or card_data.get("authentication", {}).get("scheme"),
            source=source,
        )
        self.emit_event(event)

    # --- Task lifecycle ---

    def on_task_submitted(
        self,
        task_id: str,
        receiver_url: str,
        *,
        task_type: str | None = None,
        submitter_agent_id: str | None = None,
        message_role: str = "user",
        raw_payload: dict[str, Any] | None = None,
    ) -> None:
        """Record an A2A task submission."""
        from layerlens.instrument._vendored.events_protocol import TaskSubmittedEvent

        # Check for ACP-origin patterns
        protocol_origin = "a2a"
        if raw_payload:
            normalized, is_acp = self._acp_normalizer.detect_and_normalize(raw_payload)
            if is_acp:
                protocol_origin = "acp"
                raw_payload = normalized

        self._task_start_times[task_id] = time.monotonic()
        self._task_machines[task_id] = TaskStateMachine(task_id)

        event = TaskSubmittedEvent.create(
            task_id=task_id,
            receiver_agent_url=receiver_url,
            task_type=task_type,
            submitter_agent_id=submitter_agent_id,
            protocol_origin=protocol_origin,
            message_role=message_role,
        )
        self.emit_event(event)

    def on_task_completed(
        self,
        task_id: str,
        final_status: str,
        *,
        artifacts: list[dict[str, Any]] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        share_to_agent_id: str | None = None,
    ) -> None:
        """Record an A2A task completion.

        Args:
            task_id: Unique task identifier.
            final_status: Terminal status of the task.
            artifacts: Optional list of artifact dicts.
            error_code: Error code if the task failed.
            error_message: Error message if the task failed.
            share_to_agent_id: When provided **and** a memory_service is
                configured, the task context is stored as episodic memory
                and shared to the specified agent via
                ``AgentMemoryService.share_memory()``.
        """
        from layerlens.instrument._vendored.events_protocol import TaskCompletedEvent

        duration_ms = None
        if task_id in self._task_start_times:
            duration_ms = (time.monotonic() - self._task_start_times.pop(task_id)) * 1000

        artifact_hashes = []
        if artifacts:
            for art in artifacts:
                h = hashlib.sha256(str(art).encode()).hexdigest()
                artifact_hashes.append(f"sha256:{h}")

        event = TaskCompletedEvent.create(
            task_id=task_id,
            final_status=final_status,
            artifact_count=len(artifacts or []),
            artifact_hashes=artifact_hashes,
            error_code=error_code,
            error_message=error_message,
            duration_ms=duration_ms,
        )
        self.emit_event(event)
        self._task_machines.pop(task_id, None)

        # Optionally share task context via memory
        if self._memory_service is not None and share_to_agent_id:
            self._share_task_memory(task_id, final_status, artifacts, share_to_agent_id)

    def _share_task_memory(
        self,
        task_id: str,
        final_status: str,
        artifacts: list[dict[str, Any]] | None,
        target_agent_id: str,
    ) -> None:
        """Store task context as episodic memory and share to target agent.

        Failures are logged and swallowed.
        """
        try:
            from layerlens.instrument._vendored.memory_models import MemoryEntry

            content = f"task_id={task_id}, status={final_status}"
            if artifacts:
                content += f", artifact_count={len(artifacts)}"

            entry = MemoryEntry(
                org_id="",
                agent_id="a2a",
                memory_type="episodic",
                key=f"task_context_{task_id}",
                content=content,
                importance=0.6,
                metadata={
                    "source": "a2a_adapter",
                    "task_id": task_id,
                    "shared_to": target_agent_id,
                },
            )
            stored = self._memory_service.store(entry)  # type: ignore[union-attr]
            self._memory_service.share_memory(stored.id, target_agent_id)  # type: ignore[union-attr]
        except Exception:
            logger.debug(
                "A2A: failed to share task memory for task %s to agent %s",
                task_id,
                target_agent_id,
                exc_info=True,
            )

    def on_task_delegation(
        self,
        from_agent: str,
        to_agent: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record an A2A task delegation as an agent.handoff event."""
        from layerlens.instrument._vendored.events_cross_cutting import AgentHandoffEvent

        ctx_str = str(context or {})
        ctx_hash = f"sha256:{hashlib.sha256(ctx_str.encode()).hexdigest()}"

        event = AgentHandoffEvent.create(
            from_agent=from_agent,
            to_agent=to_agent,
            handoff_context_hash=ctx_hash,
        )
        self.emit_event(event)

    # --- SSE stream handling ---

    def on_stream_event(
        self,
        sequence: int,
        payload: Any,
    ) -> None:
        """Record an A2A SSE stream event."""
        from layerlens.instrument._vendored.events_protocol import ProtocolStreamEvent

        payload_str = str(payload)
        payload_hash = f"sha256:{hashlib.sha256(payload_str.encode()).hexdigest()}"

        event = ProtocolStreamEvent.create(
            protocol="a2a",
            sequence_in_stream=sequence,
            payload_hash=payload_hash,
            payload_summary=payload_str[:200] if len(payload_str) > 200 else payload_str,
        )
        self.emit_event(event)
