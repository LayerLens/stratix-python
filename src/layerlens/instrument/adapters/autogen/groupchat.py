"""
AutoGen GroupChat Tracing

Traces GroupChat speaker selection and turn management for multi-agent
conversations.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from layerlens.instrument.adapters.autogen.lifecycle import AutoGenAdapter

logger = logging.getLogger(__name__)


class GroupChatTracer:
    """
    Traces GroupChat speaker selection and turn management.

    Wraps GroupChatManager to intercept speaker selection, message routing,
    and termination detection.
    """

    def __init__(self, adapter: AutoGenAdapter) -> None:
        self._adapter = adapter
        self._lock = threading.Lock()
        self._message_seq: int = 0
        self._original_run_chat: Callable | None = None

    @property
    def message_seq(self) -> int:
        return self._message_seq

    def wrap_manager(self, manager: Any) -> Any:
        """
        Wrap a GroupChatManager with tracing.

        Args:
            manager: An AutoGen GroupChatManager instance

        Returns:
            The wrapped manager (same object, modified in-place)
        """
        if hasattr(manager, "run_chat"):
            self._original_run_chat = manager.run_chat
            manager.run_chat = self._create_traced_run_chat(
                manager, manager.run_chat
            )
        manager._stratix_tracer = self
        return manager

    def on_speaker_selected(
        self,
        method: str | None = None,
        candidates: list[str] | None = None,
        chosen: str | None = None,
    ) -> None:
        """
        Record a speaker selection event.

        Emits agent.code (L2) dict event for the selection.
        """
        try:
            self._adapter.emit_dict_event("agent.code", {
                "framework": "autogen",
                "event_subtype": "speaker_selection",
                "method": method,
                "candidates": candidates,
                "chosen": chosen,
                "message_seq": self._message_seq,
            })
        except Exception:
            logger.warning("Error emitting speaker selection event", exc_info=True)

    def on_message_routed(
        self,
        from_agent: str,
        to_agent: str,
        message: Any = None,
    ) -> None:
        """
        Record a message routing event.

        Emits agent.handoff (cross-cutting).
        """
        with self._lock:
            self._message_seq += 1
            msg_seq = self._message_seq
        try:
            self._adapter.emit_dict_event("agent.handoff", {
                "framework": "autogen",
                "from_agent": from_agent,
                "to_agent": to_agent,
                "reason": "groupchat_routing",
                "message_seq": msg_seq,
            })
        except Exception:
            logger.warning("Error emitting message routing event", exc_info=True)

    def on_termination(
        self,
        reason: str | None = None,
        final_speaker: str | None = None,
    ) -> None:
        """
        Record conversation termination.

        Emits agent.output (L1).
        """
        try:
            self._adapter.emit_dict_event("agent.output", {
                "framework": "autogen",
                "event_subtype": "groupchat_termination",
                "termination_reason": reason,
                "final_speaker": final_speaker,
                "total_messages": self._message_seq,
            })
        except Exception:
            logger.warning("Error emitting termination event", exc_info=True)

    def _create_traced_run_chat(
        self,
        manager: Any,
        original: Callable,
    ) -> Callable:
        """Create a traced version of run_chat."""
        tracer = self

        def traced_run_chat(*args: Any, **kwargs: Any) -> Any:
            start_ns = time.time_ns()

            try:
                tracer._adapter.emit_dict_event("agent.input", {
                    "framework": "autogen",
                    "event_subtype": "groupchat_start",
                    "timestamp_ns": start_ns,
                })
            except Exception:
                logger.warning("Error emitting groupchat start", exc_info=True)

            result = original(*args, **kwargs)

            try:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                tracer.on_termination(
                    reason="run_chat_complete",
                    final_speaker=None,
                )
            except Exception:
                logger.warning("Error emitting groupchat end", exc_info=True)

            return result

        traced_run_chat._stratix_original = original
        return traced_run_chat
