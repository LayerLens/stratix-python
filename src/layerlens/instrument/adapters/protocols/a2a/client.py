"""
A2A Client-Side Wrapper

Returns a traced A2A client that instruments outgoing task submissions
and receives streamed updates.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class A2AClientWrapper:
    """
    Wraps an A2A client to trace outgoing task operations.

    All task submissions, cancellations, and subscription events are
    captured and emitted through the adapter.
    """

    def __init__(self, adapter: Any, target_url: str) -> None:
        self._adapter = adapter
        self._target_url = target_url

    def send_task(
        self,
        task_id: str,
        messages: list[dict[str, Any]],
        *,
        task_type: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        """
        Trace an outgoing tasks/send call.

        Args:
            task_id: A2A task identifier.
            messages: Task messages.
            task_type: Optional task type from skill definition.
            agent_id: Submitting agent ID.
        """
        self._adapter.on_task_submitted(
            task_id=task_id,
            receiver_url=self._target_url,
            task_type=task_type,
            submitter_agent_id=agent_id,
            message_role="user",
        )

    def complete_task(
        self,
        task_id: str,
        status: str,
        *,
        artifacts: list[dict[str, Any]] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """
        Trace task completion.

        Args:
            task_id: A2A task identifier.
            status: Terminal status (completed, failed, cancelled).
            artifacts: Output artifacts.
            error_code: Error code if failed.
            error_message: Error message if failed.
        """
        self._adapter.on_task_completed(
            task_id=task_id,
            final_status=status,
            artifacts=artifacts,
            error_code=error_code,
            error_message=error_message,
        )

    def delegate_task(
        self,
        from_agent: str,
        to_agent: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Trace an A2A task delegation (handoff)."""
        self._adapter.on_task_delegation(
            from_agent=from_agent,
            to_agent=to_agent,
            context=context,
        )
