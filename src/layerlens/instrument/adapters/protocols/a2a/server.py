"""
A2A Server-Side Wrapper

Wraps an A2A-compliant HTTP handler to intercept incoming JSON-RPC
requests and SSE streams for tracing.
"""

from __future__ import annotations

import logging
from typing import Any
from collections.abc import Callable

logger = logging.getLogger(__name__)


class A2AServerWrapper:
    """
    Wraps an A2A server handler to intercept and trace requests.

    Intercepts incoming JSON-RPC requests, extracts task lifecycle
    events, and delegates to the original handler.
    """

    # JSON-RPC methods that map to task lifecycle events
    _TASK_METHODS = frozenset(
        {
            "tasks/send",
            "tasks/sendSubscribe",
            "tasks/get",
            "tasks/cancel",
            "tasks/pushNotification/set",
            "tasks/pushNotification/get",
        }
    )

    def __init__(
        self,
        adapter: Any,
        original_handler: Callable[..., Any] | None = None,
    ) -> None:
        self._adapter = adapter
        self._original_handler = original_handler

    def handle_request(
        self,
        request_body: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """
        Process an incoming A2A JSON-RPC request.

        Extracts task lifecycle information and emits events before
        delegating to the original handler.

        Args:
            request_body: Parsed JSON-RPC request body.
            headers: HTTP headers.

        Returns:
            The response from the original handler, or None.
        """
        method = request_body.get("method", "")
        params = request_body.get("params", {})

        if method == "tasks/send" or method == "tasks/sendSubscribe":
            task = params.get("task", params)
            task_id = task.get("id", request_body.get("id", ""))
            self._adapter.on_task_submitted(
                task_id=str(task_id),
                receiver_url="self",
                raw_payload=request_body,
            )

        if self._original_handler:
            return self._original_handler(request_body)  # type: ignore[no-any-return]
        return None

    def handle_agent_card_request(self) -> dict[str, Any] | None:
        """Handle a request for the agent's Agent Card."""
        # Emit discovery event — the adapter will handle card registration
        return None
