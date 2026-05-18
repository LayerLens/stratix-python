"""Agent-to-agent handoff detection.

Used by multi-agent framework adapters (LangGraph today; CrewAI / OpenAI
Agents already detect handoffs natively) to emit ``agent.handoff`` events
when a workflow transitions from one named agent / node to another.

This is a thin, framework-agnostic helper — the framework adapter feeds
it the "next agent" each time control changes and the detector decides
whether that constitutes a handoff. ``agent.handoff`` is in
``_ALWAYS_ENABLED`` in :mod:`~layerlens.instrument._capture_config`, so
no layer gating is required.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

_TRUNCATE_AT = 500
_LIST_THRESHOLD = 10
_INTERESTING_KEYS = ("task", "current_task", "objective", "query", "messages", "next")


def scrub_context(state: Any) -> Dict[str, Any]:
    """Return a small, JSON-friendly summary of a state dict.

    - Picks a fixed allow-list of interesting keys (task / messages / etc.).
    - Truncates long strings to 500 chars.
    - Replaces long lists with a ``"[N items]"`` placeholder.
    - Returns ``{}`` if *state* is not a dict.
    """
    if not isinstance(state, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in _INTERESTING_KEYS:
        if key not in state:
            continue
        value = state[key]
        if isinstance(value, str) and len(value) > _TRUNCATE_AT:
            out[key] = value[:_TRUNCATE_AT] + "..."
        elif isinstance(value, list) and len(value) > _LIST_THRESHOLD:
            out[key] = f"[{len(value)} items]"
        else:
            out[key] = value
    return out


class HandoffDetector:
    """Stateful tracker that emits ``agent.handoff`` on agent transitions.

    Usage::

        detector = HandoffDetector()
        detector.set_current_agent("supervisor")
        ...
        # When the workflow routes to a new agent:
        detector.detect("researcher", context=state)  # emits handoff
        detector.detect("researcher", context=state)  # no-op (same agent)
        detector.detect("writer", context=state)  # emits handoff

    The detector calls ``_emit_handoff`` which routes the event through the
    currently active :class:`TraceCollector`. If no collector is active the
    event is silently dropped.
    """

    def __init__(self) -> None:
        self._current_agent: Optional[str] = None

    @property
    def current_agent(self) -> Optional[str]:
        return self._current_agent

    def set_current_agent(self, name: Optional[str]) -> None:
        """Seed the tracker with the agent that's currently running."""
        self._current_agent = name

    def reset(self) -> None:
        self._current_agent = None

    def detect(
        self,
        next_agent: str,
        *,
        context: Any = None,
        reason: Optional[str] = None,
        parent_span_id: Optional[str] = None,
    ) -> bool:
        """Record that control has moved to ``next_agent``.

        Returns ``True`` if a handoff was detected and emitted, ``False``
        if this was either the first agent observed or a re-entry into the
        same agent.
        """
        prev = self._current_agent
        if prev is None or prev == next_agent:
            self._current_agent = next_agent
            return False

        self._current_agent = next_agent
        _emit_handoff(
            from_agent=prev,
            to_agent=next_agent,
            context=context,
            reason=reason,
            parent_span_id=parent_span_id,
        )
        return True


# ----------------------------------------------------------------------
# Event emission
# ----------------------------------------------------------------------


def _emit_handoff(
    *,
    from_agent: str,
    to_agent: str,
    context: Any = None,
    reason: Optional[str] = None,
    parent_span_id: Optional[str] = None,
) -> None:
    """Emit an ``agent.handoff`` event into the active collector.

    No-op when no collector is active. Context (if any) is scrubbed and
    hashed; the hash matches the format used by the attestation chain.
    """
    # Imports kept local so this module stays cheap to import.
    import uuid

    from ..._context import _current_span_id, _current_collector
    from ....attestation._hash import compute_hash

    collector = _current_collector.get()
    if collector is None:
        return

    payload: Dict[str, Any] = {
        "from_agent": from_agent,
        "to_agent": to_agent,
        "timestamp_ns": time.time_ns(),
    }
    if reason:
        payload["reason"] = reason
    if context is not None:
        scrubbed = scrub_context(context)
        if scrubbed:
            try:
                payload["handoff_context_hash"] = compute_hash(scrubbed)
            except TypeError:
                payload["handoff_context_hash"] = compute_hash({"_repr": repr(scrubbed)})
            payload["context"] = scrubbed

    collector.emit(
        "agent.handoff",
        payload,
        span_id=uuid.uuid4().hex[:16],
        parent_span_id=parent_span_id or _current_span_id.get(),
    )
