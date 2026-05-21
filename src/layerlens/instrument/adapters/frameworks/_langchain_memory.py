"""LangChain memory tracing.

Wraps a LangChain memory object so each ``save_context`` / ``clear``
call emits an ``agent.state.change`` event whenever the memory's
contents actually change. The trigger is hashed with the same SHA-256
canonical-JSON path as the attestation chain, so before/after hashes
are comparable with other ``agent.state.change`` events emitted by the
LangGraph adapter.

Usage::

    from layerlens.instrument.adapters.frameworks.langchain import wrap_memory

    memory = ConversationBufferMemory(...)
    traced = wrap_memory(memory)

    # use exactly like the original memory
    traced.save_context({"input": "hi"}, {"output": "hello"})

``agent.state.change`` is in ``_ALWAYS_ENABLED`` so emissions bypass
layer-level gating. Events are dropped (no-op) when no
``TraceCollector`` is active.
"""

from __future__ import annotations

import time
import uuid
import logging
from typing import Any, Dict, List, Optional

from ..._context import _current_span_id, _current_collector
from ....attestation._hash import compute_hash

log = logging.getLogger(__name__)


def _hash_memory(memory: Any) -> str:
    """Return ``sha256:<hex>`` over a memory object's loaded variables.

    Falls back to a hash of ``repr(memory)`` if the variables aren't
    JSON-serializable.
    """
    try:
        variables = memory.load_memory_variables({})
    except Exception:
        variables = None

    if variables is None:
        return compute_hash({"_repr": repr(memory)})

    try:
        return compute_hash(variables)
    except TypeError:
        return compute_hash({"_repr": repr(variables)})


def _emit_state_change(
    *,
    memory_type: str,
    before_hash: str,
    after_hash: str,
    trigger: str,
) -> None:
    collector = _current_collector.get()
    if collector is None:
        return
    collector.emit(
        "agent.state.change",
        {
            "framework": "langchain",
            "memory_type": memory_type,
            "before_hash": before_hash,
            "after_hash": after_hash,
            "trigger": trigger,
            "timestamp_ns": time.time_ns(),
        },
        span_id=uuid.uuid4().hex[:16],
        parent_span_id=_current_span_id.get(),
    )


class TracedMemory:
    """Proxy wrapper around a LangChain memory object.

    Intercepts ``save_context`` and ``clear`` to emit
    ``agent.state.change`` events when the memory's loaded variables
    change. All other attribute access is forwarded to the underlying
    memory via ``__getattr__``.
    """

    def __init__(self, memory: Any) -> None:
        self._memory = memory
        self._last_hash: Optional[str] = None

    # ------------------------------------------------------------------
    # Intercepted methods
    # ------------------------------------------------------------------

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        before_hash = _hash_memory(self._memory)
        self._memory.save_context(inputs, outputs)
        after_hash = _hash_memory(self._memory)
        if before_hash != after_hash:
            _emit_state_change(
                memory_type=type(self._memory).__name__,
                before_hash=before_hash,
                after_hash=after_hash,
                trigger="save_context",
            )
        self._last_hash = after_hash

    def clear(self) -> None:
        before_hash = _hash_memory(self._memory)
        self._memory.clear()
        after_hash = _hash_memory(self._memory)
        if before_hash != after_hash:
            _emit_state_change(
                memory_type=type(self._memory).__name__,
                before_hash=before_hash,
                after_hash=after_hash,
                trigger="clear",
            )
        self._last_hash = after_hash

    # ------------------------------------------------------------------
    # Passthrough
    # ------------------------------------------------------------------

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self._memory.load_memory_variables(inputs)

    @property
    def memory_variables(self) -> List[str]:
        return self._memory.memory_variables

    def __getattr__(self, name: str) -> Any:
        # ``__getattr__`` only runs when standard attribute lookup fails,
        # so we won't recurse into ``self._memory`` / ``self._last_hash``.
        return getattr(self._memory, name)


def wrap_memory(memory: Any) -> TracedMemory:
    """Wrap *memory* in a :class:`TracedMemory` proxy."""
    return TracedMemory(memory)


# ----------------------------------------------------------------------
# Mutation tracker — context manager for explicit before/after framing.
# ----------------------------------------------------------------------


class MemoryMutationTracker:
    """Track memory mutations across explicit operation boundaries.

    Useful when ``save_context`` is called outside our control (e.g.
    inside a third-party agent) and you want a single ``agent.state.change``
    event per logical operation rather than per call::

        tracker = MemoryMutationTracker()
        with tracker.track(memory, operation="agent_turn"):
            chain.invoke(...)
    """

    def __init__(self) -> None:
        self._mutations: List[Dict[str, Any]] = []

    def track(self, memory: Any, *, operation: str = "unknown") -> "_MemoryTrackingContext":
        return _MemoryTrackingContext(memory=memory, operation=operation, tracker=self)

    def record_mutation(self, mutation: Dict[str, Any]) -> None:
        self._mutations.append(mutation)

    @property
    def mutations(self) -> List[Dict[str, Any]]:
        return list(self._mutations)

    def clear(self) -> None:
        self._mutations.clear()


class _MemoryTrackingContext:
    def __init__(self, *, memory: Any, operation: str, tracker: MemoryMutationTracker) -> None:
        self._memory = memory
        self._operation = operation
        self._tracker = tracker
        self._before_hash: Optional[str] = None

    def __enter__(self) -> "_MemoryTrackingContext":
        self._before_hash = _hash_memory(self._memory)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._before_hash is None:
            return
        after_hash = _hash_memory(self._memory)
        if after_hash == self._before_hash:
            return
        mutation = {
            "memory_type": type(self._memory).__name__,
            "before_hash": self._before_hash,
            "after_hash": after_hash,
            "operation": self._operation,
            "timestamp_ns": time.time_ns(),
        }
        self._tracker.record_mutation(mutation)
        _emit_state_change(
            memory_type=mutation["memory_type"],
            before_hash=mutation["before_hash"],
            after_hash=mutation["after_hash"],
            trigger=self._operation,
        )
