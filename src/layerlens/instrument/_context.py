"""
STRATIX Context Propagation

Provides thread-local and async-local context for trace propagation.

From Step 4 specification:
- Initialization MUST bind a tracer context (thread-local / async-local)
- Decorators and hooks do not require manual propagation
- Context propagation works across async
"""

from __future__ import annotations

import contextvars
import threading
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from layerlens.instrument._core import STRATIX

from layerlens.instrument.schema.identity import VectorClock, SequenceIdAllocator


# Context variable for async/thread-local context
_stratix_context: contextvars.ContextVar["STRATIXContext | None"] = contextvars.ContextVar(
    "stratix_context", default=None
)


class STRATIXContext:
    """
    Context for STRATIX trace propagation.

    Maintains:
    - Current evaluation/trial/trace IDs
    - Current span ID and parent span ID stack
    - Sequence ID allocator (monotonic per agent)
    - Vector clock state
    """

    def __init__(
        self,
        stratix: "STRATIX",
        evaluation_id: str | None = None,
        trial_id: str | None = None,
        trace_id: str | None = None,
    ):
        """
        Initialize context.

        Args:
            stratix: The parent STRATIX instance
            evaluation_id: Evaluation ID (generated if not provided)
            trial_id: Trial ID (generated if not provided)
            trace_id: Trace ID (generated if not provided)
        """
        self._stratix = stratix
        self._evaluation_id = evaluation_id or str(uuid.uuid4())
        self._trial_id = trial_id or str(uuid.uuid4())
        self._trace_id = trace_id or str(uuid.uuid4())
        self._span_stack: list[str] = []  # Stack of span IDs
        self._current_span_id: str | None = None
        self._sequence_allocator = SequenceIdAllocator()
        self._vector_clock = VectorClock.empty()
        self._lock = threading.Lock()

    @property
    def stratix(self) -> "STRATIX":
        """Get the parent STRATIX instance."""
        return self._stratix

    @property
    def evaluation_id(self) -> str:
        """Get the current evaluation ID."""
        return self._evaluation_id

    @property
    def trial_id(self) -> str:
        """Get the current trial ID."""
        return self._trial_id

    @property
    def trace_id(self) -> str:
        """Get the current trace ID."""
        return self._trace_id

    @property
    def current_span_id(self) -> str | None:
        """Get the current span ID."""
        return self._current_span_id

    @property
    def parent_span_id(self) -> str | None:
        """Get the parent span ID (top of the stack, current span's parent)."""
        if self._span_stack:
            return self._span_stack[-1]
        return None

    @property
    def sequence_id(self) -> int:
        """Get the current sequence ID."""
        return self._sequence_allocator.current

    @property
    def vector_clock(self) -> VectorClock:
        """Get the current vector clock."""
        return self._vector_clock

    def next_sequence_id(self) -> int:
        """Allocate the next sequence ID."""
        return self._sequence_allocator.next()

    def increment_vector_clock(self, participant_id: str | None = None) -> VectorClock:
        """
        Increment the vector clock.

        Args:
            participant_id: Participant ID (defaults to agent ID)

        Returns:
            Updated vector clock
        """
        pid = participant_id or f"agent:{self._stratix.agent_id}"
        with self._lock:
            self._vector_clock = self._vector_clock.increment(pid)
            return self._vector_clock

    def merge_vector_clock(self, other: VectorClock) -> VectorClock:
        """
        Merge with another vector clock (for handoffs/receives).

        Args:
            other: The remote vector clock

        Returns:
            Merged vector clock
        """
        pid = f"agent:{self._stratix.agent_id}"
        with self._lock:
            self._vector_clock = self._vector_clock.merge(other).increment(pid)
            return self._vector_clock

    def start_span(self, span_id: str | None = None) -> str:
        """
        Start a new span.

        Args:
            span_id: Span ID (generated if not provided)

        Returns:
            The new span ID
        """
        new_span_id = span_id or str(uuid.uuid4())
        with self._lock:
            if self._current_span_id is not None:
                self._span_stack.append(self._current_span_id)
            self._current_span_id = new_span_id
        return new_span_id

    def end_span(self) -> str | None:
        """
        End the current span and return to parent.

        Returns:
            The ended span ID, or None if no span was active
        """
        with self._lock:
            ended = self._current_span_id
            if self._span_stack:
                self._current_span_id = self._span_stack.pop()
            else:
                self._current_span_id = None
            return ended

    def create_child_context(self) -> "STRATIXContext":
        """
        Create a child context for nested operations.

        The child shares the same evaluation/trial/trace but has
        its own span stack.
        """
        child = STRATIXContext(
            stratix=self._stratix,
            evaluation_id=self._evaluation_id,
            trial_id=self._trial_id,
            trace_id=self._trace_id,
        )
        # Copy vector clock state
        child._vector_clock = self._vector_clock
        # Set parent span
        if self._current_span_id:
            child._span_stack.append(self._current_span_id)
        return child

    def to_dict(self) -> dict[str, Any]:
        """Serialize context for propagation."""
        return {
            "evaluation_id": self._evaluation_id,
            "trial_id": self._trial_id,
            "trace_id": self._trace_id,
            "span_id": self._current_span_id,
            "parent_span_id": self.parent_span_id,
            "sequence_id": self.sequence_id,
            "vector_clock": self._vector_clock.model_dump(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], stratix: "STRATIX") -> "STRATIXContext":
        """
        Restore context from serialized data.

        Args:
            data: Serialized context data
            stratix: The STRATIX instance

        Returns:
            Restored context
        """
        ctx = cls(
            stratix=stratix,
            evaluation_id=data.get("evaluation_id"),
            trial_id=data.get("trial_id"),
            trace_id=data.get("trace_id"),
        )
        if data.get("parent_span_id"):
            ctx._span_stack.append(data["parent_span_id"])
        if data.get("span_id"):
            ctx._current_span_id = data["span_id"]
        if data.get("vector_clock"):
            ctx._vector_clock = VectorClock(clock=data["vector_clock"])
        return ctx


def get_current_context() -> STRATIXContext | None:
    """
    Get the current STRATIX context.

    Returns:
        The current context, or None if not in an STRATIX context
    """
    return _stratix_context.get()


def set_current_context(ctx: STRATIXContext | None) -> contextvars.Token:
    """
    Set the current STRATIX context.

    Args:
        ctx: The context to set

    Returns:
        Token for restoring the previous context
    """
    return _stratix_context.set(ctx)


def reset_context(token: contextvars.Token) -> None:
    """
    Reset the context to a previous state.

    Args:
        token: Token from set_current_context
    """
    _stratix_context.reset(token)


class context_scope:
    """
    Context manager for scoped context.

    Usage:
        with context_scope(ctx):
            # ctx is active here
    """

    def __init__(self, ctx: STRATIXContext):
        self._ctx = ctx
        self._token: contextvars.Token | None = None

    def __enter__(self) -> STRATIXContext:
        self._token = set_current_context(self._ctx)
        return self._ctx

    def __exit__(self, *args) -> None:
        if self._token is not None:
            reset_context(self._token)

    async def __aenter__(self) -> STRATIXContext:
        return self.__enter__()

    async def __aexit__(self, *args) -> None:
        self.__exit__(*args)
