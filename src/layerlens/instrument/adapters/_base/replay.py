"""LayerLens shared replay re-execution executor.

Cross-adapter infrastructure that re-executes a captured
:class:`ReplayableTrace` through a *fresh* agent constructed by a
caller-provided factory. The shared executor:

* Provides a uniform :class:`ReplayResult` shape across all adapters
  (outputs, captured events, divergence points).
* Surfaces honest divergence detection — when re-execution cannot
  reproduce the original event sequence it reports
  :class:`ReplayDivergence` records rather than silently ignoring
  the mismatch (CLAUDE.md: "Honest divergence detection — if
  replay can't reproduce exactly, surface it").
* Routes adapter-specific stub injection through the
  :meth:`StubInjector.install_stubs` hook so each framework can
  intercept its own LLM/tool layer (LangChain runs callbacks, agno
  wraps ``Agent.run``, OpenAI Agents installs a TraceProcessor, etc.).
* Supports both synchronous and asynchronous agent factories — the
  factory is duck-typed by inspecting the return value, so a callable
  that returns a coroutine triggers the async branch automatically.

The executor is **NOT** a replacement for the existing
:meth:`BaseAdapter.execute_replay` async hook used by the replay
engine integration. That hook keeps its established signature for
engine compatibility. The factory-based path lives alongside it under
:meth:`BaseAdapter.execute_replay_via_factory` so cross-pollination
audit item §2.6 can land without breaking the engine contract.

Multi-tenancy: every replay executes on an adapter that is bound to
exactly one tenant. The :class:`ReplayResult` carries that ``org_id``
forward so downstream consumers (sinks, dashboards, audit logs) never
have to re-resolve the tenant after a replay completes.
"""

from __future__ import annotations

import time
import uuid
import inspect
import logging
from abc import ABC
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union, Optional
from contextlib import contextmanager
from dataclasses import field, dataclass
from unittest.mock import patch
from collections.abc import Callable, Iterator, Awaitable

if TYPE_CHECKING:
    from layerlens.instrument.adapters._base.adapter import (
        BaseAdapter,
        ReplayableTrace,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result + divergence dataclasses
# ---------------------------------------------------------------------------


class DivergenceKind(str, Enum):
    """Categorical reason a replay event differs from the original.

    Surfaced on :attr:`ReplayDivergence.kind` so dashboards can route
    extra-event divergences differently from missing-event ones.
    """

    MISSING_EVENT = "missing_event"
    """The original trace contained an event that the replay did not emit."""

    EXTRA_EVENT = "extra_event"
    """The replay emitted an event that the original trace did not contain."""

    EVENT_TYPE_MISMATCH = "event_type_mismatch"
    """Same position in the sequence but different event_type."""

    PAYLOAD_MISMATCH = "payload_mismatch"
    """Same event_type at the same position but mismatched payload field(s)."""

    EXECUTION_ERROR = "execution_error"
    """The replay raised before producing a comparable trace."""


@dataclass
class ReplayDivergence:
    """A single point of divergence between replay and original trace.

    The executor records one :class:`ReplayDivergence` per detected
    mismatch so callers can present a complete divergence report
    rather than failing fast on the first miss.

    Attributes:
        kind: Categorical reason for the divergence (see
            :class:`DivergenceKind`).
        index: Zero-based position in the original event sequence
            where the divergence was detected.
        event_type: The event type at the divergence point. For
            :class:`DivergenceKind.MISSING_EVENT` this is the original
            event's type; for :class:`DivergenceKind.EXTRA_EVENT` it is
            the replay event's type.
        original: The original event payload at this position (``None``
            for ``EXTRA_EVENT`` divergences).
        replay: The replay event payload at this position (``None`` for
            ``MISSING_EVENT`` divergences).
        detail: Free-form human-readable detail explaining what
            differed (e.g. ``"prompts list length differs: 1 vs 2"``).
    """

    kind: DivergenceKind
    index: int
    event_type: Optional[str] = None
    original: Optional[Dict[str, Any]] = None
    replay: Optional[Dict[str, Any]] = None
    detail: Optional[str] = None


@dataclass
class ReplayResult:
    """Outcome of a single replay execution.

    A :class:`ReplayResult` always populates :attr:`org_id` (multi-tenant
    binding propagated from the adapter), :attr:`trace_id` (replay's
    own id), :attr:`source_trace_id` (the original trace being
    replayed), and :attr:`captured_events`. :attr:`divergences` is
    empty on a perfect replay; non-empty otherwise.

    The executor never raises on divergence — divergence is data, not
    a failure. The executor *does* re-raise on framework execution
    errors (the agent factory or agent invocation itself blew up);
    callers that want to swallow execution errors can wrap the call.

    Attributes:
        trace_id: Unique id assigned to this replay run.
        source_trace_id: ``trace_id`` of the :class:`ReplayableTrace`
            that drove this replay.
        org_id: Tenant binding inherited from the adapter that ran
            the replay.
        framework: Framework name (matches the originating adapter's
            ``FRAMEWORK`` class attribute).
        outputs: Final agent outputs returned by the replayed agent.
            Shape is framework-specific.
        captured_events: Ordered event records produced by the replay
            adapter while re-executing — same shape as
            ``adapter._trace_events``.
        divergences: Per-divergence-point records computed by
            :meth:`ReplayExecutor._compute_divergences`. Empty when
            the replay reproduced the original trace exactly.
        duration_ns: Wall-clock duration of the replay in nanoseconds.
        execution_error: ``None`` on success; populated with a string
            description when the replay raised during execution. When
            populated the result also carries a single
            :class:`DivergenceKind.EXECUTION_ERROR` divergence and
            :attr:`captured_events` may be partial.
    """

    trace_id: str
    source_trace_id: str
    org_id: str
    framework: str
    outputs: Any = None
    captured_events: List[Dict[str, Any]] = field(default_factory=list)
    divergences: List[ReplayDivergence] = field(default_factory=list)
    duration_ns: int = 0
    execution_error: Optional[str] = None

    @property
    def is_exact(self) -> bool:
        """``True`` when no divergences were recorded."""
        return not self.divergences

    @property
    def succeeded(self) -> bool:
        """``True`` when the replay completed without a framework error.

        A succeeded replay may still carry divergences. ``succeeded``
        and :attr:`is_exact` together describe the two orthogonal
        outcome dimensions:

        * succeeded + is_exact: perfect reproduction
        * succeeded + not is_exact: replay ran but diverged
        * not succeeded: framework error during replay
        """
        return self.execution_error is None


# ---------------------------------------------------------------------------
# StubInjector ABC — adapter-specific intercepts
# ---------------------------------------------------------------------------


# Each stub patch is a (target, replacement) tuple that
# unittest.mock.patch can apply via patch(target, new=replacement).
StubPatch = Tuple[str, Any]


class StubInjector(ABC):  # noqa: B024 - intentionally abstract via subclass override, not @abstractmethod
    """Adapter-specific strategy for replacing LLM/tool calls during replay.

    Each framework intercepts the LLM/tool layer differently:

    * LangChain replaces the ``ChatModel.invoke`` callbacks pipeline.
    * Agno wraps ``Agent.run`` and reads results.
    * OpenAI Agents installs a TraceProcessor.
    * Bedrock Agents calls the Bedrock runtime directly.

    A :class:`StubInjector` returns the list of ``(target, replacement)``
    patches that the executor should apply for the duration of a single
    replay. The executor handles ``patch`` lifecycle (entering and
    exiting all patches in a stack); the injector only declares
    *what* to stub.

    The default :meth:`build_patches` returns an empty list — adapters
    that have no stubs to install (e.g. when the original trace did
    not contain LLM events) work without subclassing.
    """

    def build_patches(
        self,
        adapter: "BaseAdapter",  # noqa: ARG002 - subclass hook signature
        trace: "ReplayableTrace",  # noqa: ARG002 - subclass hook signature
    ) -> List[StubPatch]:
        """Return the patches to apply for one replay run.

        Default implementation returns an empty list. Adapters that
        need to stub LLM/tool calls override this method.

        Args:
            adapter: The adapter that owns the replay.
            trace: The :class:`ReplayableTrace` being replayed —
                consult ``trace.events`` to scope stubs to the LLM
                calls actually present in the original.

        Returns:
            A list of ``(target, replacement)`` tuples passed to
            ``unittest.mock.patch``.
        """
        return []


class _NullStubInjector(StubInjector):
    """Default stub injector that installs no patches."""


_NULL_STUB_INJECTOR = _NullStubInjector()


# ---------------------------------------------------------------------------
# ReplayExecutor — shared cross-adapter executor
# ---------------------------------------------------------------------------


# A factory either returns an agent synchronously or a coroutine that
# resolves to one. Both shapes are supported by the executor.
AgentFactory = Callable[[], Union[Any, Awaitable[Any]]]


class ReplayExecutor:
    """Shared cross-adapter replay re-execution engine.

    Run one :class:`ReplayableTrace` through a fresh agent built by a
    caller-supplied factory, capture the resulting events through the
    adapter, and report divergences.

    The executor is intentionally narrow: it does NOT know how to
    *invoke* a framework agent (different frameworks use ``run``,
    ``arun``, ``invoke``, ``ainvoke``, ``Runner.run``, etc.). The
    invocation is delegated to ``adapter._invoke_for_replay`` which
    each adapter overrides.

    Construction parameters live on the executor (not on each call) so
    a single executor can drive many replays of the same adapter:

        executor = ReplayExecutor(adapter, stub_injector=MyStubs())
        result = await executor.execute_replay(trace, my_factory)
    """

    def __init__(
        self,
        adapter: "BaseAdapter",
        stub_injector: Optional[StubInjector] = None,
    ) -> None:
        """Bind the executor to one adapter.

        Args:
            adapter: The adapter that will receive the replay's events.
                Its ``org_id`` is propagated onto the
                :class:`ReplayResult`.
            stub_injector: Optional adapter-specific stub strategy.
                Defaults to :class:`_NullStubInjector` (no stubs
                installed — useful for replays whose original trace
                contained no LLM/tool calls).
        """
        self._adapter = adapter
        self._stub_injector = stub_injector or _NULL_STUB_INJECTOR

    # --- Public API -------------------------------------------------------

    async def execute_replay(
        self,
        trace: "ReplayableTrace",
        agent_factory: AgentFactory,
    ) -> ReplayResult:
        """Re-execute ``trace`` through a fresh agent built by ``agent_factory``.

        Steps:

        1. Snapshot the adapter's ``_trace_events`` length so post-run
           events can be isolated even when the same adapter served
           prior runs.
        2. Install adapter-specific stubs via ``self._stub_injector``.
        3. Build a fresh agent — accepts both sync and async factories.
        4. Invoke the agent through the adapter-specific
           :meth:`_invoke_agent` hook, replaying the original inputs
           extracted from the first ``agent.input`` event.
        5. Collect captured events, compute divergences, build the
           :class:`ReplayResult`.

        Framework execution errors are captured into
        :attr:`ReplayResult.execution_error` rather than re-raised so
        downstream replay-batch consumers can collect partial results.
        Errors during stub teardown are logged at WARNING and never
        mask the original execution outcome.

        Args:
            trace: The :class:`ReplayableTrace` to replay.
            agent_factory: Callable that returns either an agent
                instance or an awaitable resolving to one. The factory
                is invoked exactly once per replay.

        Returns:
            A :class:`ReplayResult` carrying outputs, captured events,
            and any detected divergences.
        """
        adapter = self._adapter
        replay_trace_id = str(uuid.uuid4())
        events_before = len(adapter._trace_events)
        start_ns = time.time_ns()

        execution_error: Optional[str] = None
        outputs: Any = None
        agent: Any = None

        patches = self._stub_injector.build_patches(adapter, trace)
        with self._apply_patches(patches):
            try:
                agent = await self._build_agent(agent_factory)
                inputs = self._extract_inputs(trace)
                outputs = await self._invoke_agent(agent, inputs, trace)
            except Exception as exc:
                execution_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Replay execution failed for adapter %s trace %s: %s",
                    adapter.FRAMEWORK,
                    trace.trace_id,
                    execution_error,
                    exc_info=True,
                )

        duration_ns = time.time_ns() - start_ns
        captured = list(adapter._trace_events[events_before:])

        divergences = self._compute_divergences(trace, captured, execution_error)

        return ReplayResult(
            trace_id=replay_trace_id,
            source_trace_id=trace.trace_id,
            org_id=adapter.org_id,
            framework=adapter.FRAMEWORK,
            outputs=outputs,
            captured_events=captured,
            divergences=divergences,
            duration_ns=duration_ns,
            execution_error=execution_error,
        )

    # --- Hooks adapters override (or call through) ------------------------

    async def _invoke_agent(
        self,
        agent: Any,
        inputs: Any,
        trace: "ReplayableTrace",
    ) -> Any:
        """Invoke ``agent`` with ``inputs`` and return its output.

        Default implementation duck-types the agent against the
        common framework method shapes (``arun``, ``ainvoke``,
        ``run``, ``invoke``, plain callable). Adapters override
        :meth:`BaseAdapter._invoke_for_replay` to handle framework-
        specific shapes (e.g. OpenAI Agents' ``Runner.run(agent,
        input)`` static call).

        Args:
            agent: The agent built by the factory.
            inputs: Inputs extracted from the original trace.
            trace: The original trace (for context-aware adapters).

        Returns:
            Whatever the agent returns. Stored on
            :attr:`ReplayResult.outputs` unchanged.
        """
        adapter_hook = getattr(self._adapter, "_invoke_for_replay", None)
        if adapter_hook is not None:
            result = adapter_hook(agent, inputs, trace)
            # The base class default is to return ``NotImplemented`` so
            # the shared duck-typed fallback below kicks in. Anything
            # else (including ``None`` from an agent that genuinely
            # produces no output) is treated as a real result.
            if result is not NotImplemented:
                if inspect.isawaitable(result):
                    return await result
                return result

        # Generic duck-typing fallback.
        for async_method in ("arun", "ainvoke", "acall"):
            method = getattr(agent, async_method, None)
            if callable(method):
                return await method(inputs)
        for sync_method in ("run", "invoke", "call"):
            method = getattr(agent, sync_method, None)
            if callable(method):
                return method(inputs)
        if callable(agent):
            return agent(inputs)
        raise RuntimeError(
            f"ReplayExecutor cannot invoke agent of type {type(agent).__name__}: "
            "no run/arun/invoke/ainvoke/call/acall method found and the agent "
            "itself is not callable. Override _invoke_for_replay on the adapter."
        )

    # --- Internals --------------------------------------------------------

    @staticmethod
    async def _build_agent(factory: AgentFactory) -> Any:
        """Invoke ``factory`` and resolve coroutines transparently."""
        result: Any = factory()
        if inspect.isawaitable(result):
            return await result
        return result

    @staticmethod
    def _extract_inputs(trace: "ReplayableTrace") -> Any:
        """Pull the original input from the trace's first ``agent.input`` event.

        Returns the ``input`` payload field if present, otherwise an
        empty dict. Adapters that need a different extraction policy
        override :meth:`BaseAdapter._extract_replay_inputs`.
        """
        for event in trace.events:
            event_type = event.get("event_type") if isinstance(event, dict) else None
            if event_type == "agent.input":
                payload = event.get("payload", {}) if isinstance(event, dict) else {}
                if isinstance(payload, dict):
                    candidate = payload.get("input")
                    if candidate is not None:
                        return candidate
                    return payload
        return {}

    @contextmanager
    def _apply_patches(self, patches: List[StubPatch]) -> Iterator[None]:
        """Apply every patch and tear down on exit, in reverse order."""
        applied: List[Any] = []
        try:
            for target, replacement in patches:
                p = patch(target, new=replacement)
                p.start()
                applied.append(p)
            yield
        finally:
            for p in reversed(applied):
                try:
                    p.stop()
                except Exception:
                    logger.warning(
                        "ReplayExecutor stub teardown failed for %r",
                        p,
                        exc_info=True,
                    )

    @staticmethod
    def _compute_divergences(
        trace: "ReplayableTrace",
        captured: List[Dict[str, Any]],
        execution_error: Optional[str],
    ) -> List[ReplayDivergence]:
        """Compute :class:`ReplayDivergence` records for one replay run.

        When ``execution_error`` is set, returns a single
        :class:`DivergenceKind.EXECUTION_ERROR` record so callers can
        unambiguously distinguish a framework crash from a soft trace
        mismatch.

        For successful runs, divergences are computed pairwise on the
        ordered ``event_type`` sequence:

        * Length mismatch → trailing events recorded as
          :class:`DivergenceKind.MISSING_EVENT` (original longer) or
          :class:`DivergenceKind.EXTRA_EVENT` (replay longer).
        * Same position with different ``event_type`` →
          :class:`DivergenceKind.EVENT_TYPE_MISMATCH`.
        * Same ``event_type`` but a meaningful payload field differs
          (currently: ``model`` / ``provider`` / ``tool_name``) →
          :class:`DivergenceKind.PAYLOAD_MISMATCH`.
        """
        if execution_error is not None:
            return [
                ReplayDivergence(
                    kind=DivergenceKind.EXECUTION_ERROR,
                    index=0,
                    detail=execution_error,
                )
            ]

        original_events = list(trace.events)
        divergences: List[ReplayDivergence] = []

        common_len = min(len(original_events), len(captured))
        for i in range(common_len):
            o_evt = original_events[i] if isinstance(original_events[i], dict) else {}
            r_evt = captured[i] if isinstance(captured[i], dict) else {}
            o_type = o_evt.get("event_type")
            r_type = r_evt.get("event_type")
            if o_type != r_type:
                divergences.append(
                    ReplayDivergence(
                        kind=DivergenceKind.EVENT_TYPE_MISMATCH,
                        index=i,
                        event_type=str(o_type) if o_type else None,
                        original=o_evt,
                        replay=r_evt,
                        detail=f"original event_type={o_type!r} replay event_type={r_type!r}",
                    )
                )
                continue

            payload_diff = ReplayExecutor._compare_payloads(
                o_evt.get("payload"),
                r_evt.get("payload"),
            )
            if payload_diff is not None:
                divergences.append(
                    ReplayDivergence(
                        kind=DivergenceKind.PAYLOAD_MISMATCH,
                        index=i,
                        event_type=str(o_type) if o_type else None,
                        original=o_evt,
                        replay=r_evt,
                        detail=payload_diff,
                    )
                )

        # Trailing original events that the replay never emitted.
        for i in range(common_len, len(original_events)):
            evt = original_events[i] if isinstance(original_events[i], dict) else {}
            divergences.append(
                ReplayDivergence(
                    kind=DivergenceKind.MISSING_EVENT,
                    index=i,
                    event_type=evt.get("event_type"),
                    original=evt,
                    detail="original trace had this event but replay did not",
                )
            )
        # Trailing replay events the original did not contain.
        for i in range(common_len, len(captured)):
            evt = captured[i] if isinstance(captured[i], dict) else {}
            divergences.append(
                ReplayDivergence(
                    kind=DivergenceKind.EXTRA_EVENT,
                    index=i,
                    event_type=evt.get("event_type"),
                    replay=evt,
                    detail="replay emitted this event but original did not",
                )
            )

        return divergences

    @staticmethod
    def _compare_payloads(
        original: Any,
        replay: Any,
    ) -> Optional[str]:
        """Return a description of a meaningful payload difference, or ``None``.

        Honest divergence detection (CLAUDE.md): we only flag fields
        whose mismatch genuinely indicates the agent did something
        different. Fields like ``timestamp_ns``, ``run_id``, and
        ``duration_ns`` are *expected* to differ between runs and are
        therefore ignored.

        The chosen meaningful fields are the cross-adapter common set:

        * ``model`` and ``provider`` — different LLM was invoked
        * ``tool_name`` — different tool was called
        * ``agent_name`` — different agent ran
        * ``from_agent`` / ``to_agent`` — different handoff
        """
        if not isinstance(original, dict) or not isinstance(replay, dict):
            return None

        meaningful = ("model", "provider", "tool_name", "agent_name", "from_agent", "to_agent")
        diffs: List[str] = []
        for key in meaningful:
            if key not in original and key not in replay:
                continue
            o_val = original.get(key)
            r_val = replay.get(key)
            if o_val != r_val:
                diffs.append(f"{key}: {o_val!r} != {r_val!r}")
        if diffs:
            return "; ".join(diffs)
        return None


__all__ = [
    "AgentFactory",
    "DivergenceKind",
    "ReplayDivergence",
    "ReplayExecutor",
    "ReplayResult",
    "StubInjector",
    "StubPatch",
]
