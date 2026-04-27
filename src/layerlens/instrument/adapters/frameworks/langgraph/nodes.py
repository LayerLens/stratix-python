"""
STRATIX LangGraph Node Tracing

Provides node entry/exit hooks and decorators for tracing node execution.

In addition to the cross-cutting ``agent.state.change`` event emitted on
state diffs, every node execution emits an L2 ``AgentCodeEvent``
(``agent.code``) per spec ``04a-langgraph-adapter-spec.md`` table line
57. The emission carries:

* ``repo`` — module path of the node callable (PII-safe identity).
* ``commit`` — runtime marker (``"runtime"`` when no git context is
  resolvable from the callable; the upstream platform fills the real
  commit at trace finalization).
* ``artifact_hash`` — deterministic SHA-256 of the callable's bytecode
  (or qualified name when no bytecode is available, e.g. C-implemented
  builtins or wrapped callables). The same node executed twice produces
  the same ``artifact_hash`` so replay diff engines can correlate
  per-node artifacts across runs.
* ``config_hash`` — SHA-256 over a deterministic descriptor of the node
  identity (qualified name + parameter names + execution outcome key
  set + duration bucket). Carries no state values — only key NAMES — so
  no PII can leak through the L2 envelope.
* ``build_info`` — structured per-execution metadata (``node_name``,
  ``node_qualname``, ``execution_duration_ns``, ``input_state_keys``,
  ``output_state_keys``, ``status`` ∈ {``"success"``, ``"error"``},
  ``error_class`` when applicable, ``error_truncated`` when applicable).
"""

from __future__ import annotations

import time
import hashlib
import inspect
import logging
from typing import TYPE_CHECKING, Any, TypeVar
from functools import wraps
from dataclasses import field, dataclass
from collections.abc import Callable

from layerlens.instrument.adapters.frameworks.langgraph.state import LangGraphStateAdapter

if TYPE_CHECKING:
    from layerlens.instrument.adapters._base.adapter import BaseAdapter

logger = logging.getLogger(__name__)


StateT = TypeVar("StateT")
NodeFunc = Callable[[StateT], StateT]


# ---------------------------------------------------------------------------
# L2 AgentCodeEvent helpers
# ---------------------------------------------------------------------------

# Truncation budget for stringified error payloads attached to the L2
# envelope. Tracebacks are PII-risky if user input ends up in repr; we
# cap the surface so a runaway value cannot exfiltrate by accident.
_ERROR_TRUNCATION_BYTES = 512

# Sentinel used when the platform cannot resolve a git commit at the
# adapter layer. The trace finalizer / attestation path is responsible
# for replacing this with the real commit when one is available — see
# ``stratix.attestation.commit_resolver``. Emitting ``"runtime"`` here
# keeps the L2 envelope schema-valid while making it obvious downstream
# that the value was synthesized from runtime context, not git.
_RUNTIME_COMMIT_SENTINEL = "runtime"

# Sentinel used when the node callable lives outside any importable
# module (e.g. lambdas or REPL-defined functions). The repo field is
# REQUIRED by the L2 schema, so we surface a stable value instead of
# raising — runtime introspection of unreachable callables is a real
# scenario for ad-hoc graph composition.
_RUNTIME_REPO_SENTINEL = "layerlens.instrument:runtime"


def _zero_hash() -> str:
    """Return a schema-valid placeholder ``sha256:`` hash."""
    return "sha256:" + ("0" * 64)


def _sha256(data: bytes) -> str:
    """Compute a schema-prefixed SHA-256 over ``data``."""
    digest = hashlib.sha256(data).hexdigest()
    return "sha256:" + digest


def _resolve_callable_module(func: Callable[..., Any]) -> str:
    """Resolve a stable module path for ``func``, falling back to a sentinel.

    Uses ``__module__`` when present and importable. Lambdas and
    REPL-defined callables fall back to :data:`_RUNTIME_REPO_SENTINEL`.
    """
    module = getattr(func, "__module__", None)
    if isinstance(module, str) and module:
        return module
    return _RUNTIME_REPO_SENTINEL


def _resolve_callable_qualname(func: Callable[..., Any]) -> str:
    """Resolve a stable qualified name for ``func``.

    Prefers ``__qualname__``; falls back to ``__name__``; falls back to
    ``repr()`` as a last resort. The result is used both as a debug
    identity in ``build_info`` and as input to the artifact / config
    hashes when bytecode is unavailable.
    """
    qualname = getattr(func, "__qualname__", None)
    if isinstance(qualname, str) and qualname:
        return qualname
    name = getattr(func, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return repr(func)


def _resolve_callable_artifact_hash(func: Callable[..., Any]) -> str:
    """Compute a deterministic artifact hash for the node callable.

    Resolution order — each step preserves determinism so that repeated
    executions of the same callable yield the same hash, enabling
    replay-correlation across runs:

    1. ``__code__.co_code`` — CPython bytecode of the underlying
       function. Stable across invocations of the same compiled
       function within a process.
    2. ``inspect.getsource(func)`` — source text. Used when the
       callable wraps a non-Python implementation but the source is
       importable. May raise ``OSError`` for builtins or REPL-defined
       callables; we suppress and fall through.
    3. Qualified name fallback — last resort for callables with no
       bytecode and no resolvable source (C-implemented builtins,
       partially-applied callables, etc.).

    The fallback chain guarantees that ``_resolve_callable_artifact_hash``
    never raises. A deterministic placeholder is preferable to a
    non-emission because the L2 envelope is non-optional per spec.
    """
    # Step 1: bytecode of the wrapped function. ``functools.wraps``
    # exposes the original ``__wrapped__``, so we follow the chain.
    target: Any = getattr(func, "__wrapped__", func)
    code = getattr(target, "__code__", None)
    if code is not None:
        co_bytes = getattr(code, "co_code", None)
        if isinstance(co_bytes, (bytes, bytearray)) and co_bytes:
            return _sha256(bytes(co_bytes))

    # Step 2: source text. ``inspect.getsource`` raises ``OSError`` for
    # builtins and REPL-defined callables; suppress those rather than
    # failing the emission.
    source: str | None
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):
        source = None
    if isinstance(source, str) and source:
        return _sha256(source.encode("utf-8"))

    # Step 3: qualified name fallback. Always resolves.
    return _sha256(_resolve_callable_qualname(target).encode("utf-8"))


def _resolve_callable_signature_descriptor(func: Callable[..., Any]) -> str:
    """Build a stable parameter-name descriptor for ``func``.

    Returns a comma-joined list of parameter names from
    ``inspect.signature``. Returns the empty string if no signature is
    inspectable (e.g. C builtins). Never raises.
    """
    target: Any = getattr(func, "__wrapped__", func)
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return ""
    return ",".join(param.name for param in sig.parameters.values())


def _safe_state_keys(state: Any) -> list[str]:
    """Extract the top-level key NAMES from a state-like object.

    PII-safe: returns key NAMES only, never values. Falls back to an
    empty list for non-mapping states (lists, primitives, frozen
    Pydantic models without dict access).

    Handles dicts directly. For Pydantic models, attempts ``model_dump``
    (v2) / ``dict`` (v1) and pulls keys; failures yield an empty list.
    """
    if isinstance(state, dict):
        return sorted(str(k) for k in state.keys())
    # Pydantic v2
    model_dump = getattr(state, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except Exception:
            dumped = None
        if isinstance(dumped, dict):
            return sorted(str(k) for k in dumped.keys())
    # Pydantic v1 / dataclass-like
    to_dict = getattr(state, "dict", None)
    if callable(to_dict):
        try:
            dumped = to_dict()
        except Exception:
            dumped = None
        if isinstance(dumped, dict):
            return sorted(str(k) for k in dumped.keys())
    return []


def _truncate_error(err: BaseException) -> tuple[str, str, bool]:
    """Return ``(error_class, truncated_repr, was_truncated)`` for ``err``.

    The repr is capped at :data:`_ERROR_TRUNCATION_BYTES` to bound the
    PII / payload-size risk of attaching arbitrary user-formatted error
    messages to the L2 envelope.
    """
    error_class = type(err).__name__
    repr_str = str(err)
    encoded = repr_str.encode("utf-8", errors="replace")
    if len(encoded) <= _ERROR_TRUNCATION_BYTES:
        return error_class, repr_str, False
    truncated = encoded[:_ERROR_TRUNCATION_BYTES].decode("utf-8", errors="ignore")
    return error_class, truncated, True


# ---------------------------------------------------------------------------
# Node execution tracking
# ---------------------------------------------------------------------------


@dataclass
class NodeExecution:
    """Tracks a single node execution.

    Captures the identity of the node callable plus the boundary state
    hashes. The ``node_callable`` is preserved so the L2
    :class:`AgentCodeEvent` emission can derive deterministic artifact /
    config hashes from the same source on every run.
    """

    node_name: str
    start_time_ns: int
    end_time_ns: int | None = None
    state_hash_before: str | None = None
    state_hash_after: str | None = None
    error: str | None = None
    node_callable: Callable[..., Any] | None = None
    input_state_keys: list[str] = field(default_factory=list)
    output_state_keys: list[str] = field(default_factory=list)
    error_class: str | None = None
    error_truncated: bool = False


class NodeTracer:
    """
    Tracer for LangGraph node executions.

    Provides hooks for node entry/exit and automatic state change detection.

    On every node execution (success or failure) the tracer also emits
    an L2 :class:`AgentCodeEvent` per spec
    ``04a-langgraph-adapter-spec.md`` table line 57. The event carries
    the node callable's identity-derived hash so per-node artifact
    attestation and replay correlation are possible — see this module's
    docstring for the full field semantics.

    Usage:
        tracer = NodeTracer(stratix_instance)

        # Manual tracking
        with tracer.trace_node("my_node", state):
            # Node logic here
            new_state = process(state)

        # Or use the decorator
        @tracer.decorate
        def my_node(state):
            return process(state)
    """

    def __init__(
        self,
        stratix_instance: Any = None,
        state_adapter: LangGraphStateAdapter | None = None,
        adapter: BaseAdapter | None = None,
    ) -> None:
        """
        Initialize the node tracer.

        Args:
            stratix_instance: STRATIX SDK instance (legacy)
            state_adapter: State adapter for change detection
            adapter: BaseAdapter instance (new-style)
        """
        self._stratix = stratix_instance
        self._adapter = adapter
        self._state_adapter = state_adapter or LangGraphStateAdapter()
        self._executions: list[NodeExecution] = []

    def trace_node(
        self,
        node_name: str,
        state: Any,
        node_callable: Callable[..., Any] | None = None,
    ) -> _NodeContext:
        """
        Create a context manager for tracing a node.

        Args:
            node_name: Name of the node
            state: Current state
            node_callable: The node callable (used to derive the L2
                ``AgentCodeEvent`` identity hashes). Optional — when
                omitted, the L2 envelope is still emitted with sentinel
                identity values rather than being skipped.

        Returns:
            Context manager for node tracing
        """
        return _NodeContext(
            tracer=self,
            node_name=node_name,
            state=state,
            node_callable=node_callable,
        )

    def decorate(self, func: NodeFunc) -> NodeFunc:  # type: ignore[type-arg]
        """
        Decorate a node function with tracing.

        Args:
            func: Node function

        Returns:
            Decorated function
        """
        node_name = func.__name__

        @wraps(func)
        def wrapper(state: StateT) -> StateT:
            with self.trace_node(node_name, state, node_callable=func) as ctx:
                result = func(state)
                ctx.set_result(result)
                return result  # type: ignore[no-any-return]

        return wrapper

    def on_node_enter(
        self,
        node_name: str,
        state: Any,
        node_callable: Callable[..., Any] | None = None,
    ) -> NodeExecution:
        """
        Called when entering a node.

        Captures the entry state hash + the input state key set
        (PII-safe key NAMES only) for inclusion in the L2
        :class:`AgentCodeEvent`.

        Args:
            node_name: Name of the node
            state: Current state
            node_callable: The node callable (used to derive L2 identity
                hashes when the node exits)

        Returns:
            NodeExecution tracking object
        """
        execution = NodeExecution(
            node_name=node_name,
            start_time_ns=time.time_ns(),
            state_hash_before=self._state_adapter.get_hash(state),
            node_callable=node_callable,
            input_state_keys=_safe_state_keys(state),
        )
        self._executions.append(execution)

        return execution

    def on_node_exit(
        self,
        execution: NodeExecution,
        state: Any,
        error: BaseException | None = None,
    ) -> None:
        """
        Called when exiting a node.

        Always emits an L2 :class:`AgentCodeEvent` for the node
        execution — both on the success and the error path — per spec
        ``04a-langgraph-adapter-spec.md`` table line 57. Additionally
        emits the cross-cutting ``agent.state.change`` event when the
        boundary hashes differ.

        Args:
            execution: Execution tracking object
            state: State after node execution
            error: Exception if node failed
        """
        execution.end_time_ns = time.time_ns()
        execution.state_hash_after = self._state_adapter.get_hash(state)
        execution.output_state_keys = _safe_state_keys(state)

        if error is not None:
            error_class, truncated, was_truncated = _truncate_error(error)
            execution.error = truncated
            execution.error_class = error_class
            execution.error_truncated = was_truncated

        # L2: emit AgentCodeEvent on every node execution (success or
        # error). Spec 04a §3 table line 57 requires this regardless of
        # state-change outcome.
        self._emit_agent_code(execution)

        # Cross-cutting: state change only when state actually changed.
        if execution.state_hash_before != execution.state_hash_after:
            self._emit_state_change(execution)

    # --- Event emission ---

    def _emit_agent_code(self, execution: NodeExecution) -> None:
        """Emit an L2 ``AgentCodeEvent`` for a node execution.

        The emission goes through ``adapter.emit_event`` (preferred,
        circuit-breaker + CaptureConfig + multi-tenant ``org_id``
        stamping all apply) when an adapter is attached. Otherwise falls
        back to the legacy ``stratix.emit("agent.code", ...)`` dict
        path.

        On any exception, logs at DEBUG and returns silently — emission
        failures must never break a node execution.
        """
        try:
            from layerlens.instrument._vendored.events import AgentCodeEvent
        except Exception:
            logger.debug(
                "AgentCodeEvent vendored import failed; skipping L2 emission",
                exc_info=True,
            )
            return

        callable_obj = execution.node_callable
        repo = (
            _resolve_callable_module(callable_obj)
            if callable_obj is not None
            else _RUNTIME_REPO_SENTINEL
        )
        artifact_hash = (
            _resolve_callable_artifact_hash(callable_obj)
            if callable_obj is not None
            else _zero_hash()
        )
        signature_descriptor = (
            _resolve_callable_signature_descriptor(callable_obj)
            if callable_obj is not None
            else ""
        )

        duration_ns = (execution.end_time_ns or execution.start_time_ns) - execution.start_time_ns
        status = "error" if execution.error_class is not None else "success"

        # config_hash captures the per-execution identity descriptor:
        # qualified name + signature + status + sorted output key set.
        # State VALUES are never included; only key NAMES, so the hash
        # is PII-safe per the cross-cutting privacy contract.
        qualname = (
            _resolve_callable_qualname(callable_obj)
            if callable_obj is not None
            else execution.node_name
        )
        config_descriptor = "|".join(
            [
                "node_name=" + execution.node_name,
                "qualname=" + qualname,
                "signature=" + signature_descriptor,
                "status=" + status,
                "in_keys=" + ",".join(execution.input_state_keys),
                "out_keys=" + ",".join(execution.output_state_keys),
            ]
        )
        config_hash = _sha256(config_descriptor.encode("utf-8"))

        build_info: dict[str, Any] = {
            "node_name": execution.node_name,
            "node_qualname": qualname,
            "execution_duration_ns": duration_ns,
            "input_state_keys": list(execution.input_state_keys),
            "output_state_keys": list(execution.output_state_keys),
            "status": status,
        }
        if execution.error_class is not None:
            build_info["error_class"] = execution.error_class
            build_info["error_truncated"] = execution.error_truncated

        try:
            event = AgentCodeEvent.create(
                repo=repo,
                commit=_RUNTIME_COMMIT_SENTINEL,
                artifact_hash=artifact_hash,
                config_hash=config_hash,
                build_info=build_info,
            )
        except Exception:
            logger.debug(
                "AgentCodeEvent construction failed for node %s; skipping",
                execution.node_name,
                exc_info=True,
            )
            return

        # Preferred: adapter path (multi-tenant + circuit breaker).
        if self._adapter is not None:
            try:
                self._adapter.emit_event(event)
                return
            except Exception:
                logger.debug(
                    "Adapter emit_event(AgentCodeEvent) failed, falling back to legacy",
                    exc_info=True,
                )

        # Legacy fallback for tracers attached to a raw STRATIX instance.
        if self._stratix and hasattr(self._stratix, "emit"):
            try:
                payload = {
                    "code": {
                        "repo": repo,
                        "commit": _RUNTIME_COMMIT_SENTINEL,
                        "artifact_hash": artifact_hash,
                        "config_hash": config_hash,
                        "build_info": build_info,
                    },
                    "node_name": execution.node_name,
                }
                self._stratix.emit("agent.code", payload)
            except Exception:
                logger.debug("Legacy agent.code emission failed", exc_info=True)

    def _emit_state_change(self, execution: NodeExecution) -> None:
        """Emit state change event via adapter (preferred) or legacy path."""
        duration_ns = (execution.end_time_ns or 0) - execution.start_time_ns

        # New-style: route through adapter.emit_event
        if self._adapter is not None:
            try:
                from layerlens.instrument._vendored.events import (
                    StateType,
                    AgentStateChangeEvent,
                )

                typed_payload = AgentStateChangeEvent.create(
                    state_type=StateType.INTERNAL,
                    before_hash=execution.state_hash_before or _zero_hash(),
                    after_hash=execution.state_hash_after or _zero_hash(),
                )
                self._adapter.emit_event(typed_payload)
                return
            except Exception:
                logger.debug("Typed event emission failed, falling back to legacy", exc_info=True)

        # Legacy fallback
        if self._stratix and hasattr(self._stratix, "emit"):
            self._stratix.emit(
                "agent.state.change",
                {
                    "node_name": execution.node_name,
                    "before_hash": execution.state_hash_before,
                    "after_hash": execution.state_hash_after,
                    "duration_ns": duration_ns,
                },
            )


class _NodeContext:
    """Context manager for node tracing."""

    def __init__(
        self,
        tracer: NodeTracer,
        node_name: str,
        state: Any,
        node_callable: Callable[..., Any] | None = None,
    ) -> None:
        self._tracer = tracer
        self._node_name = node_name
        self._state = state
        self._node_callable = node_callable
        self._result_state: Any = None
        self._execution: NodeExecution | None = None

    def __enter__(self) -> _NodeContext:
        self._execution = self._tracer.on_node_enter(
            self._node_name,
            self._state,
            node_callable=self._node_callable,
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._execution:
            # Use result state if set, otherwise use original state
            final_state = self._result_state if self._result_state is not None else self._state
            error = exc_val if exc_val else None
            self._tracer.on_node_exit(self._execution, final_state, error)

    def set_result(self, state: Any) -> None:
        """Set the result state for tracking."""
        self._result_state = state


def trace_node(
    stratix_instance: Any = None,
    state_adapter: LangGraphStateAdapter | None = None,
    adapter: BaseAdapter | None = None,
) -> Callable[[NodeFunc], NodeFunc]:  # type: ignore[type-arg]
    """
    Decorator factory for tracing node functions.

    Usage:
        @trace_node(stratix)
        def my_node(state):
            return new_state

    Args:
        stratix_instance: STRATIX SDK instance
        state_adapter: State adapter for change detection
        adapter: BaseAdapter instance (new-style)

    Returns:
        Decorator function
    """
    tracer = NodeTracer(stratix_instance, state_adapter, adapter=adapter)

    def decorator(func: NodeFunc) -> NodeFunc:  # type: ignore[type-arg]
        return tracer.decorate(func)

    return decorator


def create_traced_node(
    func: NodeFunc,  # type: ignore[type-arg]
    stratix_instance: Any = None,
    adapter: BaseAdapter | None = None,
    node_name: str | None = None,
) -> NodeFunc:  # type: ignore[type-arg]
    """
    Create a traced version of a node function.

    This is useful when you want to trace existing functions without
    modifying them.

    Args:
        func: Original node function
        stratix_instance: STRATIX SDK instance
        adapter: BaseAdapter instance (new-style)
        node_name: Name to use for tracing (defaults to function name)

    Returns:
        Traced node function
    """
    tracer = NodeTracer(stratix_instance, adapter=adapter)
    name = node_name or func.__name__

    @wraps(func)
    def traced_func(state: Any) -> Any:
        with tracer.trace_node(name, state, node_callable=func) as ctx:
            result = func(state)
            ctx.set_result(result)
            return result

    return traced_func
