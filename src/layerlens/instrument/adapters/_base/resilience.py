"""Per-callback try/except resilience wrapper for adapter callbacks.

Mature framework adapters (CrewAI, AutoGen, OpenAI Agents, Google ADK,
Strands, Bedrock Agents) wrap every observability callback in a
try/except boundary so an exception in the adapter never escapes back
into the framework's own execution path. Lighter adapters historically
relied on outer wrappers — meaning a single bug in our callback could
crash a customer's agent run.

This module exposes a shared decorator (``resilient_callback``) and a
per-adapter failure tracker (``ResilienceTracker``) so every framework
adapter can apply the SAME resilience contract:

1. Catch ``Exception`` (NOT ``BaseException`` — KeyboardInterrupt /
   SystemExit / GeneratorExit must still propagate).
2. Log the exception via the adapter's logger with
   ``adapter_name``, ``callback_name``, and a truncated traceback.
3. Increment the adapter's ``_resilience_failures`` counter.
4. Return the framework's expected default value for the callback so
   the framework continues uninterrupted.

The failure counter is consulted by ``ResilienceTracker.health_status``
which returns ``HealthStatus.DEGRADED`` once the adapter has crossed
``DEFAULT_FAILURE_THRESHOLD`` failures within the lifetime of the run.
Adapters surface this in their ``adapter_info().metadata`` block.

This module is **adapter-internal infrastructure**. It is NOT public
API for end users — there are no version guarantees on the helpers
exposed here, only on the BaseAdapter contract.
"""

from __future__ import annotations

import enum
import logging
import functools
import threading
import traceback
from typing import Any, Dict, TypeVar, Callable, Optional, cast

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public constants & enums
# ---------------------------------------------------------------------------


DEFAULT_FAILURE_THRESHOLD: int = 5
"""Number of resilience failures before an adapter is marked DEGRADED.

Chosen as a balance between fast detection (catch persistent bugs in
adapter wiring quickly) and not flapping on transient framework quirks
(a single bad event from a flaky upstream shouldn't degrade the entire
adapter). Adapters can override this via ``ResilienceTracker(threshold=...)``.
"""


_TRACEBACK_TRUNCATION: int = 4000
"""Maximum characters of formatted traceback to log per failure.

Prevents log spam from huge tracebacks (deep async stacks under
LangGraph or LlamaIndex can produce >10kB tracebacks per failure).
"""


class HealthStatus(str, enum.Enum):
    """Adapter health states surfaced via ``adapter_info().metadata``."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"


# ---------------------------------------------------------------------------
# Default-value table for known callbacks
# ---------------------------------------------------------------------------
#
# Many framework callback APIs require the callback to RETURN something
# (not just produce side-effects). For example:
#   * Pydantic-AI ``after_model_request`` is expected to return the
#     (possibly-mutated) response object — returning ``None`` would replace
#     the LLM response with ``None`` and break the agent.
#   * Pydantic-AI ``before_tool_execute`` returns the (possibly-mutated)
#     args tuple — returning ``None`` would erase the tool args.
#   * Google ADK plugin callbacks are documented as returning ``None``
#     (no override semantics) — ``None`` is the correct default.
#   * Strands hook callbacks return ``None``.
#   * boto3 event-system handlers (Bedrock Agents) return ``None``.
#
# When a callback that needs a passthrough (e.g. Pydantic-AI mutating
# hooks) raises, returning ``None`` would corrupt the framework's data
# flow. Adapters can pass ``passthrough_arg`` to ``resilient_callback``
# so the wrapper returns that argument's value instead of the default.

_DEFAULTS: Dict[str, Any] = {
    # Google ADK plugin callbacks — all return None (no override hook).
    "before_run_callback": None,
    "after_run_callback": None,
    "before_agent_callback": None,
    "after_agent_callback": None,
    "before_model_callback": None,
    "after_model_callback": None,
    "on_model_error_callback": None,
    "before_tool_callback": None,
    "after_tool_callback": None,
    "on_tool_error_callback": None,
    "on_event_callback": None,
    # Strands hooks — sync, return None.
    "_on_agent_initialized": None,
    "_on_before_invocation": None,
    "_on_after_invocation": None,
    "_on_before_model": None,
    "_on_after_model": None,
    "_on_before_tool": None,
    "_on_after_tool": None,
    # OpenAI Agents TracingProcessor — return None.
    "on_trace_start": None,
    "on_trace_end": None,
    "on_span_start": None,
    "on_span_end": None,
    "shutdown": None,
    "force_flush": None,
    # boto3 event handlers (Bedrock Agents).
    "_before_invoke": None,
    "_after_invoke": None,
}


def get_default_for(callback_name: str) -> Any:
    """Return the framework-expected default for *callback_name*, or ``None``.

    The default of ``None`` is correct for the overwhelming majority of
    callback APIs across instrumented frameworks (boto3 event system,
    LlamaIndex span/event handlers, Strands hooks, Google ADK plugins,
    OpenAI Agents TracingProcessor). For callbacks that need to return a
    passthrough value (Pydantic-AI mutating hooks), use ``resilient_callback``
    with ``passthrough_arg`` instead.
    """
    return _DEFAULTS.get(callback_name)


# ---------------------------------------------------------------------------
# Failure tracker
# ---------------------------------------------------------------------------


class ResilienceTracker:
    """Per-adapter failure counter + degraded-health surface.

    Each framework adapter instantiates one tracker (in ``__init__``).
    ``resilient_callback`` records failures via :meth:`record_failure`.
    The adapter's ``adapter_info()`` reports current health via
    :meth:`health_status` and a snapshot of recent failures via
    :meth:`as_metadata`.

    The tracker is thread-safe: framework callbacks can fire from worker
    threads (CrewAI dispatches across threads, AutoGen group chat fans
    out, Bedrock boto3 hooks run in the request thread).
    """

    def __init__(
        self,
        adapter_name: str,
        threshold: int = DEFAULT_FAILURE_THRESHOLD,
    ) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self._adapter_name = adapter_name
        self._threshold = threshold
        self._lock = threading.Lock()
        self._total_failures: int = 0
        self._per_callback_failures: Dict[str, int] = {}
        self._last_error: Optional[str] = None
        self._last_callback: Optional[str] = None

    # -- recording --------------------------------------------------------

    def record_failure(self, callback_name: str, exc: BaseException) -> None:
        """Atomically record a failed callback invocation."""
        with self._lock:
            self._total_failures += 1
            self._per_callback_failures[callback_name] = self._per_callback_failures.get(callback_name, 0) + 1
            self._last_callback = callback_name
            self._last_error = f"{type(exc).__name__}: {exc}"[:500]

    def reset(self) -> None:
        """Clear all failure state. Adapters call this on ``disconnect()``."""
        with self._lock:
            self._total_failures = 0
            self._per_callback_failures.clear()
            self._last_error = None
            self._last_callback = None

    # -- queries ----------------------------------------------------------

    @property
    def total_failures(self) -> int:
        with self._lock:
            return self._total_failures

    @property
    def threshold(self) -> int:
        return self._threshold

    def health_status(self) -> HealthStatus:
        """Return DEGRADED once total failures cross the threshold."""
        with self._lock:
            return HealthStatus.DEGRADED if self._total_failures >= self._threshold else HealthStatus.HEALTHY

    def as_metadata(self) -> Dict[str, Any]:
        """Snapshot for inclusion in ``adapter_info().metadata``."""
        with self._lock:
            data: Dict[str, Any] = {
                "resilience_status": (
                    HealthStatus.DEGRADED.value if self._total_failures >= self._threshold else HealthStatus.HEALTHY.value
                ),
                "resilience_failures_total": self._total_failures,
                "resilience_failure_threshold": self._threshold,
            }
            if self._per_callback_failures:
                # Cap to top 20 so metadata payloads don't explode for
                # adapters with many distinct callbacks.
                top = sorted(
                    self._per_callback_failures.items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )[:20]
                data["resilience_failures_by_callback"] = dict(top)
            if self._last_error is not None:
                data["resilience_last_error"] = self._last_error
            if self._last_callback is not None:
                data["resilience_last_callback"] = self._last_callback
            return data


# ---------------------------------------------------------------------------
# The decorator
# ---------------------------------------------------------------------------


F = TypeVar("F", bound=Callable[..., Any])


def resilient_callback(
    *,
    callback_name: Optional[str] = None,
    default: Any = None,
    passthrough_arg: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Callable[[F], F]:
    """Wrap a bound adapter method so observability errors never escape.

    The decorator must be applied to *instance methods* of an adapter
    class. The adapter MUST expose:

    * ``self.name`` (or fall back to the class name) for logging context
    * ``self._resilience`` (a :class:`ResilienceTracker`) for failure
      recording

    On exception inside the wrapped method:

    1. The exception is caught (excluding ``BaseException`` subclasses
       like ``KeyboardInterrupt``).
    2. ``self._resilience.record_failure(name, exc)`` is invoked.
    3. The exception is logged via *logger* (or the adapter's module
       logger) at WARNING level with a truncated traceback.
    4. The wrapper returns ``default``, OR the value of the keyword/positional
       argument named *passthrough_arg* (so frameworks that expect a
       mutating callback to return the passed-through value still work).

    Parameters
    ----------
    callback_name:
        Name to use in failure tracking and log records. Defaults to the
        wrapped function's ``__name__``.
    default:
        Value to return when the wrapped method raises.
        Use the framework's expected return type for this callback —
        e.g. ``None`` for void handlers, ``""`` for handlers expected to
        return a string, the original ``args`` tuple for mutating hooks.
        For common callback names, the table in :func:`get_default_for`
        provides the canonical default.
    passthrough_arg:
        If set, the wrapper returns the value of this argument (looked
        up in *kwargs* first, then matched positionally if needed) on
        failure. Use this for mutating hooks (Pydantic-AI
        ``after_model_request`` returns the response object;
        ``before_tool_execute`` returns the args tuple). When both
        *passthrough_arg* and *default* are set, *passthrough_arg* wins
        when the argument is present; otherwise *default* is used.
    logger:
        Logger to emit failure messages to. Defaults to the module
        logger of the wrapped function.
    """

    def _decorate(func: F) -> F:
        cb_name = callback_name or func.__name__
        # Resolve logger lazily — the wrapped function's module is the
        # right logger context for warnings (so users can mute one
        # adapter's resilience warnings without muting all of them).
        bound_logger = logger or logging.getLogger(func.__module__)

        @functools.wraps(func)
        def _wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return func(self, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — intentional broad catch
                _on_failure(
                    self,
                    cb_name=cb_name,
                    exc=exc,
                    bound_logger=bound_logger,
                )
                return _resolve_return_value(
                    args=args,
                    kwargs=kwargs,
                    func=func,
                    passthrough_arg=passthrough_arg,
                    default=default,
                )

        return cast(F, _wrapper)

    return _decorate


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _on_failure(
    adapter: Any,
    *,
    cb_name: str,
    exc: BaseException,
    bound_logger: logging.Logger,
) -> None:
    """Record + log a callback failure on *adapter*'s resilience tracker.

    Best-effort: if the adapter doesn't have a tracker (programming
    error), we still log the failure so the user sees it.
    """
    adapter_name = getattr(adapter, "name", None) or type(adapter).__name__
    tracker = getattr(adapter, "_resilience", None)
    if isinstance(tracker, ResilienceTracker):
        tracker.record_failure(cb_name, exc)

    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    if len(tb) > _TRACEBACK_TRUNCATION:
        tb = tb[: _TRACEBACK_TRUNCATION - 24] + "\n... [traceback truncated]"

    bound_logger.warning(
        "layerlens: resilient_callback caught exception in %s.%s: %s\n%s",
        adapter_name,
        cb_name,
        exc,
        tb,
    )


def _resolve_return_value(
    *,
    args: tuple[Any, ...],
    kwargs: Dict[str, Any],
    func: Callable[..., Any],
    passthrough_arg: Optional[str],
    default: Any,
) -> Any:
    """Compute the value to return when a wrapped callback raises.

    If *passthrough_arg* names a parameter that was actually supplied,
    return its value. Otherwise return *default*.
    """
    if not passthrough_arg:
        return default

    # Keyword-supplied arguments are the most common case for callback
    # APIs (Pydantic-AI / Google ADK / Strands all use keyword-only
    # callback signatures).
    if passthrough_arg in kwargs:
        return kwargs[passthrough_arg]

    # Fall back to positional resolution by inspecting the function's
    # parameter list (skip ``self`` which is always position 0).
    try:
        params = func.__code__.co_varnames[: func.__code__.co_argcount]
    except AttributeError:
        return default
    for index, name in enumerate(params):
        if name == passthrough_arg and index >= 1 and index - 1 < len(args):
            return args[index - 1]

    return default
