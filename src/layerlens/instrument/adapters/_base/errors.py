"""Shared error-aware event emission helper.

When a framework callback raises an exception (LLM rate limit, API down,
tool exception, malformed prompt, etc.), the corresponding lifecycle event
typically appears in the trace as a "start" with no matching "end" — the
dashboard renders this as a hung request, not a failure.

:func:`emit_error_event` produces a discrete ``policy.violation`` (or
adapter-overridden) event with PII-safe context so operators can triage
the failure end-to-end. It is the cross-pollination of the AutoGen
``wrappers.py`` and LangChain ``on_*_error`` callback patterns to the
ten "lighter" runtime adapters (agno, ms_agent_framework, openai_agents,
llama_index, google_adk, strands, pydantic_ai, smolagents,
bedrock_agents, embedding).

Design rules
------------

1. **Re-raise semantics preserved.** This helper only EMITS — it never
   swallows the original exception. Wrappers in adapters call
   :func:`emit_error_event` inside a ``try/except → emit → raise``
   block.
2. **PII-safe.** The exception ``args`` may contain raw user input or
   secret values; the helper truncates the string-cast message, redacts
   common secret patterns, and never serialises the raw context dict.
   Only allow-listed framework attribution keys are forwarded.
3. **Multi-tenant.** When an ``org_id`` is propagated through the
   adapter's stratix client (or supplied via the optional ``org_id``
   parameter), it is added to the event payload so downstream sinks can
   route per-tenant.
4. **Bounded payload.** Tracebacks are truncated to ``MAX_TRACEBACK_FRAMES``
   frames and ``MAX_TRACEBACK_CHARS`` characters; the message is truncated
   to ``MAX_MESSAGE_CHARS``. This keeps events under typical sink size
   limits even when the framework raises with a megabyte-long message.
"""

from __future__ import annotations

import re
import logging
import traceback
from typing import TYPE_CHECKING, Any, Dict, Tuple, Mapping, Optional

if TYPE_CHECKING:
    from layerlens.instrument.adapters._base.adapter import BaseAdapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bounds (matched to AutoGen / Semantic Kernel / Agentforce conventions)
# ---------------------------------------------------------------------------

#: Maximum traceback frames retained on the event payload. Older frames
#: (deeper in the call chain) are dropped first.
MAX_TRACEBACK_FRAMES: int = 8

#: Maximum total traceback string length AFTER frame truncation. Hard cap
#: applied as a second pass to defend against frames that are themselves
#: very long (e.g. a single statement spanning multiple kilobytes).
MAX_TRACEBACK_CHARS: int = 4000

#: Maximum length of the exception message stored on the event payload.
#: Mirrors AutoGen ``lifecycle.py:534-538`` (500-char preview default).
MAX_MESSAGE_CHARS: int = 500

#: Allow-list of framework-context keys safe to propagate verbatim onto
#: the error event. Any other key supplied by the caller is dropped to
#: avoid accidental PII leakage.
SAFE_CONTEXT_KEYS: Tuple[str, ...] = (
    "framework",
    "agent_name",
    "agent_id",
    "agent_alias_id",
    "chat_name",
    "tool_name",
    "tool_type",
    "model",
    "model_name",
    "provider",
    "session_id",
    "span_id",
    "trace_id",
    "run_id",
    "node_name",
    "phase",
    "callback",
    "operation",
    "step",
    "latency_ms",
    "duration_ns",
)

#: Default event type emitted by :func:`emit_error_event`. Adapters may
#: override per-call (e.g. ``agent.error`` for tool / agent failures vs
#: ``policy.violation`` for guardrail trips).
DEFAULT_EVENT_TYPE: str = "policy.violation"

#: Patterns that mask common secret formats in the exception message and
#: traceback. Applied case-insensitively. Conservative — matches obvious
#: ``key="..."`` / ``Bearer <token>`` / ``Authorization: ...`` shapes
#: rather than attempting general PII scrubbing.
_SECRET_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    # Order matters: Bearer / sk-* first so they match before the more
    # generic ``Authorization: ...`` rule, which would otherwise consume
    # only the literal word "Bearer" and leave the token visible.
    (
        re.compile(r"\bBearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE),
        "Bearer ***REDACTED***",
    ),
    (
        re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
        "sk-***REDACTED***",
    ),
    (
        re.compile(
            r"\b(api[_-]?key|secret|token|password|authorization|auth[_-]?token)"
            r'\s*[:=]\s*["\']?([^"\'\s,;\}\)]+)',
            re.IGNORECASE,
        ),
        r"\1=***REDACTED***",
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scrub_secrets(text: str) -> str:
    """Redact common secret patterns from ``text``.

    Idempotent and safe on empty input.
    """
    if not text:
        return text
    scrubbed = text
    for pattern, replacement in _SECRET_PATTERNS:
        scrubbed = pattern.sub(replacement, scrubbed)
    return scrubbed


def _truncate(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` characters with an ellipsis suffix."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _format_traceback(exc: BaseException) -> str:
    """Return the last :data:`MAX_TRACEBACK_FRAMES` of ``exc``'s traceback.

    Frames closest to the raise point are KEPT (the deepest frames are
    typically the most actionable for debugging). The final string is
    further bounded to :data:`MAX_TRACEBACK_CHARS` and run through the
    secret scrubber.
    """
    tb = exc.__traceback__
    if tb is None:
        return ""
    try:
        # ``format_tb`` returns a list of strings (one per frame). We tail
        # the list to keep the most recent (deepest) frames.
        frames = traceback.format_tb(tb)
    except Exception:
        return ""
    tail = frames[-MAX_TRACEBACK_FRAMES:]
    joined = "".join(tail)
    return _scrub_secrets(_truncate(joined, MAX_TRACEBACK_CHARS))


def _safe_message(exc: BaseException) -> str:
    """Return ``str(exc)`` truncated, scrubbed, and bounded.

    Returns an empty string if ``str()`` itself raises (some user-defined
    exceptions misbehave in ``__str__`` — we never want to crash the
    error-handling path).
    """
    try:
        message = str(exc)
    except Exception:
        try:
            message = repr(exc)
        except Exception:
            return ""
    return _scrub_secrets(_truncate(message, MAX_MESSAGE_CHARS))


def _filter_context(context: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return ``context`` filtered to :data:`SAFE_CONTEXT_KEYS`.

    Coerces values to JSON-friendly primitives — strings, numbers, bools,
    or ``None`` — and stringifies anything else with bounded length. This
    prevents callers from accidentally leaking raw user input via a
    framework's payload object.
    """
    if not context:
        return {}
    filtered: Dict[str, Any] = {}
    for key in SAFE_CONTEXT_KEYS:
        if key not in context:
            continue
        value = context[key]
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            filtered[key] = value
        else:
            try:
                filtered[key] = _truncate(str(value), MAX_MESSAGE_CHARS)
            except Exception:
                continue
    return filtered


def _resolve_org_id(adapter: "BaseAdapter", explicit: Optional[str]) -> Optional[str]:
    """Determine the tenant ``org_id`` for the error event.

    Explicit argument wins; otherwise the adapter's stratix client is
    consulted (read-only attribute access — never raises). Returns
    ``None`` when neither is available so the field is simply omitted
    from the payload (sinks treat missing org_id as the platform tenant).
    """
    if explicit:
        return explicit
    stratix = getattr(adapter, "_stratix", None)
    if stratix is None:
        return None
    for attr in ("org_id", "tenant_id"):
        try:
            value = getattr(stratix, attr, None)
        except Exception:
            value = None
        if isinstance(value, str) and value:
            return value
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_error_payload(
    adapter: "BaseAdapter",
    exc: BaseException,
    context: Optional[Mapping[str, Any]] = None,
    severity: str = "error",
    org_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct the event payload that :func:`emit_error_event` would emit.

    Exposed separately so callers (or tests) can inspect / extend the
    payload before emission. The returned dict is freshly constructed —
    mutations by the caller do not leak into other invocations.
    """
    payload: Dict[str, Any] = {
        "framework": adapter.FRAMEWORK,
        "exception_type": type(exc).__name__,
        "exception_module": getattr(type(exc), "__module__", "") or "",
        "message": _safe_message(exc),
        "traceback": _format_traceback(exc),
        "severity": severity,
    }

    safe_ctx = _filter_context(context)
    # Caller-provided framework key overrides the adapter default only
    # if non-empty — adapters always have a FRAMEWORK constant.
    if "framework" in safe_ctx and not safe_ctx["framework"]:
        safe_ctx.pop("framework")
    payload.update(safe_ctx)

    org = _resolve_org_id(adapter, org_id)
    if org:
        payload["org_id"] = org

    return payload


def emit_error_event(
    adapter: "BaseAdapter",
    exc: BaseException,
    context: Optional[Mapping[str, Any]] = None,
    severity: str = "error",
    event_type: str = DEFAULT_EVENT_TYPE,
    org_id: Optional[str] = None,
) -> None:
    """Emit a structured error event for ``exc`` through ``adapter``.

    The caller is responsible for re-raising ``exc`` after calling this
    function — the helper does NOT catch or swallow exceptions. Callers
    typically wrap framework callbacks like::

        try:
            framework_call(...)
        except Exception as exc:
            emit_error_event(adapter, exc, {"tool_name": tool})
            raise

    The emission itself is best-effort: any failure inside
    :meth:`BaseAdapter.emit_dict_event` is logged at DEBUG and silently
    dropped so error tracing never masks the original framework error.

    Args:
        adapter: The :class:`BaseAdapter` whose ``emit_dict_event`` will
            receive the event. Required so the event participates in the
            adapter's circuit-breaker, capture-config, and sink dispatch.
        exc: The exception that was raised. Must be a real exception —
            ``__traceback__`` is read but may be ``None``.
        context: Optional framework attribution dict. Only keys in
            :data:`SAFE_CONTEXT_KEYS` are propagated; everything else is
            dropped to avoid PII leakage.
        severity: Logical severity (``"error"`` default, ``"warning"``,
            ``"critical"``). Forwarded to the event payload for
            downstream alerting policy.
        event_type: Event-type string. Defaults to
            :data:`DEFAULT_EVENT_TYPE` (``"policy.violation"``); adapters
            that prefer ``"agent.error"`` or ``"tool.error"`` may override.
        org_id: Tenant identifier. Defaults to ``adapter._stratix.org_id``
            (read defensively). Multi-tenant SaaS deployments rely on this
            being set on every event.
    """
    try:
        payload = build_error_payload(
            adapter=adapter,
            exc=exc,
            context=context,
            severity=severity,
            org_id=org_id,
        )
    except Exception:  # pragma: no cover - defensive
        logger.debug("Failed to build error payload", exc_info=True)
        return

    try:
        adapter.emit_dict_event(event_type, payload)
    except Exception:  # pragma: no cover - defensive
        logger.debug(
            "Failed to emit %s event from %s", event_type, adapter.FRAMEWORK, exc_info=True
        )


__all__ = [
    "DEFAULT_EVENT_TYPE",
    "MAX_MESSAGE_CHARS",
    "MAX_TRACEBACK_CHARS",
    "MAX_TRACEBACK_FRAMES",
    "SAFE_CONTEXT_KEYS",
    "build_error_payload",
    "emit_error_event",
]
