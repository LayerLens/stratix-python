"""Wrap arbitrary MCP tool-call functions for tracing.

Used when the MCP SDK surface isn't a class with ``call_tool`` (which
:class:`MCPProtocolAdapter` already handles) — e.g. a bare callable
registered as a tool. Re-wrapping is idempotent via the
``_layerlens_wrapped`` sentinel.
"""

from __future__ import annotations

import time
import inspect
import logging
import functools
from typing import Any, Dict
from collections.abc import Callable

from ...._events import MCP_TOOL_CALL

log = logging.getLogger(__name__)


def _extract_tool_name(args: tuple, kwargs: Dict[str, Any], default: str = "unknown") -> str:
    name = kwargs.get("name") or kwargs.get("tool_name")
    if name:
        return str(name)
    if args and isinstance(args[0], str):
        return args[0]
    return default


def _extract_input(args: tuple, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    raw = kwargs.get("arguments", kwargs.get("input"))
    if raw is None and len(args) >= 2:
        raw = args[1]
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    return {"args": repr(raw)}


def _coerce_output(result: Any) -> Dict[str, Any] | None:
    if result is None:
        return None
    if hasattr(result, "model_dump"):
        try:
            return dict(result.model_dump())
        except Exception:  # pragma: no cover - defensive
            pass
    if isinstance(result, dict):
        return result
    if isinstance(result, (str, int, float, bool)):
        return {"result": result}
    return {"result": repr(result)}


def _emit(
    adapter: Any,
    *,
    tool_name: str,
    input_data: Dict[str, Any],
    output_data: Dict[str, Any] | None,
    error: str | None,
    latency_ms: float,
) -> None:
    payload: Dict[str, Any] = {
        "tool_name": tool_name,
        "arguments": input_data,
        "latency_ms": latency_ms,
    }
    if output_data is not None:
        payload["result"] = output_data
    if error is not None:
        payload["error"] = error
    adapter.emit(MCP_TOOL_CALL, payload)


def wrap_mcp_tool_call(original_fn: Callable[..., Any], adapter: Any) -> Callable[..., Any]:
    """Wrap a sync MCP tool-call function for tracing."""
    if getattr(original_fn, "_layerlens_wrapped", False):
        return original_fn

    @functools.wraps(original_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tool_name = _extract_tool_name(args, kwargs)
        input_data = _extract_input(args, kwargs)
        start = time.monotonic()
        result: Any = None
        error: str | None = None
        try:
            result = original_fn(*args, **kwargs)
            return result
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            _emit(
                adapter,
                tool_name=tool_name,
                input_data=input_data,
                output_data=_coerce_output(result) if error is None else None,
                error=error,
                latency_ms=(time.monotonic() - start) * 1000,
            )

    wrapper._layerlens_wrapped = True  # type: ignore[attr-defined]
    return wrapper


def wrap_mcp_tool_call_async(original_fn: Callable[..., Any], adapter: Any) -> Callable[..., Any]:
    """Wrap an async MCP tool-call function for tracing."""
    if getattr(original_fn, "_layerlens_wrapped", False):
        return original_fn
    if not inspect.iscoroutinefunction(original_fn):
        log.debug("wrap_mcp_tool_call_async called on non-coroutine %r", original_fn)

    @functools.wraps(original_fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        tool_name = _extract_tool_name(args, kwargs)
        input_data = _extract_input(args, kwargs)
        start = time.monotonic()
        result: Any = None
        error: str | None = None
        try:
            result = await original_fn(*args, **kwargs)
            return result
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            _emit(
                adapter,
                tool_name=tool_name,
                input_data=input_data,
                output_data=_coerce_output(result) if error is None else None,
                error=error,
                latency_ms=(time.monotonic() - start) * 1000,
            )

    wrapper._layerlens_wrapped = True  # type: ignore[attr-defined]
    return wrapper
