"""
MCP Tool Call Wrapper

Wraps MCP client tool call dispatch to intercept and trace all tool
invocations automatically.
"""

from __future__ import annotations

import time
import logging
import functools
from typing import Any
from collections.abc import Callable

logger = logging.getLogger(__name__)


def wrap_mcp_tool_call(
    original_fn: Callable[..., Any],
    adapter: Any,
) -> Callable[..., Any]:
    """
    Wrap an MCP tool call function for tracing.

    The wrapper emits tool.call events for every invocation, plus
    protocol.tool.structured_output if a structured output schema
    is present.

    Args:
        original_fn: The original tool call function.
        adapter: MCPExtensionsAdapter instance.

    Returns:
        Wrapped function.
    """
    if getattr(original_fn, "_layerlens_original", False):
        return original_fn

    @functools.wraps(original_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tool_name = kwargs.get("name", kwargs.get("tool_name", "unknown"))
        input_data = kwargs.get("arguments", kwargs.get("input", {}))

        start = time.monotonic()
        error_msg = None
        result = None
        try:
            result = original_fn(*args, **kwargs)
            return result
        except Exception as exc:
            error_msg = str(exc)
            raise
        finally:
            latency_ms = (time.monotonic() - start) * 1000
            output_data = None
            if result is not None:
                if hasattr(result, "model_dump"):
                    output_data = result.model_dump()
                elif isinstance(result, dict):
                    output_data = result
                else:
                    output_data = {"result": str(result)}

            adapter.on_tool_call(
                tool_name=str(tool_name),
                input_data=input_data
                if isinstance(input_data, dict)
                else {"args": str(input_data)},
                output_data=output_data,
                error=error_msg,
                latency_ms=latency_ms,
            )

    wrapper._layerlens_original = True  # type: ignore[attr-defined]
    return wrapper


async def wrap_mcp_tool_call_async(
    original_fn: Callable[..., Any],
    adapter: Any,
) -> Callable[..., Any]:
    """
    Wrap an async MCP tool call function for tracing.

    Args:
        original_fn: The original async tool call function.
        adapter: MCPExtensionsAdapter instance.

    Returns:
        Wrapped async function.
    """
    if getattr(original_fn, "_layerlens_original", False):
        return original_fn

    @functools.wraps(original_fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        tool_name = kwargs.get("name", kwargs.get("tool_name", "unknown"))
        input_data = kwargs.get("arguments", kwargs.get("input", {}))

        start = time.monotonic()
        error_msg = None
        result = None
        try:
            result = await original_fn(*args, **kwargs)
            return result
        except Exception as exc:
            error_msg = str(exc)
            raise
        finally:
            latency_ms = (time.monotonic() - start) * 1000
            output_data = None
            if result is not None:
                if hasattr(result, "model_dump"):
                    output_data = result.model_dump()
                elif isinstance(result, dict):
                    output_data = result
                else:
                    output_data = {"result": str(result)}

            adapter.on_tool_call(
                tool_name=str(tool_name),
                input_data=input_data
                if isinstance(input_data, dict)
                else {"args": str(input_data)},
                output_data=output_data,
                error=error_msg,
                latency_ms=latency_ms,
            )

    wrapper._layerlens_original = True  # type: ignore[attr-defined]
    return wrapper
