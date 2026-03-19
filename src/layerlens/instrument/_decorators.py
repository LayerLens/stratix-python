"""
STRATIX Decorator-Based Instrumentation

From Step 4 specification:
- Tool and action instrumentation SHOULD be implemented primarily via
  decorators or wrappers, not explicit emits
- Decorators provide:
  - Automatic capture of input/output
  - Latency measurement
  - Exception capture
  - Deterministic sequence boundaries
  - Automatic privacy enforcement

The decorator MUST automatically emit:
- tool.call (L5a)
- Optional tool.logic (L5b) if registered
- Optional tool.environment (L5c) if available
"""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from layerlens.instrument.schema.events import (
    ToolCallEvent,
    ModelInvokeEvent,
)
from layerlens.instrument.schema.events.l5_tools import IntegrationType
from layerlens.instrument._context import get_current_context

if TYPE_CHECKING:
    from layerlens.instrument._core import STRATIX


T = TypeVar("T")


def trace_tool(
    stratix: "STRATIX",
    name: str,
    version: str = "unavailable",
    integration: str = "library",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for tool/action instrumentation.

    Args:
        stratix: The STRATIX instance
        name: Tool name
        version: Tool version
        integration: Integration type (library, service, agent, script)

    Returns:
        Decorator function
    """
    integration_type = IntegrationType(integration)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            ctx = get_current_context()
            if ctx is None:
                # No context, just run the function
                return func(*args, **kwargs)

            # Capture input
            input_data = _capture_input(args, kwargs)

            # Start timing
            start_time = time.perf_counter()

            # Execute
            output_data: dict[str, Any] | None = None
            error_msg: str | None = None
            try:
                result = func(*args, **kwargs)
                output_data = _capture_output(result)
                return result
            except Exception as e:
                error_msg = str(e)
                raise
            finally:
                # Calculate latency
                latency_ms = (time.perf_counter() - start_time) * 1000

                # Create and emit tool.call event
                event_payload = ToolCallEvent.create(
                    name=name,
                    version=version,
                    integration=integration_type,
                    input_data=input_data,
                    output_data=output_data,
                    error=error_msg,
                    latency_ms=latency_ms,
                )
                stratix._emit_event(ctx, event_payload)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            ctx = get_current_context()
            if ctx is None:
                return await func(*args, **kwargs)

            input_data = _capture_input(args, kwargs)
            start_time = time.perf_counter()

            output_data: dict[str, Any] | None = None
            error_msg: str | None = None
            try:
                result = await func(*args, **kwargs)
                output_data = _capture_output(result)
                return result
            except Exception as e:
                error_msg = str(e)
                raise
            finally:
                latency_ms = (time.perf_counter() - start_time) * 1000
                event_payload = ToolCallEvent.create(
                    name=name,
                    version=version,
                    integration=integration_type,
                    input_data=input_data,
                    output_data=output_data,
                    error=error_msg,
                    latency_ms=latency_ms,
                )
                stratix._emit_event(ctx, event_payload)

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper

    return decorator


def trace_model(
    stratix: "STRATIX",
    provider: str,
    name: str,
    version: str = "unavailable",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for model invocation instrumentation.

    Args:
        stratix: The STRATIX instance
        provider: Model provider (openai, anthropic, etc.)
        name: Model name
        version: Model version

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            ctx = get_current_context()
            if ctx is None:
                return func(*args, **kwargs)

            # Extract parameters if available
            parameters = _extract_model_params(kwargs)

            start_time = time.perf_counter()

            prompt_tokens: int | None = None
            completion_tokens: int | None = None
            total_tokens: int | None = None

            try:
                result = func(*args, **kwargs)

                # Try to extract token counts from result
                tokens = _extract_token_counts(result)
                if tokens:
                    prompt_tokens, completion_tokens, total_tokens = tokens

                return result
            finally:
                latency_ms = (time.perf_counter() - start_time) * 1000

                event_payload = ModelInvokeEvent.create(
                    provider=provider,
                    name=name,
                    version=version,
                    parameters=parameters,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                )
                stratix._emit_event(ctx, event_payload)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            ctx = get_current_context()
            if ctx is None:
                return await func(*args, **kwargs)

            parameters = _extract_model_params(kwargs)
            start_time = time.perf_counter()

            prompt_tokens: int | None = None
            completion_tokens: int | None = None
            total_tokens: int | None = None

            try:
                result = await func(*args, **kwargs)
                tokens = _extract_token_counts(result)
                if tokens:
                    prompt_tokens, completion_tokens, total_tokens = tokens
                return result
            finally:
                latency_ms = (time.perf_counter() - start_time) * 1000
                event_payload = ModelInvokeEvent.create(
                    provider=provider,
                    name=name,
                    version=version,
                    parameters=parameters,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                )
                stratix._emit_event(ctx, event_payload)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper

    return decorator


def _capture_input(args: tuple, kwargs: dict) -> dict[str, Any]:
    """Capture function input as a dictionary."""
    result: dict[str, Any] = {}

    # Add positional args
    if args:
        result["args"] = [_safe_serialize(a) for a in args]

    # Add keyword args
    if kwargs:
        result["kwargs"] = {k: _safe_serialize(v) for k, v in kwargs.items()}

    return result


def _capture_output(result: Any) -> dict[str, Any]:
    """Capture function output as a dictionary."""
    return {"result": _safe_serialize(result)}


def _safe_serialize(value: Any) -> Any:
    """Safely serialize a value for logging."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_serialize(v) for v in value[:100]]  # Limit list size
    if isinstance(value, dict):
        return {k: _safe_serialize(v) for k, v in list(value.items())[:100]}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return {k: _safe_serialize(v) for k, v in value.__dict__.items() if not k.startswith("_")}
    return str(value)[:1000]  # Truncate long strings


def _extract_model_params(kwargs: dict) -> dict[str, Any]:
    """Extract model parameters from kwargs."""
    param_keys = ["temperature", "max_tokens", "top_p", "top_k", "stop", "stream"]
    return {k: v for k, v in kwargs.items() if k in param_keys}


def _extract_token_counts(result: Any) -> tuple[int, int, int] | None:
    """Try to extract token counts from model response."""
    # Try OpenAI-style response
    if hasattr(result, "usage"):
        usage = result.usage
        if hasattr(usage, "prompt_tokens"):
            return (
                getattr(usage, "prompt_tokens", 0),
                getattr(usage, "completion_tokens", 0),
                getattr(usage, "total_tokens", 0),
            )

    # Try dict-style response
    if isinstance(result, dict):
        usage = result.get("usage", {})
        if "prompt_tokens" in usage:
            return (
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                usage.get("total_tokens", 0),
            )

    return None
