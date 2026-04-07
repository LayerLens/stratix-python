from __future__ import annotations

import uuid
import asyncio
import functools
from typing import Any, Dict, Tuple, Callable, Optional

from ._capture_config import CaptureConfig
from ._collector import TraceCollector
from ._context import _current_collector, _push_span, _pop_span


def trace(
    client: Any,
    *,
    name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    capture_config: Optional[CaptureConfig] = None,
) -> Callable[..., Any]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        span_name = name or fn.__name__

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                config = capture_config or CaptureConfig.standard()
                collector = TraceCollector(client, config)
                root_span_id = uuid.uuid4().hex[:16]

                col_token = _current_collector.set(collector)
                span_snapshot = _push_span(root_span_id, span_name)
                try:
                    collector.emit(
                        "agent.input",
                        {"name": span_name, "input": _capture_input(args, kwargs), **(metadata or {})},
                        span_id=root_span_id,
                        span_name=span_name,
                    )

                    result = await fn(*args, **kwargs)

                    collector.emit(
                        "agent.output",
                        {"name": span_name, "output": result, "status": "ok"},
                        span_id=root_span_id,
                        span_name=span_name,
                    )
                    collector.flush()
                    return result
                except Exception as exc:
                    collector.emit(
                        "agent.error",
                        {"name": span_name, "error": str(exc), "status": "error"},
                        span_id=root_span_id,
                        span_name=span_name,
                    )
                    collector.flush()
                    raise
                finally:
                    _pop_span(span_snapshot)
                    _current_collector.reset(col_token)

            return async_wrapper
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                config = capture_config or CaptureConfig.standard()
                collector = TraceCollector(client, config)
                root_span_id = uuid.uuid4().hex[:16]

                col_token = _current_collector.set(collector)
                span_snapshot = _push_span(root_span_id, span_name)
                try:
                    collector.emit(
                        "agent.input",
                        {"name": span_name, "input": _capture_input(args, kwargs), **(metadata or {})},
                        span_id=root_span_id,
                        span_name=span_name,
                    )

                    result = fn(*args, **kwargs)

                    collector.emit(
                        "agent.output",
                        {"name": span_name, "output": result, "status": "ok"},
                        span_id=root_span_id,
                        span_name=span_name,
                    )
                    collector.flush()
                    return result
                except Exception as exc:
                    collector.emit(
                        "agent.error",
                        {"name": span_name, "error": str(exc), "status": "error"},
                        span_id=root_span_id,
                        span_name=span_name,
                    )
                    collector.flush()
                    raise
                finally:
                    _pop_span(span_snapshot)
                    _current_collector.reset(col_token)

            return sync_wrapper

    return decorator


def _capture_input(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
    if args and kwargs:
        return {"args": list(args), "kwargs": kwargs}
    if args:
        arg_list = list(args)
        return arg_list if len(arg_list) > 1 else arg_list[0]
    if kwargs:
        return kwargs
    return None
