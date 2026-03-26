from __future__ import annotations

import asyncio
import functools
from typing import Any, Dict, Tuple, Callable, Optional

from ._types import SpanData
from ._context import _current_span, _current_recorder
from ._recorder import _SENTINEL, TraceRecorder


def trace(
    client: Any,
    *,
    name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    signing_service: Any = _SENTINEL,
) -> Callable[..., Any]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        span_name = name or fn.__name__

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                recorder = TraceRecorder(client, signing_service=signing_service)
                root = SpanData(
                    name=span_name,
                    kind="chain",
                    input=_capture_input(args, kwargs),
                    metadata=metadata or {},
                )
                recorder.root = root

                rec_token = _current_recorder.set(recorder)
                span_token = _current_span.set(root)
                try:
                    result = await fn(*args, **kwargs)
                    root.output = result
                    root.finish()
                    await recorder.async_flush()
                    return result
                except Exception as exc:
                    root.finish(error=str(exc))
                    await recorder.async_flush()
                    raise
                finally:
                    _current_span.reset(span_token)
                    _current_recorder.reset(rec_token)

            return async_wrapper
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                recorder = TraceRecorder(client, signing_service=signing_service)
                root = SpanData(
                    name=span_name,
                    kind="chain",
                    input=_capture_input(args, kwargs),
                    metadata=metadata or {},
                )
                recorder.root = root

                rec_token = _current_recorder.set(recorder)
                span_token = _current_span.set(root)
                try:
                    result = fn(*args, **kwargs)
                    root.output = result
                    root.finish()
                    recorder.flush()
                    return result
                except Exception as exc:
                    root.finish(error=str(exc))
                    recorder.flush()
                    raise
                finally:
                    _current_span.reset(span_token)
                    _current_recorder.reset(rec_token)

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
