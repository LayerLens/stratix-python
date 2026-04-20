from __future__ import annotations

import abc
import time
import logging
from typing import Any, Dict, Callable, Iterator, Optional, AsyncIterator

from .._base import AdapterInfo, BaseAdapter
from ..._context import _current_collector
from ._emit_helpers import emit_llm_error, emit_llm_events

log: logging.Logger = logging.getLogger(__name__)


class MonkeyPatchProvider(BaseAdapter):
    """Base for providers that monkey-patch SDK client or module methods."""

    name: str
    capture_params: frozenset[str]

    #: Subclasses may set a per-provider pricing table override (Azure, Bedrock).
    pricing_table: Optional[dict[str, dict[str, float]]] = None

    def __init__(self) -> None:
        self._client: Any = None
        self._originals: Dict[str, Any] = {}

    @staticmethod
    @abc.abstractmethod
    def extract_output(response: Any) -> Any: ...

    @staticmethod
    @abc.abstractmethod
    def extract_meta(response: Any) -> Dict[str, Any]: ...

    # Optional hook: providers that support tool/function calls override this.
    @staticmethod
    def extract_tool_calls(response: Any) -> list[dict[str, Any]]:  # noqa: ARG004
        return []

    # Optional hook: providers that support streaming implement this to
    # aggregate chunks into a single response-like object.
    @staticmethod
    def aggregate_stream(chunks: list[Any]) -> Any:  # noqa: ARG004
        return None

    # Optional hook: derive extra parameter fields from request kwargs that can't
    # be captured verbatim (e.g. privacy-aware flags, counts of bulky collections).
    @staticmethod
    def derive_params(kwargs: Dict[str, Any]) -> Dict[str, Any]:  # noqa: ARG004
        return {}

    def _wrap_sync(self, event_name: str, original: Any) -> Any:
        extractors = self._extractors()

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if _current_collector.get() is None:
                log.debug("layerlens.%s: no active trace context, passing through", event_name)
                return original(*args, **kwargs)
            start = time.time()
            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                latency_ms = (time.time() - start) * 1000
                emit_llm_error(event_name, exc, latency_ms)
                raise
            latency_ms = (time.time() - start) * 1000

            if kwargs.get("stream") is True:
                return self._wrap_stream_iterator(event_name, kwargs, response, start)

            emit_llm_events(
                event_name,
                kwargs,
                response,
                extractors.output,
                extractors.meta,
                self.capture_params,
                latency_ms,
                pricing_table=self.pricing_table,
                extract_tool_calls=extractors.tool_calls,
                extra_params=type(self).derive_params(kwargs),
            )
            return response

        return wrapped

    def _wrap_async(self, event_name: str, original: Any) -> Any:
        extractors = self._extractors()

        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            if _current_collector.get() is None:
                log.debug("layerlens.%s: no active trace context, passing through", event_name)
                return await original(*args, **kwargs)
            start = time.time()
            try:
                response = await original(*args, **kwargs)
            except Exception as exc:
                latency_ms = (time.time() - start) * 1000
                emit_llm_error(event_name, exc, latency_ms)
                raise
            latency_ms = (time.time() - start) * 1000

            if kwargs.get("stream") is True:
                return self._wrap_async_stream_iterator(event_name, kwargs, response, start)

            emit_llm_events(
                event_name,
                kwargs,
                response,
                extractors.output,
                extractors.meta,
                self.capture_params,
                latency_ms,
                pricing_table=self.pricing_table,
                extract_tool_calls=extractors.tool_calls,
                extra_params=type(self).derive_params(kwargs),
            )
            return response

        return wrapped

    def _wrap_stream_iterator(
        self,
        event_name: str,
        kwargs: Dict[str, Any],
        stream: Iterator[Any],
        start: float,
    ) -> Iterator[Any]:
        extractors = self._extractors()
        aggregate = type(self).aggregate_stream
        chunks: list[Any] = []

        def generator() -> Iterator[Any]:
            try:
                for chunk in stream:
                    chunks.append(chunk)
                    yield chunk
            except Exception as exc:
                emit_llm_error(event_name, exc, (time.time() - start) * 1000)
                raise
            latency_ms = (time.time() - start) * 1000
            response = aggregate(chunks)
            if response is None:
                return
            emit_llm_events(
                event_name,
                kwargs,
                response,
                extractors.output,
                extractors.meta,
                self.capture_params,
                latency_ms,
                pricing_table=self.pricing_table,
                extract_tool_calls=extractors.tool_calls,
                extra_params=type(self).derive_params(kwargs),
            )

        return generator()

    def _wrap_async_stream_iterator(
        self,
        event_name: str,
        kwargs: Dict[str, Any],
        stream: AsyncIterator[Any],
        start: float,
    ) -> AsyncIterator[Any]:
        extractors = self._extractors()
        aggregate = type(self).aggregate_stream
        chunks: list[Any] = []

        async def generator() -> AsyncIterator[Any]:
            try:
                async for chunk in stream:
                    chunks.append(chunk)
                    yield chunk
            except Exception as exc:
                emit_llm_error(event_name, exc, (time.time() - start) * 1000)
                raise
            latency_ms = (time.time() - start) * 1000
            response = aggregate(chunks)
            if response is None:
                return
            emit_llm_events(
                event_name,
                kwargs,
                response,
                extractors.output,
                extractors.meta,
                self.capture_params,
                latency_ms,
                pricing_table=self.pricing_table,
                extract_tool_calls=extractors.tool_calls,
                extra_params=type(self).derive_params(kwargs),
            )

        return generator()

    def disconnect(self) -> None:
        if self._client is None:
            return
        for key, orig in self._originals.items():
            try:
                parts = key.split(".")
                obj = self._client
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], orig)
            except Exception:
                log.warning("Could not restore %s", key)
        self._client = None
        self._originals.clear()

    def adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            name=self.name,
            adapter_type="provider",
            connected=self._client is not None,
        )

    # --- internals ---

    class _Extractors:
        __slots__ = ("output", "meta", "tool_calls")

        def __init__(
            self,
            output: Callable[[Any], Any],
            meta: Callable[[Any], Dict[str, Any]],
            tool_calls: Callable[[Any], list[dict[str, Any]]],
        ) -> None:
            self.output = output
            self.meta = meta
            self.tool_calls = tool_calls

    def _extractors(self) -> "MonkeyPatchProvider._Extractors":
        return MonkeyPatchProvider._Extractors(
            output=type(self).extract_output,
            meta=type(self).extract_meta,
            tool_calls=type(self).extract_tool_calls,
        )
