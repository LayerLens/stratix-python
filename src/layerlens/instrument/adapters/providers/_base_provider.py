from __future__ import annotations

import abc
import time
import logging
from typing import Any, Dict

from .._base import AdapterInfo, BaseAdapter
from ._emit_helpers import emit_llm_events, emit_llm_error
from ..._context import _current_collector

log: logging.Logger = logging.getLogger(__name__)


class MonkeyPatchProvider(BaseAdapter):
    """Base for providers that monkey-patch SDK client or module methods."""

    name: str
    capture_params: frozenset[str]

    def __init__(self) -> None:
        self._client: Any = None
        self._originals: Dict[str, Any] = {}

    @staticmethod
    @abc.abstractmethod
    def extract_output(response: Any) -> Any: ...

    @staticmethod
    @abc.abstractmethod
    def extract_meta(response: Any) -> Dict[str, Any]: ...

    def _wrap_sync(self, event_name: str, original: Any) -> Any:
        extract_output = self.extract_output
        extract_meta = self.extract_meta
        capture_params = self.capture_params

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
            emit_llm_events(
                event_name, kwargs, response,
                extract_output, extract_meta, capture_params, latency_ms,
            )
            return response

        return wrapped

    def _wrap_async(self, event_name: str, original: Any) -> Any:
        extract_output = self.extract_output
        extract_meta = self.extract_meta
        capture_params = self.capture_params

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
            emit_llm_events(
                event_name, kwargs, response,
                extract_output, extract_meta, capture_params, latency_ms,
            )
            return response

        return wrapped

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
