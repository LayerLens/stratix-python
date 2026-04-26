"""HTTP-based event sinks.

The SDK ships adapter telemetry to atlas-app via httpx. Two sinks are
provided:

* :class:`HttpEventSink` — synchronous, immediate or batched flush, used
  by sync adapters (LangChain callbacks, OpenAI client wrappers).
* :class:`AsyncHttpEventSink` — async equivalent for adapters that run
  inside an event loop (LangGraph, AsyncAnthropic).

Both sinks:

* Reuse the SDK's existing ``X-API-Key`` auth and base-URL conventions.
* Apply exponential backoff (0.5s → 8s) on 429/5xx, matching
  ``layerlens._base_client.MAX_RETRY_DELAY``.
* Treat the network as best-effort: a sink that cannot deliver an event
  logs at DEBUG and drops the batch. The adapter's circuit breaker is
  the authority on persistent transport failures.

The default endpoint is ``/api/v1/telemetry/spans`` on the atlas-app
control plane. The path is configurable so the same sink can be
pointed at a self-hosted ingest gateway.
"""

from __future__ import annotations

import os
import time
import asyncio
import logging
import threading
from typing import Any, Dict, List, Optional

import httpx

from layerlens.instrument.adapters._base.sinks import EventSink

logger = logging.getLogger(__name__)


_DEFAULT_BASE_URL = os.environ.get(
    "LAYERLENS_STRATIX_BASE_URL",
    "https://api.layerlens.ai/api/v1",
)
_DEFAULT_PATH = "/telemetry/spans"
_DEFAULT_TIMEOUT_S = 10.0
_DEFAULT_MAX_BATCH = 50
_DEFAULT_FLUSH_INTERVAL_S = 1.0

_MAX_RETRIES = 2
_INITIAL_RETRY_DELAY_S = 0.5
_MAX_RETRY_DELAY_S = 8.0
_RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def _format_event(
    event_type: str,
    payload: Dict[str, Any],
    timestamp_ns: int,
    adapter_name: str,
    trace_id: Optional[str],
) -> Dict[str, Any]:
    """Build the wire JSON object for a single event.

    ``timestamp_ns`` is wall-clock UTC nanoseconds since the Unix epoch
    (the value returned by :func:`time.time_ns`). The atlas-app worker
    consuming this field assumes UTC nanoseconds — do not change this
    contract without coordinating a wire-schema version bump.
    """
    return {
        "event_type": event_type,
        "payload": payload,
        "timestamp_ns": timestamp_ns,
        "adapter": adapter_name,
        "trace_id": trace_id,
    }


def _post_with_retry(
    client: httpx.Client,
    path: str,
    body: Dict[str, Any],
) -> bool:
    """POST ``body`` to ``path`` with backoff on 429 / 5xx.

    Returns True if the post succeeded (2xx), False if it gave up after
    retries or hit a 4xx (which we don't retry — the body is bad).
    """
    delay = _INITIAL_RETRY_DELAY_S
    retries_left = _MAX_RETRIES

    while True:
        try:
            resp = client.post(path, json=body)
        except httpx.HTTPError as exc:
            if retries_left > 0:
                logger.debug(
                    "HttpEventSink transport error: %s (retries left: %d)",
                    exc,
                    retries_left,
                )
                time.sleep(delay)
                delay = min(delay * 2, _MAX_RETRY_DELAY_S)
                retries_left -= 1
                continue
            logger.debug("HttpEventSink giving up after transport errors", exc_info=True)
            return False

        if resp.status_code in _RETRY_STATUS_CODES and retries_left > 0:
            retry_after = resp.headers.get("retry-after")
            try:
                sleep_for = float(retry_after) if retry_after else delay
            except ValueError:
                sleep_for = delay
            sleep_for = min(sleep_for, _MAX_RETRY_DELAY_S)
            time.sleep(sleep_for)
            delay = min(delay * 2, _MAX_RETRY_DELAY_S)
            retries_left -= 1
            continue

        if 200 <= resp.status_code < 300:
            return True

        # Non-retriable — log and drop.
        logger.debug(
            "HttpEventSink got non-retriable status %d body=%s",
            resp.status_code,
            resp.text[:500],
        )
        return False


_CONSECUTIVE_DROP_WARN_THRESHOLD = 3
_DROP_WARN_LOG_CODE = "layerlens.sink.batch_dropped"


class HttpEventSink(EventSink):
    """Synchronous HTTP sink that POSTs events to atlas-app.

    Args:
        adapter_name: Tag inserted into every event so the server can
            attribute the source adapter. Required.
        api_key: LayerLens API key (``X-API-Key`` header). Falls back to
            the ``LAYERLENS_STRATIX_API_KEY`` env var.
        base_url: Base URL for atlas-app. Defaults to
            ``$LAYERLENS_STRATIX_BASE_URL`` or
            ``https://api.layerlens.ai/api/v1``.
        path: Endpoint path. Defaults to ``"/telemetry/spans"``.
        trace_id: Optional trace-id to tag all events from this sink.
        max_batch: Max events to buffer before forced flush. Default 50.
        flush_interval_s: Wall-clock interval after which a partial
            buffer is flushed. Default 1.0 second.
        timeout_s: Per-request HTTP timeout in seconds. Default 10.0.
        client: Optional pre-built ``httpx.Client``. If supplied, the
            sink will not close it on :meth:`close`.
        background_flush: If True (default), spawn a daemon thread that
            wakes every ``flush_interval_s`` and forces a flush of any
            buffered events. This bounds telemetry latency for adapters
            that emit sporadically. Set False for deterministic
            single-thread tests.

    The sink buffers events and flushes either when ``max_batch`` is
    reached, when ``flush_interval_s`` elapses since the last flush, or
    when :meth:`flush` / :meth:`close` is called.

    After 3 consecutive batch failures the sink logs at WARN once with
    error code ``layerlens.sink.batch_dropped`` so log alerting can pick
    it up. Subsequent failures within the same window stay at DEBUG.

    :meth:`stats` exposes ``batches_sent``, ``batches_dropped``, and
    ``buffer_size`` for callers that want to surface sink health.
    """

    def __init__(
        self,
        adapter_name: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        path: str = _DEFAULT_PATH,
        trace_id: Optional[str] = None,
        max_batch: int = _DEFAULT_MAX_BATCH,
        flush_interval_s: float = _DEFAULT_FLUSH_INTERVAL_S,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        client: Optional[httpx.Client] = None,
        background_flush: bool = True,
    ) -> None:
        self._adapter_name = adapter_name
        self._trace_id = trace_id
        self._path = path
        self._max_batch = max_batch
        self._flush_interval_s = flush_interval_s

        self._buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self._closed = False
        self._batches_sent = 0
        self._batches_dropped = 0
        self._consecutive_drops = 0

        self._owns_client = client is None
        if client is not None:
            self._client = client
        else:
            resolved_key = api_key or os.environ.get("LAYERLENS_STRATIX_API_KEY", "")
            resolved_base = base_url or _DEFAULT_BASE_URL
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            if resolved_key:
                headers["X-API-Key"] = resolved_key
            self._client = httpx.Client(
                base_url=resolved_base,
                headers=headers,
                timeout=timeout_s,
            )

        # Daemon timer thread bounds idle-flush latency so a sporadic
        # adapter does not leave events buffered until process exit.
        self._stop_event = threading.Event()
        self._timer_thread: Optional[threading.Thread] = None
        if background_flush and flush_interval_s > 0:
            self._timer_thread = threading.Thread(
                target=self._timer_loop,
                name=f"layerlens-sink-{adapter_name}",
                daemon=True,
            )
            self._timer_thread.start()

    @property
    def buffer_size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def stats(self) -> Dict[str, int]:
        """Snapshot of sink-level counters for observability."""
        with self._lock:
            return {
                "batches_sent": self._batches_sent,
                "batches_dropped": self._batches_dropped,
                "buffer_size": len(self._buffer),
                "consecutive_drops": self._consecutive_drops,
            }

    def send(self, event_type: str, payload: Dict[str, Any], timestamp_ns: int) -> None:
        if self._closed:
            return

        event = _format_event(
            event_type=event_type,
            payload=payload,
            timestamp_ns=timestamp_ns,
            adapter_name=self._adapter_name,
            trace_id=self._trace_id,
        )

        should_flush = False
        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) >= self._max_batch:
                should_flush = True

        if should_flush:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            if not self._buffer:
                self._last_flush = time.monotonic()
                return
            batch = list(self._buffer)
            self._buffer.clear()
            self._last_flush = time.monotonic()

        body = {"events": batch}
        ok = _post_with_retry(self._client, self._path, body)
        if ok:
            with self._lock:
                self._batches_sent += 1
                self._consecutive_drops = 0
        else:
            with self._lock:
                self._batches_dropped += 1
                self._consecutive_drops += 1
                consecutive = self._consecutive_drops
            if consecutive == _CONSECUTIVE_DROP_WARN_THRESHOLD:
                logger.warning(
                    "%s: HttpEventSink for adapter %s dropped %d consecutive batches "
                    "(latest had %d events). Telemetry pipeline may be degraded.",
                    _DROP_WARN_LOG_CODE,
                    self._adapter_name,
                    consecutive,
                    len(batch),
                )
            else:
                logger.debug(
                    "HttpEventSink dropped batch of %d events for adapter %s "
                    "(consecutive=%d)",
                    len(batch),
                    self._adapter_name,
                    consecutive,
                )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._timer_thread is not None:
            self._stop_event.set()
            self._timer_thread.join(timeout=max(self._flush_interval_s * 2, 1.0))
        try:
            self.flush()
        finally:
            if self._owns_client:
                try:
                    self._client.close()
                except Exception:
                    logger.debug("HttpEventSink client.close() failed", exc_info=True)

    def _timer_loop(self) -> None:
        """Background daemon: wake every ``flush_interval_s`` and flush."""
        while not self._stop_event.wait(self._flush_interval_s):
            if self._closed:
                return
            try:
                with self._lock:
                    has_data = bool(self._buffer)
                if has_data:
                    self.flush()
            except Exception:
                logger.debug("HttpEventSink timer flush failed", exc_info=True)


class AsyncHttpEventSink(EventSink):
    """Async HTTP sink for adapters running inside an event loop.

    The :meth:`send`, :meth:`flush`, and :meth:`close` methods on this
    class are synchronous (matching :class:`EventSink`) but they
    schedule work on the running loop. A separate :meth:`asend`,
    :meth:`aflush`, and :meth:`aclose` are provided for callers that
    can ``await`` the result.

    For callers that emit events from synchronous code paths inside an
    async program (a common shape for OpenAI's sync client used inside
    a FastAPI handler), use :class:`HttpEventSink` instead — it does
    not require the loop.
    """

    def __init__(
        self,
        adapter_name: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        path: str = _DEFAULT_PATH,
        trace_id: Optional[str] = None,
        max_batch: int = _DEFAULT_MAX_BATCH,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._adapter_name = adapter_name
        self._trace_id = trace_id
        self._path = path
        self._max_batch = max_batch
        self._buffer: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._closed = False

        self._owns_client = client is None
        if client is not None:
            self._client = client
        else:
            resolved_key = api_key or os.environ.get("LAYERLENS_STRATIX_API_KEY", "")
            resolved_base = base_url or _DEFAULT_BASE_URL
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            if resolved_key:
                headers["X-API-Key"] = resolved_key
            self._client = httpx.AsyncClient(
                base_url=resolved_base,
                headers=headers,
                timeout=timeout_s,
            )

    def send(self, event_type: str, payload: Dict[str, Any], timestamp_ns: int) -> None:
        """Sync entrypoint compatible with :class:`EventSink`.

        Schedules the post on the running event loop without blocking.
        If no loop is running, falls back to ``asyncio.run`` for the
        single send (slow path; prefer :meth:`asend`).
        """
        if self._closed:
            return
        coro = self.asend(event_type, payload, timestamp_ns)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            asyncio.run(coro)

    async def asend(
        self,
        event_type: str,
        payload: Dict[str, Any],
        timestamp_ns: int,
    ) -> None:
        if self._closed:
            return

        event = _format_event(
            event_type=event_type,
            payload=payload,
            timestamp_ns=timestamp_ns,
            adapter_name=self._adapter_name,
            trace_id=self._trace_id,
        )

        async with self._lock:
            self._buffer.append(event)
            should_flush = len(self._buffer) >= self._max_batch

        if should_flush:
            await self.aflush()

    def flush(self) -> None:
        if self._closed:
            return
        coro = self.aflush()
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            asyncio.run(coro)

    async def aflush(self) -> None:
        async with self._lock:
            if not self._buffer:
                return
            batch = list(self._buffer)
            self._buffer.clear()

        delay = _INITIAL_RETRY_DELAY_S
        retries_left = _MAX_RETRIES
        body = {"events": batch}

        while True:
            try:
                resp = await self._client.post(self._path, json=body)
            except httpx.HTTPError:
                if retries_left > 0:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, _MAX_RETRY_DELAY_S)
                    retries_left -= 1
                    continue
                logger.debug(
                    "AsyncHttpEventSink dropped batch of %d events", len(batch)
                )
                return

            if resp.status_code in _RETRY_STATUS_CODES and retries_left > 0:
                retry_after = resp.headers.get("retry-after")
                try:
                    sleep_for = float(retry_after) if retry_after else delay
                except ValueError:
                    sleep_for = delay
                await asyncio.sleep(min(sleep_for, _MAX_RETRY_DELAY_S))
                delay = min(delay * 2, _MAX_RETRY_DELAY_S)
                retries_left -= 1
                continue

            if 200 <= resp.status_code < 300:
                return

            logger.debug(
                "AsyncHttpEventSink got non-retriable status %d", resp.status_code
            )
            return

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        coro = self.aclose()
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            asyncio.run(coro)

    async def aclose(self) -> None:
        await self.aflush()
        if self._owns_client:
            try:
                await self._client.aclose()
            except Exception:
                logger.debug("AsyncHttpEventSink client.aclose() failed", exc_info=True)
