"""Background trace upload: per-client channels, queue, circuit breaker.

Delivery is best-effort by design — instrumentation must never crash or block
the host app. The loss modes are deliberate and bounded:

* queue overflow (64/client): new traces dropped with a WARNING;
* circuit breaker (10 consecutive failures → open 60 s): traces discarded,
  including items already dequeued by the worker, until the half-open retry;
* process exit: ``atexit`` drains via :func:`shutdown_uploads`; abrupt kills
  can lose queued traces (no public flush API yet — D1 decision).
"""

from __future__ import annotations

import os
import json
import time
import queue
import atexit
import logging
import weakref
import tempfile
import threading
from typing import Any, Dict, Tuple, Optional

log: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-client upload channel
# ---------------------------------------------------------------------------


class UploadChannel:
    """Per-client upload state: circuit breaker + background worker + queue.

    Each ``client`` gets its own channel so that a failing backend A
    doesn't trip the breaker for a healthy backend B.
    """

    _THRESHOLD = 10
    _COOLDOWN_S = 60.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._error_count = 0
        self._circuit_open = False
        self._opened_at: float = 0.0
        self._queue: queue.Queue[Optional[Tuple[Any, Dict[str, Any]]]] = queue.Queue(maxsize=64)
        self._worker: Optional[threading.Thread] = None

    # -- Circuit breaker --

    def _allow(self) -> bool:
        with self._lock:
            if not self._circuit_open:
                return True
            if time.monotonic() - self._opened_at >= self._COOLDOWN_S:
                self._circuit_open = False
                self._error_count = 0
                log.info("layerlens: upload circuit breaker half-open, retrying")
                return True
            return False

    def _on_success(self) -> None:
        with self._lock:
            if self._error_count > 0:
                self._error_count = 0
                self._circuit_open = False

    def _on_failure(self) -> None:
        with self._lock:
            self._error_count += 1
            if self._error_count >= self._THRESHOLD and not self._circuit_open:
                self._circuit_open = True
                self._opened_at = time.monotonic()
                log.warning(
                    "layerlens: upload circuit breaker OPEN after %d errors (cooldown %.0fs)",
                    self._error_count,
                    self._COOLDOWN_S,
                )

    # -- Worker thread --

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            client, payload = item
            if not self._allow():
                continue
            try:
                path = _write_trace_file(payload)
            except Exception:
                # A serialize failure must not kill the daemon worker (every
                # queued trace behind it would be lost) — count it and continue.
                self._on_failure()
                log.warning("layerlens: failed to serialize trace for upload", exc_info=True)
                continue
            try:
                client.traces.upload(path)
                self._on_success()
            except Exception:
                self._on_failure()
                log.warning("layerlens: background trace upload failed", exc_info=True)
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    log.debug("Failed to remove temp trace file: %s", path)

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="layerlens-upload",
            )
            self._worker.start()

    def enqueue(self, client: Any, payload: Dict[str, Any]) -> bool:
        """Enqueue a trace for background upload. Returns False if dropped."""
        if _sync_mode:
            self._upload_sync(client, payload)
            return True
        if not self._allow():
            return False
        self._ensure_worker()
        try:
            self._queue.put_nowait((client, payload))
            return True
        except queue.Full:
            log.warning(
                "layerlens: upload queue full, dropping trace %s",
                payload.get("trace_id", "?"),
            )
            return False

    def _upload_sync(self, client: Any, payload: Dict[str, Any]) -> None:
        """Synchronous upload (used in tests)."""
        if not self._allow():
            return
        try:
            path = _write_trace_file(payload)
        except Exception:
            self._on_failure()
            log.warning("layerlens: failed to serialize trace for upload", exc_info=True)
            return
        try:
            client.traces.upload(path)
            self._on_success()
        except Exception:
            self._on_failure()
            log.warning("layerlens: trace upload failed", exc_info=True)
        finally:
            try:
                os.unlink(path)
            except OSError:
                log.debug("Failed to remove temp trace file: %s", path)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Drain the queue and stop the worker thread."""
        if self._worker is None or not self._worker.is_alive():
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout)
        self._worker = None


# ---------------------------------------------------------------------------
# Channel registry (one per client)
# ---------------------------------------------------------------------------

_ATTR = "_layerlens_upload_channel"
_channels: list[UploadChannel] = []  # keeps refs for shutdown_uploads
_registry_lock = threading.Lock()
# Fallback for clients that reject attribute injection (slotted/frozen):
# weakly keyed so a dropped client doesn't pin its channel forever.
_side_channels: "weakref.WeakKeyDictionary[Any, UploadChannel]" = weakref.WeakKeyDictionary()
# Last resort for clients that are not weak-referenceable either; deliberately
# pins the client (bounded by the number of distinct clients in a process).
_pinned_side_channels: Dict[int, Tuple[Any, UploadChannel]] = {}


def _lookup_side_channel(client: Any) -> Optional[UploadChannel]:
    try:
        ch = _side_channels.get(client)
    except TypeError:
        ch = None
    if ch is not None:
        return ch
    pinned = _pinned_side_channels.get(id(client))
    if pinned is not None and pinned[0] is client:
        return pinned[1]
    return None


def _get_channel(client: Any) -> UploadChannel:
    """Return (or create) the UploadChannel for *client*.

    The channel is stored directly on the client object so that identity
    is tied to the object's lifetime, not its ``id()`` (which can be
    reused after garbage collection). Slotted/frozen clients that reject
    attribute injection fall back to a weak side dict instead.
    """
    ch = getattr(client, _ATTR, None)
    if isinstance(ch, UploadChannel):
        return ch
    with _registry_lock:
        # Double-check under lock
        ch = getattr(client, _ATTR, None)
        if isinstance(ch, UploadChannel):
            return ch
        ch = _lookup_side_channel(client)
        if ch is not None:
            return ch
        ch = UploadChannel()
        try:
            object.__setattr__(client, _ATTR, ch)
        except (AttributeError, TypeError):
            try:
                _side_channels[client] = ch
            except TypeError:
                _pinned_side_channels[id(client)] = (client, ch)
        _channels.append(ch)
        return ch


# ---------------------------------------------------------------------------
# Public API (used by TraceCollector)
# ---------------------------------------------------------------------------

_sync_mode = False


def enqueue_upload(client: Any, payload: Dict[str, Any]) -> bool:
    """Enqueue a trace for background upload via the client's channel."""
    return _get_channel(client).enqueue(client, payload)


def shutdown_uploads(timeout: float = 5.0) -> None:
    """Shut down all upload channels."""
    with _registry_lock:
        channels = list(_channels)
    for ch in channels:
        ch.shutdown(timeout)


atexit.register(shutdown_uploads)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_trace_file(payload: Dict[str, Any]) -> str:
    fd, path = tempfile.mkstemp(suffix=".json", prefix="layerlens_trace_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump([payload], f, default=str)
    except Exception:
        # A serialization failure (e.g. a circular reference) must not leave the
        # mkstemp'd file orphaned on disk (LAY-3572 / B37).
        try:
            os.unlink(path)
        except OSError:
            log.debug("Failed to remove temp trace file after write failure: %s", path)
        raise
    return path


def upload_trace(client: Any, payload: Dict[str, Any]) -> None:
    """Synchronous upload (testing convenience)."""
    _get_channel(client)._upload_sync(client, payload)
