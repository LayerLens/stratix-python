from __future__ import annotations

import os
import json
import time
import asyncio
import logging
import tempfile
import threading
from typing import Any, Dict

log: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_error_count = 0
_circuit_open = False
_opened_at: float = 0.0

_THRESHOLD = 10
_COOLDOWN_S = 60.0


def _allow() -> bool:
    global _circuit_open, _error_count
    with _lock:
        if not _circuit_open:
            return True
        if time.monotonic() - _opened_at >= _COOLDOWN_S:
            _circuit_open = False
            _error_count = 0
            log.info("layerlens: upload circuit breaker half-open, retrying")
            return True
        return False


def _on_success() -> None:
    global _error_count, _circuit_open
    with _lock:
        if _error_count > 0:
            _error_count = 0
            _circuit_open = False


def _on_failure() -> None:
    global _error_count, _circuit_open, _opened_at
    with _lock:
        _error_count += 1
        if _error_count >= _THRESHOLD and not _circuit_open:
            _circuit_open = True
            _opened_at = time.monotonic()
            log.warning(
                "layerlens: upload circuit breaker OPEN after %d errors (cooldown %.0fs)",
                _error_count,
                _COOLDOWN_S,
            )


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def _write_trace_file(payload: Dict[str, Any]) -> str:
    fd, path = tempfile.mkstemp(suffix=".json", prefix="layerlens_trace_")
    with os.fdopen(fd, "w") as f:
        json.dump([payload], f, default=str)
    return path


def upload_trace(client: Any, payload: Dict[str, Any]) -> None:
    if not _allow():
        return
    path = _write_trace_file(payload)
    try:
        client.traces.upload(path)
        _on_success()
    except Exception:
        _on_failure()
        log.warning("layerlens: trace upload failed", exc_info=True)
    finally:
        try:
            os.unlink(path)
        except OSError:
            log.debug("Failed to remove temp trace file: %s", path)


async def async_upload_trace(client: Any, payload: Dict[str, Any]) -> None:
    if not _allow():
        return
    path = await asyncio.to_thread(_write_trace_file, payload)
    try:
        await client.traces.upload(path)
        _on_success()
    except Exception:
        _on_failure()
        log.warning("layerlens: async trace upload failed", exc_info=True)
    finally:
        try:
            os.unlink(path)
        except OSError:
            log.debug("Failed to remove temp trace file: %s", path)
