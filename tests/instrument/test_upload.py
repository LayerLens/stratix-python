"""Upload-channel lifecycle tests (LAY-3579 / T5).

Covers the delivery path end to end: per-client channel identity (incl. the
N3 frozen/slotted-client fallback), queue-overflow drop semantics, the
circuit breaker state machine, breaker-open discard of already-dequeued
items, shutdown drain, and sync mode.
"""

from __future__ import annotations

import time
import logging
from typing import Any, List
from unittest.mock import Mock

import pytest

from layerlens.instrument import _upload
from layerlens.instrument._upload import UploadChannel, _get_channel

_LOGGER = "layerlens.instrument._upload"


@pytest.fixture(autouse=True)
def _isolate_channel_registry():
    """Keep channels created here from leaking into other tests."""
    before = list(_upload._channels)
    yield
    added = [ch for ch in _upload._channels if ch not in before]
    for ch in added:
        try:
            ch.shutdown(timeout=1.0)
        except Exception:
            pass
    with _upload._registry_lock:
        _upload._channels[:] = before


def _make_client() -> Mock:
    client = Mock()
    client.traces = Mock()
    client.traces.upload = Mock()
    return client


class _SlottedClient:
    """Simulates an SDK client that rejects attribute injection (N3)."""

    __slots__ = ("traces", "__weakref__")

    def __init__(self) -> None:
        self.traces = Mock()


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        time.sleep(0.01)


# ---------------------------------------------------------------------------
# Channel identity (incl. N3)
# ---------------------------------------------------------------------------


class TestChannelIdentity:
    def test_same_client_returns_same_channel(self) -> None:
        client = _make_client()
        assert _get_channel(client) is _get_channel(client)

    def test_distinct_clients_get_distinct_channels(self) -> None:
        a, b = _make_client(), _make_client()
        assert _get_channel(a) is not _get_channel(b)

    def test_channel_is_stored_on_the_client(self) -> None:
        client = _make_client()
        ch = _get_channel(client)
        assert getattr(client, _upload._ATTR) is ch

    def test_slotted_client_gets_stable_channel(self) -> None:
        """N3: attribute injection fails on slotted clients — the promised
        side-dict fallback must still return ONE channel per client."""
        client = _SlottedClient()
        assert _get_channel(client) is _get_channel(client)

    def test_slotted_client_does_not_grow_channel_registry(self) -> None:
        """N3: without the fallback every call creates (and pins) a fresh
        channel + daemon worker, growing ``_channels`` without bound."""
        client = _SlottedClient()
        before = len(_upload._channels)
        for _ in range(5):
            _get_channel(client)
        assert len(_upload._channels) - before == 1


# ---------------------------------------------------------------------------
# Queue overflow (documented designed loss, N4)
# ---------------------------------------------------------------------------


class TestQueueOverflow:
    def test_enqueue_drops_when_queue_full(self, monkeypatch: Any, caplog: Any) -> None:
        monkeypatch.setattr(_upload, "_sync_mode", False)
        ch = UploadChannel()
        monkeypatch.setattr(ch, "_ensure_worker", lambda: None)  # never drain
        client = _make_client()

        capacity = ch._queue.maxsize
        for i in range(capacity):
            assert ch.enqueue(client, {"trace_id": f"t{i}"}) is True

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            assert ch.enqueue(client, {"trace_id": "overflow"}) is False
        assert any("queue full" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Circuit breaker state machine
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_opens_after_threshold_consecutive_failures(self) -> None:
        ch = UploadChannel()
        for _ in range(ch._THRESHOLD - 1):
            ch._on_failure()
        assert ch._allow() is True
        ch._on_failure()
        assert ch._circuit_open is True
        assert ch._allow() is False

    def test_half_open_after_cooldown(self) -> None:
        ch = UploadChannel()
        for _ in range(ch._THRESHOLD):
            ch._on_failure()
        assert ch._allow() is False
        ch._opened_at = time.monotonic() - ch._COOLDOWN_S - 1
        assert ch._allow() is True  # half-open: one retry allowed
        assert ch._circuit_open is False

    def test_success_resets_error_count(self) -> None:
        ch = UploadChannel()
        for _ in range(ch._THRESHOLD - 1):
            ch._on_failure()
        ch._on_success()
        assert ch._error_count == 0
        for _ in range(ch._THRESHOLD - 1):
            ch._on_failure()
        assert ch._allow() is True  # consecutive count restarted


# ---------------------------------------------------------------------------
# Breaker-open discard of already-dequeued items (documented designed loss, N4)
# ---------------------------------------------------------------------------


class TestBreakerOpenDiscard:
    def test_open_breaker_discards_dequeued_items(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(_upload, "_sync_mode", False)
        ch = UploadChannel()
        client = _make_client()
        for i in range(3):
            ch._queue.put_nowait((client, {"trace_id": f"t{i}"}))

        # Open the breaker, then start the worker: items are dequeued and
        # dropped without an upload attempt.
        ch._circuit_open = True
        ch._opened_at = time.monotonic()
        ch._ensure_worker()
        _wait_until(lambda: ch._queue.empty())

        assert ch._queue.empty()
        assert client.traces.upload.call_count == 0
        ch.shutdown(timeout=2.0)


# ---------------------------------------------------------------------------
# Shutdown drain
# ---------------------------------------------------------------------------


class TestShutdownDrain:
    def test_shutdown_drains_pending_uploads(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(_upload, "_sync_mode", False)
        ch = UploadChannel()
        client = _make_client()
        for i in range(5):
            assert ch.enqueue(client, {"trace_id": f"t{i}"}) is True

        ch.shutdown(timeout=5.0)

        assert client.traces.upload.call_count == 5
        assert ch._worker is None

    def test_shutdown_without_worker_is_noop(self) -> None:
        ch = UploadChannel()
        ch.shutdown(timeout=1.0)  # never started — must not raise
        assert ch._worker is None

    def test_shutdown_uploads_covers_registered_channels(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(_upload, "_sync_mode", False)
        client = _make_client()
        ch = _get_channel(client)
        assert ch.enqueue(client, {"trace_id": "t"}) is True

        _upload.shutdown_uploads(timeout=5.0)

        assert client.traces.upload.call_count == 1


# ---------------------------------------------------------------------------
# Sync mode
# ---------------------------------------------------------------------------


class TestSyncMode:
    def test_sync_mode_uploads_inline(self) -> None:
        # The autouse conftest fixture sets _sync_mode = True.
        client = _make_client()
        assert _upload.enqueue_upload(client, {"trace_id": "t"}) is True
        assert client.traces.upload.call_count == 1

    def test_sync_mode_swallows_upload_failure(self) -> None:
        client = _make_client()
        client.traces.upload.side_effect = RuntimeError("backend down")
        assert _upload.enqueue_upload(client, {"trace_id": "t"}) is True  # never raises


# ---------------------------------------------------------------------------
# Worker resilience
# ---------------------------------------------------------------------------


class TestWorkerResilience:
    def test_failed_upload_does_not_kill_worker(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(_upload, "_sync_mode", False)
        ch = UploadChannel()
        client = _make_client()
        calls: List[str] = []

        def _upload_fn(path: str) -> None:
            calls.append(path)
            if len(calls) == 1:
                raise RuntimeError("transient")

        client.traces.upload.side_effect = _upload_fn
        assert ch.enqueue(client, {"trace_id": "t1"}) is True
        assert ch.enqueue(client, {"trace_id": "t2"}) is True
        _wait_until(lambda: len(calls) >= 2)

        assert len(calls) == 2  # second item still processed
        assert ch._worker is not None and ch._worker.is_alive()
        ch.shutdown(timeout=2.0)
