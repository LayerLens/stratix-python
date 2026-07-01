"""Upload data-loss observability: opt-in callback + per-reason drop counter
(LAY-3636, F-L7-001). Trace loss was previously log-only; the host app had no
programmatic signal. These exercise the deterministic loss paths."""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from layerlens.instrument import _upload
from layerlens.instrument._upload import UploadChannel


def _make_client() -> Mock:
    client = Mock()
    client.traces = Mock()
    client.traces.upload = Mock()
    return client


@pytest.fixture(autouse=True)
def _reset_loss_state():
    prev_sync = _upload._sync_mode
    _upload._sync_mode = False  # exercise the async enqueue path deterministically
    _upload.set_upload_loss_callback(None)
    _upload.reset_upload_loss_stats()
    yield
    _upload._sync_mode = prev_sync
    _upload.set_upload_loss_callback(None)
    _upload.reset_upload_loss_stats()


@pytest.mark.invariant
class TestUploadLossObservability:
    def test_circuit_open_enqueue_records_loss(self):
        seen = []
        _upload.set_upload_loss_callback(lambda reason, payload: seen.append((reason, payload.get("trace_id"))))
        ch = UploadChannel()
        ch._circuit_open = True
        ch._opened_at = time.monotonic()

        ok = ch.enqueue(_make_client(), {"trace_id": "t1"})

        assert ok is False
        assert ("circuit_open", "t1") in seen
        assert _upload.get_upload_loss_stats().get("circuit_open") == 1

    def test_rejected_upload_records_loss(self):
        seen = []
        _upload.set_upload_loss_callback(lambda r, p: seen.append(r))
        client = _make_client()
        client.traces.upload.return_value = Mock(trace_ids=[])  # backend rejected (no ids)
        ch = UploadChannel()

        ch._upload_sync(client, {"trace_id": "t2"})

        assert "rejected" in seen
        assert _upload.get_upload_loss_stats().get("rejected") == 1

    def test_upload_exception_records_loss(self):
        seen = []
        _upload.set_upload_loss_callback(lambda r, p: seen.append(r))
        client = _make_client()
        client.traces.upload.side_effect = RuntimeError("boom")
        ch = UploadChannel()

        ch._upload_sync(client, {"trace_id": "t3"})

        assert "upload_error" in seen
        assert _upload.get_upload_loss_stats().get("upload_error") == 1

    def test_callback_exception_is_swallowed(self):
        def bad(reason, payload):
            raise ValueError("callback boom")

        _upload.set_upload_loss_callback(bad)
        client = _make_client()
        client.traces.upload.side_effect = RuntimeError("boom")
        ch = UploadChannel()

        # A misbehaving callback must never break the (already best-effort) upload path.
        ch._upload_sync(client, {"trace_id": "t4"})

        assert _upload.get_upload_loss_stats().get("upload_error") == 1

    def test_no_callback_by_default_is_compatible(self):
        # Default: no callback registered. Losses are still counted; nothing raises.
        client = _make_client()
        client.traces.upload.return_value = None  # rejected
        ch = UploadChannel()

        ch._upload_sync(client, {"trace_id": "t5"})

        assert _upload.get_upload_loss_stats().get("rejected") == 1
