"""Tests for STRATIX Event Sinks (SDK-side: APIUploadSink, LoggingSink)."""

import json
import logging
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from layerlens.instrument.adapters._sinks import (
    APIUploadSink,
    EventSink,
    LoggingSink,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ConcreteEventSink(EventSink):
    """Minimal concrete sink for ABC tests."""

    def __init__(self):
        self.sent: list[tuple] = []
        self.flushed = False
        self.closed = False

    def send(self, event_type, payload, timestamp_ns):
        self.sent.append((event_type, payload, timestamp_ns))

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# EventSink ABC
# ---------------------------------------------------------------------------


class TestEventSinkABC:
    """Tests for the EventSink abstract base class."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            EventSink()

    def test_concrete_subclass_satisfies_abc(self):
        sink = ConcreteEventSink()
        sink.send("test.event", {"key": "val"}, time.time_ns())
        sink.flush()
        sink.close()
        assert len(sink.sent) == 1
        assert sink.flushed
        assert sink.closed


# ---------------------------------------------------------------------------
# APIUploadSink
# ---------------------------------------------------------------------------


class TestAPIUploadSink:
    """Tests for APIUploadSink."""

    def _make_client(self):
        """Create a mock layerlens client with traces.upload()."""
        client = MagicMock()
        client.traces = MagicMock()
        client.traces.upload = MagicMock()
        return client

    def test_send_buffers_events(self):
        client = self._make_client()
        sink = APIUploadSink(client=client, buffer_size=0)  # disable auto-flush
        ts = time.time_ns()

        sink.send("model.invoke", {"model": "gpt-4"}, ts)
        sink.send("tool.call", {"tool_name": "search"}, ts)

        # Not uploaded yet (auto-flush disabled)
        client.traces.upload.assert_not_called()
        assert len(sink._buffer) == 2

    def test_flush_uploads_jsonl(self):
        client = self._make_client()
        sink = APIUploadSink(client=client, buffer_size=0)
        ts = time.time_ns()

        sink.send("model.invoke", {"model": "gpt-4"}, ts)
        sink.flush()

        client.traces.upload.assert_called_once()
        call_kwargs = client.traces.upload.call_args
        assert "file" in call_kwargs.kwargs or len(call_kwargs.args) > 0
        assert len(sink._buffer) == 0  # buffer cleared

    def test_auto_flush_on_buffer_size(self):
        client = self._make_client()
        sink = APIUploadSink(client=client, buffer_size=2)
        ts = time.time_ns()

        sink.send("event.a", {}, ts)
        client.traces.upload.assert_not_called()

        sink.send("event.b", {}, ts)
        client.traces.upload.assert_called_once()

    def test_close_flushes_buffer(self):
        client = self._make_client()
        sink = APIUploadSink(client=client, buffer_size=0)
        ts = time.time_ns()

        sink.send("model.invoke", {"model": "gpt-4"}, ts)
        sink.close()

        client.traces.upload.assert_called_once()

    def test_send_after_close_is_noop(self):
        client = self._make_client()
        sink = APIUploadSink(client=client, buffer_size=0)

        sink.close()
        sink.send("late.event", {"should": "be ignored"}, time.time_ns())

        assert len(sink._buffer) == 0

    def test_custom_trace_id(self):
        client = self._make_client()
        sink = APIUploadSink(client=client, trace_id="my-custom-trace")

        assert sink.trace_id == "my-custom-trace"

    def test_auto_generated_trace_id(self):
        client = self._make_client()
        sink = APIUploadSink(client=client)

        assert sink.trace_id is not None
        assert len(sink.trace_id) > 0

    def test_sequence_increments(self):
        client = self._make_client()
        sink = APIUploadSink(client=client, buffer_size=0)
        ts = time.time_ns()

        sink.send("event.a", {}, ts)
        sink.send("event.b", {}, ts + 1000)
        sink.send("event.c", {}, ts + 2000)

        seqs = [e["sequence_id"] for e in sink._buffer]
        assert seqs == [1, 2, 3]

    def test_upload_failure_does_not_raise(self):
        """If the upload fails, APIUploadSink logs but doesn't propagate."""
        client = self._make_client()
        client.traces.upload.side_effect = RuntimeError("network error")

        sink = APIUploadSink(client=client, buffer_size=0)
        sink.send("test", {}, time.time_ns())

        # Should not raise
        sink.flush()

    def test_agent_id_included_in_records(self):
        client = self._make_client()
        sink = APIUploadSink(client=client, agent_id="test-agent", buffer_size=0)

        sink.send("test", {}, time.time_ns())

        assert sink._buffer[0]["agent_id"] == "test-agent"


# ---------------------------------------------------------------------------
# LoggingSink
# ---------------------------------------------------------------------------


class TestLoggingSink:
    """Tests for LoggingSink."""

    def test_send_logs_event(self, caplog):
        sink = LoggingSink(level=logging.INFO)
        ts = time.time_ns()

        with caplog.at_level(logging.INFO, logger="layerlens.instrument.events"):
            sink.send("model.invoke", {"model": "gpt-4"}, ts)

        assert len(caplog.records) == 1
        assert "model.invoke" in caplog.records[0].message

    def test_close_prevents_further_logging(self, caplog):
        sink = LoggingSink(level=logging.INFO)

        sink.close()

        with caplog.at_level(logging.INFO, logger="layerlens.instrument.events"):
            sink.send("late.event", {}, time.time_ns())

        assert len(caplog.records) == 0

    def test_flush_is_noop(self):
        sink = LoggingSink()
        sink.flush()  # Should not raise

    def test_custom_logger_name(self, caplog):
        sink = LoggingSink(logger_name="my.custom.logger", level=logging.WARNING)
        ts = time.time_ns()

        with caplog.at_level(logging.WARNING, logger="my.custom.logger"):
            sink.send("test", {"key": "val"}, ts)

        assert len(caplog.records) == 1
