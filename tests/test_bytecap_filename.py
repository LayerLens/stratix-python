"""Per-event byte cap + fail-fast upload filename sanitize (LAY-3639,
F-L12-003/004)."""

from __future__ import annotations

import json
import tempfile
from unittest.mock import Mock, patch

import pytest

from layerlens.instrument._collector import TraceCollector
from layerlens.resources.traces.traces import Traces, _validate_upload_filename
from layerlens.instrument._capture_config import CaptureConfig


@pytest.mark.invariant
class TestPerEventByteCap:
    def test_oversized_event_payload_truncated(self):
        col = TraceCollector(Mock(), CaptureConfig.standard())
        # "big" is not a content key for agent.step, so it survives redaction and
        # is large enough (~300 KiB) to exceed the per-event byte cap.
        col.emit("agent.step", {"big": "x" * (300 * 1024)}, span_id="s1")
        ev = col.events[0]
        assert ev["payload"].get("_truncated") is True
        blob = json.dumps(ev["payload"])
        assert "x" * 1024 not in blob  # the oversized data is gone
        assert len(blob.encode("utf-8")) < 4096  # marker is small

    def test_normal_event_not_truncated(self):
        col = TraceCollector(Mock(), CaptureConfig.standard())
        col.emit("agent.step", {"note": "hello"}, span_id="s1")
        ev = col.events[0]
        assert ev["payload"].get("_truncated") is not True
        assert ev["payload"].get("note") == "hello"


class TestUploadFilenameValidation:
    @pytest.mark.parametrize("name", ["trace.json", "my-trace_2.jsonl", "a.b.c.json"])
    def test_valid_filenames_pass(self, name):
        assert _validate_upload_filename(name) == name

    @pytest.mark.invariant
    @pytest.mark.parametrize("name", ["", ".", "..", "a/b.json", "a\\b.json", "bad\x00.json", "x\n.json"])
    def test_unsafe_filenames_rejected(self, name):
        with pytest.raises(ValueError):
            _validate_upload_filename(name)

    @pytest.mark.invariant
    def test_upload_rejects_unsafe_filename_before_network(self):
        client = Mock()
        client.organization_id = "org-1"
        client.project_id = "proj-1"
        client.post_cast = Mock()
        tr = Traces(client)
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("[]")
            path = f.name
        with patch("layerlens.resources.traces.traces.os.path.basename", return_value="bad\x00.json"):
            with pytest.raises(ValueError):
                tr.upload(path)
        tr._post.assert_not_called()  # fail-fast: rejected before the presign request
