"""
Tests for ACP-origin payload detection and normalization.
"""

import pytest

from layerlens.instrument.adapters.protocols.a2a.acp_normalizer import ACPNormalizer


class TestACPDetection:
    def setup_method(self):
        self.normalizer = ACPNormalizer()

    def test_detect_via_header(self):
        assert self.normalizer.detect_acp_origin(
            payload={},
            headers={"X-ACP-Version": "1.0"},
        )

    def test_detect_via_lowercase_header(self):
        assert self.normalizer.detect_acp_origin(
            payload={},
            headers={"x-acp-version": "1.0"},
        )

    def test_detect_via_payload_namespace(self):
        assert self.normalizer.detect_acp_origin(
            payload={"acp": {"version": "1.0"}},
        )

    def test_detect_via_task_run(self):
        assert self.normalizer.detect_acp_origin(
            payload={"params": {"task_run": {"id": "tr-1"}}},
        )

    def test_no_detection(self):
        assert not self.normalizer.detect_acp_origin(
            payload={"method": "tasks/send", "params": {"task": {"id": "t-1"}}},
        )


class TestACPNormalization:
    def setup_method(self):
        self.normalizer = ACPNormalizer()

    def test_normalize_task_run_to_task(self):
        payload = {
            "params": {
                "task_run": {
                    "id": "tr-001",
                    "input": {"messages": [{"role": "user", "content": "Hello"}]},
                    "output": {"artifacts": [{"type": "text", "data": "Hi"}]},
                    "status": "running",
                    "metadata": {"source": "acp-agent"},
                }
            }
        }
        result = self.normalizer.normalize(payload)
        task = result["params"]["task"]
        assert task["id"] == "tr-001"
        assert task["history"] == [{"role": "user", "content": "Hello"}]
        assert task["artifacts"] == [{"type": "text", "data": "Hi"}]
        assert task["status"]["state"] == "working"  # running → working
        assert task["metadata"]["source"] == "acp-agent"

    def test_status_mapping(self):
        mappings = {
            "running": "working",
            "completed": "completed",
            "failed": "failed",
            "cancelled": "cancelled",
            "pending": "submitted",
        }
        for acp_status, expected_a2a in mappings.items():
            payload = {"params": {"task_run": {"id": "t", "status": acp_status}}}
            result = self.normalizer.normalize(payload)
            assert result["params"]["task"]["status"]["state"] == expected_a2a

    def test_acp_namespace_removal(self):
        payload = {"acp": {"version": "1.0"}, "data": "test"}
        result = self.normalizer.normalize(payload)
        assert "acp" not in result
        assert result["metadata"]["acp_version"] == "1.0"

    def test_detect_and_normalize(self):
        payload = {
            "acp": {"version": "1.0"},
            "params": {"task_run": {"id": "tr-1", "status": "running"}},
        }
        normalized, is_acp = self.normalizer.detect_and_normalize(payload)
        assert is_acp
        assert "task" in normalized["params"]
        assert "task_run" not in normalized["params"]

    def test_detect_and_normalize_non_acp(self):
        payload = {"method": "tasks/send", "params": {"task": {"id": "t-1"}}}
        normalized, is_acp = self.normalizer.detect_and_normalize(payload)
        assert not is_acp
        assert normalized is payload

    def test_dict_status_normalization(self):
        payload = {
            "params": {
                "task_run": {
                    "id": "tr-2",
                    "status": {"state": "running", "details": "Processing"},
                }
            }
        }
        result = self.normalizer.normalize(payload)
        assert result["params"]["task"]["status"]["state"] == "working"
