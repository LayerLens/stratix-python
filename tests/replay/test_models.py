from __future__ import annotations

from layerlens.replay.models import (
    ReplayDiff,
    ReplayResult,
    ReplayStatus,
    ReplayRequest,
    BatchReplayFilter,
)


class TestReplayType:
    def test_basic_when_no_overrides(self):
        assert ReplayRequest(trace_id="t1").replay_type == "basic"

    def test_checkpoint_takes_precedence(self):
        req = ReplayRequest(
            trace_id="t1",
            checkpoint_id="cp1",
            model_override="gpt-4o-mini",
            prompt_overrides={"system": "hi"},
        )
        assert req.replay_type == "checkpoint"

    def test_model_swap(self):
        assert ReplayRequest(trace_id="t1", model_override="gpt-4o").replay_type == "model_swap"

    def test_prompt_optimization(self):
        assert ReplayRequest(trace_id="t1", prompt_overrides={"system": "x"}).replay_type == "prompt_optimization"

    def test_mock_replay(self):
        assert ReplayRequest(trace_id="t1", mock_config={"tool": {"enabled": True}}).replay_type == "mock"

    def test_parameterized(self):
        assert ReplayRequest(trace_id="t1", input_overrides={"q": "x"}).replay_type == "parameterized"
        assert ReplayRequest(trace_id="t1", config_overrides={"temperature": 0.2}).replay_type == "parameterized"
        assert ReplayRequest(trace_id="t1", tool_overrides={"web": {"timeout": 5}}).replay_type == "parameterized"


class TestParameterOverrides:
    def test_flattens_only_set_fields(self):
        req = ReplayRequest(
            trace_id="t1",
            model_override="gpt-4o-mini",
            input_overrides={"x": 1},
            state_overrides={"s": 2},
        )
        out = req.parameter_overrides()
        assert out == {
            "model": "gpt-4o-mini",
            "input_overrides": {"x": 1},
            "state_overrides": {"s": 2},
        }

    def test_empty_when_nothing_set(self):
        assert ReplayRequest(trace_id="t1").parameter_overrides() == {}


class TestReplayResultDefaults:
    def test_completed_is_default_status(self):
        r = ReplayResult(original_trace_id="a", replay_trace_id="b")
        assert r.status == ReplayStatus.COMPLETED
        assert r.diff == ReplayDiff()
        assert r.error is None


class TestBatchReplayFilter:
    def test_all_fields_optional(self):
        f = BatchReplayFilter()
        assert f.model is None and f.tags == [] and f.trace_ids == []
