from __future__ import annotations

import pytest

from layerlens.replay.diff_engine import DiffEngine, similarity

from .conftest import make_trace


class TestSimilarity:
    def test_both_empty_is_identical(self):
        assert similarity(None, None) == 1.0
        assert similarity("", "") == 1.0

    def test_one_empty_is_zero(self):
        assert similarity("hi", None) == 0.0
        assert similarity(None, "hi") == 0.0

    def test_identical_strings(self):
        assert similarity("abc", "abc") == 1.0

    def test_partial_overlap(self):
        assert 0 < similarity("hello", "world") < 1.0


class TestDiffEngine:
    def test_identical_traces_have_no_diff(self):
        t = make_trace(events=[{"type": "a"}, {"type": "b"}], output="x")
        diff = DiffEngine().diff(t, t)
        assert diff.output_changed is False
        assert diff.output_similarity == 1.0
        assert diff.event_diff.missing_event_types == []
        assert diff.event_diff.extra_event_types == []
        assert diff.event_diff.reordered is False

    def test_different_output(self):
        a = make_trace(output="hello")
        b = make_trace(output="goodbye")
        diff = DiffEngine().diff(a, b)
        assert diff.output_changed is True
        assert diff.output_similarity < 1.0

    def test_missing_and_extra_event_types(self):
        a = make_trace(events=[{"type": "x"}, {"type": "y"}])
        b = make_trace(events=[{"type": "y"}, {"type": "z"}])
        diff = DiffEngine().diff(a, b)
        assert diff.event_diff.missing_event_types == ["x"]
        assert diff.event_diff.extra_event_types == ["z"]

    def test_reorder_detected(self):
        a = make_trace(events=[{"type": "x"}, {"type": "y"}])
        b = make_trace(events=[{"type": "y"}, {"type": "x"}])
        diff = DiffEngine().diff(a, b)
        assert diff.event_diff.reordered is True

    def test_cost_and_latency_deltas(self):
        a = make_trace()
        b = make_trace()
        diff = DiffEngine().diff(
            a,
            b,
            cost_original=0.01,
            cost_replay=0.015,
            latency_original_ms=100.0,
            latency_replay_ms=140.0,
        )
        assert diff.cost_diff_usd == pytest.approx(0.005)
        assert diff.latency_diff_ms == pytest.approx(40.0)

    def test_cost_delta_is_none_when_either_missing(self):
        diff = DiffEngine().diff(make_trace(), make_trace(), cost_original=0.01)
        assert diff.cost_diff_usd is None

    def test_reorder_false_when_event_types_differ(self):
        a = make_trace(events=[{"type": "x"}])
        b = make_trace(events=[{"type": "y"}])
        diff = DiffEngine().diff(a, b)
        assert diff.event_diff.reordered is False
