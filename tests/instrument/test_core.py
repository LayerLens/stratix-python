from __future__ import annotations

import os

import pytest

from layerlens.instrument import SpanData, span, trace
from layerlens.instrument._context import _current_span, _current_recorder
from layerlens.instrument._recorder import TraceRecorder


class TestTraceDecorator:
    def test_basic_trace(self, mock_client):
        @trace(mock_client)
        def my_func(x):
            return x * 2

        result = my_func(5)
        assert result == 10
        mock_client.traces.upload.assert_called_once()

    def test_trace_with_custom_name(self, mock_client, capture_trace):
        @trace(mock_client, name="custom_name")
        def my_func():
            return "ok"

        my_func()
        assert capture_trace["trace"][0]["name"] == "custom_name"

    def test_trace_captures_input(self, mock_client, capture_trace):
        @trace(mock_client)
        def my_func(query):
            return "result"

        my_func("hello")
        assert capture_trace["trace"][0]["input"] == "hello"

    def test_trace_captures_output(self, mock_client, capture_trace):
        @trace(mock_client)
        def my_func():
            return {"answer": 42}

        my_func()
        assert capture_trace["trace"][0]["output"] == {"answer": 42}

    def test_trace_on_error(self, mock_client, capture_trace):
        @trace(mock_client)
        def my_func():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            my_func()

        assert capture_trace["trace"][0]["status"] == "error"
        assert capture_trace["trace"][0]["error"] == "boom"

    def test_trace_cleans_up_context(self, mock_client):
        @trace(mock_client)
        def my_func():
            return "ok"

        my_func()
        assert _current_recorder.get() is None
        assert _current_span.get() is None

    def test_trace_cleans_up_context_on_error(self, mock_client):
        @trace(mock_client)
        def my_func():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            my_func()

        assert _current_recorder.get() is None
        assert _current_span.get() is None


class TestSpanContextManager:
    def test_span_creates_child(self, mock_client, capture_trace):
        @trace(mock_client)
        def my_func():
            with span("child_span", kind="llm") as s:
                s.output = "child output"
            return "done"

        my_func()
        root = capture_trace["trace"][0]
        assert len(root["children"]) == 1
        child = root["children"][0]
        assert child["name"] == "child_span"
        assert child["kind"] == "llm"
        assert child["output"] == "child output"
        assert child["parent_id"] == root["span_id"]

    def test_nested_spans(self, mock_client, capture_trace):
        @trace(mock_client)
        def my_func():
            with span("outer", kind="chain") as s1:
                s1.output = "outer"
                with span("inner", kind="llm") as s2:
                    s2.output = "inner"
            return "done"

        my_func()
        root = capture_trace["trace"][0]
        outer = root["children"][0]
        assert outer["name"] == "outer"
        inner = outer["children"][0]
        assert inner["name"] == "inner"
        assert inner["parent_id"] == outer["span_id"]

    def test_span_on_error(self, mock_client, capture_trace):
        @trace(mock_client)
        def my_func():
            try:
                with span("failing") as s:
                    raise ValueError("span error")
            except ValueError:
                pass
            return "recovered"

        my_func()
        child = capture_trace["trace"][0]["children"][0]
        assert child["status"] == "error"
        assert child["error"] == "span error"

    def test_span_without_trace_noops(self):
        with span("orphan", kind="llm") as s:
            s.output = "test"
        assert s.output == "test"

    def test_multiple_sibling_spans(self, mock_client, capture_trace):
        @trace(mock_client)
        def my_func():
            with span("retrieve", kind="retriever") as s:
                s.output = ["doc1", "doc2"]
            with span("generate", kind="llm") as s:
                s.output = "answer"
            return "done"

        my_func()
        root = capture_trace["trace"][0]
        assert len(root["children"]) == 2
        assert root["children"][0]["name"] == "retrieve"
        assert root["children"][1]["name"] == "generate"


class TestTraceRecorder:
    def test_flush_calls_upload(self, mock_client):
        recorder = TraceRecorder(mock_client)
        recorder.root = SpanData(name="root")
        recorder.root.finish()

        recorder.flush()
        mock_client.traces.upload.assert_called_once()

        path = mock_client.traces.upload.call_args[0][0]
        assert not os.path.exists(path)

    def test_flush_noop_without_root(self, mock_client):
        recorder = TraceRecorder(mock_client)
        recorder.flush()
        mock_client.traces.upload.assert_not_called()
