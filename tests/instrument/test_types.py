from __future__ import annotations

import time

from layerlens.instrument._types import SpanData


class TestSpanData:
    def test_defaults(self):
        s = SpanData(name="test")
        assert s.name == "test"
        assert len(s.span_id) == 16
        assert s.parent_id is None
        assert s.status == "ok"
        assert s.kind == "internal"
        assert s.input is None
        assert s.output is None
        assert s.error is None
        assert s.metadata == {}
        assert s.children == []
        assert s.end_time is None
        assert s.start_time <= time.time()

    def test_finish_ok(self):
        s = SpanData(name="test")
        s.finish()
        assert s.end_time is not None
        assert s.status == "ok"
        assert s.error is None

    def test_finish_error(self):
        s = SpanData(name="test")
        s.finish(error="something broke")
        assert s.end_time is not None
        assert s.status == "error"
        assert s.error == "something broke"

    def test_to_dict(self):
        parent = SpanData(name="parent")
        child = SpanData(name="child", parent_id=parent.span_id)
        parent.children.append(child)

        d = parent.to_dict()
        assert d["name"] == "parent"
        assert d["parent_id"] is None
        assert len(d["children"]) == 1
        assert d["children"][0]["name"] == "child"
        assert d["children"][0]["parent_id"] == parent.span_id

    def test_to_dict_nested(self):
        root = SpanData(name="root")
        child1 = SpanData(name="c1", parent_id=root.span_id)
        child2 = SpanData(name="c2", parent_id=child1.span_id)
        root.children.append(child1)
        child1.children.append(child2)

        d = root.to_dict()
        assert d["children"][0]["children"][0]["name"] == "c2"
