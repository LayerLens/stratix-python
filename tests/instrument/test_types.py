from __future__ import annotations

from layerlens.instrument._span import span
from layerlens.instrument._context import _parent_span_id, _current_span_id, _current_span_name


class TestSpan:
    def test_yields_string_span_id(self):
        with span("test") as span_id:
            assert isinstance(span_id, str)
            assert len(span_id) == 16

    def test_sets_current_span_id(self):
        with span("test") as span_id:
            assert _current_span_id.get() == span_id

    def test_sets_parent_span_id(self):
        _current_span_id.set("parent123")
        try:
            with span("test") as span_id:
                assert _parent_span_id.get() == "parent123"
                assert _current_span_id.get() == span_id
        finally:
            _current_span_id.set(None)

    def test_stores_span_name(self):
        with span("retrieval"):
            assert _current_span_name.get() == "retrieval"

    def test_restores_context_after(self):
        original_span = _current_span_id.get()
        original_name = _current_span_name.get()
        with span("test"):
            pass
        assert _current_span_id.get() == original_span
        assert _current_span_name.get() == original_name
