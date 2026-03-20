from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from layerlens.cli._formatter import (
    to_dict,
    _truncate,
    format_table,
    _format_value,
    format_output,
    format_single,
)


class SampleModel(BaseModel):
    id: str
    name: str
    score: float = 0.0


class TestToDict:
    """Test to_dict conversion for various input types."""

    def test_pydantic_v2_model(self):
        """Pydantic model with model_dump is converted."""
        m = SampleModel(id="1", name="test", score=0.5)
        result = to_dict(m)
        assert result == {"id": "1", "name": "test", "score": 0.5}

    def test_dict_passthrough(self):
        """Dict input is returned as-is."""
        d = {"key": "value"}
        assert to_dict(d) is d

    def test_other_type_passthrough(self):
        """Non-model, non-dict input is returned as-is."""
        assert to_dict("hello") == "hello"
        assert to_dict(42) == 42


class TestFormatValue:
    """Test _format_value display conversion."""

    def test_none(self):
        assert _format_value(None) == "-"

    def test_bool_true(self):
        assert _format_value(True) == "Yes"

    def test_bool_false(self):
        assert _format_value(False) == "No"

    def test_float(self):
        assert _format_value(3.14159) == "3.1416"

    def test_dict(self):
        result = _format_value({"a": 1})
        assert json.loads(result) == {"a": 1}

    def test_list(self):
        result = _format_value([1, 2])
        assert json.loads(result) == [1, 2]

    def test_string(self):
        assert _format_value("hello") == "hello"

    def test_int(self):
        assert _format_value(42) == "42"


class TestTruncate:
    """Test _truncate string truncation."""

    def test_short_string(self):
        assert _truncate("abc", 10) == "abc"

    def test_exact_width(self):
        assert _truncate("abcde", 5) == "abcde"

    def test_long_string(self):
        result = _truncate("abcdefgh", 5)
        assert len(result) == 5
        assert result.endswith("\u2026")

    def test_single_char_width(self):
        result = _truncate("abcdef", 1)
        assert result == "\u2026"


class TestFormatTable:
    """Test format_table rendering."""

    @pytest.fixture
    def columns(self):
        return [("id", "ID"), ("name", "Name")]

    def test_empty_list(self, columns):
        result = format_table([], columns)
        assert result == "No results found."

    def test_single_row(self, columns):
        items = [{"id": "1", "name": "Alice"}]
        result = format_table(items, columns)
        lines = result.split("\n")
        assert len(lines) == 3  # header, separator, row
        assert "ID" in lines[0]
        assert "Name" in lines[0]
        assert "1" in lines[2]
        assert "Alice" in lines[2]

    def test_multiple_rows(self, columns):
        items = [{"id": "1", "name": "Alice"}, {"id": "2", "name": "Bob"}]
        result = format_table(items, columns)
        lines = result.split("\n")
        assert len(lines) == 4  # header, separator, 2 rows

    def test_pydantic_models(self, columns):
        items = [SampleModel(id="1", name="Test")]
        result = format_table(items, columns)
        assert "Test" in result

    def test_column_width_adapts(self):
        columns = [("val", "V")]
        items = [{"val": "short"}, {"val": "a much longer value here"}]
        result = format_table(items, columns)
        lines = result.split("\n")
        # Header should be at least as wide as longest value
        assert len(lines[0]) >= len("a much longer value here")

    def test_truncation_at_max_width(self):
        columns = [("val", "V")]
        items = [{"val": "x" * 100}]
        result = format_table(items, columns, max_col_width=20)
        data_line = result.split("\n")[2]
        assert len(data_line.strip()) <= 20


class TestFormatOutput:
    """Test format_output dispatch."""

    @pytest.fixture
    def columns(self):
        return [("id", "ID"), ("name", "Name")]

    def test_json_format_list(self, columns):
        items = [{"id": "1", "name": "A"}]
        result = format_output(items, "json", columns)
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert parsed[0]["id"] == "1"

    def test_json_format_single(self):
        item = {"id": "1", "name": "A"}
        result = format_output(item, "json")
        parsed = json.loads(result)
        assert parsed["id"] == "1"

    def test_table_format_list(self, columns):
        items = [{"id": "1", "name": "A"}]
        result = format_output(items, "table", columns)
        assert "ID" in result
        assert "Name" in result

    def test_table_format_single(self):
        item = {"id": "1", "name": "Test"}
        result = format_output(item, "table")
        assert "Test" in result


class TestFormatSingle:
    """Test format_single key-value rendering."""

    def test_dict_input(self):
        result = format_single({"name": "Alice", "age": 30})
        assert "Name" in result
        assert "Alice" in result
        assert "Age" in result
        assert "30" in result

    def test_pydantic_model(self):
        m = SampleModel(id="1", name="Test", score=0.5)
        result = format_single(m)
        assert "Id" in result
        assert "1" in result

    def test_non_dict(self):
        result = format_single("just a string")
        assert result == "just a string"
