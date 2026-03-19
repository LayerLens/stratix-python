"""
Tests for MCP structured output schema validation and event emission.
"""

import pytest

from layerlens.instrument.adapters.protocols.mcp.structured_output import (
    validate_structured_output,
    compute_output_hash,
    compute_schema_hash,
)


class TestStructuredOutputValidation:
    def test_valid_output(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        output = {"name": "test"}
        is_valid, errors = validate_structured_output(output, schema)
        assert is_valid
        assert errors == []

    def test_missing_required_field(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        output = {"other": "value"}
        is_valid, errors = validate_structured_output(output, schema)
        assert not is_valid
        assert len(errors) > 0

    def test_wrong_type(self):
        schema = {"type": "object"}
        output = "not an object"
        is_valid, errors = validate_structured_output(output, schema)
        assert not is_valid

    def test_array_type(self):
        schema = {"type": "array"}
        output = [1, 2, 3]
        is_valid, errors = validate_structured_output(output, schema)
        assert is_valid

    def test_string_type(self):
        schema = {"type": "string"}
        is_valid, errors = validate_structured_output("hello", schema)
        assert is_valid

    def test_number_type(self):
        schema = {"type": "number"}
        is_valid, errors = validate_structured_output(42, schema)
        assert is_valid

    def test_boolean_type(self):
        schema = {"type": "boolean"}
        is_valid, errors = validate_structured_output(True, schema)
        assert is_valid


class TestOutputHashing:
    def test_compute_output_hash(self):
        h = compute_output_hash({"key": "value"})
        assert h.startswith("sha256:")
        assert len(h) == 71  # sha256: + 64 hex chars

    def test_compute_schema_hash(self):
        h = compute_schema_hash({"type": "object"})
        assert h.startswith("sha256:")

    def test_hash_deterministic(self):
        h1 = compute_output_hash({"a": 1, "b": 2})
        h2 = compute_output_hash({"b": 2, "a": 1})
        assert h1 == h2  # sort_keys ensures deterministic


class TestStructuredOutputEvents:
    def test_on_structured_output_valid(self, mcp_adapter, mock_stratix):
        mcp_adapter.on_structured_output(
            tool_name="search",
            output={"results": ["r1"]},
            schema={"$id": "search-output", "type": "object"},
            validation_passed=True,
        )
        assert len(mock_stratix.events) == 1
        event = mock_stratix.events[0][0]
        assert event.event_type == "protocol.tool.structured_output"
        assert event.validation_passed is True
        assert event.schema_id == "search-output"

    def test_on_structured_output_invalid(self, mcp_adapter, mock_stratix):
        mcp_adapter.on_structured_output(
            tool_name="search",
            output="not an object",
            schema={"type": "object"},
            validation_passed=False,
            validation_errors=["Expected object, got string"],
        )
        event = mock_stratix.events[0][0]
        assert event.validation_passed is False
        assert len(event.validation_errors) == 1
