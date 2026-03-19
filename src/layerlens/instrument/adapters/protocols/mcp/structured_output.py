"""
MCP Structured Output Handler

Handles schema validation of MCP structured tool outputs and
emits protocol.tool.structured_output events.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def validate_structured_output(
    output: Any,
    schema: dict[str, Any],
) -> tuple[bool, list[str]]:
    """
    Validate a structured output against a JSON Schema.

    Uses basic type checking when jsonschema is not available.

    Args:
        output: The structured output value.
        schema: The JSON Schema to validate against.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors: list[str] = []

    try:
        import jsonschema
        try:
            jsonschema.validate(instance=output, schema=schema)
            return True, []
        except jsonschema.ValidationError as exc:
            errors.append(str(exc.message))
            return False, errors
        except jsonschema.SchemaError as exc:
            errors.append(f"Invalid schema: {exc.message}")
            return False, errors
    except ImportError:
        # Fallback: basic type validation
        return _basic_type_check(output, schema)


def _basic_type_check(
    output: Any,
    schema: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Basic type check when jsonschema is not available."""
    errors: list[str] = []
    schema_type = schema.get("type")

    if schema_type == "object" and not isinstance(output, dict):
        errors.append(f"Expected object, got {type(output).__name__}")
    elif schema_type == "array" and not isinstance(output, list):
        errors.append(f"Expected array, got {type(output).__name__}")
    elif schema_type == "string" and not isinstance(output, str):
        errors.append(f"Expected string, got {type(output).__name__}")
    elif schema_type == "number" and not isinstance(output, (int, float)):
        errors.append(f"Expected number, got {type(output).__name__}")
    elif schema_type == "boolean" and not isinstance(output, bool):
        errors.append(f"Expected boolean, got {type(output).__name__}")

    # Check required fields for objects
    if schema_type == "object" and isinstance(output, dict):
        required = schema.get("required", [])
        for field in required:
            if field not in output:
                errors.append(f"Missing required field: {field}")

    return len(errors) == 0, errors


def compute_output_hash(output: Any) -> str:
    """Compute SHA-256 hash of a structured output value."""
    output_str = json.dumps(output, sort_keys=True, default=str)
    h = hashlib.sha256(output_str.encode()).hexdigest()
    return f"sha256:{h}"


def compute_schema_hash(schema: dict[str, Any]) -> str:
    """Compute SHA-256 hash of a JSON Schema."""
    schema_str = json.dumps(schema, sort_keys=True)
    h = hashlib.sha256(schema_str.encode()).hexdigest()
    return f"sha256:{h}"
