"""MCP structured-output validation helpers.

Validates tool responses against a JSON Schema (uses ``jsonschema`` when
available, falls back to a minimal type/required check) and computes stable
SHA-256 hashes for the output value and schema. Used by the MCP adapter to
emit ``mcp.structured_output`` events with validation results.
"""

from __future__ import annotations

import json
import hashlib
import logging
from typing import Any

log = logging.getLogger(__name__)


def validate_structured_output(
    output: Any,
    schema: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Validate ``output`` against ``schema``. Returns ``(is_valid, errors)``."""
    try:
        import jsonschema  # type: ignore[import-untyped]
    except ImportError:
        return _basic_type_check(output, schema)

    try:
        jsonschema.validate(instance=output, schema=schema)
        return True, []
    except jsonschema.ValidationError as exc:
        return False, [str(exc.message)]
    except jsonschema.SchemaError as exc:
        return False, [f"Invalid schema: {exc.message}"]


def _basic_type_check(output: Any, schema: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    schema_type = schema.get("type")
    type_map: dict[str, type | tuple[type, ...]] = {
        "object": dict,
        "array": list,
        "string": str,
        "number": (int, float),
        "boolean": bool,
    }
    expected = type_map.get(schema_type) if schema_type else None
    if expected is not None and not isinstance(output, expected):
        errors.append(f"Expected {schema_type}, got {type(output).__name__}")
    if schema_type == "object" and isinstance(output, dict):
        for field in schema.get("required", []) or []:
            if field not in output:
                errors.append(f"Missing required field: {field}")
    return not errors, errors


def compute_output_hash(output: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(output, sort_keys=True, default=str).encode()).hexdigest()


def compute_schema_hash(schema: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
