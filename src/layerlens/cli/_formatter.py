from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, Optional


def to_dict(obj: Any) -> Any:
    """Convert a Pydantic model (v1 or v2) to a dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif hasattr(obj, "dict"):
        return obj.dict()
    elif isinstance(obj, dict):
        return obj
    return obj


def format_table(items: List[Any], columns: List[Tuple[str, str]], max_col_width: int = 40) -> str:
    """Render items as a fixed-width text table.

    Args:
        items: List of Pydantic models or dicts.
        columns: List of (field_key, header_label) tuples.
        max_col_width: Maximum column width before truncation.

    Returns:
        Formatted table string.
    """
    if not items:
        return "No results found."

    rows: List[Dict[str, str]] = []
    for item in items:
        d = to_dict(item) if not isinstance(item, dict) else item
        row: Dict[str, str] = {}
        for key, _ in columns:
            val = d.get(key)
            row[key] = _format_value(val)
        rows.append(row)

    # Compute column widths
    widths: Dict[str, int] = {}
    for key, header in columns:
        widths[key] = min(max(len(header), max(len(r[key]) for r in rows)), max_col_width)

    # Build header
    header_parts = [header.ljust(widths[key]) for key, header in columns]
    header_line = "  ".join(header_parts)
    separator = "  ".join("-" * widths[key] for key, _ in columns)

    # Build rows
    lines = [header_line, separator]
    for row in rows:
        parts = [_truncate(row[key], widths[key]).ljust(widths[key]) for key, _ in columns]
        lines.append("  ".join(parts))

    return "\n".join(lines)


def format_output(data: Any, output_format: str, columns: Optional[List[Tuple[str, str]]] = None) -> str:
    """Format data as table or JSON.

    Args:
        data: A list of items, a single item, or a dict.
        output_format: "table" or "json".
        columns: For table format, list of (field_key, header_label) tuples.

    Returns:
        Formatted string.
    """
    if output_format == "json":
        return _format_json(data)

    # Table format
    if isinstance(data, list):
        if columns:
            return format_table(data, columns)
        return _format_json(data)

    # Single item
    return format_single(data)


def format_single(item: Any) -> str:
    """Format a single item as key-value pairs."""
    d = to_dict(item) if not isinstance(item, dict) else item
    if not isinstance(d, dict):
        return str(d)

    lines = []
    max_key_len = max(len(k) for k in d) if d else 0
    for key, value in d.items():
        label = key.replace("_", " ").title()
        lines.append(f"{label:<{max_key_len + 4}} {_format_value(value)}")
    return "\n".join(lines)


def _format_json(data: Any) -> str:
    """Format data as pretty-printed JSON."""
    if isinstance(data, list):
        return json.dumps([to_dict(item) for item in data], indent=2, default=str)
    return json.dumps(to_dict(data), indent=2, default=str)


def _format_value(val: Any) -> str:
    """Convert a value to a display string."""
    if val is None:
        return "-"
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, float):
        return f"{val:.4f}"
    if isinstance(val, (dict, list)):
        return json.dumps(val, default=str)
    return str(val)


def _truncate(s: str, width: int) -> str:
    """Truncate a string to width, adding ellipsis if needed."""
    if len(s) <= width:
        return s
    return s[: width - 1] + "\u2026"
