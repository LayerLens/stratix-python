from __future__ import annotations

import json
import hashlib
from enum import Enum
from typing import Any
from datetime import datetime
from dataclasses import asdict


def _json_default(obj: Any) -> Any:
    """Handle non-standard types for canonical JSON serialization."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def canonical_json(data: Any) -> str:
    """Serialize data to canonical JSON: sorted keys, compact, deterministic."""
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )


def compute_hash(data: Any) -> str:
    """Compute SHA-256 hash of canonicalized data. Returns 'sha256:<64 hex chars>'."""
    raw = canonical_json(data)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
