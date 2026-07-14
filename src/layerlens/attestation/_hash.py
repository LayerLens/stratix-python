from __future__ import annotations

import json
import hashlib
from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime
from dataclasses import asdict

#: Reserved wire-event fields that carry the per-event attestation hash itself.
#: They are attached to the wire event AFTER it is hashed (see
#: ``TraceCollector._append_locked``) so they MUST be excluded from the hash
#: input — an object's hash can never include itself.
SELF_HASH_FIELDS = ("hash", "previous_hash")


def event_hash_input(data: Dict[str, Any], previous_hash: Optional[str]) -> Dict[str, Any]:
    """Canonical hash input for one chained event.

    Strips the self-referential hash fields (``hash``/``previous_hash``, which
    are attached to the wire event only AFTER it is hashed) and injects
    ``_previous_hash`` for chain linkage. The chain builder and the tamper
    verifier both go through this one function so they agree byte-for-byte, and
    a verifier that rebuilds the chain from the wire events reproduces the
    identical hashes. Events that never carry the self-hash fields (the
    collector's own call, every pre-existing trace) are unaffected.
    """
    hashable = {k: v for k, v in data.items() if k not in SELF_HASH_FIELDS}
    hashable["_previous_hash"] = previous_hash
    return hashable


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
