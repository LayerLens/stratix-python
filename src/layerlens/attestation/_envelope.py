from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime, timezone
from dataclasses import field, dataclass


class HashScope(Enum):
    """Level at which a hash was computed."""

    EVENT = "event"
    TRIAL = "trial"


@dataclass
class AttestationEnvelope:
    """Single entry in a hash chain."""

    hash: str
    scope: HashScope
    previous_hash: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    signature: Optional[str] = None
    signing_key_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "hash": self.hash,
            "scope": self.scope.value,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.signature is not None:
            d["signature"] = self.signature
        if self.signing_key_id is not None:
            d["signing_key_id"] = self.signing_key_id
        return d
