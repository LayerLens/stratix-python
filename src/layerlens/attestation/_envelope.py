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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hash": self.hash,
            "scope": self.scope.value,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp.isoformat(),
        }
