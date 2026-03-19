"""Content cache for LLM-generated content.

Hash-keyed, JSON-serialized disk cache for Tier 3 LLM content
to avoid redundant API calls.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


class ContentCache:
    """Disk-backed content cache.

    Caches LLM-generated content keyed by hash of the request parameters.
    Default location: ~/.stratix/simulator/cache/
    """

    def __init__(self, cache_dir: str | None = None):
        if cache_dir:
            self._cache_dir = Path(cache_dir)
        else:
            home = Path(os.environ.get("STRATIX_HOME", Path.home() / ".stratix"))
            self._cache_dir = home / "simulator" / "cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, Any] = {}

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def _make_key(self, **kwargs: Any) -> str:
        """Generate a hash key from keyword arguments."""
        key_str = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def get(self, **kwargs: Any) -> Any | None:
        """Get cached value, or None if not cached."""
        key = self._make_key(**kwargs)
        # Memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key]
        # Disk cache
        path = self._cache_path(key)
        if path.exists():
            try:
                with open(path) as f:
                    value = json.load(f)
                self._memory_cache[key] = value
                return value
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def set(self, value: Any, **kwargs: Any) -> None:
        """Store value in cache."""
        key = self._make_key(**kwargs)
        self._memory_cache[key] = value
        path = self._cache_path(key)
        try:
            with open(path, "w") as f:
                json.dump(value, f)
        except OSError:
            pass

    def clear(self) -> int:
        """Clear all cached entries. Returns count of entries cleared."""
        count = 0
        for path in self._cache_dir.glob("*.json"):
            try:
                path.unlink()
                count += 1
            except OSError:
                pass
        self._memory_cache.clear()
        return count

    @property
    def size(self) -> int:
        """Number of cached entries on disk."""
        return len(list(self._cache_dir.glob("*.json")))
