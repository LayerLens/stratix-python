"""
AG-UI State Delta Handler

Handles AG-UI STATE_DELTA events (JSON Patch operations, RFC 6902)
and translates them into Stratix agent.state.change events with
proper before/after hash computation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class StateDeltaHandler:
    """
    Manages AG-UI state snapshots and deltas.

    Maintains a cached copy of the current state. When a STATE_DELTA
    event arrives (containing JSON Patch operations), applies the patch
    to compute the new state and generates before/after hashes.
    """

    def __init__(self) -> None:
        self._current_state: dict[str, Any] = {}

    @property
    def current_state(self) -> dict[str, Any]:
        return copy.deepcopy(self._current_state)

    def apply_snapshot(self, state: dict[str, Any]) -> tuple[str, str]:
        """
        Apply a full state snapshot (STATE_SNAPSHOT event).

        Args:
            state: The complete state snapshot.

        Returns:
            Tuple of (before_hash, after_hash).
        """
        before_hash = self._hash_state(self._current_state)
        self._current_state = copy.deepcopy(state)
        after_hash = self._hash_state(self._current_state)
        return before_hash, after_hash

    def apply_delta(self, operations: list[dict[str, Any]]) -> tuple[str, str]:
        """
        Apply JSON Patch operations (STATE_DELTA event).

        Implements a subset of RFC 6902 JSON Patch:
        - add: Add a value at a path
        - remove: Remove a value at a path
        - replace: Replace a value at a path

        Args:
            operations: List of JSON Patch operations.

        Returns:
            Tuple of (before_hash, after_hash).
        """
        before_hash = self._hash_state(self._current_state)

        for op in operations:
            op_type = op.get("op", "")
            path = op.get("path", "")
            value = op.get("value")

            try:
                if op_type == "add":
                    self._patch_add(path, value)
                elif op_type == "remove":
                    self._patch_remove(path)
                elif op_type == "replace":
                    self._patch_replace(path, value)
                else:
                    logger.debug("Unsupported JSON Patch op: %s", op_type)
            except Exception as exc:
                logger.warning("Failed to apply JSON Patch op %s at %s: %s", op_type, path, exc)

        after_hash = self._hash_state(self._current_state)
        return before_hash, after_hash

    # --- JSON Patch operations ---

    def _patch_add(self, path: str, value: Any) -> None:
        keys = self._parse_path(path)
        if not keys:
            self._current_state = value if isinstance(value, dict) else self._current_state
            return
        target = self._current_state
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = value

    def _patch_remove(self, path: str) -> None:
        keys = self._parse_path(path)
        if not keys:
            return
        target = self._current_state
        for key in keys[:-1]:
            if key not in target:
                return
            target = target[key]
        target.pop(keys[-1], None)

    def _patch_replace(self, path: str, value: Any) -> None:
        self._patch_add(path, value)

    @staticmethod
    def _parse_path(path: str) -> list[str]:
        """Parse a JSON Pointer path (e.g. '/foo/bar') into keys."""
        if not path or path == "/":
            return []
        parts = path.lstrip("/").split("/")
        # Unescape JSON Pointer tokens (RFC 6901)
        return [p.replace("~1", "/").replace("~0", "~") for p in parts]

    @staticmethod
    def _hash_state(state: dict[str, Any]) -> str:
        """Compute SHA-256 hash of a state dict."""
        state_json = json.dumps(state, sort_keys=True, default=str)
        h = hashlib.sha256(state_json.encode()).hexdigest()
        return f"sha256:{h}"

    def reset(self) -> None:
        """Clear cached state."""
        self._current_state.clear()
