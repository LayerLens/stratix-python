"""Apply AG-UI ``STATE_DELTA`` JSON Patch operations and hash results.

``StateDeltaHandler`` keeps a cached snapshot of the agent's UI state so
that when a ``STATE_DELTA`` event arrives (RFC 6902 JSON Patch ops) we can
compute deterministic before/after SHA-256 hashes and return them to the
adapter for inclusion in ``agent.state.change`` payloads.

Supports the core subset of RFC 6902: ``add``, ``remove``, ``replace``.
``move``, ``copy``, and ``test`` are not implemented.
"""

from __future__ import annotations

import copy
import json
import hashlib
import logging
from typing import Any

log = logging.getLogger(__name__)


class StateDeltaHandler:
    """Maintains a cached AG-UI state and applies JSON Patch deltas to it."""

    def __init__(self) -> None:
        self._current_state: dict[str, Any] = {}

    @property
    def current_state(self) -> dict[str, Any]:
        return copy.deepcopy(self._current_state)

    def apply_snapshot(self, state: dict[str, Any]) -> tuple[str, str]:
        before_hash = self._hash_state(self._current_state)
        self._current_state = copy.deepcopy(state)
        after_hash = self._hash_state(self._current_state)
        return before_hash, after_hash

    def apply_delta(self, operations: list[dict[str, Any]]) -> tuple[str, str]:
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
                    self._patch_add(path, value)
                else:
                    log.debug("Unsupported JSON Patch op: %s", op_type)
            except Exception as exc:
                log.warning("JSON Patch %s @ %s failed: %s", op_type, path, exc)
        return before_hash, self._hash_state(self._current_state)

    def reset(self) -> None:
        self._current_state.clear()

    # --- internals ---

    def _patch_add(self, path: str, value: Any) -> None:
        keys = self._parse_path(path)
        if not keys:
            if isinstance(value, dict):
                self._current_state = dict(value)
            return
        target = self._current_state
        for key in keys[:-1]:
            nxt = target.setdefault(key, {})
            if not isinstance(nxt, dict):
                return
            target = nxt
        target[keys[-1]] = value

    def _patch_remove(self, path: str) -> None:
        keys = self._parse_path(path)
        if not keys:
            return
        target = self._current_state
        for key in keys[:-1]:
            nxt = target.get(key)
            if not isinstance(nxt, dict):
                return
            target = nxt
        target.pop(keys[-1], None)

    @staticmethod
    def _parse_path(path: str) -> list[str]:
        if not path or path == "/":
            return []
        return [p.replace("~1", "/").replace("~0", "~") for p in path.lstrip("/").split("/")]

    @staticmethod
    def _hash_state(state: dict[str, Any]) -> str:
        return "sha256:" + hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()
