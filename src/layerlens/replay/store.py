"""Pluggable store for completed replay results."""

from __future__ import annotations

from typing import Dict, List, Iterable, Optional, Protocol

from .models import ReplayResult


class ReplayStore(Protocol):
    def save(self, result: ReplayResult) -> None: ...
    def get(self, replay_trace_id: str) -> Optional[ReplayResult]: ...
    def list_for_original(self, original_trace_id: str) -> List[ReplayResult]: ...
    def all(self) -> Iterable[ReplayResult]: ...


class InMemoryReplayStore:
    """Default store — useful for tests, notebooks, and short-lived jobs."""

    def __init__(self) -> None:
        self._by_id: Dict[str, ReplayResult] = {}
        self._by_original: Dict[str, List[str]] = {}

    def save(self, result: ReplayResult) -> None:
        self._by_id[result.replay_trace_id] = result
        self._by_original.setdefault(result.original_trace_id, []).append(result.replay_trace_id)

    def get(self, replay_trace_id: str) -> Optional[ReplayResult]:
        return self._by_id.get(replay_trace_id)

    def list_for_original(self, original_trace_id: str) -> List[ReplayResult]:
        ids = self._by_original.get(original_trace_id, [])
        return [self._by_id[i] for i in ids if i in self._by_id]

    def all(self) -> Iterable[ReplayResult]:
        return self._by_id.values()

    def clear(self) -> None:
        self._by_id.clear()
        self._by_original.clear()
