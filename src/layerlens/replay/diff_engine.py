"""Structural + textual diffing between an original trace and a replay."""

from __future__ import annotations

import difflib
from typing import Any, Dict, List, Optional

from .models import ReplayDiff, EventDiffDetail
from ..models.trace import Trace


def similarity(a: Optional[str], b: Optional[str]) -> float:
    """SequenceMatcher ratio, safe for ``None`` / empty inputs."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


class DiffEngine:
    """Produce a :class:`ReplayDiff` from two traces.

    Kept stateless so callers can reuse one engine across batches. The
    event shape assumed here is the ateam-style ``{"events": [...]}``
    payload stored on :attr:`Trace.data`; richer schemas degrade
    gracefully to an empty event diff rather than raising.
    """

    def diff(
        self,
        original: Trace,
        replay: Trace,
        *,
        cost_original: Optional[float] = None,
        cost_replay: Optional[float] = None,
        latency_original_ms: Optional[float] = None,
        latency_replay_ms: Optional[float] = None,
    ) -> ReplayDiff:
        orig_output = self._extract_output(original)
        repl_output = self._extract_output(replay)
        sim = similarity(orig_output, repl_output)

        cost_diff = (cost_replay - cost_original) if cost_original is not None and cost_replay is not None else None
        latency_diff = (
            (latency_replay_ms - latency_original_ms)
            if latency_original_ms is not None and latency_replay_ms is not None
            else None
        )

        return ReplayDiff(
            output_changed=orig_output != repl_output,
            output_similarity=sim,
            event_diff=self._event_diff(original, replay),
            cost_diff_usd=cost_diff,
            latency_diff_ms=latency_diff,
        )

    def _extract_output(self, trace: Trace) -> Optional[str]:
        data = trace.data or {}
        for key in ("output", "final_output", "response"):
            val = data.get(key)
            if isinstance(val, str):
                return val
            if val is not None:
                return str(val)
        return None

    def _event_diff(self, original: Trace, replay: Trace) -> EventDiffDetail:
        orig_events = _events(original)
        repl_events = _events(replay)
        orig_types = [e.get("type") or e.get("event") for e in orig_events]
        repl_types = [e.get("type") or e.get("event") for e in repl_events]
        orig_set = set(t for t in orig_types if t)
        repl_set = set(t for t in repl_types if t)
        return EventDiffDetail(
            event_count_original=len(orig_events),
            event_count_replay=len(repl_events),
            missing_event_types=sorted(orig_set - repl_set),
            extra_event_types=sorted(repl_set - orig_set),
            reordered=orig_types != repl_types and orig_set == repl_set and len(orig_events) == len(repl_events),
        )


def _events(trace: Trace) -> List[Dict[str, Any]]:
    data = trace.data or {}
    events = data.get("events")
    if isinstance(events, list):
        return [e for e in events if isinstance(e, dict)]
    return []
