"""Single-trace replay controller.

Given a callable that knows how to re-run an agent/LLM pipeline, the
controller applies the :class:`ReplayRequest`'s overrides, invokes the
callable, diffs the result against the original trace, and writes to
the :class:`ReplayStore`.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Callable, Optional

from .store import ReplayStore, InMemoryReplayStore
from .models import ReplayDiff, ReplayResult, ReplayStatus, ReplayRequest
from .diff_engine import DiffEngine
from ..models.trace import Trace

ReplayFn = Callable[[Trace, ReplayRequest], Trace]
"""User-provided replay callable.

Receives the original trace and the request (overrides already flattened
on the request), returns the replayed trace. Should raise on failure.
"""


class ReplayController:
    """Orchestrates replay of a single trace."""

    def __init__(
        self,
        replay_fn: ReplayFn,
        *,
        store: Optional[ReplayStore] = None,
        diff_engine: Optional[DiffEngine] = None,
    ) -> None:
        self._replay_fn = replay_fn
        self._store: ReplayStore = store or InMemoryReplayStore()
        self._diff_engine = diff_engine or DiffEngine()

    @property
    def store(self) -> ReplayStore:
        return self._store

    def run(
        self,
        original: Trace,
        request: ReplayRequest,
        *,
        cost_original: Optional[float] = None,
        cost_replay_fn: Optional[Callable[[Trace], float]] = None,
        latency_original_ms: Optional[float] = None,
    ) -> ReplayResult:
        start = time.time()
        replay_trace_id = f"replay_{uuid.uuid4().hex[:16]}"
        metadata: Dict[str, Any] = {
            "replay_type": request.replay_type,
            "overrides": request.parameter_overrides(),
        }
        try:
            replayed = self._replay_fn(original, request)
        except Exception as exc:
            duration_ms = (time.time() - start) * 1000
            result = ReplayResult(
                original_trace_id=original.id,
                replay_trace_id=replay_trace_id,
                status=ReplayStatus.FAILED,
                diff=ReplayDiff(),
                duration_ms=duration_ms,
                error=str(exc),
                metadata=metadata,
            )
            self._store.save(result)
            return result

        duration_ms = (time.time() - start) * 1000
        cost_replay = cost_replay_fn(replayed) if cost_replay_fn else None
        latency_replay_ms = _latency_from(replayed)

        diff = self._diff_engine.diff(
            original,
            replayed,
            cost_original=cost_original,
            cost_replay=cost_replay,
            latency_original_ms=latency_original_ms,
            latency_replay_ms=latency_replay_ms,
        )

        result = ReplayResult(
            original_trace_id=original.id,
            replay_trace_id=replay_trace_id,
            status=ReplayStatus.COMPLETED,
            diff=diff,
            duration_ms=duration_ms,
            error=None,
            metadata=metadata,
        )
        self._store.save(result)
        return result


def _latency_from(trace: Trace) -> Optional[float]:
    data = trace.data or {}
    val = data.get("latency_ms") or data.get("duration_ms")
    return float(val) if isinstance(val, (int, float)) else None
