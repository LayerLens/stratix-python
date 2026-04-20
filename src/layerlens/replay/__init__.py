"""Trace replay engine.

Replay an existing :class:`~layerlens.models.trace.Trace` with optional
overrides (model swap, input/config/prompt overrides, mocks, checkpoints)
and diff the result against the original.

The public surface mirrors the ateam replay package but is intentionally
narrower: a dataset-producing pipeline, not a full experiment platform.
Additional ateam primitives (A/B tests, cost analysis, prompt suggestions)
are out of scope until we have concrete product requirements.
"""

from __future__ import annotations

from .batch import BatchReplayer, BatchReplayResult, BatchReplayRequest, BatchReplaySummary
from .store import ReplayStore, InMemoryReplayStore
from .models import (
    ReplayDiff,
    ReplayResult,
    ReplayStatus,
    ReplayRequest,
    EventDiffDetail,
    BatchReplayFilter,
)
from .controller import ReplayFn, ReplayController
from .diff_engine import DiffEngine, similarity

__all__ = [
    "BatchReplayFilter",
    "BatchReplayRequest",
    "BatchReplayResult",
    "BatchReplaySummary",
    "BatchReplayer",
    "DiffEngine",
    "EventDiffDetail",
    "InMemoryReplayStore",
    "ReplayController",
    "ReplayDiff",
    "ReplayFn",
    "ReplayRequest",
    "ReplayResult",
    "ReplayStatus",
    "ReplayStore",
    "similarity",
]
