"""Dataset lifecycle management.

A dataset here is a versioned collection of evaluation items — an item
is typically ``{input, expected_output, metadata}``. Datasets can be
derived from replays, synthetic generation, or uploaded by hand, and
are the unit of input for :mod:`layerlens.evaluation_runs`.
"""

from __future__ import annotations

from .store import DatasetStore, InMemoryDatasetStore
from .models import Dataset, DatasetItem, DatasetVersion, DatasetVisibility

__all__ = [
    "Dataset",
    "DatasetItem",
    "DatasetStore",
    "DatasetVersion",
    "DatasetVisibility",
    "InMemoryDatasetStore",
]
