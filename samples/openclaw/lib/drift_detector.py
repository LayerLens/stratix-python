"""
DriftDetector -- Performance Drift Detection Engine
=====================================================

Manages rolling baselines per model/task pair and detects statistically
significant performance drift.  Tracks both score drift (quality
degradation) and latency drift (response time regressions).

Drift types:
  - **score_regression**: Model quality dropped below the rolling baseline.
  - **score_improvement**: Model quality improved (informational, not alerted).
  - **latency_regression**: Model response time increased significantly.
  - **latency_improvement**: Model response time decreased (informational).
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class DriftAlert(BaseModel):
    """A single drift detection alert."""

    model_id: str
    task_id: str
    drift_type: str
    severity: str
    current_value: float
    baseline_mean: float
    baseline_std: float
    delta: float
    sigma_distance: float
    window_size: int
    message: str


class BaselineSnapshot(BaseModel):
    """Snapshot of the rolling baseline for a model/task pair."""

    model_id: str
    task_id: str
    score_mean: float = 0.0
    score_std: float = 0.0
    score_count: int = 0
    latency_mean: float = 0.0
    latency_std: float = 0.0
    latency_count: int = 0


# ---------------------------------------------------------------------------
# Rolling statistics helper
# ---------------------------------------------------------------------------


class _RollingStats:
    """Maintains a fixed-size window of values for computing rolling mean and std."""

    def __init__(self, window_size: int = 20) -> None:
        self.window_size = window_size
        self._values: list[float] = []

    def add(self, value: float) -> None:
        self._values.append(value)
        if len(self._values) > self.window_size:
            self._values.pop(0)

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def mean(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    @property
    def std(self) -> float:
        if len(self._values) < 2:
            return 0.0
        m = self.mean
        variance = sum((v - m) ** 2 for v in self._values) / len(self._values)
        return math.sqrt(variance)

    @property
    def values(self) -> list[float]:
        return list(self._values)


# ---------------------------------------------------------------------------
# Drift Detector
# ---------------------------------------------------------------------------


class DriftDetector:
    """
    Performance drift detection engine with rolling baselines.

    Maintains per-(model, task) baselines and generates alerts when
    new observations deviate significantly from the historical norm.
    """

    def __init__(
        self,
        window_size: int = 20,
        sigma_threshold: float = 2.0,
        min_observations: int = 5,
        latency_sigma_threshold: float | None = None,
    ) -> None:
        self.window_size = window_size
        self.sigma_threshold = sigma_threshold
        self.min_observations = min_observations
        self.latency_sigma_threshold = latency_sigma_threshold or sigma_threshold

        self._score_stats: dict[tuple[str, str], _RollingStats] = defaultdict(
            lambda: _RollingStats(window_size)
        )
        self._latency_stats: dict[tuple[str, str], _RollingStats] = defaultdict(
            lambda: _RollingStats(window_size)
        )
        self._alerts: list[DriftAlert] = []

    def record_and_check(
        self,
        model_id: str,
        task_id: str,
        score: float,
        latency_ms: int = 0,
    ) -> list[DriftAlert]:
        key = (model_id, task_id)
        alerts: list[DriftAlert] = []

        score_stats = self._score_stats[key]
        if score_stats.count >= self.min_observations:
            alert = self._check_drift(
                model_id=model_id, task_id=task_id, value=score,
                stats=score_stats, metric_type="score",
                sigma_threshold=self.sigma_threshold,
            )
            if alert:
                alerts.append(alert)
                self._alerts.append(alert)
        score_stats.add(score)

        if latency_ms > 0:
            latency_stats = self._latency_stats[key]
            if latency_stats.count >= self.min_observations:
                alert = self._check_drift(
                    model_id=model_id, task_id=task_id,
                    value=float(latency_ms), stats=latency_stats,
                    metric_type="latency",
                    sigma_threshold=self.latency_sigma_threshold,
                )
                if alert:
                    alerts.append(alert)
                    self._alerts.append(alert)
            latency_stats.add(float(latency_ms))

        return alerts

    def get_baseline(self, model_id: str, task_id: str) -> BaselineSnapshot:
        key = (model_id, task_id)
        score_s = self._score_stats[key]
        latency_s = self._latency_stats[key]
        return BaselineSnapshot(
            model_id=model_id, task_id=task_id,
            score_mean=round(score_s.mean, 3), score_std=round(score_s.std, 3),
            score_count=score_s.count,
            latency_mean=round(latency_s.mean, 1), latency_std=round(latency_s.std, 1),
            latency_count=latency_s.count,
        )

    def get_all_baselines(self) -> list[BaselineSnapshot]:
        keys = set(self._score_stats.keys()) | set(self._latency_stats.keys())
        return [self.get_baseline(m, t) for m, t in sorted(keys)]

    def get_alert_history(self) -> list[dict[str, Any]]:
        return [a.model_dump() for a in self._alerts]

    def clear_baselines(self) -> None:
        self._score_stats.clear()
        self._latency_stats.clear()
        self._alerts.clear()

    @property
    def total_alerts(self) -> int:
        return len(self._alerts)

    def _check_drift(
        self, model_id: str, task_id: str, value: float,
        stats: _RollingStats, metric_type: str, sigma_threshold: float,
    ) -> DriftAlert | None:
        mean = stats.mean
        std = stats.std
        if std < 0.001:
            return None
        delta = value - mean
        sigma_distance = abs(delta) / std
        if sigma_distance < sigma_threshold:
            return None

        if metric_type == "score":
            if delta < 0:
                drift_type = "score_regression"
                severity = "critical" if sigma_distance > 3.0 else "warning"
            else:
                drift_type = "score_improvement"
                severity = "info"
        else:
            if delta > 0:
                drift_type = "latency_regression"
                severity = "critical" if sigma_distance > 3.0 else "warning"
            else:
                drift_type = "latency_improvement"
                severity = "info"

        message = (
            f"{drift_type.replace('_', ' ').title()} detected for "
            f"{model_id}/{task_id}: "
            f"current={value:.2f}, baseline={mean:.2f} +/- {std:.2f}, "
            f"delta={delta:+.2f} ({sigma_distance:.1f} sigma)"
        )
        logger.warning(message)

        return DriftAlert(
            model_id=model_id, task_id=task_id, drift_type=drift_type,
            severity=severity, current_value=round(value, 3),
            baseline_mean=round(mean, 3), baseline_std=round(std, 3),
            delta=round(delta, 3), sigma_distance=round(sigma_distance, 2),
            window_size=stats.count, message=message,
        )
