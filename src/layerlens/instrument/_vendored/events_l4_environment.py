"""Vendored snapshot of ``stratix.core.events.l4_environment``.

Source: ``A:/github/layerlens/ateam/stratix/core/events/l4_environment.py``
Source SHA: 7359c0e38d74e02aa1b27c34daef7a958abbd002

Compatibility shims applied for Python 3.9 + Pydantic 2:
- ``enum.StrEnum`` (added in Python 3.11) replaced with
  ``(str, Enum)`` mixin.
- PEP-604 union syntax (``X | None``) on Pydantic field annotations
  rewritten as ``Optional[X]``.

Updates require re-vendoring — see ``__init__.py`` for the workflow.
"""

# STRATIX Layer 4 Events - Environment Configuration & Metrics
#
# Layer 4a - Environment Configuration:
# {
#     "event_type": "environment.config",
#     "layer": "L4a",
#     "environment": {
#         "type": "cloud | on_prem | simulated",
#         "region": "string",
#         "attributes": { }
#     }
# }
#
# Layer 4b - Environment Metrics:
# {
#     "event_type": "environment.metrics",
#     "layer": "L4b",
#     "metrics": {
#         "cpu_pct": 42.1,
#         "gpu_pct": 77.0,
#         "latency_ms": 812
#     }
# }

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import Field, BaseModel


class EnvironmentType(str, Enum):
    """Type of execution environment."""

    CLOUD = "cloud"
    ON_PREM = "on_prem"
    SIMULATED = "simulated"


class EnvironmentInfo(BaseModel):
    """Environment information for L4a events."""

    type: EnvironmentType = Field(description="Type of environment")
    region: Optional[str] = Field(default=None, description="Geographic region")
    attributes: dict[str, Any] = Field(
        default_factory=dict, description="Additional environment attributes"
    )


class EnvironmentConfigEvent(BaseModel):
    """Layer 4a Event: Environment Configuration.

    Represents the execution environment configuration.

    NORMATIVE: Must be emitted at trial start or on runtime change.
    """

    event_type: str = Field(default="environment.config", description="Event type identifier")
    layer: str = Field(default="L4a", description="Layer identifier")
    environment: EnvironmentInfo = Field(description="Environment configuration")

    @classmethod
    def create(
        cls,
        env_type: EnvironmentType,
        region: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> EnvironmentConfigEvent:
        """Create an environment configuration event."""
        return cls(
            environment=EnvironmentInfo(
                type=env_type,
                region=region,
                attributes=attributes or {},
            )
        )


class EnvironmentMetrics(BaseModel):
    """Environment metrics for L4b events."""

    cpu_pct: Optional[float] = Field(
        default=None, ge=0, le=100, description="CPU utilization percentage"
    )
    gpu_pct: Optional[float] = Field(
        default=None, ge=0, le=100, description="GPU utilization percentage"
    )
    memory_pct: Optional[float] = Field(
        default=None, ge=0, le=100, description="Memory utilization percentage"
    )
    latency_ms: Optional[float] = Field(default=None, ge=0, description="Latency in milliseconds")
    additional_metrics: dict[str, float] = Field(
        default_factory=dict, description="Additional custom metrics"
    )


class EnvironmentMetricsEvent(BaseModel):
    """Layer 4b Event: Environment Metrics.

    Represents environment resource metrics during execution.
    """

    event_type: str = Field(default="environment.metrics", description="Event type identifier")
    layer: str = Field(default="L4b", description="Layer identifier")
    metrics: EnvironmentMetrics = Field(description="Environment metrics")

    @classmethod
    def create(
        cls,
        cpu_pct: Optional[float] = None,
        gpu_pct: Optional[float] = None,
        memory_pct: Optional[float] = None,
        latency_ms: Optional[float] = None,
        additional_metrics: Optional[dict[str, float]] = None,
    ) -> EnvironmentMetricsEvent:
        """Create an environment metrics event."""
        return cls(
            metrics=EnvironmentMetrics(
                cpu_pct=cpu_pct,
                gpu_pct=gpu_pct,
                memory_pct=memory_pct,
                latency_ms=latency_ms,
                additional_metrics=additional_metrics or {},
            )
        )


__all__ = [
    "EnvironmentType",
    "EnvironmentInfo",
    "EnvironmentConfigEvent",
    "EnvironmentMetrics",
    "EnvironmentMetricsEvent",
]
