"""Pydantic models for replay requests, diffs and results."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, BaseModel


class ReplayStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ReplayRequest(BaseModel):
    """A single-trace replay request.

    Supports basic, parameterized, model-swap, prompt-optimization,
    checkpoint and mock replays. The ``replay_type`` property resolves
    which category this request falls into based on which overrides
    are set — callers don't need to specify it explicitly.
    """

    trace_id: str = Field(description="ID of the original trace to replay")
    input_overrides: Dict[str, Any] = Field(default_factory=dict)
    model_override: Optional[str] = None
    config_overrides: Dict[str, Any] = Field(default_factory=dict)
    prompt_overrides: Dict[str, Any] = Field(default_factory=dict)
    tool_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    mock_config: Dict[str, Any] = Field(default_factory=dict)
    checkpoint_id: Optional[str] = None
    state_overrides: Dict[str, Any] = Field(default_factory=dict)

    @property
    def replay_type(self) -> str:
        if self.checkpoint_id:
            return "checkpoint"
        if self.model_override:
            return "model_swap"
        if self.prompt_overrides:
            return "prompt_optimization"
        if self.mock_config:
            return "mock"
        if self.input_overrides or self.config_overrides or self.tool_overrides:
            return "parameterized"
        return "basic"

    def parameter_overrides(self) -> Dict[str, Any]:
        """Flatten set overrides into one dict (for event metadata)."""
        out: Dict[str, Any] = {}
        if self.input_overrides:
            out["input_overrides"] = self.input_overrides
        if self.model_override:
            out["model"] = self.model_override
        if self.config_overrides:
            out["config_overrides"] = self.config_overrides
        if self.prompt_overrides:
            out["prompt_overrides"] = self.prompt_overrides
        if self.tool_overrides:
            out["tool_overrides"] = self.tool_overrides
        if self.mock_config:
            out["mock_config"] = self.mock_config
        if self.state_overrides:
            out["state_overrides"] = self.state_overrides
        return out


class EventDiffDetail(BaseModel):
    event_count_original: int = 0
    event_count_replay: int = 0
    missing_event_types: List[str] = Field(default_factory=list)
    extra_event_types: List[str] = Field(default_factory=list)
    reordered: bool = False


class ReplayDiff(BaseModel):
    output_changed: bool = False
    output_similarity: float = 1.0
    event_diff: EventDiffDetail = Field(default_factory=EventDiffDetail)
    cost_diff_usd: Optional[float] = None
    latency_diff_ms: Optional[float] = None


class ReplayResult(BaseModel):
    original_trace_id: str
    replay_trace_id: str
    status: ReplayStatus = ReplayStatus.COMPLETED
    diff: ReplayDiff = Field(default_factory=ReplayDiff)
    duration_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchReplayFilter(BaseModel):
    """Selection filter for :class:`BatchReplayRequest`."""

    model: Optional[str] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    score_lt: Optional[float] = None
    score_gt: Optional[float] = None
    tags: List[str] = Field(default_factory=list)
    trace_ids: List[str] = Field(default_factory=list)
    framework: Optional[str] = None
