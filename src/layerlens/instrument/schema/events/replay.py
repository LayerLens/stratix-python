"""
STRATIX Replay Events

Defines event types for trace checkpoint/replay operations (Epic 2).

Event Types:
- trace.checkpoint: Resumable execution checkpoints
- trace.replay.start: Replay session start with parameter overrides
- trace.replay.end: Replay session end with diff summary
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TraceCheckpointEvent(BaseModel):
    """
    Replay Event: Trace Checkpoint

    Emitted at execution points where the agent's state can be serialized
    and later resumed. Framework adapters emit this event at natural
    boundaries (e.g., after a LangGraph node completes, after a CrewAI
    task finishes).

    NORMATIVE:
    - Adapters with REPLAY capability MUST emit at resumable boundaries
    - When privacy level is hashed or not_provided, state_snapshot is
      set to empty dict and only state_hash is stored
    - framework_checkpoint_id bridges STRATIX checkpoint to framework-native
      checkpoint (e.g., LangGraph thread_ts)
    """
    event_type: str = Field(
        default="trace.checkpoint",
        description="Event type identifier",
    )
    checkpoint_id: str = Field(
        description="Unique checkpoint identifier (UUID)",
    )
    state_snapshot: dict[str, Any] = Field(
        default_factory=dict,
        description="Serialized state at this point",
    )
    state_hash: str = Field(
        description="SHA-256 hash of state_snapshot ('sha256:<hex64>')",
    )
    resumable: bool = Field(
        description="Whether execution can resume from here",
    )
    framework_checkpoint_id: str | None = Field(
        default=None,
        description="Framework-native checkpoint ID (e.g., LangGraph thread_ts)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional checkpoint metadata",
    )

    @classmethod
    def create(
        cls,
        checkpoint_id: str,
        state_hash: str,
        resumable: bool,
        state_snapshot: dict[str, Any] | None = None,
        framework_checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceCheckpointEvent:
        """
        Create a trace checkpoint event.

        Args:
            checkpoint_id: Unique checkpoint identifier
            state_hash: SHA-256 hash of the state snapshot
            resumable: Whether execution can resume from here
            state_snapshot: Serialized state (omit for hashed privacy)
            framework_checkpoint_id: Framework-native checkpoint ID
            metadata: Additional checkpoint metadata

        Returns:
            TraceCheckpointEvent instance
        """
        return cls(
            checkpoint_id=checkpoint_id,
            state_snapshot=state_snapshot or {},
            state_hash=state_hash,
            resumable=resumable,
            framework_checkpoint_id=framework_checkpoint_id,
            metadata=metadata or {},
        )


class TraceReplayStartEvent(BaseModel):
    """
    Replay Event: Trace Replay Start

    Emitted at the beginning of a replay execution. Links the new replay
    trace to the original trace and records what parameters were changed.

    NORMATIVE:
    - replay_type must be one of: basic, parameterized, checkpoint,
      model_swap, batch
    - checkpoint_id is required when replay_type is "checkpoint"
    """
    event_type: str = Field(
        default="trace.replay.start",
        description="Event type identifier",
    )
    original_trace_id: str = Field(
        description="The trace being replayed (UUID)",
    )
    replay_trace_id: str = Field(
        description="New trace ID for this replay (UUID)",
    )
    replay_type: str = Field(
        description=(
            "Replay mode: 'basic' | 'parameterized' | 'checkpoint' "
            "| 'model_swap' | 'batch'"
        ),
    )
    parameter_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="What was changed for this replay (e.g., model, temperature)",
    )
    checkpoint_id: str | None = Field(
        default=None,
        description="Checkpoint to resume from (required for checkpoint replay)",
    )

    @classmethod
    def create(
        cls,
        original_trace_id: str,
        replay_trace_id: str,
        replay_type: str,
        parameter_overrides: dict[str, Any] | None = None,
        checkpoint_id: str | None = None,
    ) -> TraceReplayStartEvent:
        """
        Create a trace replay start event.

        Args:
            original_trace_id: The trace being replayed
            replay_trace_id: New trace ID for this replay
            replay_type: Replay mode
            parameter_overrides: What was changed for this replay
            checkpoint_id: Checkpoint to resume from

        Returns:
            TraceReplayStartEvent instance
        """
        return cls(
            original_trace_id=original_trace_id,
            replay_trace_id=replay_trace_id,
            replay_type=replay_type,
            parameter_overrides=parameter_overrides or {},
            checkpoint_id=checkpoint_id,
        )


class TraceReplayEndEvent(BaseModel):
    """
    Replay Event: Trace Replay End

    Emitted at the end of a replay execution. Contains a diff summary
    comparing the replayed trace to the original.

    NORMATIVE:
    - status must be one of: completed, failed, timeout
    - error is required when status is "failed"
    """
    event_type: str = Field(
        default="trace.replay.end",
        description="Event type identifier",
    )
    original_trace_id: str = Field(
        description="The trace that was replayed (UUID)",
    )
    replay_trace_id: str = Field(
        description="The replay trace ID (UUID)",
    )
    diff_summary: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Comparison metrics: output_changed (bool), "
            "event_count_diff (int), cost_diff_usd (float), "
            "latency_diff_ms (float)"
        ),
    )
    status: str = Field(
        description="Replay status: 'completed' | 'failed' | 'timeout'",
    )
    error: str | None = Field(
        default=None,
        description="Error message if status != 'completed'",
    )

    @classmethod
    def create(
        cls,
        original_trace_id: str,
        replay_trace_id: str,
        status: str,
        diff_summary: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> TraceReplayEndEvent:
        """
        Create a trace replay end event.

        Args:
            original_trace_id: The trace that was replayed
            replay_trace_id: The replay trace ID
            status: Replay status
            diff_summary: Comparison metrics
            error: Error message if failed

        Returns:
            TraceReplayEndEvent instance
        """
        return cls(
            original_trace_id=original_trace_id,
            replay_trace_id=replay_trace_id,
            diff_summary=diff_summary or {},
            status=status,
            error=error,
        )
