"""
STRATIX Trace Container

Provides SerializedTrace — a portable, hashable representation of a
complete trace suitable for storage, replay, and cross-adapter transfer.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SerializedTrace(BaseModel):
    """
    A fully serialized trace record.

    Contains the ordered list of event dicts, checkpoint metadata,
    and integrity information needed to verify and replay a trace.
    """

    trace_id: str = Field(description="Trace ID (UUID)")
    evaluation_id: str | None = Field(default=None, description="Evaluation ID")
    trial_id: str | None = Field(default=None, description="Trial ID")
    events: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered event records (dicts)",
    )
    checkpoints: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Checkpoint snapshots collected during the trace",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata (adapter name, framework, etc.)",
    )
    hash_chain_verified: bool = Field(
        default=False,
        description="True if the hash chain was verified at serialization time",
    )
    schema_version: str = Field(
        default="1.2.0",
        description="Schema version for forward compatibility",
    )

    @classmethod
    def from_event_records(
        cls,
        events: list[dict[str, Any]],
        trace_id: str,
        evaluation_id: str | None = None,
        trial_id: str | None = None,
        checkpoints: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        hash_chain_verified: bool = False,
    ) -> SerializedTrace:
        """
        Build a SerializedTrace from raw event records.

        Args:
            events: Ordered list of event dicts.
            trace_id: The trace ID.
            evaluation_id: Optional evaluation ID.
            trial_id: Optional trial ID.
            checkpoints: Optional checkpoint snapshots.
            metadata: Arbitrary metadata.
            hash_chain_verified: Whether the hash chain was verified.

        Returns:
            SerializedTrace instance
        """
        return cls(
            trace_id=trace_id,
            evaluation_id=evaluation_id,
            trial_id=trial_id,
            events=events,
            checkpoints=checkpoints or [],
            metadata=metadata or {},
            hash_chain_verified=hash_chain_verified,
        )
