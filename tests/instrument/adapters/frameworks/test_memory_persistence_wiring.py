"""Per-adapter memory persistence wiring smoke tests.

For each of the six target adapters wired in this PR (cross-poll #1),
verify that:

* The adapter inherits a per-instance :class:`MemoryRecorder` from
  :class:`BaseAdapter`.
* The recorder is bound to the adapter's ``org_id`` (multi-tenancy).
* :meth:`serialize_for_replay` includes a content-addressable
  ``memory_snapshot`` dict in the :class:`ReplayableTrace` metadata.
* The snapshot is restorable into a fresh recorder (replay round-trip).

These tests exercise the wiring contract — the deeper behavioural
unit tests for the recorder itself live in
``tests/instrument/adapters/_base/test_memory.py`` (27 tests).

Browser_use is intentionally **not** included — that adapter is not
present on this PR's base branch (see
``docs/adapters/memory-contract.md`` "Honest scope disclosure"). When
the histories merge, this test module should be extended with a
``BrowserUseAdapter`` parametrize entry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

import pytest

from layerlens.instrument.adapters._base import (
    BaseAdapter,
    MemoryRecorder,
    MemorySnapshot,
)


class _RecordingStratix:
    """Minimal stratix stand-in carrying a tenant binding."""

    org_id: str = "test-org-mem"

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


def _adapter_classes() -> List[Type[BaseAdapter]]:
    """Return the list of target adapter classes for this PR.

    Imported lazily and individually so a missing adapter on the base
    branch (e.g. ``browser_use``) is reported as a clean skip rather
    than a collection error.
    """
    classes: List[Type[BaseAdapter]] = []
    from layerlens.instrument.adapters.frameworks.agno import AgnoAdapter
    from layerlens.instrument.adapters.frameworks.google_adk import GoogleADKAdapter
    from layerlens.instrument.adapters.frameworks.llama_index import LlamaIndexAdapter
    from layerlens.instrument.adapters.frameworks.openai_agents import OpenAIAgentsAdapter
    from layerlens.instrument.adapters.frameworks.bedrock_agents import BedrockAgentsAdapter
    from layerlens.instrument.adapters.frameworks.ms_agent_framework import MSAgentAdapter

    classes.extend([
        AgnoAdapter,
        BedrockAgentsAdapter,
        GoogleADKAdapter,
        LlamaIndexAdapter,
        MSAgentAdapter,
        OpenAIAgentsAdapter,
    ])
    return classes


@pytest.fixture(params=_adapter_classes(), ids=lambda c: c.__name__)
def adapter_cls(request: pytest.FixtureRequest) -> Type[BaseAdapter]:
    return request.param  # type: ignore[no-any-return]


def test_adapter_owns_memory_recorder_bound_to_org(adapter_cls: Type[BaseAdapter]) -> None:
    """Every wired adapter exposes a recorder bound to its tenant."""
    stratix = _RecordingStratix()
    adapter = adapter_cls(stratix=stratix)

    assert isinstance(adapter.memory_recorder, MemoryRecorder)
    assert adapter.memory_recorder.org_id == "test-org-mem"
    assert adapter.org_id == adapter.memory_recorder.org_id


def test_record_memory_turn_advances_episodic_buffer(adapter_cls: Type[BaseAdapter]) -> None:
    """The BaseAdapter helper feeds the recorder."""
    stratix = _RecordingStratix()
    adapter = adapter_cls(stratix=stratix)
    initial = adapter.memory_recorder.snapshot()
    assert initial.turn_index == 0
    assert initial.episodic == []

    adapter.record_memory_turn(
        agent_name="test-agent",
        input_data="hello",
        output_data="world",
        tools=["search"],
    )

    snap = adapter.memory_recorder.snapshot()
    assert snap.turn_index == 1
    assert len(snap.episodic) == 1
    assert snap.episodic[0]["agent_name"] == "test-agent"
    assert snap.episodic[0]["input"] == "hello"
    assert snap.episodic[0]["output"] == "world"
    assert snap.episodic[0]["tools"] == ["search"]


def test_serialize_for_replay_embeds_memory_snapshot(adapter_cls: Type[BaseAdapter]) -> None:
    """Each adapter ships ``metadata["memory_snapshot"]`` in the replay trace."""
    stratix = _RecordingStratix()
    adapter = adapter_cls(stratix=stratix)
    adapter.record_memory_turn(
        agent_name="agent",
        input_data="i",
        output_data="o",
        tools=["t"],
    )

    trace = adapter.serialize_for_replay()
    assert "memory_snapshot" in trace.metadata, (
        f"{adapter_cls.__name__}.serialize_for_replay() must embed "
        "metadata['memory_snapshot'] for replay-safe memory restoration"
    )

    snapshot_dict = trace.metadata["memory_snapshot"]
    assert isinstance(snapshot_dict, dict)
    # The dict must round-trip into a MemorySnapshot.
    snapshot = MemorySnapshot.from_dict(snapshot_dict)
    assert snapshot.turn_index == 1
    assert snapshot.org_id == "test-org-mem"
    assert len(snapshot.episodic) == 1


def test_replay_engine_can_restore_recorder_from_serialized_trace(
    adapter_cls: Type[BaseAdapter],
) -> None:
    """Replay-safety smoke: serialise → from_dict → restore → snapshot match."""
    stratix = _RecordingStratix()
    src_adapter = adapter_cls(stratix=stratix)
    for i in range(3):
        src_adapter.record_memory_turn(
            agent_name="a",
            input_data=f"in-{i}",
            output_data=f"out-{i}",
            tools=["t1"],
        )
    src_adapter.memory_recorder.set_semantic("session_summary", "user asked about pricing")
    src_trace = src_adapter.serialize_for_replay()
    src_snapshot = src_adapter.memory_recorder.snapshot()

    # Replay-side: fresh adapter + restore from the trace.
    replay_adapter = adapter_cls(stratix=_RecordingStratix())
    snapshot = MemorySnapshot.from_dict(src_trace.metadata["memory_snapshot"])
    replay_adapter.memory_recorder.restore(snapshot)
    restored_snapshot = replay_adapter.memory_recorder.snapshot()

    assert restored_snapshot.content_hash == src_snapshot.content_hash
    assert restored_snapshot.turn_index == src_snapshot.turn_index
    assert restored_snapshot.episodic == src_snapshot.episodic
    assert restored_snapshot.semantic == src_snapshot.semantic


def test_recorder_rejects_cross_tenant_snapshot_via_adapter(adapter_cls: Type[BaseAdapter]) -> None:
    """Tenant-A adapter cannot accept a tenant-B snapshot at the recorder boundary."""
    stratix_a = _RecordingStratix()
    stratix_a.org_id = "tenant-A"
    adapter_a = adapter_cls(stratix=stratix_a)
    adapter_a.record_memory_turn(agent_name="x", input_data="i", output_data="o")
    snap_a = adapter_a.memory_recorder.snapshot()

    stratix_b = _RecordingStratix()
    stratix_b.org_id = "tenant-B"
    adapter_b = adapter_cls(stratix=stratix_b)

    with pytest.raises(ValueError, match="Cross-tenant restore is prohibited"):
        adapter_b.memory_recorder.restore(snap_a)
