"""Tests for the shared :class:`ReplayExecutor` and result types.

Covers:

* Happy-path round-trip — an adapter serializes a trace, the executor
  re-runs it through a fresh agent, the captured events match.
* Divergence detection — every :class:`DivergenceKind` is exercised
  with a deliberately mismatched replay.
* Sync + async agent factory support.
* Multi-tenant ``org_id`` propagation onto :class:`ReplayResult`.
* Stub injection lifecycle (patch enter + exit, even on errors).
* Execution-error capture (no re-raise; carried as data).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from dataclasses import dataclass

import pytest

from layerlens.instrument.adapters._base import (
    AdapterInfo,
    BaseAdapter,
    AdapterHealth,
    AdapterStatus,
    DivergenceKind,
    ReplayExecutor,
    ReplayableTrace,
    AdapterCapability,
)
from layerlens.instrument.adapters._base.replay import (
    StubInjector,
)

# ---------------------------------------------------------------------------
# Test fixtures: minimal adapter + recording stratix
# ---------------------------------------------------------------------------


class _RecordingStratix:
    """Stratix double that records emitted events for assertion."""

    org_id: str = "tenant-A"

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


class _ReplayTestAdapter(BaseAdapter):
    """Tiny BaseAdapter subclass that exposes the executor for tests."""

    FRAMEWORK = "replay_test"
    VERSION = "0.0.1"

    def connect(self) -> None:
        self._connected = True
        self._status = AdapterStatus.HEALTHY

    def disconnect(self) -> None:
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED

    def health_check(self) -> AdapterHealth:
        return AdapterHealth(
            status=self._status,
            framework_name=self.FRAMEWORK,
            adapter_version=self.VERSION,
        )

    def get_adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            name="ReplayTestAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            capabilities=[AdapterCapability.REPLAY],
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        return ReplayableTrace(
            adapter_name="ReplayTestAdapter",
            framework=self.FRAMEWORK,
            trace_id="replay-test-trace",
            events=list(self._trace_events),
        )


@dataclass
class _FakeAgent:
    """Sync agent stub used by the executor's generic invocation path."""

    output: str = "ok"
    raises: bool = False

    def run(self, inputs: Any) -> Any:
        if self.raises:
            raise RuntimeError("boom")
        return self.output


@dataclass
class _FakeAsyncAgent:
    """Async agent stub used to exercise the awaitable invocation branch."""

    output: str = "async-ok"

    async def arun(self, inputs: Any) -> Any:
        await asyncio.sleep(0)
        return self.output


# ---------------------------------------------------------------------------
# Happy-path round-trip
# ---------------------------------------------------------------------------


def _emit_simple_run(adapter: _ReplayTestAdapter) -> None:
    """Simulate an agent run by emitting a canonical 3-event sequence."""
    adapter.emit_dict_event(
        "agent.input",
        {"agent_name": "alpha", "input": "hello"},
    )
    adapter.emit_dict_event(
        "model.invoke",
        {"model": "gpt-5.3", "provider": "openai"},
    )
    adapter.emit_dict_event(
        "agent.output",
        {"agent_name": "alpha", "output": "ok"},
    )


def test_executor_returns_replay_result_with_org_id() -> None:
    adapter = _ReplayTestAdapter(stratix=_RecordingStratix())
    adapter.connect()
    _emit_simple_run(adapter)
    trace = adapter.serialize_for_replay()

    def factory() -> _FakeAgent:
        return _FakeAgent(output="ok")

    executor = ReplayExecutor(adapter)

    async def replay_inline() -> Any:
        # Adapter must re-emit the same 3 events for an exact replay.
        async def _factory_with_emit() -> _FakeAgent:
            _emit_simple_run(adapter)  # simulate framework re-emission
            return _FakeAgent(output="ok")

        return await executor.execute_replay(trace, _factory_with_emit)

    result = asyncio.run(replay_inline())
    assert result.org_id == "tenant-A"
    assert result.framework == "replay_test"
    assert result.source_trace_id == "replay-test-trace"
    assert result.execution_error is None
    assert result.succeeded is True


def test_happy_path_exact_round_trip() -> None:
    """A perfectly reproduced trace yields zero divergences."""
    adapter = _ReplayTestAdapter(stratix=_RecordingStratix())
    adapter.connect()
    _emit_simple_run(adapter)
    trace = adapter.serialize_for_replay()

    async def factory() -> _FakeAgent:
        # Re-emit the exact same sequence so divergences are empty.
        _emit_simple_run(adapter)
        return _FakeAgent(output="ok")

    executor = ReplayExecutor(adapter)
    result = asyncio.run(executor.execute_replay(trace, factory))

    assert result.is_exact is True
    assert result.divergences == []
    assert len(result.captured_events) == 3
    assert result.outputs == "ok"


def test_async_factory_supported() -> None:
    """The factory may return a coroutine — executor awaits it."""
    adapter = _ReplayTestAdapter(stratix=_RecordingStratix())
    adapter.connect()
    _emit_simple_run(adapter)
    trace = adapter.serialize_for_replay()

    async def factory() -> _FakeAsyncAgent:
        await asyncio.sleep(0)
        _emit_simple_run(adapter)
        return _FakeAsyncAgent(output="async-ok")

    executor = ReplayExecutor(adapter)
    result = asyncio.run(executor.execute_replay(trace, factory))

    assert result.outputs == "async-ok"
    assert result.is_exact is True


def test_sync_factory_supported() -> None:
    """A plain sync factory works without wrapping in a coroutine."""
    adapter = _ReplayTestAdapter(stratix=_RecordingStratix())
    adapter.connect()
    _emit_simple_run(adapter)
    trace = adapter.serialize_for_replay()

    def factory() -> _FakeAgent:
        _emit_simple_run(adapter)
        return _FakeAgent(output="ok")

    executor = ReplayExecutor(adapter)
    result = asyncio.run(executor.execute_replay(trace, factory))

    assert result.is_exact is True
    assert result.outputs == "ok"


# ---------------------------------------------------------------------------
# Divergence detection
# ---------------------------------------------------------------------------


def test_missing_event_divergence() -> None:
    """When replay emits FEWER events than original, missing events are flagged."""
    adapter = _ReplayTestAdapter(stratix=_RecordingStratix())
    adapter.connect()
    _emit_simple_run(adapter)
    trace = adapter.serialize_for_replay()

    async def factory() -> _FakeAgent:
        # Only emit the first two events instead of three.
        adapter.emit_dict_event(
            "agent.input",
            {"agent_name": "alpha", "input": "hello"},
        )
        adapter.emit_dict_event(
            "model.invoke",
            {"model": "gpt-5.3", "provider": "openai"},
        )
        return _FakeAgent()

    executor = ReplayExecutor(adapter)
    result = asyncio.run(executor.execute_replay(trace, factory))

    assert result.is_exact is False
    assert len(result.divergences) == 1
    div = result.divergences[0]
    assert div.kind is DivergenceKind.MISSING_EVENT
    assert div.event_type == "agent.output"
    assert div.index == 2


def test_extra_event_divergence() -> None:
    """When replay emits MORE events than original, extras are flagged."""
    adapter = _ReplayTestAdapter(stratix=_RecordingStratix())
    adapter.connect()
    _emit_simple_run(adapter)
    trace = adapter.serialize_for_replay()

    async def factory() -> _FakeAgent:
        _emit_simple_run(adapter)
        # One extra event the original did not contain.
        adapter.emit_dict_event(
            "tool.call",
            {"tool_name": "search"},
        )
        return _FakeAgent()

    executor = ReplayExecutor(adapter)
    result = asyncio.run(executor.execute_replay(trace, factory))

    extras = [d for d in result.divergences if d.kind is DivergenceKind.EXTRA_EVENT]
    assert len(extras) == 1
    assert extras[0].event_type == "tool.call"
    assert extras[0].index == 3


def test_event_type_mismatch_divergence() -> None:
    """Same position, different event_type → EVENT_TYPE_MISMATCH."""
    adapter = _ReplayTestAdapter(stratix=_RecordingStratix())
    adapter.connect()
    _emit_simple_run(adapter)
    trace = adapter.serialize_for_replay()

    async def factory() -> _FakeAgent:
        adapter.emit_dict_event("agent.input", {"agent_name": "alpha"})
        # Original had model.invoke at position 1; replay emits tool.call.
        adapter.emit_dict_event("tool.call", {"tool_name": "search"})
        adapter.emit_dict_event("agent.output", {"agent_name": "alpha"})
        return _FakeAgent()

    executor = ReplayExecutor(adapter)
    result = asyncio.run(executor.execute_replay(trace, factory))

    mismatches = [d for d in result.divergences if d.kind is DivergenceKind.EVENT_TYPE_MISMATCH]
    assert len(mismatches) == 1
    assert mismatches[0].index == 1
    assert mismatches[0].detail is not None
    assert "model.invoke" in mismatches[0].detail
    assert "tool.call" in mismatches[0].detail


def test_payload_mismatch_divergence() -> None:
    """Same event_type but a meaningful payload field differs → PAYLOAD_MISMATCH."""
    adapter = _ReplayTestAdapter(stratix=_RecordingStratix())
    adapter.connect()
    _emit_simple_run(adapter)
    trace = adapter.serialize_for_replay()

    async def factory() -> _FakeAgent:
        adapter.emit_dict_event("agent.input", {"agent_name": "alpha"})
        # Different model invoked — meaningful divergence.
        adapter.emit_dict_event(
            "model.invoke",
            {"model": "claude-opus-4.6", "provider": "anthropic"},
        )
        adapter.emit_dict_event("agent.output", {"agent_name": "alpha"})
        return _FakeAgent()

    executor = ReplayExecutor(adapter)
    result = asyncio.run(executor.execute_replay(trace, factory))

    mismatches = [d for d in result.divergences if d.kind is DivergenceKind.PAYLOAD_MISMATCH]
    assert len(mismatches) == 1
    assert mismatches[0].index == 1
    assert mismatches[0].detail is not None
    assert "model" in mismatches[0].detail
    assert "provider" in mismatches[0].detail


def test_irrelevant_payload_difference_is_not_flagged() -> None:
    """Honest divergence: differing timestamps/ids are NOT flagged.

    Only meaningful fields (model, provider, tool_name, agent_name,
    handoff endpoints) are compared. Two replays of the same agent
    will always have different ``timestamp_ns``/``run_id`` — flagging
    those would render every replay "divergent" and hide real issues.
    """
    adapter = _ReplayTestAdapter(stratix=_RecordingStratix())
    adapter.connect()
    adapter.emit_dict_event(
        "model.invoke",
        {"model": "gpt-5.3", "provider": "openai", "duration_ns": 1000, "run_id": "r1"},
    )
    trace = adapter.serialize_for_replay()

    async def factory() -> _FakeAgent:
        adapter.emit_dict_event(
            "model.invoke",
            # Same model + provider, different run_id and duration.
            {"model": "gpt-5.3", "provider": "openai", "duration_ns": 9999, "run_id": "r2"},
        )
        return _FakeAgent()

    executor = ReplayExecutor(adapter)
    result = asyncio.run(executor.execute_replay(trace, factory))

    assert result.is_exact is True


# ---------------------------------------------------------------------------
# Execution errors
# ---------------------------------------------------------------------------


def test_execution_error_captured_not_raised() -> None:
    """A framework error is captured into ``execution_error``, not raised."""
    adapter = _ReplayTestAdapter(stratix=_RecordingStratix())
    adapter.connect()
    _emit_simple_run(adapter)
    trace = adapter.serialize_for_replay()

    def factory() -> _FakeAgent:
        return _FakeAgent(raises=True)

    executor = ReplayExecutor(adapter)
    result = asyncio.run(executor.execute_replay(trace, factory))

    assert result.execution_error is not None
    assert "boom" in result.execution_error
    assert result.succeeded is False
    assert len(result.divergences) == 1
    assert result.divergences[0].kind is DivergenceKind.EXECUTION_ERROR


def test_factory_error_captured_as_execution_error() -> None:
    """An exception from the factory itself is captured, not raised."""
    adapter = _ReplayTestAdapter(stratix=_RecordingStratix())
    adapter.connect()
    _emit_simple_run(adapter)
    trace = adapter.serialize_for_replay()

    def factory() -> _FakeAgent:
        raise ValueError("factory broke")

    executor = ReplayExecutor(adapter)
    result = asyncio.run(executor.execute_replay(trace, factory))

    assert result.execution_error is not None
    assert "factory broke" in result.execution_error
    assert result.divergences[0].kind is DivergenceKind.EXECUTION_ERROR


# ---------------------------------------------------------------------------
# Stub injection
# ---------------------------------------------------------------------------


# Patch target lives on the layerlens package so unittest.mock.patch's
# string-importer (which only follows ``__import__``-discoverable
# packages) can resolve it. The ``tests`` directory is not on sys.path
# as a package, so a ``tests.instrument.adapters._base.test_replay.X``
# patch target would raise ModuleNotFoundError. We attach the target
# attribute to the executor module at runtime instead.
import layerlens.instrument.adapters._base.replay as _replay_module

_replay_module._STUB_TARGET_VALUE = ["original"]


class _RecordingStubInjector(StubInjector):
    """Stub injector that swaps a list element so we can verify lifecycle."""

    def __init__(self) -> None:
        self.build_calls: int = 0

    def build_patches(
        self,
        adapter: BaseAdapter,
        trace: ReplayableTrace,
    ) -> Any:
        self.build_calls += 1
        return [
            (
                "layerlens.instrument.adapters._base.replay._STUB_TARGET_VALUE",
                ["stubbed"],
            ),
        ]


def test_stub_injector_applied_then_torn_down() -> None:
    """Stubs are active during the replay and removed afterward."""
    adapter = _ReplayTestAdapter(stratix=_RecordingStratix())
    adapter.connect()
    _emit_simple_run(adapter)
    trace = adapter.serialize_for_replay()

    injector = _RecordingStubInjector()
    inside_value: List[str] = []

    async def factory() -> _FakeAgent:
        # During the replay the patched value should be visible.
        inside_value.extend(_replay_module._STUB_TARGET_VALUE)
        _emit_simple_run(adapter)
        return _FakeAgent()

    executor = ReplayExecutor(adapter, stub_injector=injector)
    asyncio.run(executor.execute_replay(trace, factory))

    assert injector.build_calls == 1
    assert inside_value == ["stubbed"], "stub should have been live during replay"
    # After replay completes the original value is restored.
    assert _replay_module._STUB_TARGET_VALUE == ["original"]


def test_stub_injector_torn_down_on_execution_error() -> None:
    """Stub teardown happens even if the agent raises mid-replay."""
    adapter = _ReplayTestAdapter(stratix=_RecordingStratix())
    adapter.connect()
    _emit_simple_run(adapter)
    trace = adapter.serialize_for_replay()

    injector = _RecordingStubInjector()

    def factory() -> _FakeAgent:
        return _FakeAgent(raises=True)

    executor = ReplayExecutor(adapter, stub_injector=injector)
    asyncio.run(executor.execute_replay(trace, factory))

    # Even after the framework error, the patch was torn down.
    assert _replay_module._STUB_TARGET_VALUE == ["original"]


# ---------------------------------------------------------------------------
# Multi-tenancy
# ---------------------------------------------------------------------------


def test_replay_result_does_not_leak_across_tenants() -> None:
    """A replay started on adapter for tenant A returns ``org_id=A``."""
    a = _ReplayTestAdapter(org_id="tenant-A")
    b = _ReplayTestAdapter(org_id="tenant-B")
    a.connect()
    b.connect()
    _emit_simple_run(a)
    trace = a.serialize_for_replay()

    async def factory_a() -> _FakeAgent:
        _emit_simple_run(a)
        return _FakeAgent()

    async def factory_b() -> _FakeAgent:
        _emit_simple_run(b)
        return _FakeAgent()

    result_a = asyncio.run(ReplayExecutor(a).execute_replay(trace, factory_a))
    result_b = asyncio.run(ReplayExecutor(b).execute_replay(trace, factory_b))

    assert result_a.org_id == "tenant-A"
    assert result_b.org_id == "tenant-B"
    # Captured events on both replays carry the bound tenant.
    for evt in result_a.captured_events:
        assert evt.get("org_id") == "tenant-A"
    for evt in result_b.captured_events:
        assert evt.get("org_id") == "tenant-B"


# ---------------------------------------------------------------------------
# BaseAdapter.execute_replay_via_factory hook
# ---------------------------------------------------------------------------


def test_base_adapter_execute_replay_via_factory_is_not_implemented_by_default() -> None:
    """Adapters opt in by overriding the method; default raises."""
    adapter = _ReplayTestAdapter(stratix=_RecordingStratix())
    trace = adapter.serialize_for_replay()

    async def factory() -> _FakeAgent:
        return _FakeAgent()

    async def call() -> Any:
        return await adapter.execute_replay_via_factory(trace, factory)

    with pytest.raises(NotImplementedError):
        asyncio.run(call())


def test_base_adapter_execute_replay_keeps_engine_signature() -> None:
    """Existing engine-facing ``execute_replay`` still raises NotImplementedError.

    Confirms the factory path was added *alongside* the engine path,
    not in place of it.
    """
    adapter = _ReplayTestAdapter(stratix=_RecordingStratix())

    async def call() -> Any:
        return await adapter.execute_replay({}, None, None, "rid")

    with pytest.raises(NotImplementedError):
        asyncio.run(call())


# ---------------------------------------------------------------------------
# Generic invocation duck-typing
# ---------------------------------------------------------------------------


def test_executor_falls_back_to_callable_agent() -> None:
    """An agent that is itself callable is invoked directly."""

    class CallableAgent:
        def __call__(self, inputs: Any) -> str:
            return f"called:{inputs}"

    adapter = _ReplayTestAdapter(stratix=_RecordingStratix())
    adapter.connect()
    adapter.emit_dict_event("agent.input", {"input": "hello"})
    trace = adapter.serialize_for_replay()

    def factory() -> CallableAgent:
        adapter.emit_dict_event("agent.input", {"input": "hello"})
        return CallableAgent()

    executor = ReplayExecutor(adapter)
    result = asyncio.run(executor.execute_replay(trace, factory))

    assert result.outputs == "called:hello"


def test_executor_raises_when_agent_has_no_invocation_method() -> None:
    """An object with no run/invoke/__call__ → caught as execution error."""

    class OpaqueAgent:
        pass

    adapter = _ReplayTestAdapter(stratix=_RecordingStratix())
    adapter.connect()
    _emit_simple_run(adapter)
    trace = adapter.serialize_for_replay()

    def factory() -> OpaqueAgent:
        return OpaqueAgent()

    executor = ReplayExecutor(adapter)
    result = asyncio.run(executor.execute_replay(trace, factory))

    assert result.execution_error is not None
    assert "cannot invoke agent" in result.execution_error.lower()
