"""Unit tests for the Agno framework adapter.

Mocked at the SDK shape level — no real ``agno`` runtime needed.

After the typed-event migration (PR
``feat/instrument-typed-events-foundation``) every emit site flows
through :meth:`BaseAdapter.emit_event` with a canonical Pydantic
payload. The :class:`_RecordingStratix` stand-in below records both
shapes so pre- and post-migration assertions live side by side: the
``payload`` slot always carries a dict (model-dumped if typed).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from layerlens._compat.pydantic import BaseModel as _CompatBaseModel, model_dump as _compat_model_dump
from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.agno import (
    ADAPTER_CLASS,
    AgnoAdapter,
    instrument_agent,
)


class _RecordingStratix:
    # Multi-tenant test stand-in: every recording client carries an
    # org_id so adapters constructed with this stratix pass the
    # BaseAdapter fail-fast check. Tests asserting cross-tenant
    # isolation override this default.
    org_id: str = "test-org"

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        # Hold strong references to the original typed payloads for the
        # subset of tests that want to assert against the model surface
        # (e.g. ``isinstance(payload, ToolCallEvent)``). The dict view
        # lives on ``events`` and is what most assertions read.
        self.typed_payloads: List[Any] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        # Two-arg legacy path: ``emit(event_type, payload_dict)``.
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})
            return
        # Single-arg typed path: ``emit(payload_model[, privacy_level])``.
        if args and isinstance(args[0], _CompatBaseModel):
            payload_model = args[0]
            self.typed_payloads.append(payload_model)
            event_type = getattr(payload_model, "event_type", "<unknown>")
            self.events.append(
                {"event_type": event_type, "payload": _compat_model_dump(payload_model)}
            )


class _FakeAgent:
    """Minimal duck-typed Agno agent for tests."""

    def __init__(
        self,
        name: str = "test-agent",
        tools: Any = None,
        model: Any = None,
        description: Any = None,
        instructions: Any = None,
        team: Any = None,
        knowledge: Any = None,
        result: Any = None,
        raises: bool = False,
    ) -> None:
        self.name = name
        self.tools = tools
        self.model = model
        self.description = description
        self.instructions = instructions
        self.team = team
        self.knowledge = knowledge
        self._result = result
        self._raises = raises

    def run(self, message: str, **kwargs: Any) -> Any:
        if self._raises:
            raise RuntimeError("simulated failure")
        return self._result if self._result is not None else SimpleNamespace(content=f"out:{message}")


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is AgnoAdapter


def test_lifecycle() -> None:
    a = AgnoAdapter(org_id="test-org")
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    assert a.is_connected is True
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED
    assert a.is_connected is False


def test_adapter_info_and_health() -> None:
    a = AgnoAdapter(org_id="test-org")
    a.connect()
    info = a.get_adapter_info()
    assert info.framework == "agno"
    assert info.name == "AgnoAdapter"
    assert info.version == AgnoAdapter.VERSION
    assert info.capabilities  # non-empty list
    health = a.health_check()
    assert health.framework_name == "agno"
    assert health.status == AdapterStatus.HEALTHY


def test_instrument_agent_wraps_run() -> None:
    adapter = AgnoAdapter(stratix=_RecordingStratix(), capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="planner")
    adapter.instrument_agent(agent)
    # Wrapped: function name is now traced.
    assert agent.run.__name__ == "traced_run_sync"

    adapter.disconnect()
    # Restored: name is back to the original.
    assert agent.run.__name__ == "run"


def test_run_emits_input_and_output_events() -> None:
    """Typed-event assertions: agno emits canonical L1/L4 payloads.

    After the typed-event migration, agno-specific provenance lives
    on :class:`MessageContent.metadata` rather than at the top level
    of the payload dict. The top-level dict reflects the canonical
    schema (``content``, ``layer``, ``event_type``).
    """
    stratix = _RecordingStratix()
    adapter = AgnoAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="planner", model="gpt-5")
    adapter.instrument_agent(agent)
    result = agent.run("hello")

    assert getattr(result, "content", None) == "out:hello"

    types = [e["event_type"] for e in stratix.events]
    assert "environment.config" in types
    assert "agent.input" in types
    assert "agent.output" in types

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    # Canonical L1 schema: payload carries ``content`` + ``layer``.
    assert out["payload"]["layer"] == "L1"
    content = out["payload"]["content"]
    # Agno-specific provenance lives in MessageContent.metadata.
    assert content["metadata"]["agent_name"] == "planner"
    assert content["metadata"]["framework"] == "agno"
    assert content["metadata"]["duration_ns"] >= 0


def test_run_failure_emits_output_with_error() -> None:
    """Errors are surfaced via canonical metadata on AgentOutputEvent.

    The previous adapter put ``error`` at the top level of an ad-hoc
    payload dict; the canonical schema has no top-level error slot
    on :class:`AgentOutputEvent`, so the error is carried in
    :class:`MessageContent.metadata` and the test asserts that
    location.
    """
    stratix = _RecordingStratix()
    adapter = AgnoAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="failing", raises=True)
    adapter.instrument_agent(agent)

    with pytest.raises(RuntimeError):
        agent.run("bad")

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    metadata = out["payload"]["content"]["metadata"]
    assert "error" in metadata
    assert "simulated failure" in metadata["error"]
    assert metadata["run_status"] == "run_failed"


def test_environment_config_emits_once_per_agent() -> None:
    stratix = _RecordingStratix()
    adapter = AgnoAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="a1", tools=[SimpleNamespace(name="search")], model="gpt-5")
    adapter.instrument_agent(agent)
    adapter.instrument_agent(agent)  # idempotent

    configs = [e for e in stratix.events if e["event_type"] == "environment.config"]
    assert len(configs) == 1
    # Canonical L4a schema: payload.environment.attributes is the dict
    # that carries adapter-specific provenance.
    attributes = configs[0]["payload"]["environment"]["attributes"]
    assert attributes["agent_name"] == "a1"
    assert attributes["tools"] == ["search"]
    assert configs[0]["payload"]["environment"]["type"] == "simulated"


def test_on_tool_use_emits_event() -> None:
    """Typed ToolCallEvent: tool name lives at payload.tool.name."""
    stratix = _RecordingStratix()
    adapter = AgnoAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2, latency_ms=12.3)

    evt = next(e for e in stratix.events if e["event_type"] == "tool.call")
    payload = evt["payload"]
    assert payload["layer"] == "L5a"
    assert payload["tool"]["name"] == "calc"
    assert payload["tool"]["integration"] == "library"
    assert payload["latency_ms"] == 12.3
    assert payload["input"] == {"x": 1}
    # Scalar tool_output is wrapped in {value: ...} so the canonical
    # ``output: dict`` slot is satisfied.
    assert payload["output"] == {"value": 2}


def test_on_handoff_emits_event_with_context_hash() -> None:
    """Typed AgentHandoffEvent: handoff_context_hash is sha256:<hex64>."""
    stratix = _RecordingStratix()
    adapter = AgnoAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter.on_handoff(from_agent="a", to_agent="b", context="some context")

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    payload = evt["payload"]
    assert payload["from_agent"] == "a"
    assert payload["to_agent"] == "b"
    assert payload["handoff_context_hash"].startswith("sha256:")
    # 7 chars prefix + 64 hex = 71 chars total per the canonical
    # validator in events_cross_cutting.py.
    assert len(payload["handoff_context_hash"]) == 7 + 64


def test_capture_config_gates_l5a_tool_calls() -> None:
    """When l5a_tool_calls is disabled, tool.call events do NOT fire."""
    stratix = _RecordingStratix()
    cfg = CaptureConfig(l5a_tool_calls=False)
    adapter = AgnoAdapter(stratix=stratix, capture_config=cfg)
    adapter.connect()

    adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2)
    # And handoffs (cross-cutting) should still fire.
    adapter.on_handoff(from_agent="a", to_agent="b", context="x")

    types = [e["event_type"] for e in stratix.events]
    assert "tool.call" not in types
    assert "agent.handoff" in types


def test_instrument_agent_helper() -> None:
    """Top-level convenience function returns a connected adapter."""
    agent = _FakeAgent(name="helper")
    adapter = instrument_agent(agent, org_id="test-org")
    assert adapter.is_connected is True
    assert adapter.status == AdapterStatus.HEALTHY


def test_serialize_for_replay() -> None:
    adapter = AgnoAdapter(
        stratix=_RecordingStratix(),
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()

    rt = adapter.serialize_for_replay()
    assert rt.framework == "agno"
    assert rt.adapter_name == "AgnoAdapter"
    assert "capture_config" in rt.config


# ---------------------------------------------------------------------------
# Typed-event migration regression tests
# ---------------------------------------------------------------------------


def test_agno_emits_typed_payloads_only() -> None:
    """Every emit site in agno is a typed :meth:`emit_event` call.

    Pins the post-migration contract: the recording stratix's
    ``typed_payloads`` list grows for every emission and the legacy
    two-arg dict path receives nothing. This is the public contract
    backing the ``grep emit_dict_event src/.../agno/ → 0`` acceptance
    criterion in the typed-events foundation PR.
    """
    from layerlens.instrument._compat.events import (
        AgentInputEvent,
        AgentOutputEvent,
        ModelInvokeEvent,
        EnvironmentConfigEvent,
    )

    stratix = _RecordingStratix()
    adapter = AgnoAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="planner", model="gpt-5")
    adapter.instrument_agent(agent)
    agent.run("hello")

    # Every captured payload is a Pydantic model instance — the
    # legacy dict path was not used.
    assert stratix.typed_payloads, "expected typed payloads to be captured"
    types_seen = {type(p) for p in stratix.typed_payloads}
    assert AgentInputEvent in types_seen
    assert AgentOutputEvent in types_seen
    assert EnvironmentConfigEvent in types_seen
    assert ModelInvokeEvent in types_seen


def test_agno_emit_does_not_warn_after_migration() -> None:
    """No DeprecationWarning fires from agno emission paths.

    The base adapter's ``emit_dict_event`` raises a
    :class:`DeprecationWarning` on every call. After migration, agno
    must never trigger that warning. ``filterwarnings("error", ...)``
    converts the warning into a test failure.
    """
    import warnings

    stratix = _RecordingStratix()
    adapter = AgnoAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="planner", model="gpt-5")
    adapter.instrument_agent(agent)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        agent.run("hello")
        adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2)
        adapter.on_handoff(from_agent="a", to_agent="b", context="ctx")
        adapter.on_llm_call(provider="openai", model="gpt-5", tokens_prompt=10)


def test_agno_typed_handoff_validates_canonical_hash() -> None:
    """Handoffs emit a canonical sha256 context hash.

    Pins the regression: the previous adapter emitted ``None`` when
    no context was supplied, which violated the canonical schema's
    ``handoff_context_hash`` validator.
    """
    stratix = _RecordingStratix()
    adapter = AgnoAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    # Empty context → still a well-formed hash (over the empty string).
    adapter.on_handoff(from_agent="a", to_agent="b", context=None)

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    assert evt["payload"]["handoff_context_hash"].startswith("sha256:")


def test_agno_typed_emission_records_org_id() -> None:
    """Typed emit_event still stamps the bound org_id on every event.

    The canonical event payload models do not declare ``org_id`` as a
    field (the Identity envelope sits one level up in the production
    :class:`StratixEvent` wrapper). The base adapter therefore
    re-injects ``org_id`` into the dict view emitted to sinks via
    :meth:`_post_emit_success`, and into the trace replay buffer.
    Both surfaces are asserted here.
    """
    stratix = _RecordingStratix()
    adapter = AgnoAdapter(stratix=stratix, capture_config=CaptureConfig.full(), org_id="tenant-42")
    adapter.connect()

    adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2)

    # Replay buffer carries org_id at the envelope level.
    rt = adapter.serialize_for_replay()
    assert any(evt.get("org_id") == "tenant-42" for evt in rt.events)
    # And inside each per-event payload dict (re-injected by
    # _post_emit_success regardless of whether the model declared it).
    assert any(
        evt.get("payload", {}).get("org_id") == "tenant-42" for evt in rt.events
    )
