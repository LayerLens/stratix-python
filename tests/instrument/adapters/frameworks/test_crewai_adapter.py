"""Unit tests for the CrewAI framework adapter.

Mocked at the SDK shape level — no real ``crewai`` runtime needed.

After the typed-event migration (PR #129 follow-up — bundle 1) every
emit site in :mod:`layerlens.instrument.adapters.frameworks.crewai.lifecycle`
flows through :meth:`BaseAdapter.emit_event` with a canonical Pydantic
payload. The :class:`_RecordingStratix` stand-in below records both
shapes so pre- and post-migration assertions live side by side: the
``payload`` slot always carries a dict (model-dumped if typed), and
``typed_payloads`` holds the original Pydantic instances for tests
that want to assert against the model surface.

NOTE: The ``on_delegation`` flow still routes through
:mod:`crewai.delegation` (untracked on this branch — covered by a
future follow-up PR). Tests that exercise the delegation path are
expected to still see the legacy dict-emit DeprecationWarning until
that follow-up lands.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

from layerlens._compat.pydantic import (
    BaseModel as _CompatBaseModel,
    model_dump as _compat_model_dump,
)
from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.crewai import (
    ADAPTER_CLASS,
    CrewAIAdapter,
    LayerLensCrewCallback,
    instrument_crew,
)


class _RecordingStratix:
    # Multi-tenant test stand-in: every recording client carries an
    # org_id so adapters constructed with this stratix pass the
    # BaseAdapter fail-fast check. Tests asserting cross-tenant
    # isolation override this default.
    org_id: str = "test-org"

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        # Hold strong references to the original typed payloads for
        # the subset of tests that want to assert against the model
        # surface (e.g. ``isinstance(payload, ToolCallEvent)``). The
        # dict view lives on ``events`` and is what most assertions
        # read.
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


class _FakeCrew:
    def __init__(self, agents: Any = None, process: Any = None) -> None:
        self.agents = agents or []
        self.process = process
        self.step_callback: Any = None
        self.task_callback: Any = None


def _make_agent(role: str = "researcher", tools: Any = None, llm: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        role=role,
        goal="goal",
        backstory="back",
        verbose=False,
        allow_delegation=False,
        max_iter=5,
        memory=False,
        tools=tools,
        llm=llm,
    )


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is CrewAIAdapter


def test_lifecycle() -> None:
    a = CrewAIAdapter(org_id="test-org")
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_adapter_info_and_health() -> None:
    a = CrewAIAdapter(org_id="test-org")
    a.connect()
    info = a.get_adapter_info()
    assert info.framework == "crewai"
    assert info.name == "CrewAIAdapter"
    health = a.health_check()
    assert health.framework_name == "crewai"


def test_instrument_crew_attaches_callback_and_emits_config() -> None:
    """Typed EnvironmentConfigEvent: agent_role lives on
    payload.environment.attributes.
    """
    stratix = _RecordingStratix()
    adapter = CrewAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    crew = _FakeCrew(
        agents=[_make_agent(role="researcher"), _make_agent(role="writer")],
        process="sequential",
    )
    instrumented = adapter.instrument_crew(crew)

    # Callbacks attached.
    assert instrumented.step_callback is not None
    assert instrumented.task_callback is not None
    assert isinstance(instrumented._stratix_callback, LayerLensCrewCallback)

    # Two environment.config events — one per agent role.
    configs = [e for e in stratix.events if e["event_type"] == "environment.config"]
    assert len(configs) == 2
    # Canonical L4a schema: payload.environment.attributes is the dict
    # that carries adapter-specific provenance (agent_role).
    roles = {c["payload"]["environment"]["attributes"]["agent_role"] for c in configs}
    assert roles == {"researcher", "writer"}
    for cfg in configs:
        assert cfg["payload"]["environment"]["type"] == "simulated"


def test_environment_config_idempotent_per_role() -> None:
    """Re-instrumenting a crew with same agents should not re-emit configs."""
    stratix = _RecordingStratix()
    adapter = CrewAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    crew = _FakeCrew(agents=[_make_agent(role="researcher")])
    adapter.instrument_crew(crew)
    adapter.instrument_crew(crew)

    configs = [e for e in stratix.events if e["event_type"] == "environment.config"]
    assert len(configs) == 1


def test_on_crew_start_end_emits_input_output() -> None:
    """Typed AgentInputEvent + AgentOutputEvent for the crew lifecycle.
    CrewAI-specific provenance lives on MessageContent.metadata.
    """
    stratix = _RecordingStratix()
    adapter = CrewAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.on_crew_start(crew_input="research topic")
    adapter.on_crew_end(crew_output="report")

    types = [e["event_type"] for e in stratix.events]
    assert "agent.input" in types
    assert "agent.output" in types

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    payload = out["payload"]
    assert payload["layer"] == "L1"
    assert payload["content"]["message"] == "report"
    metadata = payload["content"]["metadata"]
    assert metadata["framework"] == "crewai"
    assert metadata["duration_ns"] >= 0
    assert metadata["run_status"] == "crew_complete"


def test_on_task_start_end_emits_input_output_and_cost() -> None:
    """Typed migration: task-start → AgentInputEvent (role=AGENT,
    event_subtype=task_start). task-end → AgentOutputEvent
    (run_status=task_complete) + canonical CostRecordEvent.
    """
    stratix = _RecordingStratix()
    adapter = CrewAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.on_task_start("research", agent_role="researcher", task_order=1)

    # Build a task_output with token_usage to also verify cost.record fires.
    task_output = SimpleNamespace(token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    adapter.on_task_end(task_output=task_output, agent_role="researcher", task_order=1)

    # Task-start → AgentInputEvent (role=AGENT, event_subtype=task_start).
    inputs = [e for e in stratix.events if e["event_type"] == "agent.input"]
    task_inputs = [
        e for e in inputs
        if e["payload"]["content"]["metadata"].get("event_subtype") == "task_start"
    ]
    assert len(task_inputs) == 1
    ts_payload = task_inputs[0]["payload"]
    assert ts_payload["content"]["message"] == "research"
    assert ts_payload["content"]["role"] == "agent"
    assert ts_payload["content"]["metadata"]["agent_role"] == "researcher"
    assert ts_payload["content"]["metadata"]["task_order"] == 1

    # Task-end → AgentOutputEvent (run_status=task_complete).
    outputs = [e for e in stratix.events if e["event_type"] == "agent.output"]
    task_outputs = [
        e for e in outputs
        if e["payload"]["content"]["metadata"].get("event_subtype") == "task_complete"
    ]
    assert len(task_outputs) == 1
    te_metadata = task_outputs[0]["payload"]["content"]["metadata"]
    assert te_metadata["agent_role"] == "researcher"
    assert te_metadata["run_status"] == "task_complete"

    # Canonical cost.record: tokens via prompt_tokens / completion_tokens / tokens.
    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    cost_payload = cost["payload"]
    assert cost_payload["cost"]["prompt_tokens"] == 10
    assert cost_payload["cost"]["completion_tokens"] == 5
    assert cost_payload["cost"]["tokens"] == 15


def test_on_tool_use_emits_event() -> None:
    """Typed ToolCallEvent: tool name lives at payload.tool.name."""
    stratix = _RecordingStratix()
    adapter = CrewAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2, latency_ms=12.3)

    evt = next(e for e in stratix.events if e["event_type"] == "tool.call")
    payload = evt["payload"]
    assert payload["layer"] == "L5a"
    assert payload["tool"]["name"] == "calc"
    assert payload["tool"]["integration"] == "library"
    assert payload["latency_ms"] == 12.3
    assert payload["input"] == {"x": 1}
    # Scalar tool_output is wrapped in {"value": ...} so the canonical
    # ``output: dict`` slot is satisfied.
    assert payload["output"] == {"value": 2}


def test_on_delegation_emits_handoff() -> None:
    """Delegation flows through the (untracked) delegation tracker.

    The delegation tracker still uses ``emit_dict_event`` on this
    branch — it lives in ``crewai/delegation.py`` which is untracked
    and outside this PR's scope (per
    ``docs/adapters/typed-events-followups.md``). We assert the
    handoff event lands at the dict shape that the tracker emits.
    """
    import warnings

    stratix = _RecordingStratix()
    adapter = CrewAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    # Suppress the expected DeprecationWarning from delegation.py — it
    # is a known follow-up site documented in the bundle 1 PR body.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        adapter.on_delegation(from_agent="researcher", to_agent="writer", context="findings")

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    assert evt["payload"]["from_agent"] == "researcher"
    assert evt["payload"]["to_agent"] == "writer"


def test_capture_config_gates_l5a_tool_calls() -> None:
    """When l5a_tool_calls is disabled, tool.call events do NOT fire,
    but cross-cutting handoff events (from the untracked delegation
    tracker) still emit.
    """
    import warnings

    stratix = _RecordingStratix()
    cfg = CaptureConfig(l5a_tool_calls=False)
    adapter = CrewAIAdapter(stratix=stratix, capture_config=cfg)
    adapter.connect()

    adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2)
    # Delegation goes through the untracked dict-emit tracker — see
    # test_on_delegation_emits_handoff for the rationale.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        adapter.on_delegation(from_agent="a", to_agent="b", context="x")

    types = [e["event_type"] for e in stratix.events]
    assert "tool.call" not in types
    # handoff is cross-cutting / always enabled.
    assert "agent.handoff" in types


def test_instrument_crew_helper() -> None:
    """Top-level convenience function returns the instrumented crew."""
    crew = _FakeCrew(agents=[_make_agent(role="r1")])
    result = instrument_crew(crew, org_id="test-org")
    # The helper returns the crew itself (with callbacks attached).
    assert result is crew
    assert result._stratix_callback is not None


def test_serialize_for_replay() -> None:
    adapter = CrewAIAdapter(
        stratix=_RecordingStratix(),
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()
    rt = adapter.serialize_for_replay()
    assert rt.framework == "crewai"
    assert rt.adapter_name == "CrewAIAdapter"
    assert "capture_config" in rt.config


# ---------------------------------------------------------------------------
# Typed-event migration regression tests (PR #129 follow-up — bundle 1)
# ---------------------------------------------------------------------------


def test_crewai_lifecycle_emits_typed_payloads_only() -> None:
    """Every emit site in crewai lifecycle.py is a typed emit_event call.

    Pins the post-migration contract for the lifecycle module: the
    recording stratix's ``typed_payloads`` list grows for every
    emission and the legacy two-arg dict path receives nothing.
    Delegation is excluded — it routes through the untracked
    ``crewai/delegation.py`` (covered by a future follow-up PR).
    """
    from layerlens.instrument._compat.events import (
        ToolCallEvent,
        AgentInputEvent,
        CostRecordEvent,
        AgentOutputEvent,
        ModelInvokeEvent,
        EnvironmentConfigEvent,
    )

    stratix = _RecordingStratix()
    adapter = CrewAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    crew = _FakeCrew(agents=[_make_agent(role="researcher")], process="sequential")
    adapter.instrument_crew(crew)
    adapter.on_crew_start(crew_input="topic")
    adapter.on_task_start("research", agent_role="researcher", task_order=1)
    task_output = SimpleNamespace(
        token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    adapter.on_task_end(task_output=task_output, agent_role="researcher", task_order=1)
    adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2)
    adapter.on_llm_call(provider="openai", model="gpt-5", tokens_prompt=8)
    adapter.on_crew_end(crew_output="report")

    # Every captured payload from lifecycle paths is a Pydantic model.
    assert stratix.typed_payloads, "expected typed payloads to be captured"
    types_seen = {type(p) for p in stratix.typed_payloads}
    assert AgentInputEvent in types_seen
    assert AgentOutputEvent in types_seen
    assert CostRecordEvent in types_seen
    assert EnvironmentConfigEvent in types_seen
    assert ModelInvokeEvent in types_seen
    assert ToolCallEvent in types_seen


def test_crewai_lifecycle_emit_does_not_warn_after_migration() -> None:
    """No DeprecationWarning fires from crewai lifecycle.py paths.

    Excludes ``on_delegation`` which still routes through the
    untracked delegation tracker.
    """
    import warnings

    stratix = _RecordingStratix()
    adapter = CrewAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    crew = _FakeCrew(agents=[_make_agent(role="researcher")], process="sequential")

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        adapter.instrument_crew(crew)
        adapter.on_crew_start(crew_input="topic")
        adapter.on_task_start("research", agent_role="researcher", task_order=1)
        task_output = SimpleNamespace(
            token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
        adapter.on_task_end(task_output=task_output, agent_role="researcher", task_order=1)
        adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2)
        adapter.on_llm_call(provider="openai", model="gpt-5", tokens_prompt=8)
        adapter.on_crew_end(crew_output="report")
