"""Typed-event regression tests for the langgraph adapter.

Bundle #6 of the typed-events migration ports the langgraph
:class:`LayerLensLangGraphAdapter` lifecycle hooks
(:meth:`on_graph_start`, :meth:`on_graph_end`, :meth:`on_node_end`)
from :meth:`emit_dict_event` to typed
:meth:`BaseAdapter.emit_event` calls against the canonical Pydantic
models in :mod:`layerlens.instrument._compat.events`.

The full langgraph test suite is untracked on PR #129's foundation
branch (``test_langgraph_adapter.py`` does not exist on this
branch). Additionally, ``langgraph/state.py`` and
``langgraph/handoff.py`` are also untracked — the lifecycle module
imports :class:`LangGraphStateAdapter` from ``state.py`` at module
load. We use :func:`pytest.importorskip` to defer collection on
branches missing those submodules; the test runs cleanly once the
submodules merge in.

Mirrors the per-adapter typed-event regression pattern from
PR #138 / #151 / #152.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List

import pytest

from layerlens._compat.pydantic import (
    BaseModel as _CompatBaseModel,
    model_dump as _compat_model_dump,
)
from layerlens.instrument._compat.events import (
    AgentInputEvent,
    AgentOutputEvent,
    AgentStateChangeEvent,
    EnvironmentConfigEvent,
)
from layerlens.instrument.adapters._base.capture import CaptureConfig

# langgraph/state.py + langgraph/handoff.py are untracked on PR
# #129's foundation branch — lifecycle.py imports them at module
# load. Defer collection until those submodules land.
lifecycle_module = pytest.importorskip(
    "layerlens.instrument.adapters.frameworks.langgraph.lifecycle"
)
LayerLensLangGraphAdapter = lifecycle_module.LayerLensLangGraphAdapter


class _RecordingStratix:
    org_id: str = "test-org"

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.typed_payloads: List[Any] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})
            return
        if args and isinstance(args[0], _CompatBaseModel):
            payload_model = args[0]
            self.typed_payloads.append(payload_model)
            event_type = getattr(payload_model, "event_type", "<unknown>")
            self.events.append(
                {"event_type": event_type, "payload": _compat_model_dump(payload_model)}
            )


@pytest.fixture
def adapter() -> Any:
    a = LayerLensLangGraphAdapter(
        stratix=_RecordingStratix(),
        capture_config=CaptureConfig.full(),
        org_id="test-org",
    )
    a.connect()
    return a


class TestLanggraphTypedEvents:
    def test_on_graph_start_emits_typed_environment_and_input(
        self, adapter: Any
    ) -> None:
        adapter.on_graph_start(
            graph_id="my_graph",
            execution_id="exec-1",
            initial_state={"messages": ["hi"]},
            config={"thread_id": "t-1"},
        )

        stratix = adapter._stratix
        env_payloads = [p for p in stratix.typed_payloads if isinstance(p, EnvironmentConfigEvent)]
        input_payloads = [p for p in stratix.typed_payloads if isinstance(p, AgentInputEvent)]
        assert len(env_payloads) == 1
        assert len(input_payloads) == 1
        # Canonical environment shape: env_type=simulated, attributes carry framework provenance.
        env = stratix.events[0]
        assert env["event_type"] == "environment.config"
        assert env["payload"]["environment"]["type"] == "simulated"
        assert env["payload"]["environment"]["attributes"]["graph_id"] == "my_graph"
        # Canonical agent.input: role=agent, metadata carries graph_id / execution_id.
        agent_input = stratix.events[1]
        assert agent_input["event_type"] == "agent.input"
        assert agent_input["payload"]["content"]["role"] == "agent"
        assert agent_input["payload"]["content"]["metadata"]["graph_id"] == "my_graph"
        assert agent_input["payload"]["content"]["metadata"]["execution_id"] == "exec-1"

    def test_on_graph_end_emits_typed_output_and_state_change(
        self, adapter: Any
    ) -> None:
        execution = adapter.on_graph_start(
            graph_id="g", execution_id="e1", initial_state={"v": 1},
        )
        adapter.on_graph_end(execution=execution, final_state={"v": 2})

        stratix = adapter._stratix
        output_payloads = [p for p in stratix.typed_payloads if isinstance(p, AgentOutputEvent)]
        state_payloads = [p for p in stratix.typed_payloads if isinstance(p, AgentStateChangeEvent)]
        assert len(output_payloads) == 1
        # State changed: v: 1 → v: 2.
        assert len(state_payloads) == 1
        # Canonical state hash format: ``sha256:<hex64>``.
        state_change = state_payloads[0]
        before = state_change.state.before_hash
        after = state_change.state.after_hash
        assert before.startswith("sha256:")
        assert after.startswith("sha256:")
        assert len(before) == 7 + 64
        assert len(after) == 7 + 64
        # Canonical run_status marker on the agent.output metadata.
        out_event = next(e for e in stratix.events if e["event_type"] == "agent.output")
        assert out_event["payload"]["content"]["metadata"]["run_status"] == "run_complete"

    def test_on_graph_end_with_error_emits_run_failed(self, adapter: Any) -> None:
        execution = adapter.on_graph_start(
            graph_id="g", execution_id="e1", initial_state={"v": 1},
        )
        adapter.on_graph_end(
            execution=execution, final_state={"v": 1}, error=RuntimeError("boom"),
        )

        stratix = adapter._stratix
        out_event = next(e for e in stratix.events if e["event_type"] == "agent.output")
        assert out_event["payload"]["content"]["metadata"]["run_status"] == "run_failed"
        assert out_event["payload"]["content"]["metadata"]["error"] == "boom"

    def test_no_deprecation_warning_after_migration(self, adapter: Any) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            execution = adapter.on_graph_start(
                graph_id="g", execution_id="e1", initial_state={"v": 1},
            )
            adapter.on_graph_end(execution=execution, final_state={"v": 2})

    def test_canonicalize_state_hash_handles_raw_hex(self) -> None:
        """Helper coverage: raw hex64 → sha256-prefixed canonical."""
        canon = lifecycle_module._canonicalize_state_hash
        raw_hex = "a" * 64
        assert canon(raw_hex) == "sha256:" + raw_hex
        already = "sha256:" + ("b" * 64)
        assert canon(already) == already
        # Garbage input still yields a valid canonical hash (re-hashed).
        result = canon("not-a-hash")
        assert result.startswith("sha256:")
        assert len(result) == 7 + 64
