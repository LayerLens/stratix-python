"""L2 ``AgentCodeEvent`` emission tests for the LangGraph adapter.

Validates the gap closure called out in the depth audit
(``A:/tmp/adapter-depth-audit.md`` §1.3) and demanded by spec
``04a-langgraph-adapter-spec.md`` §3 mapping table line 57:

    | Node execution (each node in the graph) | AgentCodeEvent | L2 |

Every node execution — including the error path — must emit one
``AgentCodeEvent`` carrying repo / commit / artifact_hash / config_hash
plus per-execution ``build_info`` (duration, key sets, status, error
class). ``artifact_hash`` must be deterministic across re-execution so
replay diff engines can correlate per-node artifacts.

Two emission paths are exercised here:

* :class:`NodeTracer` decorator path (``trace_node`` / ``decorate``)
  used by callers that wrap individual node functions.
* :class:`LayerLensLangGraphAdapter.on_node_start` /
  :meth:`on_node_end` lifecycle path used by callers driving the
  adapter directly (handoff detector / custom orchestrators).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument._vendored.events import CodeInfo, AgentCodeEvent
from layerlens.instrument.adapters._base.adapter import AdapterStatus
from layerlens.instrument.adapters.frameworks.langgraph import (
    NodeTracer,
    LayerLensLangGraphAdapter,
)
from layerlens.instrument.adapters.frameworks.langgraph.nodes import (
    _RUNTIME_REPO_SENTINEL,
    _RUNTIME_COMMIT_SENTINEL,
    _resolve_callable_artifact_hash,
)

# ---------------------------------------------------------------------------
# Recording stratix client + adapter helpers
# ---------------------------------------------------------------------------


class _RecordingStratix:
    """Minimal STRATIX client double that captures every emit() call.

    Mirrors the pattern used by ``test_autogen_adapter`` so assertions
    look the same across framework adapter test suites.
    """

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.typed_events: List[Any] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        # The adapter path emits typed Pydantic payloads directly:
        #   stratix.emit(event_payload)         (single arg)
        #   stratix.emit(event_payload, lvl)    (typed + privacy_level)
        # The legacy / dict path emits (event_type, payload_dict):
        #   stratix.emit("agent.code", {...})
        if len(args) == 1:
            payload = args[0]
            event_type = getattr(payload, "event_type", None)
            self.typed_events.append(payload)
            if event_type:
                # Build a dict view for assertion symmetry with the
                # legacy path. ``model_dump`` is available on every
                # Pydantic v2 BaseModel; the vendored modules are v2.
                model_dump = getattr(payload, "model_dump", None)
                payload_dict = model_dump() if callable(model_dump) else {}
                self.events.append({"event_type": event_type, "payload": payload_dict})
            return
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})
            return


# Sample node callables used across the suite. Defined at module scope
# so ``__qualname__`` is stable and the artifact hash assertion is not
# perturbed by closure-scope qualname differences (which would defeat
# the determinism check).


def _double_x_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Sample success-path node: doubles ``state['x']``."""
    out = dict(state)
    out["x"] = state.get("x", 0) * 2
    return out


def _failing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Sample error-path node: always raises."""
    raise RuntimeError("synthetic node failure")


def _identity_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Sample no-op node: returns state unchanged."""
    return state


# ---------------------------------------------------------------------------
# NodeTracer decorator path
# ---------------------------------------------------------------------------


class TestNodeTracerEmitsAgentCodeOnSuccess:
    """The decorator + context-manager path emits L2 ``AgentCodeEvent``."""

    def _build_adapter(self) -> tuple[_RecordingStratix, LayerLensLangGraphAdapter]:
        stratix = _RecordingStratix()
        adapter = LayerLensLangGraphAdapter(
            stratix=stratix,
            capture_config=CaptureConfig.full(),
        )
        adapter.connect()
        return stratix, adapter

    def test_one_agent_code_event_per_successful_node(self) -> None:
        stratix, adapter = self._build_adapter()
        tracer = NodeTracer(adapter=adapter)

        traced = tracer.decorate(_double_x_node)
        traced({"x": 1})

        agent_code_events = [e for e in stratix.events if e["event_type"] == "agent.code"]
        assert len(agent_code_events) == 1, (
            f"expected exactly 1 agent.code event, got {len(agent_code_events)}: "
            f"{[e['event_type'] for e in stratix.events]}"
        )

    def test_required_l2_envelope_fields_are_present(self) -> None:
        """Spec 04a §3 + 05 trace schema: code.{repo,commit,artifact_hash,config_hash}."""
        stratix, adapter = self._build_adapter()
        tracer = NodeTracer(adapter=adapter)

        traced = tracer.decorate(_double_x_node)
        traced({"x": 1})

        evt = next(e for e in stratix.events if e["event_type"] == "agent.code")
        assert evt["payload"]["layer"] == "L2"
        code = evt["payload"]["code"]
        # Required CodeInfo fields per ateam stratix.core.events.l2_code:
        for key in ("repo", "commit", "artifact_hash", "config_hash"):
            assert key in code, f"L2 code envelope missing {key!r}: {code}"
        # Hash format guard from CodeInfo.validate_hash.
        assert code["artifact_hash"].startswith("sha256:")
        assert code["config_hash"].startswith("sha256:")
        assert len(code["artifact_hash"]) == len("sha256:") + 64
        assert len(code["config_hash"]) == len("sha256:") + 64

    def test_repo_resolves_to_module_path_for_module_callable(self) -> None:
        stratix, adapter = self._build_adapter()
        tracer = NodeTracer(adapter=adapter)

        traced = tracer.decorate(_double_x_node)
        traced({"x": 1})

        evt = next(e for e in stratix.events if e["event_type"] == "agent.code")
        # _double_x_node is module-defined → repo is the module path.
        assert evt["payload"]["code"]["repo"] == _double_x_node.__module__

    def test_repo_falls_back_for_non_module_callable(self) -> None:
        """Lambdas with stripped ``__module__`` fall back to the runtime sentinel."""
        stratix, adapter = self._build_adapter()
        tracer = NodeTracer(adapter=adapter)

        # Strip __module__ to simulate a callable from a module-less
        # context (such as REPL-defined functions).
        anon: Any = lambda state: state  # noqa: E731
        anon.__module__ = ""

        with tracer.trace_node("anon_node", {"x": 1}, node_callable=anon):
            pass

        evt = next(e for e in stratix.events if e["event_type"] == "agent.code")
        assert evt["payload"]["code"]["repo"] == _RUNTIME_REPO_SENTINEL

    def test_build_info_contains_per_execution_metadata(self) -> None:
        stratix, adapter = self._build_adapter()
        tracer = NodeTracer(adapter=adapter)

        traced = tracer.decorate(_double_x_node)
        traced({"x": 1, "trace_id": "abc"})

        evt = next(e for e in stratix.events if e["event_type"] == "agent.code")
        build_info = evt["payload"]["code"]["build_info"]
        assert build_info["status"] == "success"
        assert build_info["node_name"] == "_double_x_node"
        assert build_info["execution_duration_ns"] >= 0
        # PII-safe: keys NAMES only — values must NOT appear anywhere
        # in the L2 envelope.
        assert sorted(build_info["input_state_keys"]) == ["trace_id", "x"]
        assert sorted(build_info["output_state_keys"]) == ["trace_id", "x"]


class TestNodeTracerEmitsAgentCodeOnError:
    """Error path: AgentCodeEvent must still be emitted, with error_class."""

    def _build_adapter(self) -> tuple[_RecordingStratix, LayerLensLangGraphAdapter]:
        stratix = _RecordingStratix()
        adapter = LayerLensLangGraphAdapter(
            stratix=stratix,
            capture_config=CaptureConfig.full(),
        )
        adapter.connect()
        return stratix, adapter

    def test_agent_code_emitted_when_node_raises(self) -> None:
        stratix, adapter = self._build_adapter()
        tracer = NodeTracer(adapter=adapter)

        with pytest.raises(RuntimeError, match="synthetic node failure"):
            with tracer.trace_node("failing", {"x": 1}, node_callable=_failing_node):
                _failing_node({"x": 1})

        agent_code_events = [e for e in stratix.events if e["event_type"] == "agent.code"]
        assert len(agent_code_events) == 1, (
            "L2 emission must occur on the error path — "
            "spec 04a §3 table line 57"
        )

    def test_error_class_in_build_info(self) -> None:
        stratix, adapter = self._build_adapter()
        tracer = NodeTracer(adapter=adapter)

        with pytest.raises(RuntimeError):
            with tracer.trace_node("failing", {"x": 1}, node_callable=_failing_node):
                _failing_node({"x": 1})

        evt = next(e for e in stratix.events if e["event_type"] == "agent.code")
        build_info = evt["payload"]["code"]["build_info"]
        assert build_info["status"] == "error"
        assert build_info["error_class"] == "RuntimeError"
        # Truncation flag present on the error path.
        assert "error_truncated" in build_info
        assert build_info["error_truncated"] is False

    def test_error_message_truncated_when_oversized(self) -> None:
        """Long error messages are truncated to bound PII / payload risk."""
        stratix, adapter = self._build_adapter()
        tracer = NodeTracer(adapter=adapter)

        # Define a callable that raises a >>512 byte message.
        big_msg = "x" * 2000

        def _big_error_node(state: Dict[str, Any]) -> Dict[str, Any]:
            raise ValueError(big_msg)

        with pytest.raises(ValueError):
            with tracer.trace_node("big_err", {}, node_callable=_big_error_node):
                _big_error_node({})

        evt = next(e for e in stratix.events if e["event_type"] == "agent.code")
        build_info = evt["payload"]["code"]["build_info"]
        assert build_info["error_class"] == "ValueError"
        assert build_info["error_truncated"] is True


class TestArtifactHashDeterminism:
    """Replay correlation requires deterministic per-node artifact hashes."""

    def test_same_callable_yields_same_artifact_hash_across_executions(self) -> None:
        stratix = _RecordingStratix()
        adapter = LayerLensLangGraphAdapter(
            stratix=stratix,
            capture_config=CaptureConfig.full(),
        )
        adapter.connect()
        tracer = NodeTracer(adapter=adapter)

        traced = tracer.decorate(_double_x_node)
        traced({"x": 1})
        traced({"x": 5})
        traced({"x": 99})

        agent_code_events = [e for e in stratix.events if e["event_type"] == "agent.code"]
        assert len(agent_code_events) == 3
        hashes = {e["payload"]["code"]["artifact_hash"] for e in agent_code_events}
        assert len(hashes) == 1, (
            "artifact_hash must be deterministic across re-executions — "
            "replay diff engines correlate per-node artifacts via this hash"
        )

    def test_helper_artifact_hash_matches_emitted_hash(self) -> None:
        """The helper used by tests of replay tooling produces the same value."""
        stratix = _RecordingStratix()
        adapter = LayerLensLangGraphAdapter(
            stratix=stratix,
            capture_config=CaptureConfig.full(),
        )
        adapter.connect()
        tracer = NodeTracer(adapter=adapter)

        tracer.decorate(_double_x_node)({"x": 1})

        evt = next(e for e in stratix.events if e["event_type"] == "agent.code")
        expected = _resolve_callable_artifact_hash(_double_x_node)
        assert evt["payload"]["code"]["artifact_hash"] == expected

    def test_different_callables_yield_different_artifact_hashes(self) -> None:
        stratix = _RecordingStratix()
        adapter = LayerLensLangGraphAdapter(
            stratix=stratix,
            capture_config=CaptureConfig.full(),
        )
        adapter.connect()
        tracer = NodeTracer(adapter=adapter)

        tracer.decorate(_double_x_node)({"x": 1})
        tracer.decorate(_identity_node)({"x": 1})

        agent_code_events = [e for e in stratix.events if e["event_type"] == "agent.code"]
        assert len(agent_code_events) == 2
        h1 = agent_code_events[0]["payload"]["code"]["artifact_hash"]
        h2 = agent_code_events[1]["payload"]["code"]["artifact_hash"]
        assert h1 != h2, "Distinct node bodies must yield distinct artifact hashes"


class TestPiiSafety:
    """Spec privacy contract: only key NAMES, never values, in L2 envelope."""

    def test_state_values_never_appear_in_emitted_envelope(self) -> None:
        stratix = _RecordingStratix()
        adapter = LayerLensLangGraphAdapter(
            stratix=stratix,
            capture_config=CaptureConfig.full(),
        )
        adapter.connect()
        tracer = NodeTracer(adapter=adapter)

        secret = "SECRET_API_KEY_DO_NOT_LEAK"
        traced = tracer.decorate(_double_x_node)
        traced({"x": 1, "auth": secret})

        evt = next(e for e in stratix.events if e["event_type"] == "agent.code")
        # Serialize the entire payload (including build_info) and assert
        # the secret does not appear anywhere in the L2 envelope.
        import json

        serialized = json.dumps(evt["payload"], default=str)
        assert secret not in serialized, (
            "L2 AgentCodeEvent must not carry state VALUES — only key NAMES"
        )
        # The key NAME ('auth') is allowed and expected.
        build_info = evt["payload"]["code"]["build_info"]
        assert "auth" in build_info["input_state_keys"]
        assert "auth" in build_info["output_state_keys"]


class TestCommitSentinel:
    """Adapter-level emission uses the runtime commit sentinel."""

    def test_commit_field_uses_runtime_sentinel(self) -> None:
        stratix = _RecordingStratix()
        adapter = LayerLensLangGraphAdapter(
            stratix=stratix,
            capture_config=CaptureConfig.full(),
        )
        adapter.connect()
        tracer = NodeTracer(adapter=adapter)

        tracer.decorate(_double_x_node)({"x": 1})

        evt = next(e for e in stratix.events if e["event_type"] == "agent.code")
        assert evt["payload"]["code"]["commit"] == _RUNTIME_COMMIT_SENTINEL


# ---------------------------------------------------------------------------
# Lifecycle (on_node_start / on_node_end) path
# ---------------------------------------------------------------------------


class TestLifecycleOnNodeEmitsAgentCode:
    """Direct on_node_start / on_node_end calls also emit AgentCodeEvent."""

    def _build(self) -> tuple[_RecordingStratix, LayerLensLangGraphAdapter]:
        stratix = _RecordingStratix()
        adapter = LayerLensLangGraphAdapter(
            stratix=stratix,
            capture_config=CaptureConfig.full(),
        )
        adapter.connect()
        return stratix, adapter

    def test_on_node_lifecycle_emits_agent_code_per_execution(self) -> None:
        stratix, adapter = self._build()
        graph_exec = adapter.on_graph_start(
            graph_id="g1",
            execution_id="g1:1",
            initial_state={"x": 0},
            config=None,
        )

        # First node
        ctx_a = adapter.on_node_start(
            graph_exec,
            "node_a",
            state={"x": 0},
            node_callable=_double_x_node,
        )
        adapter.on_node_end(graph_exec, ctx_a, state={"x": 0})

        # Second node, different callable
        ctx_b = adapter.on_node_start(
            graph_exec,
            "node_b",
            state={"x": 0},
            node_callable=_identity_node,
        )
        adapter.on_node_end(graph_exec, ctx_b, state={"x": 0})

        agent_code_events = [e for e in stratix.events if e["event_type"] == "agent.code"]
        assert len(agent_code_events) == 2, (
            "lifecycle on_node_end must emit one agent.code per node execution"
        )
        names = {e["payload"]["code"]["build_info"]["node_name"] for e in agent_code_events}
        assert names == {"node_a", "node_b"}

    def test_lifecycle_emits_on_error_path(self) -> None:
        stratix, adapter = self._build()
        graph_exec = adapter.on_graph_start(
            graph_id="g1",
            execution_id="g1:1",
            initial_state={"x": 0},
            config=None,
        )
        ctx = adapter.on_node_start(
            graph_exec,
            "boom",
            state={"x": 0},
            node_callable=_failing_node,
        )
        adapter.on_node_end(
            graph_exec,
            ctx,
            state={"x": 0},
            error=RuntimeError("kaboom"),
        )

        evt = next(e for e in stratix.events if e["event_type"] == "agent.code")
        assert evt["payload"]["code"]["build_info"]["status"] == "error"
        assert evt["payload"]["code"]["build_info"]["error_class"] == "RuntimeError"


# ---------------------------------------------------------------------------
# CaptureConfig L2 gating — disabled by default
# ---------------------------------------------------------------------------


class TestCaptureConfigGating:
    """L2 events are gated by CaptureConfig.l2_agent_code (default: False).

    Spec 04a §2 step 3: ``L2 for code``. The cross-cutting state.change
    event is always enabled, but ``agent.code`` honours the L2 flag.
    """

    def test_l2_disabled_drops_agent_code(self) -> None:
        stratix = _RecordingStratix()
        # Standard config has l2_agent_code=False.
        adapter = LayerLensLangGraphAdapter(
            stratix=stratix,
            capture_config=CaptureConfig.standard(),
        )
        adapter.connect()
        tracer = NodeTracer(adapter=adapter)

        tracer.decorate(_double_x_node)({"x": 1})

        agent_code_events = [e for e in stratix.events if e["event_type"] == "agent.code"]
        assert agent_code_events == [], (
            "CaptureConfig.standard() has l2_agent_code=False — "
            "L2 events must be dropped by the BaseAdapter gate"
        )

    def test_l2_enabled_emits_agent_code(self) -> None:
        stratix = _RecordingStratix()
        adapter = LayerLensLangGraphAdapter(
            stratix=stratix,
            capture_config=CaptureConfig(l2_agent_code=True),
        )
        adapter.connect()
        tracer = NodeTracer(adapter=adapter)

        tracer.decorate(_double_x_node)({"x": 1})

        agent_code_events = [e for e in stratix.events if e["event_type"] == "agent.code"]
        assert len(agent_code_events) == 1


# ---------------------------------------------------------------------------
# Pydantic round-trip (typed event)
# ---------------------------------------------------------------------------


class TestPydanticRoundTrip:
    """The vendored AgentCodeEvent model accepts the emitted payload."""

    def test_round_trip_through_pydantic(self) -> None:
        stratix = _RecordingStratix()
        adapter = LayerLensLangGraphAdapter(
            stratix=stratix,
            capture_config=CaptureConfig.full(),
        )
        adapter.connect()
        tracer = NodeTracer(adapter=adapter)

        tracer.decorate(_double_x_node)({"x": 1})

        # Pull the typed event the recording client captured.
        typed = next(
            e for e in stratix.typed_events if getattr(e, "event_type", None) == "agent.code"
        )
        assert isinstance(typed, AgentCodeEvent)
        assert isinstance(typed.code, CodeInfo)
        assert typed.layer == "L2"
        assert typed.code.commit == _RUNTIME_COMMIT_SENTINEL


# ---------------------------------------------------------------------------
# State change still emitted alongside L2
# ---------------------------------------------------------------------------


class TestNoRegressionOnStateChange:
    """L2 emission must not displace the cross-cutting state.change.

    Tested through the lifecycle path because the lifecycle adapter
    routes ``agent.state.change`` through :meth:`emit_dict_event` (the
    dict-emission path). The :class:`NodeTracer` decorator path emits
    state.change via the typed-event path which has a pre-existing
    bug unrelated to this PR (state hashes lack the ``sha256:`` prefix
    required by :class:`AgentStateChangeEvent`'s validator) — that path
    is not exercised here so this PR's L2 wiring is the sole focus.
    """

    def test_state_change_still_fires_when_state_mutated(self) -> None:
        stratix = _RecordingStratix()
        adapter = LayerLensLangGraphAdapter(
            stratix=stratix,
            capture_config=CaptureConfig.full(),
        )
        adapter.connect()
        graph_exec = adapter.on_graph_start(
            graph_id="g1",
            execution_id="g1:1",
            initial_state={"x": 1},
            config=None,
        )
        ctx = adapter.on_node_start(
            graph_exec,
            "doubler",
            state={"x": 1},
            node_callable=_double_x_node,
        )
        # State mutated between start and end — different hash.
        adapter.on_node_end(graph_exec, ctx, state={"x": 2})

        agent_code_events = [e for e in stratix.events if e["event_type"] == "agent.code"]
        state_change_events = [
            e for e in stratix.events if e["event_type"] == "agent.state.change"
        ]
        assert len(agent_code_events) == 1
        assert len(state_change_events) >= 1, (
            "Cross-cutting agent.state.change must still emit on state mutation"
        )

    def test_state_change_not_fired_when_state_unchanged_but_l2_still_emits(
        self,
    ) -> None:
        stratix = _RecordingStratix()
        adapter = LayerLensLangGraphAdapter(
            stratix=stratix,
            capture_config=CaptureConfig.full(),
        )
        adapter.connect()
        graph_exec = adapter.on_graph_start(
            graph_id="g2",
            execution_id="g2:1",
            initial_state={"x": 1},
            config=None,
        )
        ctx = adapter.on_node_start(
            graph_exec,
            "identity",
            state={"x": 1},
            node_callable=_identity_node,
        )
        # State unchanged between start and end — same hash.
        adapter.on_node_end(graph_exec, ctx, state={"x": 1})

        # Look only at events emitted FROM the node lifecycle (not the
        # graph-start environment.config / agent.input which always fire).
        agent_code_events = [e for e in stratix.events if e["event_type"] == "agent.code"]
        # Filter state.change events to those carrying a node_name —
        # that distinguishes node-boundary state changes from
        # graph-boundary state changes emitted by on_graph_end.
        node_state_change_events = [
            e
            for e in stratix.events
            if e["event_type"] == "agent.state.change"
            and e.get("payload", {}).get("node_name") == "identity"
        ]
        # L2 emission is unconditional per spec; node-boundary
        # state.change is gated on actual mutation.
        assert len(agent_code_events) == 1
        assert node_state_change_events == []


# ---------------------------------------------------------------------------
# Adapter wiring sanity (defensive)
# ---------------------------------------------------------------------------


def test_adapter_starts_healthy() -> None:
    """Sanity guard: tests above assume HEALTHY adapter status."""
    stratix = _RecordingStratix()
    adapter = LayerLensLangGraphAdapter(
        stratix=stratix,
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()
    assert adapter.status == AdapterStatus.HEALTHY
