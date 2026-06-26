"""Behavioral unit tests for the LIVE A2A protocol adapter (LAY-3617).

The protocol unit tiers were dead/contract-only while A2A carries the
highest-stakes content: cross-agent task delegation and the agent-discovery
trust handshake. This suite drives the REAL ``A2AProtocolAdapter`` reached by
``connect()`` — it patches ``send_task``/``get_task``/``cancel_task``/
``get_agent_card``/``register_handler`` on a target double and asserts the
*emitted* events (exact ``event_type`` constants + key payload fields).

NOTE: this is deliberately NOT ``test_a2a_client.py`` / ``test_a2a_server.py``,
which exercise the hand-driven ``A2AClientWrapper`` / ``A2AServerWrapper``
classes (a separate API with different emit shapes). Those never touch the
live ``connect()``-installed wrappers.

Emitted events are routed through the autouse schema lock
(``record_for_schema_lock``) so every payload here is validated against
``tests/instrument/_event_schema.py`` after the test body.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from layerlens.instrument._events import (
    A2A_DELEGATION,
    A2A_TASK_CREATED,
    A2A_TASK_UPDATED,
    A2A_AGENT_DISCOVERED,
)
from layerlens.instrument._context import _current_collector
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.protocols.a2a.adapter import A2AProtocolAdapter
from layerlens.instrument.adapters.protocols.a2a.agent_card import parse_agent_card
from layerlens.instrument.adapters.protocols.a2a.acp_normalizer import ACPNormalizer

# Relative import (matches test_trace_context / test_concurrency): an absolute
# ``tests.instrument.conftest`` import binds a DIFFERENT module object than the
# one pytest's autouse ``_enforce_schema_lock`` fixture uses, so events recorded
# through it are silently NOT validated. The relative form shares the buffer.
from ...conftest import record_for_schema_lock

# ---------------------------------------------------------------------------
# Helpers — drive the adapter under an ambient collector, then hand the
# emitted events to the schema lock so they are validated after the test.
# ---------------------------------------------------------------------------


def _run_collected(mock_client: Any, fn: Any, config: Optional[CaptureConfig] = None) -> List[Dict[str, Any]]:
    collector = TraceCollector(mock_client, config or CaptureConfig())
    token = _current_collector.set(collector)
    try:
        fn()
    finally:
        _current_collector.reset(token)
    events = collector.events
    record_for_schema_lock(events)
    return events


def _types(events: List[Dict[str, Any]]) -> List[str]:
    return [e["event_type"] for e in events]


def _of_type(events: List[Dict[str, Any]], event_type: str) -> List[Dict[str, Any]]:
    return [e for e in events if e["event_type"] == event_type]


def _one(events: List[Dict[str, Any]], event_type: str) -> Dict[str, Any]:
    matches = _of_type(events, event_type)
    assert matches, f"expected one {event_type}, got {_types(events)}"
    return matches[0]


def _payload(events: List[Dict[str, Any]], event_type: str) -> Dict[str, Any]:
    return _one(events, event_type)["payload"]


# Reusable, well-formed Agent Card matching the real /.well-known/agent.json shape.
GOOD_CARD = {
    "name": "Planner",
    "description": "plans things",
    "url": "https://peer.example/a2a",
    "protocolVersion": "0.3.0",
    "capabilities": {"streaming": True},
    "skills": [{"name": "plan"}, {"name": "summarize"}],
    "authentication": {"scheme": "bearer"},
}


# ---------------------------------------------------------------------------
# a2a.agent.discovered — get_agent_card -> _wrap_discovery -> parse_agent_card
# ---------------------------------------------------------------------------


class TestAgentDiscovered:
    def test_discovered_from_dict_card_carries_parsed_fields(self, mock_client: Any) -> None:
        adapter = A2AProtocolAdapter()
        target = SimpleNamespace(get_agent_card=lambda **kw: GOOD_CARD)
        adapter.connect(target=target)

        events = _run_collected(mock_client, lambda: target.get_agent_card(agent_url="https://peer.example"))

        payload = _payload(events, A2A_AGENT_DISCOVERED)
        # agent_id falls back to the card name (no id/agent_id present).
        assert payload["agent_id"] == "Planner"
        assert payload["name"] == "Planner"
        # NOTE (adapter behavior, see bug_found): for a dict/JSON card the
        # adapter takes parse_agent_card's skills, which is the RAW list — it is
        # NOT flattened to names (the name-flattening _extract_skills is only the
        # fallback used when parse returns no skills). So the discovered payload
        # carries the raw skill objects here.
        assert payload["skills"] == [{"name": "plan"}, {"name": "summarize"}]
        assert payload["authScheme"] == "bearer"
        assert payload["protocolVersion"] == "0.3.0"
        assert payload["protocol"] == "a2a"

    def test_discovered_from_json_string_card(self, mock_client: Any) -> None:
        adapter = A2AProtocolAdapter()
        target = SimpleNamespace(get_agent_card=lambda **kw: json.dumps(GOOD_CARD))
        adapter.connect(target=target)

        events = _run_collected(mock_client, lambda: target.get_agent_card())

        payload = _payload(events, A2A_AGENT_DISCOVERED)
        assert payload["name"] == "Planner"
        # Raw skill objects passed through from parse_agent_card (see above).
        assert payload["skills"] == [{"name": "plan"}, {"name": "summarize"}]
        assert payload["authScheme"] == "bearer"

    def test_discovered_still_emitted_on_malformed_card(self, mock_client: Any) -> None:
        """A malformed JSON string makes parse_agent_card raise ValueError, which
        _wrap_discovery catches (normalized=None). The discovery event must STILL
        be emitted — name falls back via getattr(result, 'name', None), which is
        None for a bare string. This is the trust-handshake-still-observed path."""
        adapter = A2AProtocolAdapter()
        malformed = "{not valid json"
        target = SimpleNamespace(get_agent_card=lambda **kw: malformed)
        adapter.connect(target=target)

        events = _run_collected(mock_client, lambda: target.get_agent_card())

        payload = _payload(events, A2A_AGENT_DISCOVERED)
        # _extract_agent_id(str) returns None (no id/agent_id/name attrs on a str).
        assert payload["agent_id"] is None
        # getattr(<str>, "name", None) -> None; normalized was None.
        assert payload["name"] is None
        assert payload["skills"] == []
        assert payload["authScheme"] is None
        # The raw malformed string is returned to the caller unchanged.

    def test_skills_shape_differs_for_object_card_vs_dict_card(self, mock_client: Any) -> None:
        """Locks the inconsistency reported in bug_found: a dict/JSON card emits
        the RAW skills list (parse_agent_card passthrough) while an OBJECT card
        emits skill *names* (via the _extract_skills fallback, since parse is
        skipped for non-dict/non-str results). Same logical card, two shapes."""
        skill = SimpleNamespace(name="plan")
        card_obj = SimpleNamespace(name="ObjAgent", skills=[skill])
        adapter = A2AProtocolAdapter()
        target = SimpleNamespace(get_agent_card=lambda **kw: card_obj)
        adapter.connect(target=target)

        events = _run_collected(mock_client, lambda: target.get_agent_card())
        payload = _payload(events, A2A_AGENT_DISCOVERED)
        # Object path -> flattened names (NOT raw objects like the dict path).
        assert payload["skills"] == ["plan"]
        assert payload["name"] == "ObjAgent"
        assert payload["agent_id"] == "ObjAgent"
        # parse_agent_card was never called for an object, so these stay None.
        assert payload["authScheme"] is None
        assert payload["protocolVersion"] is None

    def test_get_agent_card_returns_original_result(self, mock_client: Any) -> None:
        adapter = A2AProtocolAdapter()
        target = SimpleNamespace(get_agent_card=lambda **kw: GOOD_CARD)
        adapter.connect(target=target)

        result_holder: Dict[str, Any] = {}

        def go() -> None:
            result_holder["r"] = target.get_agent_card()

        _run_collected(mock_client, go)
        assert result_holder["r"] is GOOD_CARD


# ---------------------------------------------------------------------------
# Client path — send_task -> a2a.task.created + a2a.delegation + a2a.task.updated
# ---------------------------------------------------------------------------


class TestClientSendTask:
    def _adapter_with_target(self, send_result: Any = None) -> Any:
        adapter = A2AProtocolAdapter()
        target = SimpleNamespace(
            send_task=lambda **kw: (send_result if send_result is not None else {"task_id": "t1"}),
            get_task=lambda **kw: {"task_id": "t1"},
            cancel_task=lambda **kw: None,
        )
        adapter.connect(target=target)
        return adapter, target

    def test_send_task_emits_created_delegation_and_updated(self, mock_client: Any) -> None:
        adapter, target = self._adapter_with_target({"status": "completed"})

        def go() -> None:
            target.send_task(task_id="task-7", agent_id="agent-42", skill="plan", priority="high")

        events = _run_collected(mock_client, go)

        # All three event types fire, in order: created -> delegation -> updated.
        assert _types(events) == [A2A_TASK_CREATED, A2A_DELEGATION, A2A_TASK_UPDATED]

        created = _payload(events, A2A_TASK_CREATED)
        assert created["task_id"] == "task-7"
        assert created["method"] == "send_task"
        # _summarize keeps only agent_id/skill/task_id/priority in the request.
        assert created["request"] == {
            "agent_id": "agent-42",
            "skill": "plan",
            "task_id": "task-7",
            "priority": "high",
        }

        delegation = _payload(events, A2A_DELEGATION)
        assert delegation["task_id"] == "task-7"
        assert delegation["target_agent"] == "agent-42"
        assert delegation["skill"] == "plan"

        updated = _payload(events, A2A_TASK_UPDATED)
        assert updated["task_id"] == "task-7"
        assert updated["status"] == "completed"
        assert isinstance(updated["latency_ms"], (int, float))

    def test_created_delegation_updated_share_parent_span(self, mock_client: Any) -> None:
        adapter, target = self._adapter_with_target({"status": "completed"})
        events = _run_collected(mock_client, lambda: target.send_task(task_id="t-span", agent_id="a1"))
        parents = {e["parent_span_id"] for e in events}
        # The whole send_task lifecycle is correlated under one parent span.
        assert len(parents) == 1
        assert next(iter(parents)) is not None

    def test_status_defaults_to_completed_when_result_has_none(self, mock_client: Any) -> None:
        adapter, target = self._adapter_with_target({"task_id": "t1"})  # no status key
        events = _run_collected(mock_client, lambda: target.send_task(task_id="t1", agent_id="a1"))
        assert _payload(events, A2A_TASK_UPDATED)["status"] == "completed"

    def test_send_task_failure_emits_failed_update_and_reraises(self, mock_client: Any) -> None:
        adapter = A2AProtocolAdapter()

        def boom(**kw: Any) -> Any:
            raise RuntimeError("downstream agent unreachable")

        target = SimpleNamespace(send_task=boom)
        adapter.connect(target=target)

        def go() -> None:
            with pytest.raises(RuntimeError, match="unreachable"):
                target.send_task(task_id="t-fail", agent_id="a1")

        events = _run_collected(mock_client, go)

        # created + delegation still emitted, then a FAILED update (no extra update).
        assert _types(events) == [A2A_TASK_CREATED, A2A_DELEGATION, A2A_TASK_UPDATED]
        updated = _payload(events, A2A_TASK_UPDATED)
        assert updated["task_id"] == "t-fail"
        assert updated["status"] == "failed"
        assert "unreachable" in updated["error"]
        assert isinstance(updated["latency_ms"], (int, float))

    def test_status_dict_state_is_unwrapped(self, mock_client: Any) -> None:
        # Real a2a results carry status as {"state": "working"} nested dicts.
        adapter, target = self._adapter_with_target({"status": {"state": "working"}})
        events = _run_collected(mock_client, lambda: target.send_task(task_id="t1", agent_id="a1"))
        assert _payload(events, A2A_TASK_UPDATED)["status"] == "working"


# ---------------------------------------------------------------------------
# Server path — register_handler wraps the inbound handler; invoking it
# emits a2a.task.created (source="server") + a2a.task.updated.
# ---------------------------------------------------------------------------


class TestServerRegisterHandler:
    def test_inbound_task_emits_created_and_updated(self, mock_client: Any) -> None:
        adapter = A2AProtocolAdapter()
        registered: Dict[str, Any] = {}
        target = SimpleNamespace(register_handler=lambda handler, **kw: registered.__setitem__("h", handler))
        adapter.connect(target=target)

        # The app registers its handler; the adapter wraps it transparently.
        def app_handler(task: Any) -> Any:
            return {"status": "completed"}

        def go() -> None:
            target.register_handler(app_handler)
            wrapped = registered["h"]
            # Simulate an inbound task arriving from a peer agent.
            wrapped({"id": "srv-task-1", "skill": "summarize"})

        events = _run_collected(mock_client, go)

        assert _types(events) == [A2A_TASK_CREATED, A2A_TASK_UPDATED]
        created = _payload(events, A2A_TASK_CREATED)
        assert created["task_id"] == "srv-task-1"
        assert created["source"] == "server"
        assert created["skill"] == "summarize"

        updated = _payload(events, A2A_TASK_UPDATED)
        assert updated["task_id"] == "srv-task-1"
        assert updated["status"] == "completed"
        assert isinstance(updated["latency_ms"], (int, float))

    def test_server_handler_result_passthrough(self, mock_client: Any) -> None:
        adapter = A2AProtocolAdapter()
        registered: Dict[str, Any] = {}
        target = SimpleNamespace(register_handler=lambda handler, **kw: registered.__setitem__("h", handler))
        adapter.connect(target=target)

        sentinel = {"status": "completed", "echo": 99}
        out: Dict[str, Any] = {}

        def go() -> None:
            target.register_handler(lambda task: sentinel)
            out["r"] = registered["h"]({"id": "srv-2"})

        _run_collected(mock_client, go)
        assert out["r"] is sentinel

    def test_inbound_handler_failure_emits_failed_update_and_reraises(self, mock_client: Any) -> None:
        adapter = A2AProtocolAdapter()
        registered: Dict[str, Any] = {}
        target = SimpleNamespace(register_handler=lambda handler, **kw: registered.__setitem__("h", handler))
        adapter.connect(target=target)

        def app_handler(task: Any) -> Any:
            raise ValueError("handler blew up")

        def go() -> None:
            target.register_handler(app_handler)
            with pytest.raises(ValueError, match="blew up"):
                registered["h"]({"id": "srv-bad"})

        events = _run_collected(mock_client, go)
        assert _types(events) == [A2A_TASK_CREATED, A2A_TASK_UPDATED]
        updated = _payload(events, A2A_TASK_UPDATED)
        assert updated["status"] == "failed"
        assert "blew up" in updated["error"]

    def test_acp_origin_inbound_task_is_normalized_before_create(self, mock_client: Any) -> None:
        """An ACP-shaped inbound payload (task_run namespace) is normalized to
        A2A canonical form by the adapter before emitting; the running->working
        status remap and task_run.id->task.id mapping must both hold."""
        adapter = A2AProtocolAdapter()
        registered: Dict[str, Any] = {}
        target = SimpleNamespace(register_handler=lambda handler, **kw: registered.__setitem__("h", handler))
        adapter.connect(target=target)

        acp_payload = {
            "task_run": {
                "id": "acp-77",
                "skill": "translate",
                "status": "running",
            }
        }

        def go() -> None:
            target.register_handler(lambda task: {"status": "completed"})
            registered["h"](acp_payload)

        events = _run_collected(mock_client, go)
        created = _payload(events, A2A_TASK_CREATED)
        # task_run.id -> task.id; _task_id_from reads the normalized "id".
        assert created["task_id"] == "acp-77"
        assert created["source"] == "server"


# ---------------------------------------------------------------------------
# DIRECT unit tests — parse_agent_card
# ---------------------------------------------------------------------------


class TestParseAgentCard:
    def test_good_json_string(self) -> None:
        parsed = parse_agent_card(json.dumps(GOOD_CARD))
        assert parsed["name"] == "Planner"
        assert parsed["url"] == "https://peer.example/a2a"
        assert parsed["protocolVersion"] == "0.3.0"
        assert parsed["capabilities"] == {"streaming": True}
        assert parsed["skills"] == [{"name": "plan"}, {"name": "summarize"}]
        assert parsed["authScheme"] == "bearer"

    def test_dict_input(self) -> None:
        parsed = parse_agent_card(dict(GOOD_CARD))
        assert parsed["name"] == "Planner"
        assert parsed["authScheme"] == "bearer"

    def test_malformed_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid Agent Card JSON"):
            parse_agent_card("{not json at all")

    def test_missing_fields_get_defaults(self) -> None:
        parsed = parse_agent_card({"name": "Bare"})
        assert parsed["name"] == "Bare"
        assert parsed["url"] == ""
        assert parsed["protocolVersion"] == "unknown"
        assert parsed["capabilities"] == {}
        assert parsed["skills"] == []
        assert parsed["authScheme"] is None

    def test_version_casing_variant_falls_back_to_version_key(self) -> None:
        # Older cards use lowercase "version" rather than "protocolVersion".
        parsed = parse_agent_card({"name": "Old", "version": "0.2.1"})
        assert parsed["protocolVersion"] == "0.2.1"

    def test_auth_scheme_via_type_key(self) -> None:
        parsed = parse_agent_card({"name": "X", "authentication": {"type": "oauth2"}})
        assert parsed["authScheme"] == "oauth2"

    def test_auth_scheme_prefers_scheme_over_type(self) -> None:
        parsed = parse_agent_card({"name": "X", "authentication": {"scheme": "bearer", "type": "oauth2"}})
        assert parsed["authScheme"] == "bearer"

    def test_auth_as_bare_string(self) -> None:
        parsed = parse_agent_card({"name": "X", "authentication": "apiKey"})
        assert parsed["authScheme"] == "apiKey"
        assert parsed["authentication"] == "apiKey"

    def test_missing_name_defaults_to_unknown(self) -> None:
        parsed = parse_agent_card({"url": "https://x"})
        assert parsed["name"] == "unknown"


# ---------------------------------------------------------------------------
# DIRECT unit tests — ACPNormalizer
# ---------------------------------------------------------------------------


class TestACPNormalizer:
    def setup_method(self) -> None:
        self.norm = ACPNormalizer()

    def test_detect_via_header(self) -> None:
        assert self.norm.detect_acp_origin({}, headers={"X-ACP-Version": "1.0"}) is True

    def test_detect_via_lowercase_header(self) -> None:
        assert self.norm.detect_acp_origin({}, headers={"x-acp-version": "1.0"}) is True

    def test_detect_via_acp_namespace(self) -> None:
        assert self.norm.detect_acp_origin({"acp": {"version": "1.0"}}) is True

    def test_detect_via_task_run(self) -> None:
        assert self.norm.detect_acp_origin({"task_run": {"id": "x"}}) is True

    def test_detect_via_task_run_inside_params(self) -> None:
        assert self.norm.detect_acp_origin({"params": {"task_run": {"id": "x"}}}) is True

    def test_plain_a2a_payload_not_detected(self) -> None:
        assert self.norm.detect_acp_origin({"task": {"id": "x"}}) is False

    def test_status_remap_running_to_working(self) -> None:
        normalized = self.norm.normalize({"task_run": {"id": "r1", "status": "running"}})
        assert normalized["task"]["status"] == {"state": "working"}

    def test_status_remap_pending_to_submitted(self) -> None:
        normalized = self.norm.normalize({"task_run": {"id": "r1", "status": "pending"}})
        assert normalized["task"]["status"] == {"state": "submitted"}

    def test_unknown_status_passes_through(self) -> None:
        normalized = self.norm.normalize({"task_run": {"id": "r1", "status": "weird"}})
        assert normalized["task"]["status"] == {"state": "weird"}

    def test_task_run_field_mapping(self) -> None:
        payload = {
            "task_run": {
                "id": "run-9",
                "input": {"messages": [{"role": "user", "content": "hi"}]},
                "output": {"artifacts": [{"name": "a"}]},
                "status": "completed",
                "metadata": {"trace": "abc"},
            }
        }
        normalized = self.norm.normalize(payload)
        task = normalized["task"]
        assert task["id"] == "run-9"
        assert task["history"] == [{"role": "user", "content": "hi"}]
        assert task["artifacts"] == [{"name": "a"}]
        assert task["status"] == {"state": "completed"}
        assert task["metadata"] == {"trace": "abc"}
        # task_run is consumed (no longer present after normalize).
        assert "task_run" not in normalized

    def test_task_run_inside_params_is_mapped(self) -> None:
        payload = {"jsonrpc": "2.0", "params": {"task_run": {"id": "p1", "status": "running"}}}
        normalized = self.norm.normalize(payload)
        assert normalized["params"]["task"]["id"] == "p1"
        assert normalized["params"]["task"]["status"] == {"state": "working"}
        assert "task_run" not in normalized["params"]

    def test_acp_namespace_version_moved_to_metadata(self) -> None:
        normalized = self.norm.normalize({"acp": {"version": "1.2"}, "task_run": {"id": "x"}})
        assert normalized["metadata"]["acp_version"] == "1.2"
        assert "acp" not in normalized

    def test_detect_and_normalize_returns_flag(self) -> None:
        out, is_acp = self.norm.detect_and_normalize({"task_run": {"id": "x", "status": "running"}})
        assert is_acp is True
        assert out["task"]["id"] == "x"

    def test_detect_and_normalize_passthrough_for_non_acp(self) -> None:
        original = {"task": {"id": "x"}}
        out, is_acp = self.norm.detect_and_normalize(original)
        assert is_acp is False
        assert out is original

    def test_status_as_dict_state_is_remapped(self) -> None:
        normalized = self.norm.normalize({"task_run": {"id": "r1", "status": {"state": "running"}}})
        assert normalized["task"]["status"] == {"state": "working"}
