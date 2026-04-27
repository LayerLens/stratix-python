"""Tests for LangChain LCEL (LangChain Expression Language) tracing.

Covers spec 04b §1 weakness #4 and §4 (LCEL Support):

* Detection of all five LCEL primitives from the ``name`` kwarg
* Tag parsing for composition position (``seq:step:N``, ``map:key:K``,
  ``branch:N``, ``condition:N``)
* The :class:`LCELRunnableTracker` state machine: begin/end, hierarchy,
  root detection, depth, status transitions, payload generation
* Synthetic ``chain.composition`` event emission via ``agent.code``
* Per-runnable ``agent.input`` / ``agent.output`` / ``agent.code`` events
* Nested composition correctness (sequence-in-parallel-in-sequence)
* Error path handling
* No regression in LangGraph node attribution

These tests run without invoking the upstream LangChain runtime —
callbacks are driven directly with the same kwargs LangChain would
pass. This keeps the test suite hermetic and fast (~50ms total).
"""

from __future__ import annotations

import uuid
from typing import Any

from layerlens.instrument.adapters._base.capture import CaptureConfig
from layerlens.instrument.adapters.frameworks.langchain import (
    LCELNode,
    RunnableKind,
    CompositionPosition,
    LCELRunnableTracker,
    LayerLensCallbackHandler,
    detect_runnable_kind,
    parse_composition_tag,
    parse_parallel_branches,
)


class _RecordingStratix:
    """Test double that records every emitted event."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event_type: Any, payload: Any = None) -> None:
        if isinstance(event_type, str):
            self.events.append({"type": event_type, "payload": payload})


def _new_id() -> uuid.UUID:
    return uuid.uuid4()


def _full_capture() -> CaptureConfig:
    return CaptureConfig.full()


# ---------------------------------------------------------------------------
# detect_runnable_kind
# ---------------------------------------------------------------------------


class TestDetectRunnableKind:
    def test_sequence(self) -> None:
        assert detect_runnable_kind("RunnableSequence") == RunnableKind.SEQUENCE

    def test_parallel_with_keys(self) -> None:
        assert detect_runnable_kind("RunnableParallel<a,b>") == RunnableKind.PARALLEL

    def test_parallel_no_keys(self) -> None:
        assert detect_runnable_kind("RunnableParallel") == RunnableKind.PARALLEL

    def test_lambda(self) -> None:
        assert detect_runnable_kind("RunnableLambda") == RunnableKind.LAMBDA

    def test_passthrough(self) -> None:
        assert detect_runnable_kind("RunnablePassthrough") == RunnableKind.PASSTHROUGH

    def test_branch(self) -> None:
        assert detect_runnable_kind("RunnableBranch") == RunnableKind.BRANCH

    def test_other_classifies_non_lcel(self) -> None:
        assert detect_runnable_kind("ChatOpenAI") == RunnableKind.OTHER
        assert detect_runnable_kind("StrOutputParser") == RunnableKind.OTHER
        # User-supplied lambda __name__ is NOT classified as LAMBDA from
        # the name alone — the parent's ``map:key:`` / ``seq:step:`` tag
        # is what tells us it's an LCEL child. Detection alone returns
        # OTHER for arbitrary callable names; classification as a lambda
        # happens only via the ``RunnableLambda`` literal name string.
        assert detect_runnable_kind("my_func") == RunnableKind.OTHER

    def test_empty_or_none(self) -> None:
        assert detect_runnable_kind(None) == RunnableKind.OTHER
        assert detect_runnable_kind("") == RunnableKind.OTHER


# ---------------------------------------------------------------------------
# parse_composition_tag
# ---------------------------------------------------------------------------


class TestParseCompositionTag:
    def test_sequence_step_tag(self) -> None:
        pos = parse_composition_tag(["seq:step:2"])
        assert pos == CompositionPosition(parent_kind=RunnableKind.SEQUENCE, label="2", role="step")

    def test_parallel_key_tag(self) -> None:
        pos = parse_composition_tag(["map:key:context"])
        assert pos is not None
        assert pos.parent_kind == RunnableKind.PARALLEL
        assert pos.label == "context"
        assert pos.role == "key"

    def test_branch_body_tag(self) -> None:
        pos = parse_composition_tag(["branch:1"])
        assert pos is not None
        assert pos.parent_kind == RunnableKind.BRANCH
        assert pos.role == "body"

    def test_branch_condition_tag(self) -> None:
        pos = parse_composition_tag(["condition:0"])
        assert pos is not None
        assert pos.parent_kind == RunnableKind.BRANCH
        assert pos.role == "condition"

    def test_unrelated_tags_ignored(self) -> None:
        assert parse_composition_tag(["user:tag", "another"]) is None

    def test_first_known_tag_wins(self) -> None:
        # Mix of unrelated + known: the known tag is decoded.
        pos = parse_composition_tag(["custom-tag", "seq:step:5"])
        assert pos is not None
        assert pos.role == "step"
        assert pos.label == "5"

    def test_empty_and_none(self) -> None:
        assert parse_composition_tag(None) is None
        assert parse_composition_tag([]) is None


# ---------------------------------------------------------------------------
# parse_parallel_branches
# ---------------------------------------------------------------------------


class TestParseParallelBranches:
    def test_two_keys(self) -> None:
        assert parse_parallel_branches("RunnableParallel<context,question>") == [
            "context",
            "question",
        ]

    def test_three_keys_with_whitespace(self) -> None:
        assert parse_parallel_branches("RunnableParallel<a, b , c>") == ["a", "b", "c"]

    def test_no_brackets_returns_empty(self) -> None:
        assert parse_parallel_branches("RunnableParallel") == []

    def test_empty_brackets_returns_empty(self) -> None:
        assert parse_parallel_branches("RunnableParallel<>") == []


# ---------------------------------------------------------------------------
# LCELRunnableTracker
# ---------------------------------------------------------------------------


class TestLCELRunnableTracker:
    def test_begin_root_marks_root(self) -> None:
        t = LCELRunnableTracker()
        node = t.begin(run_id="r1", parent_run_id=None, name="RunnableSequence", tags=None)
        assert isinstance(node, LCELNode)
        assert node.kind == RunnableKind.SEQUENCE
        assert node.depth == 0
        assert t.is_root("r1")

    def test_child_under_root_records_parent_and_depth(self) -> None:
        t = LCELRunnableTracker()
        t.begin(run_id="r1", parent_run_id=None, name="RunnableSequence", tags=None)
        child = t.begin(
            run_id="r2",
            parent_run_id="r1",
            name="RunnableLambda",
            tags=["seq:step:1"],
        )
        assert child.depth == 1
        assert child.parent_run_id == "r1"
        assert not t.is_root("r2")
        # Parent should record the child.
        parent = t.get_node("r1")
        assert parent is not None
        assert "r2" in parent.child_run_ids

    def test_end_marks_completion(self) -> None:
        t = LCELRunnableTracker()
        t.begin(run_id="r1", parent_run_id=None, name="RunnableSequence", tags=None)
        node = t.end("r1")
        assert node is not None
        assert node.status == "ok"
        assert node.end_time_ns is not None
        assert node.end_time_ns >= node.start_time_ns

    def test_end_with_error_marks_error(self) -> None:
        t = LCELRunnableTracker()
        t.begin(run_id="r1", parent_run_id=None, name="RunnableSequence", tags=None)
        node = t.end("r1", error="boom")
        assert node is not None
        assert node.status == "error"
        assert node.error == "boom"

    def test_composition_payload_for_root(self) -> None:
        t = LCELRunnableTracker()
        t.begin(run_id="r1", parent_run_id=None, name="RunnableSequence", tags=None)
        t.begin(run_id="r2", parent_run_id="r1", name="RunnableLambda", tags=["seq:step:1"])
        t.begin(run_id="r3", parent_run_id="r1", name="RunnableLambda", tags=["seq:step:2"])
        t.end("r2")
        t.end("r3")
        t.end("r1")

        payload = t.composition_payload("r1")
        assert payload is not None
        assert payload["root_run_id"] == "r1"
        assert payload["root_kind"] == "sequence"
        assert payload["node_count"] == 3
        assert payload["max_depth"] == 1
        assert payload["kind_counts"] == {"sequence": 1, "lambda": 2}
        assert payload["status"] == "ok"

    def test_composition_payload_none_for_non_root(self) -> None:
        t = LCELRunnableTracker()
        t.begin(run_id="r1", parent_run_id=None, name="RunnableSequence", tags=None)
        t.begin(run_id="r2", parent_run_id="r1", name="RunnableLambda", tags=["seq:step:1"])
        # r2 is not a root.
        assert t.composition_payload("r2") is None

    def test_reset_clears_state(self) -> None:
        t = LCELRunnableTracker()
        t.begin(run_id="r1", parent_run_id=None, name="RunnableSequence", tags=None)
        t.reset()
        assert t.get_node("r1") is None
        assert not t.is_root("r1")

    def test_input_payload_includes_composition(self) -> None:
        t = LCELRunnableTracker()
        node = t.begin(
            run_id="r1",
            parent_run_id="parent",
            name="RunnableLambda",
            tags=["seq:step:3"],
        )
        payload = t.runnable_input_payload(node, {"x": 1})
        assert payload["runnable"]["kind"] == "lambda"
        assert payload["runnable"]["depth"] == 0  # parent isn't tracked, so root
        assert payload["runnable"]["position"]["role"] == "step"
        assert payload["runnable"]["position"]["label"] == "3"
        assert "fingerprint" in payload["runnable"]
        assert payload["input"] == {"x": 1}

    def test_parallel_branches_in_input_payload(self) -> None:
        t = LCELRunnableTracker()
        node = t.begin(
            run_id="r1",
            parent_run_id=None,
            name="RunnableParallel<context,question>",
            tags=None,
        )
        payload = t.runnable_input_payload(node, "input")
        assert payload["runnable"]["parallel_branches"] == ["context", "question"]

    def test_code_payload_marks_passthrough(self) -> None:
        t = LCELRunnableTracker()
        node = t.begin(run_id="r1", parent_run_id=None, name="RunnablePassthrough", tags=None)
        t.end("r1")
        payload = t.runnable_code_payload(node)
        assert payload["passthrough"] is True
        assert payload["kind"] == "passthrough"

    def test_consume_root_completion_only_after_end(self) -> None:
        t = LCELRunnableTracker()
        t.begin(run_id="r1", parent_run_id=None, name="RunnableSequence", tags=None)
        # Before end:
        assert t.consume_root_completion("r1") is None
        t.end("r1")
        # After end:
        assert t.consume_root_completion("r1") is not None
        # Idempotent (still returns the node since data persists until reset).
        assert t.consume_root_completion("r1") is not None


# ---------------------------------------------------------------------------
# Callback integration — drive the handler with synthetic chain events
# ---------------------------------------------------------------------------


class TestCallbackIntegration:
    def test_sequence_emits_input_output_and_composition(self) -> None:
        stratix = _RecordingStratix()
        handler = LayerLensCallbackHandler(stratix=stratix, capture_config=_full_capture())
        handler.connect()

        root_id = _new_id()
        step1_id = _new_id()
        step2_id = _new_id()

        handler.on_chain_start({}, {"q": "hi"}, run_id=root_id, name="RunnableSequence")
        handler.on_chain_start(
            {},
            {"q": "hi"},
            run_id=step1_id,
            parent_run_id=root_id,
            name="RunnableLambda",
            tags=["seq:step:1"],
        )
        handler.on_chain_end({"out": 1}, run_id=step1_id, parent_run_id=root_id)
        handler.on_chain_start(
            {},
            {"out": 1},
            run_id=step2_id,
            parent_run_id=root_id,
            name="RunnableLambda",
            tags=["seq:step:2"],
        )
        handler.on_chain_end({"out": 2}, run_id=step2_id, parent_run_id=root_id)
        handler.on_chain_end({"out": 2}, run_id=root_id)

        types = [e["type"] for e in stratix.events]
        # Three agent.input (root + two children) — three agent.output.
        assert types.count("agent.input") == 3
        assert types.count("agent.output") == 3
        # Three agent.code (per runnable) plus one composition snapshot.
        assert types.count("agent.code") == 4
        composition = next(
            e for e in stratix.events if e["type"] == "agent.code" and e["payload"].get("kind") == "chain.composition"
        )
        comp = composition["payload"]["composition"]
        assert comp["root_kind"] == "sequence"
        assert comp["node_count"] == 3

    def test_parallel_records_branches(self) -> None:
        stratix = _RecordingStratix()
        handler = LayerLensCallbackHandler(stratix=stratix, capture_config=_full_capture())
        handler.connect()

        root_id = _new_id()
        a_id = _new_id()
        b_id = _new_id()

        handler.on_chain_start(
            {},
            {"x": 1},
            run_id=root_id,
            name="RunnableParallel<context,question>",
        )
        handler.on_chain_start(
            {},
            {"x": 1},
            run_id=a_id,
            parent_run_id=root_id,
            name="RunnableLambda",
            tags=["map:key:context"],
        )
        handler.on_chain_end({"context": "C"}, run_id=a_id, parent_run_id=root_id)
        handler.on_chain_start(
            {},
            {"x": 1},
            run_id=b_id,
            parent_run_id=root_id,
            name="RunnableLambda",
            tags=["map:key:question"],
        )
        handler.on_chain_end({"question": "Q"}, run_id=b_id, parent_run_id=root_id)
        handler.on_chain_end({"context": "C", "question": "Q"}, run_id=root_id)

        # Find the root agent.input event and verify it carries the branch keys.
        root_inputs = [
            e
            for e in stratix.events
            if e["type"] == "agent.input"
            and e["payload"].get("runnable", {}).get("kind") == "parallel"
        ]
        assert len(root_inputs) == 1
        assert root_inputs[0]["payload"]["runnable"]["parallel_branches"] == ["context", "question"]

        # Composition event captures both children.
        composition = next(
            e for e in stratix.events if e["type"] == "agent.code" and e["payload"].get("kind") == "chain.composition"
        )
        comp = composition["payload"]["composition"]
        assert comp["kind_counts"]["lambda"] == 2

    def test_lambda_fingerprint_present(self) -> None:
        stratix = _RecordingStratix()
        handler = LayerLensCallbackHandler(stratix=stratix, capture_config=_full_capture())
        handler.connect()

        root_id = _new_id()
        handler.on_chain_start({}, "in", run_id=root_id, name="RunnableLambda")
        handler.on_chain_end("out", run_id=root_id)

        agent_inputs = [e for e in stratix.events if e["type"] == "agent.input"]
        assert any("fingerprint" in e["payload"]["runnable"] for e in agent_inputs)
        agent_codes = [
            e
            for e in stratix.events
            if e["type"] == "agent.code" and e["payload"].get("kind") != "chain.composition"
        ]
        assert any("fingerprint" in e["payload"] for e in agent_codes)

    def test_passthrough_marked_in_code_event(self) -> None:
        stratix = _RecordingStratix()
        handler = LayerLensCallbackHandler(stratix=stratix, capture_config=_full_capture())
        handler.connect()

        root_id = _new_id()
        handler.on_chain_start({}, {"k": "v"}, run_id=root_id, name="RunnablePassthrough")
        handler.on_chain_end({"k": "v"}, run_id=root_id)

        passthrough_codes = [
            e
            for e in stratix.events
            if e["type"] == "agent.code" and e["payload"].get("passthrough")
        ]
        assert len(passthrough_codes) == 1

    def test_branch_records_condition_and_body(self) -> None:
        stratix = _RecordingStratix()
        handler = LayerLensCallbackHandler(stratix=stratix, capture_config=_full_capture())
        handler.connect()

        root_id = _new_id()
        cond_id = _new_id()
        body_id = _new_id()

        handler.on_chain_start({}, 10, run_id=root_id, name="RunnableBranch")
        handler.on_chain_start(
            {},
            10,
            run_id=cond_id,
            parent_run_id=root_id,
            name="RunnableLambda",
            tags=["condition:0"],
        )
        handler.on_chain_end(True, run_id=cond_id, parent_run_id=root_id)
        handler.on_chain_start(
            {},
            10,
            run_id=body_id,
            parent_run_id=root_id,
            name="RunnableLambda",
            tags=["branch:0"],
        )
        handler.on_chain_end(20, run_id=body_id, parent_run_id=root_id)
        handler.on_chain_end(20, run_id=root_id)

        positions = [
            e["payload"]["runnable"]["position"]
            for e in stratix.events
            if e["type"] == "agent.input"
            and e["payload"].get("runnable", {}).get("position") is not None
        ]
        roles = {p["role"] for p in positions}
        assert roles == {"condition", "body"}

    def test_nested_composition_preserves_hierarchy(self) -> None:
        """A sequence containing a parallel containing a sequence — the
        composition payload's max_depth must reflect the full nesting.
        """
        stratix = _RecordingStratix()
        handler = LayerLensCallbackHandler(stratix=stratix, capture_config=_full_capture())
        handler.connect()

        ids = {key: _new_id() for key in ("root", "par", "a", "inner_seq", "inner_a", "inner_b", "b")}

        # outer sequence (root)
        handler.on_chain_start({}, "x", run_id=ids["root"], name="RunnableSequence")

        # step 1: a parallel
        handler.on_chain_start(
            {},
            "x",
            run_id=ids["par"],
            parent_run_id=ids["root"],
            name="RunnableParallel<a,b>",
            tags=["seq:step:1"],
        )

        # branch a: just a lambda
        handler.on_chain_start(
            {},
            "x",
            run_id=ids["a"],
            parent_run_id=ids["par"],
            name="RunnableLambda",
            tags=["map:key:a"],
        )
        handler.on_chain_end("A", run_id=ids["a"], parent_run_id=ids["par"])

        # branch b: a sequence containing two lambdas
        handler.on_chain_start(
            {},
            "x",
            run_id=ids["inner_seq"],
            parent_run_id=ids["par"],
            name="RunnableSequence",
            tags=["map:key:b"],
        )
        handler.on_chain_start(
            {},
            "x",
            run_id=ids["inner_a"],
            parent_run_id=ids["inner_seq"],
            name="RunnableLambda",
            tags=["seq:step:1"],
        )
        handler.on_chain_end("ia", run_id=ids["inner_a"], parent_run_id=ids["inner_seq"])
        handler.on_chain_start(
            {},
            "ia",
            run_id=ids["inner_b"],
            parent_run_id=ids["inner_seq"],
            name="RunnableLambda",
            tags=["seq:step:2"],
        )
        handler.on_chain_end("ib", run_id=ids["inner_b"], parent_run_id=ids["inner_seq"])
        handler.on_chain_end({"a": "A", "b": "ib"}, run_id=ids["inner_seq"], parent_run_id=ids["par"])

        # close parallel
        handler.on_chain_end({"a": "A", "b": "ib"}, run_id=ids["par"], parent_run_id=ids["root"])

        # step 2: passthrough
        handler.on_chain_start(
            {},
            {"a": "A", "b": "ib"},
            run_id=ids["b"],
            parent_run_id=ids["root"],
            name="RunnablePassthrough",
            tags=["seq:step:2"],
        )
        handler.on_chain_end({"a": "A", "b": "ib"}, run_id=ids["b"], parent_run_id=ids["root"])

        handler.on_chain_end({"a": "A", "b": "ib"}, run_id=ids["root"])

        composition = next(
            e for e in stratix.events if e["type"] == "agent.code" and e["payload"].get("kind") == "chain.composition"
        )
        comp = composition["payload"]["composition"]
        # root(0) → par(1) → inner_seq(2) → inner_a(3)/inner_b(3)
        assert comp["max_depth"] == 3
        assert comp["node_count"] == 7
        # inner_seq is itself a sequence — counts should reflect both seqs.
        assert comp["kind_counts"]["sequence"] == 2
        assert comp["kind_counts"]["parallel"] == 1
        assert comp["kind_counts"]["passthrough"] == 1
        assert comp["kind_counts"]["lambda"] == 3

    def test_error_in_runnable_records_error_in_composition(self) -> None:
        stratix = _RecordingStratix()
        handler = LayerLensCallbackHandler(stratix=stratix, capture_config=_full_capture())
        handler.connect()

        root_id = _new_id()
        bad_id = _new_id()

        handler.on_chain_start({}, "x", run_id=root_id, name="RunnableSequence")
        handler.on_chain_start(
            {},
            "x",
            run_id=bad_id,
            parent_run_id=root_id,
            name="RunnableLambda",
            tags=["seq:step:1"],
        )
        handler.on_chain_error(ValueError("kaboom"), run_id=bad_id, parent_run_id=root_id)
        handler.on_chain_error(ValueError("kaboom"), run_id=root_id)

        # Per-step agent.code with error.
        codes_with_error = [
            e
            for e in stratix.events
            if e["type"] == "agent.code"
            and e["payload"].get("kind") != "chain.composition"
            and e["payload"].get("error")
        ]
        assert len(codes_with_error) == 2  # one per failed runnable
        # Composition event reflects root error status.
        composition = next(
            e for e in stratix.events if e["type"] == "agent.code" and e["payload"].get("kind") == "chain.composition"
        )
        assert composition["payload"]["composition"]["status"] == "error"

    def test_l2_disabled_drops_code_and_composition(self) -> None:
        stratix = _RecordingStratix()
        # L2 off → no agent.code events.
        cfg = CaptureConfig(
            l1_agent_io=True,
            l2_agent_code=False,
            l3_model_metadata=True,
            l5a_tool_calls=True,
        )
        handler = LayerLensCallbackHandler(stratix=stratix, capture_config=cfg)
        handler.connect()

        root_id = _new_id()
        handler.on_chain_start({}, "x", run_id=root_id, name="RunnableSequence")
        handler.on_chain_end("y", run_id=root_id)

        types = [e["type"] for e in stratix.events]
        assert "agent.input" in types
        assert "agent.output" in types
        # No agent.code at all when L2 is off.
        assert "agent.code" not in types

    def test_langgraph_node_path_unaffected_by_lcel(self) -> None:
        """LangGraph node attribution must continue working — LCEL
        tracking only kicks in when no langgraph_node marker is present.
        """
        stratix = _RecordingStratix()
        handler = LayerLensCallbackHandler(stratix=stratix, capture_config=_full_capture())
        handler.connect()

        node_id = _new_id()
        handler.on_chain_start(
            {},
            {"q": "hi"},
            run_id=node_id,
            metadata={"langgraph_node": "research", "langgraph_step": 1},
            # name is also set by LangGraph but the langgraph marker
            # MUST take precedence — LCEL tracking should not engage.
            name="RunnableSequence",
        )
        handler.on_chain_end({"out": "ok"}, run_id=node_id)

        # Should look like the existing langgraph path — node_name in
        # the agent.input payload, NOT a runnable kind.
        agent_inputs = [e for e in stratix.events if e["type"] == "agent.input"]
        assert len(agent_inputs) == 1
        assert agent_inputs[0]["payload"]["node_name"] == "research"
        assert "runnable" not in agent_inputs[0]["payload"]
        # No composition snapshot (this was a langgraph node, not LCEL).
        composition_events = [
            e
            for e in stratix.events
            if e["type"] == "agent.code" and e["payload"].get("kind") == "chain.composition"
        ]
        assert composition_events == []

    def test_disconnect_resets_lcel_tracker(self) -> None:
        handler = LayerLensCallbackHandler(stratix=_RecordingStratix(), capture_config=_full_capture())
        handler.connect()
        handler.on_chain_start({}, "x", run_id=_new_id(), name="RunnableSequence")
        # Tracker has at least one node prior to disconnect.
        assert any(handler._lcel.get_node(rid) is not None for rid in handler._lcel._nodes)
        handler.disconnect()
        assert handler._lcel._nodes == {}

    def test_runnable_config_propagates_via_kwargs(self) -> None:
        """RunnableConfig is opaque to the adapter — propagation is
        validated by ensuring callbacks fire with the same run_id under
        the same parent_run_id when a config is in play. We simulate
        that here by driving with explicit ids and asserting hierarchy.
        """
        stratix = _RecordingStratix()
        handler = LayerLensCallbackHandler(stratix=stratix, capture_config=_full_capture())
        handler.connect()

        root_id = _new_id()
        child_id = _new_id()

        # The "config" RunnableConfig would surface as metadata + tags
        # on the child — we validate the adapter forwards both.
        handler.on_chain_start({}, "x", run_id=root_id, name="RunnableSequence")
        handler.on_chain_start(
            {},
            "x",
            run_id=child_id,
            parent_run_id=root_id,
            name="RunnableLambda",
            tags=["seq:step:1", "user:tag:foo"],
            metadata={"user_metadata_key": "value"},
        )
        handler.on_chain_end("y", run_id=child_id, parent_run_id=root_id)
        handler.on_chain_end("y", run_id=root_id)

        composition = next(
            e for e in stratix.events if e["type"] == "agent.code" and e["payload"].get("kind") == "chain.composition"
        )
        # The composition should still see the child as a depth-1 lambda
        # inside the root sequence — the user-supplied tag must NOT
        # confuse position parsing.
        nodes = composition["payload"]["composition"]["nodes"]
        child_node = next(n for n in nodes if n["depth"] == 1)
        assert child_node["kind"] == "lambda"
        assert child_node["position"]["role"] == "step"
        assert child_node["position"]["label"] == "1"


# ---------------------------------------------------------------------------
# fingerprint_lambda determinism
# ---------------------------------------------------------------------------


class TestFingerprintLambda:
    def test_same_inputs_same_hash(self) -> None:
        from layerlens.instrument.adapters.frameworks.langchain import fingerprint_lambda

        a = fingerprint_lambda("my_func", 2, CompositionPosition(RunnableKind.SEQUENCE, "1", "step"))
        b = fingerprint_lambda("my_func", 2, CompositionPosition(RunnableKind.SEQUENCE, "1", "step"))
        assert a == b
        assert len(a) == 16

    def test_different_inputs_different_hash(self) -> None:
        from layerlens.instrument.adapters.frameworks.langchain import fingerprint_lambda

        a = fingerprint_lambda("foo", 1, None)
        b = fingerprint_lambda("bar", 1, None)
        assert a != b
