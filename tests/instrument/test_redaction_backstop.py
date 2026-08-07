"""Universal ``capture_content=False`` backstop (LAY-3567 / LAY-3578 family).

The privacy promise — "no content leaves the SDK under ``capture_content=False``"
— must hold at the *collector* boundary for EVERY event type, not only at each
adapter's emit-time gate. Content suppression can be (and historically was)
forgotten in the adapters; the discovery sweep (2026-06-24) found the same class
of leak in ~15 places across protocol, framework and provider adapters:

* free-text ``error`` / ``error_message`` carrying ``str(exc)`` (which routinely
  echoes the failing arguments — card numbers, amounts, merchant, prompts) on
  ``agent.error`` (ALWAYS-ENABLED, so it also bypasses layer gating),
  ``mcp.tool.call``, ``mcp.async_task``, ``a2a.task.updated``; and
* content surfaces set UNCONDITIONALLY (not behind a ``capture_content`` gate):
  ``tool.call`` arguments (provider + langchain), ``agent.handoff`` context
  (langgraph graph state), ``agent.state.change`` status text (langfuse),
  ``model.invoke`` ``parameters.tools`` (the caller's tool JSON-Schema).

These tests drive the REAL redaction path (``TraceCollector.emit`` ->
``CaptureConfig.redact_payload``) and the REAL provider emit helpers — no mock
stand-ins. Each asserts the SENTINEL content is gone AND that category/metadata
(``error_type``, ``tool_name``, topology, hashes, ``reason_code``) survives, so
the fix redacts without going blind.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from layerlens.instrument._context import _current_collector
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig

SENTINEL = "SENTINEL-leak-canary-4111111111111111"
NO_CONTENT = CaptureConfig(capture_content=False)


def _emit(event_type: str, payload: Dict[str, Any], config: Optional[CaptureConfig] = None) -> Dict[str, Any]:
    """Emit one event through a real collector and return the stored payload.

    Returns ``{}`` if the event was gated out entirely (so a missing event reads
    as "no content" rather than KeyError).
    """
    collector = TraceCollector(object(), config or NO_CONTENT)
    collector.emit(event_type, payload, span_id="span-1")
    events = collector.events
    return events[0]["payload"] if events else {}


def _blob(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# agent.error — ALWAYS-ENABLED + str(exc) free text (the most pervasive leak)
# ---------------------------------------------------------------------------


def test_agent_error_strips_message_keeps_type() -> None:
    payload = _emit(
        "agent.error",
        {
            "name": "openai.chat",
            "error": f"BadRequest: bad card {SENTINEL}",
            "error_type": "ValueError",
            "latency_ms": 5,
        },
    )
    assert SENTINEL not in _blob(payload), "agent.error free-text leaked under capture_content=False"
    assert payload.get("error_type") == "ValueError", "category over-stripped — observability blinded"
    assert "latency_ms" in payload


def test_agent_error_strips_error_message_variant() -> None:
    payload = _emit(
        "agent.error",
        {"error_message": f"Salesforce step failed for {SENTINEL}", "error_code": "E_STEP"},
    )
    assert SENTINEL not in _blob(payload), "agent.error error_message variant leaked"
    assert payload.get("error_code") == "E_STEP"


def test_provider_emit_llm_error_real_path() -> None:
    """Drive the REAL provider error helper (providers/_emit_helpers.emit_llm_error)."""
    from layerlens.instrument.adapters.providers._emit_helpers import emit_llm_error

    collector = TraceCollector(object(), NO_CONTENT)
    token = _current_collector.set(collector)
    try:
        emit_llm_error("openai.chat.completions", ValueError(f"invalid request body {SENTINEL}"), 12.0)
    finally:
        _current_collector.reset(token)
    events = collector.events
    assert events, "provider error produced no event"
    blob = json.dumps([e["payload"] for e in events], default=str)
    assert SENTINEL not in blob, "provider agent.error str(exc) leaked under capture_content=False"
    assert events[0]["payload"].get("error_type") == "ValueError"


# ---------------------------------------------------------------------------
# tool.call — model-generated arguments / results (content surface)
# ---------------------------------------------------------------------------


def test_tool_call_strips_arguments_keeps_name() -> None:
    payload = _emit(
        "tool.call",
        {"provider": "openai", "tool_name": "charge_card", "arguments": {"card": SENTINEL, "amount": 500}},
    )
    assert SENTINEL not in _blob(payload), "tool.call arguments leaked under capture_content=False"
    assert payload.get("tool_name") == "charge_card", "tool_name (metadata) over-stripped"


def test_tool_result_strips_result() -> None:
    payload = _emit("tool.result", {"tool_name": "lookup", "result": f"balance {SENTINEL}"})
    assert SENTINEL not in _blob(payload), "tool.result content leaked"
    assert payload.get("tool_name") == "lookup"


def test_provider_emit_tool_call_real_path() -> None:
    from layerlens.instrument.adapters.providers._emit_helpers import emit_tool_call

    collector = TraceCollector(object(), NO_CONTENT)
    token = _current_collector.set(collector)
    try:
        emit_tool_call(
            provider="openai", model="gpt-4o", tool_name="charge", arguments={"pan": SENTINEL}, result={"ok": SENTINEL}
        )
    finally:
        _current_collector.reset(token)
    blob = json.dumps([e["payload"] for e in collector.events], default=str)
    assert SENTINEL not in blob, "provider tool.call arguments/result leaked under capture_content=False"
    # metadata must survive (redact without going blind)
    tool_calls = [e["payload"] for e in collector.events if e["event_type"] == "tool.call"]
    assert tool_calls and tool_calls[0].get("tool_name") == "charge", "tool_name over-stripped"
    assert tool_calls[0].get("provider") == "openai", "provider metadata over-stripped"


# ---------------------------------------------------------------------------
# retrieval.query — user query content
# ---------------------------------------------------------------------------


def test_retrieval_query_strips_query() -> None:
    payload = _emit("retrieval.query", {"source": "vectordb", "query": f"who is {SENTINEL}", "top_k": 5})
    assert SENTINEL not in _blob(payload), "retrieval.query leaked under capture_content=False"
    assert payload.get("source") == "vectordb" and payload.get("top_k") == 5


# ---------------------------------------------------------------------------
# agent.handoff — ALWAYS-ENABLED; context = graph state (langgraph)
# ---------------------------------------------------------------------------


def test_agent_handoff_strips_context_keeps_topology() -> None:
    payload = _emit(
        "agent.handoff",
        {
            "from_agent": "router",
            "to_agent": "billing",
            "reason": "supervisor_delegation",
            "handoff_context_hash": "sha256:abc",
            "context": {"messages": [f"user said {SENTINEL}"], "query": SENTINEL},
        },
    )
    assert SENTINEL not in _blob(payload), "agent.handoff context (graph state) leaked"
    assert payload.get("from_agent") == "router" and payload.get("to_agent") == "billing"
    assert payload.get("reason") == "supervisor_delegation", "handoff category over-stripped"
    assert payload.get("handoff_context_hash") == "sha256:abc"


# ---------------------------------------------------------------------------
# agent.state.change — ALWAYS-ENABLED; status_message / state (langfuse)
# ---------------------------------------------------------------------------


def test_agent_state_change_strips_status_message_keeps_hash() -> None:
    payload = _emit(
        "agent.state.change",
        {
            "node": "billing_node",
            "after_hash": "sha256:def",
            "status_message": f"failed: {SENTINEL}",
            "state": {"x": SENTINEL},
        },
    )
    assert SENTINEL not in _blob(payload), "agent.state.change status/state leaked"
    assert payload.get("node") == "billing_node" and payload.get("after_hash") == "sha256:def"


# ---------------------------------------------------------------------------
# model.invoke — parameters.tools = caller tool JSON-Schema (#17)
# ---------------------------------------------------------------------------


def test_model_invoke_strips_tools_param_keeps_metrics() -> None:
    payload = _emit(
        "model.invoke",
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": SENTINEL}],
            "parameters": {"temperature": 0.5, "tools": [{"name": f"fn_{SENTINEL}", "description": SENTINEL}]},
        },
    )
    assert SENTINEL not in _blob(payload), "model.invoke messages/tools schema leaked"
    assert payload.get("model") == "gpt-4o"
    assert payload.get("parameters", {}).get("temperature") == 0.5, "safe parameter over-stripped"


# ---------------------------------------------------------------------------
# Protocol error surfaces — error: str(exc) (L3 cluster)
# ---------------------------------------------------------------------------


def test_mcp_tool_call_strips_error() -> None:
    payload = _emit(
        "mcp.tool.call",
        {
            "tool_name": "charge",
            "arguments": {"card": SENTINEL},
            "error": f"charge failed card={SENTINEL}",
            "latency_ms": 3,
        },
    )
    assert SENTINEL not in _blob(payload), "mcp.tool.call error/arguments leaked"
    assert payload.get("tool_name") == "charge"


def test_mcp_async_task_strips_error() -> None:
    payload = _emit("mcp.async_task", {"async_task_id": "a1", "status": "failed", "error": f"boom {SENTINEL}"})
    assert SENTINEL not in _blob(payload), "mcp.async_task error leaked"
    # only `error` is content; the failure signal + id must survive.
    assert payload.get("status") == "failed" and payload.get("async_task_id") == "a1", "metadata over-stripped"


def test_a2a_task_updated_strips_error() -> None:
    payload = _emit(
        "a2a.task.updated", {"task_id": "t1", "status": "failed", "error": f"invalid card {SENTINEL}", "latency_ms": 9}
    )
    assert SENTINEL not in _blob(payload), "a2a.task.updated error leaked"
    assert payload.get("status") == "failed" and payload.get("task_id") == "t1"


# ---------------------------------------------------------------------------
# a2a.delegation — delegation PROVENANCE (A15 / D3, user-approved 2026-06-25).
# OVERTURN of the old lock: the delegation TOPOLOGY (from_agent/to_agent/
# target_agent ids) + the keyed-HMAC fp must SURVIVE capture_content=False so
# cross-agent provenance is auditable under privacy-on — mirroring
# agent.handoff (above) which keeps from_agent/to_agent. Only the free-text
# skill DESCRIPTION + target_url + context are content.
# ---------------------------------------------------------------------------


def test_a2a_delegation_keeps_topology_strips_skill_description() -> None:
    payload = _emit(
        "a2a.delegation",
        {
            "task_id": "t1",
            "target_agent": "billing-agent-7",
            "to_agent": "billing-agent-7",
            "from_agent": "orchestrator-1",
            "delegation_fp": "sha256:abc123",
            "skill_description": f"do {SENTINEL}",
            "target_url": f"https://{SENTINEL}.example.com",
            "context": {"note": SENTINEL},
        },
    )
    assert SENTINEL not in _blob(payload), "a2a.delegation skill_description/target_url/context leaked"
    # Topology + provenance metadata SURVIVE (A15).
    assert payload.get("task_id") == "t1", "delegation task id (metadata) over-stripped"
    assert payload.get("from_agent") == "orchestrator-1", "delegator id stripped (A15 provenance loss)"
    assert payload.get("to_agent") == "billing-agent-7", "delegatee id stripped (A15 provenance loss)"
    assert payload.get("target_agent") == "billing-agent-7", "target_agent stripped (A15 provenance loss)"
    assert payload.get("delegation_fp") == "sha256:abc123", "delegation_fp (provenance) over-stripped (A15)"


# ---------------------------------------------------------------------------
# payment.mandate_signed — blocked reason (L1); reason_code must survive
# ---------------------------------------------------------------------------


def test_payment_mandate_signed_strips_reason_keeps_code() -> None:
    payload = _emit(
        "payment.mandate_signed",
        {
            "mandate_id": "m1",
            "status": "blocked",
            "reason": f"merchant '{SENTINEL}' not in whitelist",
            "reason_code": "MERCHANT_NOT_WHITELISTED",
        },
    )
    assert SENTINEL not in _blob(payload), "payment.mandate_signed blocked reason leaked merchant/amount"
    assert payload.get("status") == "blocked"
    assert payload.get("reason_code") == "MERCHANT_NOT_WHITELISTED", "reason_code (why blocked) over-stripped"


# ---------------------------------------------------------------------------
# Residual leaks (census 2026-06-24): free-text error/content attached to an
# event type whose content keys didn't cover the field -> survived redaction.
# Fixed by extending _CONTENT_KEYS (one-file backstop, covers every adapter).
# ---------------------------------------------------------------------------


def test_agent_output_strips_error() -> None:
    # smolagents/haystack/agno tack str(exc) onto agent.output (not agent.error).
    payload = _emit("agent.output", {"output": "ok", "error": f"failed {SENTINEL}", "error_type": "ValueError"})
    assert SENTINEL not in _blob(payload), "agent.output error leaked (smolagents/haystack/agno class)"
    assert payload.get("error_type") == "ValueError", "category over-stripped"


def test_agent_step_strips_error_and_code() -> None:
    payload = _emit("agent.step", {"step": 1, "error": f"boom {SENTINEL}", "code_action": f"run({SENTINEL})"})
    assert SENTINEL not in _blob(payload), "agent.step error/code_action leaked (smolagents class)"
    assert payload.get("step") == 1


def test_model_invoke_strips_error() -> None:
    # strands / langchain attach error onto model.invoke (in addition to agent.error).
    payload = _emit("model.invoke", {"model": "gpt-4o", "error": f"timeout {SENTINEL}", "latency_ms": 5})
    assert SENTINEL not in _blob(payload), "model.invoke error leaked (strands/langchain class)"
    assert payload.get("model") == "gpt-4o" and payload.get("latency_ms") == 5


def test_tool_result_strips_error() -> None:
    payload = _emit("tool.result", {"tool_name": "charge", "error": f"declined {SENTINEL}"})
    assert SENTINEL not in _blob(payload), "tool.result error leaked (strands/haystack class)"
    assert payload.get("tool_name") == "charge"


def test_agent_node_exit_strips_error_and_io() -> None:
    # langgraph on_chain_error puts error onto agent.node.exit (was content-free -> no-op redact).
    payload = _emit(
        "agent.node.exit",
        {"node": "billing", "error": f"node failed {SENTINEL}", "output": {"x": SENTINEL}},
    )
    assert SENTINEL not in _blob(payload), "agent.node.exit error/output leaked (langgraph class)"
    assert payload.get("node") == "billing"


def test_agent_node_enter_strips_input() -> None:
    payload = _emit("agent.node.enter", {"node": "billing", "input": {"q": SENTINEL}})
    assert SENTINEL not in _blob(payload), "agent.node.enter input leaked"
    assert payload.get("node") == "billing"


def test_instrument_decorator_error_keeps_type_under_no_content(mock_client, capture_trace) -> None:
    """The public @trace decorator (highest reach) must keep a surviving
    error_type/status when the backstop strips the free-text error (Batch 2)."""
    import pytest

    from layerlens.instrument import trace

    @trace(mock_client, capture_config=CaptureConfig(capture_content=False))
    def boom() -> None:
        raise ValueError(f"secret detail {SENTINEL}")

    with pytest.raises(ValueError):
        boom()

    errors = [e for e in capture_trace["events"] if e["event_type"] == "agent.error"]
    assert errors, "no agent.error emitted by decorator"
    payload = errors[0]["payload"]
    assert SENTINEL not in _blob(payload), "decorator leaked the error message under capture_content=False"
    assert payload.get("error_type") == "ValueError", "error_type did not survive — failure now untyped"
    assert payload.get("status") == "error"


def test_agent_code_strips_code_and_output() -> None:
    # bedrock_agents/semantic_kernel/langfuse emit code/rendered_prompt/exec output on agent.code.
    # agent.code rides l2_agent_code (off by default) — enable it so the event is emitted.
    payload = _emit(
        "agent.code",
        {
            "language": "python",
            "code": f"charge({SENTINEL})",
            "rendered_prompt": f"prompt {SENTINEL}",
            "execution_error": f"err {SENTINEL}",
            "output": f"out {SENTINEL}",
        },
        config=CaptureConfig(capture_content=False, l2_agent_code=True),
    )
    assert SENTINEL not in _blob(payload), "agent.code code/prompt/exec content leaked"
    assert payload.get("language") == "python"


# ---------------------------------------------------------------------------
# Nested content (LAY-3572 / R1 / B17) — redaction must recurse, not strip only
# top-level keys. Content nested under a non-content key (a `metadata`/`extra`
# wrapper, a list element) historically survived capture_content=False because
# `redact_payload` only filtered the top-level dict.
# ---------------------------------------------------------------------------


def test_nested_content_under_unlisted_key_is_stripped() -> None:
    payload = _emit("agent.input", {"name": "agent-x", "metadata": {"messages": [{"content": SENTINEL}]}})
    assert SENTINEL not in _blob(payload), "nested content leaked under capture_content=False (non-recursive redaction)"
    assert payload.get("name") == "agent-x", "non-content sibling over-stripped"


def test_nested_content_in_list_element_is_stripped() -> None:
    payload = _emit("tool.call", {"tool_name": "search", "trace": [{"arguments": {"q": SENTINEL}}]})
    assert SENTINEL not in _blob(payload), "content nested in a list element leaked under capture_content=False"
    assert payload.get("tool_name") == "search", "tool_name metadata over-stripped"


def test_deeply_nested_content_key_is_stripped() -> None:
    payload = _emit("agent.output", {"status": "ok", "wrapper": {"inner": {"output": SENTINEL}}})
    assert SENTINEL not in _blob(payload), "deeply nested content key 'output' leaked"
    assert payload.get("status") == "ok", "status metadata over-stripped"


# ---------------------------------------------------------------------------
# UNREGISTERED event types must fail CLOSED (F2 / live probe unregistered_type_bypass).
# is_layer_enabled fail-opens for unknown types and redact_payload previously no-op'd
# when _CONTENT_KEYS had no entry -> a custom/future event_type leaked its content
# verbatim under capture_content=False (live-proven: sentinel survived to the server).
# A union-of-known-content-keys backstop is INSUFFICIENT (an arbitrary field name like
# 'secret_notes' is in no content-key set), so the type is DENY-BY-DEFAULT: keep only a
# vetted safe-metadata allowlist and drop everything else, mirroring _keep_safe_params.
# ---------------------------------------------------------------------------


def test_unregistered_event_type_denies_arbitrary_content_keeps_metadata() -> None:
    payload = _emit(
        "custom.acme.reasoning",
        {
            "messages": [{"role": "user", "content": SENTINEL}],  # known content key
            "secret_notes": SENTINEL,  # ARBITRARY key not in any content-key set
            "model": "gpt-4o",  # safe metadata -> kept
            "step": 3,  # safe metadata -> kept
        },
    )
    assert SENTINEL not in _blob(payload), "unregistered event_type leaked content under capture_content=False"
    assert payload.get("model") == "gpt-4o" and payload.get("step") == 3, "safe metadata over-stripped on unknown type"


def test_unregistered_event_type_nested_arbitrary_content_is_stripped() -> None:
    payload = _emit(
        "vendor.telemetry.v2",
        {"trace_id": "t1", "custom_blob": {"reasoning": SENTINEL, "scratch": {"x": SENTINEL}}},
    )
    assert SENTINEL not in _blob(payload), "nested arbitrary content on an unregistered type leaked"
    assert payload.get("trace_id") == "t1", "id metadata over-stripped on unknown type"


def test_unregistered_type_content_preserved_under_capture_content_true() -> None:
    payload = _emit("custom.acme.reasoning", {"secret_notes": SENTINEL}, config=CaptureConfig.full())
    assert SENTINEL in _blob(payload), "capture_content=True must not strip an unregistered type's content"


def test_unregistered_type_source_body_and_bytes_blob_are_stripped() -> None:
    # re-vet residual: 'source' (a document body) and any '*_bytes' (a content blob)
    # must NOT be kept as "safe metadata" on an unregistered type. A byte COUNT is
    # still allowed via '*_count'; a step counter stays.
    payload = _emit(
        "custom.multimodal.v1",
        {"source": SENTINEL, "image_bytes": SENTINEL, "audio_bytes": SENTINEL, "step": 2, "frame_count": 9},
    )
    assert SENTINEL not in _blob(payload), "source/_bytes content leaked on an unregistered type"
    assert payload.get("step") == 2 and payload.get("frame_count") == 9, "short metadata over-stripped"


def test_registered_content_free_type_is_not_over_stripped() -> None:
    # A REGISTERED type with no _CONTENT_KEYS entry (cost.record) keeps its curated
    # (no deny-by-default) behavior — the backstop targets ONLY unregistered types.
    payload = _emit("cost.record", {"model": "gpt-4o", "cost_usd": 0.01, "total_tokens": 42, "vendor_field": "x"})
    assert payload.get("vendor_field") == "x", "registered content-free type wrongly deny-by-defaulted"
    assert payload.get("cost_usd") == 0.01


# ---------------------------------------------------------------------------
# An ALREADY-EMPTY content key is kept, not deleted (LAY-3622 F2).
#
# An adapter that gates at emit time may set a field it treats as required to ""
# rather than omitting it (the openinference agent-turn convention). Deleting that
# key redacts NOTHING while destroying the field's presence, so a strict consumer
# reads the event as malformed and drops the turn. The exemption is privacy-neutral
# by construction — there is no content in an empty string — and it is deliberately
# narrow: a NON-empty value is still deleted even when the adapter forgot to gate,
# because that is a gating bug the backstop must keep failing closed on.
# ---------------------------------------------------------------------------


def test_an_already_empty_content_key_survives_the_backstop() -> None:
    payload = _emit("agent.input", {"agent_id": "a1", "input_text": ""})
    assert payload.get("input_text") == "", (
        "an empty content key was deleted — nothing was redacted, only presence lost"
    )
    assert payload.get("agent_id") == "a1"


def test_a_non_empty_content_key_is_STILL_deleted() -> None:
    # The privacy guarantee. If the exemption ever widens to "delete nothing on a
    # content key", this is the assertion that catches it.
    payload = _emit("agent.input", {"agent_id": "a1", "input_text": SENTINEL})
    assert "input_text" not in payload, "a populated content key survived capture_content=False"
    assert SENTINEL not in _blob(payload)


def test_a_none_valued_content_key_is_still_deleted() -> None:
    # Deliberately NOT exempt: a null key satisfies no consumer's required-field
    # check, so keeping it would be noise rather than a mitigation.
    payload = _emit("agent.output", {"status": "ok", "output_text": None})
    assert "output_text" not in payload


def test_a_whitespace_only_content_key_is_still_deleted() -> None:
    # Only "" is content-free. Anything else is a value the adapter obtained from
    # somewhere, and the backstop does not get to judge whether it matters.
    payload = _emit("agent.output", {"status": "ok", "output_text": "   "})
    assert "output_text" not in payload


def test_the_empty_exemption_applies_at_every_depth() -> None:
    # _strip_content_keys is recursive, so the exemption must be too — otherwise the
    # rule would depend on where in the tree the adapter put the field.
    payload = _emit("agent.output", {"status": "ok", "wrapper": {"output": "", "output_text": SENTINEL}})
    assert payload["wrapper"].get("output") == "", "the exemption did not reach a nested empty content key"
    assert SENTINEL not in _blob(payload), "a nested populated content key leaked"


def test_an_empty_content_key_still_goes_under_capture_content_true_untouched() -> None:
    # Vacuity control: capture_content=True returns the payload unredacted, so the
    # empty value must survive there too — otherwise the tests above could pass
    # because redaction never ran.
    payload = _emit("agent.input", {"agent_id": "a1", "input_text": ""}, CaptureConfig.full())
    assert payload.get("input_text") == ""
