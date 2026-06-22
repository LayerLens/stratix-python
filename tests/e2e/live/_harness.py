"""The L1 + L2 engine: run a scenario, assert on the local payload, upload, read back.

Design note (why we assert locally first): the trace payload we upload
(``{trace_id, events, capture_config, attestation}``) is the SDK's instrumentation
schema. Whether the backend echoes ``events`` back in ``trace.get(id).data`` is not
guaranteed, so the deep adapter assertions run on the **local pre-upload payload**
(whose shape we fully control). The round-trip then proves *persistence*: the upload
was accepted (non-empty ``trace_ids`` — a rejected schema fails loudly here) and the
trace is fetchable. If ``data`` does echo events, we re-check the event count as a bonus.
"""

from __future__ import annotations

import os
import json
import time
import uuid
import tempfile
from typing import Any, Dict, List, Optional

from layerlens.attestation._verify import verify_chain
from layerlens.instrument._context import _pop_span, _push_span, _current_collector
from layerlens.attestation._envelope import HashScope, AttestationEnvelope
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.providers.pricing import calculate_cost
from layerlens.instrument.adapters.providers.token_usage import NormalizedTokenUsage

from ._timing import TRACE_READBACK_DELAY_S, TRACE_READBACK_ATTEMPTS
from ._linkage import verify_linkage
from ._registry import ProviderCase, resolve_pricing_table
from ._scenarios import SENTINEL

_COST_TOLERANCE = 1e-6


def run_case(client: Any, case: ProviderCase, variant: str) -> Dict[str, Any]:
    """Run one (provider, variant), assert, round-trip, tear down. Returns a report row."""
    cost_cap = float(os.environ.get("LAYERLENS_LIVE_COST_CAP_USD", "0.25"))
    config = CaptureConfig(capture_content=False) if variant == "redaction" else CaptureConfig.standard()
    flow = "default" if variant == "redaction" else variant

    payload = _collect(client, case, flow, config)
    events = payload.get("events", [])
    by_type = _events_by_type(events)

    _assert_attestation(payload, events)
    total_cost = _assert_contract(case, variant, events, by_type, cost_cap)

    backend_id = _upload_capture(client, payload)
    trace = _poll_get(client, backend_id)
    if trace is None:
        raise AssertionError(f"[{case.id}/{variant}] trace {backend_id} not found after polling")
    data_has_events = _reassert_roundtrip(trace, events, variant)

    # Inbound-connector linkage. Records the stamped integration_id; asserts an
    # exact match + Healthy only when LAYERLENS_LIVE_INTEGRATION_ID is set. The
    # provider harness previously skipped this entirely (LAY-3618 / gap G7), so a
    # provider trace's connector linkage was never verifiable like a framework's.
    linkage = verify_linkage(client, backend_id)

    _teardown(client, backend_id)

    model = _first_model(by_type)
    return {
        "provider": case.id,
        "variant": variant,
        "model": model,
        "trace_id": backend_id,
        "org_id": getattr(client, "organization_id", None),
        "project_id": getattr(client, "project_id", None),
        "n_events": len(events),
        "event_types": {k: len(v) for k, v in by_type.items()},
        "tool_calls": len(by_type.get("tool.call", [])),
        "total_cost_usd": round(total_cost, 8),
        "redaction_ok": variant == "redaction",
        "attestation_ok": True,
        "data_has_events": data_has_events,
        "linked": linkage.get("linked"),
        "integration_id": linkage.get("integration_id"),
        "linkage_status": linkage.get("status"),
        "status": "pass",
    }


# --------------------------------------------------------------------------- #
# L1: run the scenario under a local collector (no auto-flush)
# --------------------------------------------------------------------------- #
def _collect(client: Any, case: ProviderCase, flow: str, config: CaptureConfig) -> Dict[str, Any]:
    collector = TraceCollector(client, config)
    col_token = _current_collector.set(collector)
    snapshot = _push_span(uuid.uuid4().hex[:16], f"{case.id}-{flow}")
    try:
        case.runner(flow)
    finally:
        _pop_span(snapshot)
        _current_collector.reset(col_token)
    return collector.to_replay_dict()


# --------------------------------------------------------------------------- #
# Local assertions (the reliable core)
# --------------------------------------------------------------------------- #
def _assert_contract(
    case: ProviderCase,
    variant: str,
    events: List[Dict[str, Any]],
    by_type: Dict[str, List[Dict[str, Any]]],
    cost_cap: float,
) -> float:
    tag = f"[{case.id}/{variant}]"
    model_invokes = by_type.get("model.invoke", [])
    cost_records = by_type.get("cost.record", [])

    if variant == "error":
        assert by_type.get("agent.error"), f"{tag} expected an agent.error event, got types {sorted(by_type)}"
        return _sum_cost(cost_records)

    if variant in ("streaming", "async-streaming"):
        assert model_invokes, f"{tag} streaming produced no model.invoke event"
        assert any(_payload(e).get("ttft_ms") is not None for e in model_invokes), (
            f"{tag} no model.invoke carried ttft_ms (streaming path did not run)"
        )
        return _sum_cost(cost_records)

    # default / redaction / async: the full canonical contract
    # (async runs the same tool loop as default, through the async client)
    c = case.contract
    n = len(events)
    assert c.min_events <= n <= c.max_events, f"{tag} event count {n} outside [{c.min_events}, {c.max_events}]"
    assert model_invokes, f"{tag} no model.invoke events"

    if c.requires_tool_call:
        assert by_type.get("tool.call"), f"{tag} expected tool.call, got types {sorted(by_type)}"
    if c.requires_cost_record:
        assert cost_records, f"{tag} expected cost.record, got types {sorted(by_type)}"

    _assert_cost_math(tag, case, cost_records)

    if variant == "redaction":
        _assert_redaction(tag, events, model_invokes)

    total = _sum_cost(cost_records)
    assert total <= cost_cap, f"{tag} run cost ${total:.4f} exceeded cap ${cost_cap:.4f}"
    return total


def _assert_cost_math(tag: str, case: ProviderCase, cost_records: List[Dict[str, Any]]) -> None:
    table = resolve_pricing_table(case)
    for cr in cost_records:
        p = _payload(cr)
        cost_usd = p.get("cost_usd")
        if case.contract.cost_priced:
            assert cost_usd is not None and cost_usd > 0, (
                f"{tag} cost.record has unpriced cost_usd={cost_usd!r} (model {p.get('model')!r})"
            )
        if cost_usd is None:
            continue
        usage = NormalizedTokenUsage(
            prompt_tokens=int(p.get("prompt_tokens") or 0),
            completion_tokens=int(p.get("completion_tokens") or 0),
            total_tokens=int(p.get("total_tokens") or 0),
            cached_tokens=p.get("cached_tokens"),
            cache_creation_tokens=p.get("cache_creation_tokens"),
            reasoning_tokens=p.get("reasoning_tokens"),
            thinking_tokens=p.get("thinking_tokens"),
        )
        recomputed = calculate_cost(p.get("model") or "", usage, table)
        if recomputed is not None:
            assert abs(recomputed - cost_usd) < _COST_TOLERANCE, (
                f"{tag} cost mismatch for {p.get('model')!r}: emitted {cost_usd}, recomputed {recomputed}"
            )


def _assert_redaction(tag: str, events: List[Dict[str, Any]], model_invokes: List[Dict[str, Any]]) -> None:
    for e in model_invokes:
        p = _payload(e)
        assert "messages" not in p, f"{tag} redaction failed: 'messages' present in model.invoke"
        assert "output_message" not in p, f"{tag} redaction failed: 'output_message' present in model.invoke"
    blob = json.dumps(events, default=str)
    assert SENTINEL not in blob, f"{tag} redaction failed: sentinel secret leaked into the trace payload"


def _assert_attestation(payload: Dict[str, Any], events: List[Dict[str, Any]]) -> None:
    chain = (payload.get("attestation") or {}).get("chain") or {}
    raw = chain.get("events") or []
    envelopes = [
        AttestationEnvelope(
            hash=e["hash"],
            scope=HashScope(e["scope"]),
            previous_hash=e.get("previous_hash"),
        )
        for e in raw
    ]
    assert len(envelopes) == len(events), f"attestation chain has {len(envelopes)} envelopes for {len(events)} events"
    result = verify_chain(envelopes)
    assert result.valid, f"attestation chain invalid: {result.error}"


# --------------------------------------------------------------------------- #
# L2: upload, read back, tear down
# --------------------------------------------------------------------------- #
def _upload_capture(client: Any, payload: Dict[str, Any]) -> str:
    fd, path = tempfile.mkstemp(suffix=".json", prefix="ll_live_trace_")
    with os.fdopen(fd, "w") as fh:
        json.dump([payload], fh, default=str)
    try:
        result = client.traces.upload(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    assert result is not None and result.trace_ids, (
        "traces.upload returned no trace_ids — the backend rejected the instrumentation "
        "schema (it accepts bytes silently and returns empty on parse failure). This is the "
        "L2 schema-acceptance gate; a serializer to the documented trace schema may be needed."
    )
    assert len(result.trace_ids) == 1, f"expected 1 trace_id, got {len(result.trace_ids)}"
    return result.trace_ids[0]


def _poll_get(
    client: Any,
    trace_id: str,
    attempts: int = TRACE_READBACK_ATTEMPTS,
    delay: float = TRACE_READBACK_DELAY_S,
) -> Any:
    for _ in range(attempts):
        trace = client.traces.get(trace_id)
        if trace is not None:
            return trace
        time.sleep(delay)
    return None


def _reassert_roundtrip(trace: Any, local_events: List[Dict[str, Any]], variant: str) -> bool:
    """Bonus: if the backend echoes events in trace.data, sanity-check the count."""
    data = getattr(trace, "data", None) or {}
    echoed = data.get("events") if isinstance(data, dict) else None
    if not isinstance(echoed, list) or not echoed:
        return False
    if variant in ("default", "redaction", "async"):
        assert len(echoed) == len(local_events), f"round-trip event count {len(echoed)} != local {len(local_events)}"
    return True


def _teardown(client: Any, trace_id: str) -> None:
    # Keep traces in the backend for the manual L3 UI pass when requested.
    if os.environ.get("LAYERLENS_LIVE_KEEP_TRACES") == "1":
        return
    try:
        client.traces.delete(trace_id)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _payload(event: Dict[str, Any]) -> Dict[str, Any]:
    return event.get("payload") or {}


def _events_by_type(events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for e in events:
        out.setdefault(e.get("event_type", "?"), []).append(e)
    return out


def _sum_cost(cost_records: List[Dict[str, Any]]) -> float:
    total = 0.0
    for cr in cost_records:
        val = _payload(cr).get("cost_usd")
        if isinstance(val, (int, float)):
            total += float(val)
    return total


def _first_model(by_type: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
    for e in by_type.get("model.invoke", []):
        model = _payload(e).get("model")
        if model:
            return str(model)
    return None
