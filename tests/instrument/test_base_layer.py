"""Unit tests for the shared base layer of the Instrument package.

Covers :class:`BaseAdapter` (circuit breaker + capture gating + sink
dispatch), :class:`CaptureConfig` (layer enable/disable + presets),
:class:`AdapterRegistry` (singleton + lazy load), and the EventSink
hierarchy.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List
from unittest import mock

import pytest

from layerlens._compat.pydantic import model_dump
from layerlens.instrument.adapters._base import (
    DEFAULT_POLICY,
    ALWAYS_ENABLED_EVENT_TYPES,
    EventSink,
    AdapterInfo,
    BaseAdapter,
    AdapterHealth,
    AdapterStatus,
    CaptureConfig,
    TraceStoreSink,
    AdapterRegistry,
    ReplayableTrace,
    AdapterCapability,
    IngestionPipelineSink,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeStratix:
    """Records emit() calls for assertions."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: List[Any] = []
        self.fail = fail

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if self.fail:
            raise RuntimeError("simulated emit failure")
        self.calls.append((args, kwargs))


class _RecordingSink(EventSink):
    """Captures every (event_type, payload, ts) the adapter dispatches."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.flushed = 0
        self.closed = 0

    def send(self, event_type: str, payload: Dict[str, Any], timestamp_ns: int) -> None:
        self.events.append(
            {"event_type": event_type, "payload": payload, "timestamp_ns": timestamp_ns}
        )

    def flush(self) -> None:
        self.flushed += 1

    def close(self) -> None:
        self.closed += 1


class _MinimalAdapter(BaseAdapter):
    """Minimal concrete adapter used for testing the base class."""

    FRAMEWORK = "test"
    VERSION = "1.0.0"

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
            error_count=self._error_count,
            circuit_open=self._circuit_open,
        )

    def get_adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            name="MinimalAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            capabilities=[AdapterCapability.TRACE_TOOLS],
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        return ReplayableTrace(
            adapter_name="MinimalAdapter",
            framework=self.FRAMEWORK,
            trace_id="test-trace",
            events=list(self._trace_events),
        )


# ---------------------------------------------------------------------------
# CaptureConfig
# ---------------------------------------------------------------------------


class TestCaptureConfig:
    def test_defaults(self) -> None:
        c = CaptureConfig()
        assert c.l1_agent_io is True
        assert c.l3_model_metadata is True
        assert c.l2_agent_code is False  # off by default

    def test_minimal_preset(self) -> None:
        c = CaptureConfig.minimal()
        assert c.l1_agent_io is True
        assert c.l3_model_metadata is False
        assert c.l5a_tool_calls is False
        assert c.capture_content is False

    def test_standard_preset(self) -> None:
        c = CaptureConfig.standard()
        assert c.l1_agent_io is True
        assert c.l3_model_metadata is True
        assert c.l5a_tool_calls is True

    def test_full_preset(self) -> None:
        c = CaptureConfig.full()
        assert all(
            [
                c.l1_agent_io,
                c.l2_agent_code,
                c.l3_model_metadata,
                c.l4a_environment_config,
                c.l4b_environment_metrics,
                c.l5a_tool_calls,
                c.l5b_tool_logic,
                c.l5c_tool_environment,
                c.l6a_protocol_discovery,
                c.l6b_protocol_streams,
                c.l6c_protocol_lifecycle,
            ]
        )

    def test_is_layer_enabled_attribute(self) -> None:
        c = CaptureConfig.standard()
        assert c.is_layer_enabled("l1_agent_io")
        assert c.is_layer_enabled("l3_model_metadata")
        assert not c.is_layer_enabled("l2_agent_code")

    def test_is_layer_enabled_short_label(self) -> None:
        c = CaptureConfig.standard()
        assert c.is_layer_enabled("L1")
        assert c.is_layer_enabled("L3")
        assert c.is_layer_enabled("L5a")
        assert not c.is_layer_enabled("L2")

    def test_is_layer_enabled_event_type(self) -> None:
        c = CaptureConfig.standard()
        assert c.is_layer_enabled("agent.input")
        assert c.is_layer_enabled("model.invoke")
        assert c.is_layer_enabled("tool.call")
        assert not c.is_layer_enabled("agent.code")

    def test_cross_cutting_always_enabled(self) -> None:
        c = CaptureConfig.minimal()
        for et in ALWAYS_ENABLED_EVENT_TYPES:
            assert c.is_layer_enabled(et), f"{et} must always be enabled"

    def test_unknown_layer_disabled(self) -> None:
        c = CaptureConfig.full()
        assert c.is_layer_enabled("not_a_real_layer") is False


# ---------------------------------------------------------------------------
# BaseAdapter: emission, gating, circuit breaker
# ---------------------------------------------------------------------------


class TestBaseAdapterEmission:
    def test_emit_dict_event_dispatches_to_stratix(self) -> None:
        stratix = _FakeStratix()
        adapter = _MinimalAdapter(stratix=stratix, capture_config=CaptureConfig.full())

        adapter.emit_dict_event("model.invoke", {"model": "gpt-4o"})

        assert len(stratix.calls) == 1

    def test_emit_dict_event_records_for_replay(self) -> None:
        adapter = _MinimalAdapter(
            stratix=_FakeStratix(),
            capture_config=CaptureConfig.full(),
        )
        adapter.emit_dict_event("tool.call", {"tool_name": "calculator"})

        assert len(adapter._trace_events) == 1
        evt = adapter._trace_events[0]
        assert evt["event_type"] == "tool.call"
        assert evt["payload"]["tool_name"] == "calculator"
        assert evt["timestamp_ns"] > 0

    def test_capture_config_gates_disabled_layer(self) -> None:
        """A layer that is disabled must drop events silently."""
        stratix = _FakeStratix()
        adapter = _MinimalAdapter(
            stratix=stratix,
            capture_config=CaptureConfig(l3_model_metadata=False),
        )
        adapter.emit_dict_event("model.invoke", {"model": "gpt-4o"})
        assert stratix.calls == []
        assert adapter._trace_events == []

    def test_cross_cutting_event_bypasses_gating(self) -> None:
        """Cross-cutting events MUST emit even when most layers are off."""
        stratix = _FakeStratix()
        adapter = _MinimalAdapter(
            stratix=stratix,
            capture_config=CaptureConfig.minimal(),
        )
        adapter.emit_dict_event("cost.record", {"api_cost_usd": 0.01})
        adapter.emit_dict_event("policy.violation", {"violation_type": "safety"})
        assert len(stratix.calls) == 2

    def test_sink_receives_events(self) -> None:
        sink = _RecordingSink()
        adapter = _MinimalAdapter(
            stratix=_FakeStratix(),
            capture_config=CaptureConfig.full(),
            event_sinks=[sink],
        )
        adapter.emit_dict_event("model.invoke", {"model": "gpt-4o"})
        assert len(sink.events) == 1
        assert sink.events[0]["event_type"] == "model.invoke"

    def test_sink_failure_does_not_break_adapter(self) -> None:
        class _BrokenSink(EventSink):
            def send(
                self, event_type: str, payload: Dict[str, Any], timestamp_ns: int
            ) -> None:
                raise RuntimeError("broken")

            def flush(self) -> None:
                raise RuntimeError("broken flush")

            def close(self) -> None:
                raise RuntimeError("broken close")

        adapter = _MinimalAdapter(
            stratix=_FakeStratix(),
            capture_config=CaptureConfig.full(),
            event_sinks=[_BrokenSink()],
        )
        # Must not raise.
        adapter.emit_dict_event("model.invoke", {"model": "gpt-4o"})
        adapter._close_sinks()  # Must not raise even with broken sink.


class TestBaseAdapterTruncation:
    """The truncation policy is opt-in per-adapter (cross-poll audit §2.4).

    These tests assert the BaseAdapter wiring contract: when an
    adapter sets ``self._truncation_policy``, the dict-emit path must
    rewrite the payload and attach a ``_truncated_fields`` audit
    list. When no policy is set, payloads pass through unchanged.
    """

    def _make_adapter(
        self, stratix: _FakeStratix, policy: Any = None
    ) -> _MinimalAdapter:
        adapter = _MinimalAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        if policy is not None:
            adapter._truncation_policy = policy
        return adapter

    def test_no_policy_passes_payload_through_unchanged(self) -> None:
        """Without a policy set, payloads are emitted byte-for-byte."""
        stratix = _FakeStratix()
        adapter = self._make_adapter(stratix, policy=None)
        large = "z" * 50000
        adapter.emit_dict_event("model.invoke", {"prompt": large})
        # Stratix received the full payload.
        emitted = stratix.calls[0][0][1]
        assert emitted["prompt"] == large
        assert "_truncated_fields" not in emitted

    def test_policy_truncates_long_prompt(self) -> None:
        """With DEFAULT_POLICY, ``prompt`` longer than 4096 is truncated."""
        stratix = _FakeStratix()
        adapter = self._make_adapter(stratix, policy=DEFAULT_POLICY)
        adapter.emit_dict_event("model.invoke", {"prompt": "p" * 10000})
        emitted = stratix.calls[0][0][1]
        assert emitted["prompt"].startswith("p" * 4096)
        assert "more chars truncated" in emitted["prompt"]
        # Audit list MUST be attached.
        audit = emitted["_truncated_fields"]
        assert isinstance(audit, list)
        assert any("prompt:chars-10000->4096" in entry for entry in audit)

    def test_policy_drops_screenshot_with_hash_reference(self) -> None:
        """Binary-ish fields are replaced with deterministic hash refs."""
        stratix = _FakeStratix()
        adapter = self._make_adapter(stratix, policy=DEFAULT_POLICY)
        adapter.emit_dict_event(
            "tool.call",
            {"tool_name": "browser_use", "screenshot": b"PNG_BYTES" * 1000},
        )
        emitted = stratix.calls[0][0][1]
        assert isinstance(emitted["screenshot"], str)
        assert emitted["screenshot"].startswith("<dropped:screenshot:sha256:")
        audit = emitted["_truncated_fields"]
        assert any("screenshot:dropped" in entry for entry in audit)

    def test_policy_short_payload_no_audit_attached(self) -> None:
        """If no field exceeds its cap, no ``_truncated_fields`` is attached."""
        stratix = _FakeStratix()
        adapter = self._make_adapter(stratix, policy=DEFAULT_POLICY)
        adapter.emit_dict_event(
            "model.invoke", {"prompt": "short", "tool_output": "ok"}
        )
        emitted = stratix.calls[0][0][1]
        assert emitted == {"prompt": "short", "tool_output": "ok"}
        assert "_truncated_fields" not in emitted

    def test_policy_does_not_mutate_caller_payload(self) -> None:
        """The original payload dict supplied by the adapter MUST NOT be mutated."""
        stratix = _FakeStratix()
        adapter = self._make_adapter(stratix, policy=DEFAULT_POLICY)
        payload = {"prompt": "p" * 10000, "tokens_prompt": 1234}
        snapshot = dict(payload)
        snapshot["prompt"] = payload["prompt"]
        adapter.emit_dict_event("model.invoke", payload)
        assert payload == snapshot

    def test_custom_policy_overrides_caps(self) -> None:
        """Adapters can pass a stricter policy via ``with_overrides``."""
        stratix = _FakeStratix()
        strict = DEFAULT_POLICY.with_overrides(field_caps={"prompt": 16})
        adapter = self._make_adapter(stratix, policy=strict)
        adapter.emit_dict_event("model.invoke", {"prompt": "p" * 100})
        emitted = stratix.calls[0][0][1]
        assert emitted["prompt"].startswith("p" * 16)
        assert "more chars truncated" in emitted["prompt"]

    def test_policy_applies_to_nested_dict(self) -> None:
        """Nested dicts: each field is checked against the policy."""
        stratix = _FakeStratix()
        adapter = self._make_adapter(stratix, policy=DEFAULT_POLICY)
        adapter.emit_dict_event(
            "agent.input",
            {
                "framework": "test",
                "metadata": {"completion": "c" * 10000},
            },
        )
        emitted = stratix.calls[0][0][1]
        assert emitted["metadata"]["completion"].startswith("c" * 4096)

    def test_truncation_failure_falls_back_to_raw_payload(self) -> None:
        """If the policy raises (defensive), emit the raw payload."""

        class _BrokenPolicy:
            def cap_for(self, _name: str) -> int:
                raise RuntimeError("simulated policy failure")

        stratix = _FakeStratix()
        adapter = self._make_adapter(stratix, policy=_BrokenPolicy())
        adapter.emit_dict_event("model.invoke", {"prompt": "p" * 100})
        # Stratix received the original payload despite policy failure.
        emitted = stratix.calls[0][0][1]
        assert emitted == {"prompt": "p" * 100}

    def test_policy_truncates_html_for_browser_use_pattern(self) -> None:
        """Browser_use HTML/DOM payloads get a 16 KiB cap."""
        stratix = _FakeStratix()
        adapter = self._make_adapter(stratix, policy=DEFAULT_POLICY)
        big_html = "<div>x</div>" * 10000  # > 100 KB
        adapter.emit_dict_event("tool.call", {"tool_name": "browse", "html": big_html})
        emitted = stratix.calls[0][0][1]
        assert isinstance(emitted["html"], str)
        # Capped at 16384.
        assert len(emitted["html"]) <= 16384 + 100  # cap + suffix
        audit = emitted["_truncated_fields"]
        assert any("html:chars-" in entry for entry in audit)


class TestCircuitBreaker:
    def test_successful_emit_resets_error_count(self) -> None:
        stratix = _FakeStratix()
        adapter = _MinimalAdapter(stratix=stratix, capture_config=CaptureConfig.full())

        # Manually set degraded state.
        adapter._error_count = 3
        adapter._status = AdapterStatus.DEGRADED

        adapter.emit_dict_event("model.invoke", {"model": "gpt-4o"})

        assert adapter._error_count == 0
        assert adapter._status == AdapterStatus.HEALTHY

    def test_emit_failures_open_circuit(self) -> None:
        stratix = _FakeStratix(fail=True)
        adapter = _MinimalAdapter(stratix=stratix, capture_config=CaptureConfig.full())

        # Threshold is 10 — trigger 10 failures.
        for _ in range(10):
            adapter.emit_dict_event("model.invoke", {"model": "gpt-4o"})

        assert adapter._circuit_open is True
        assert adapter._status == AdapterStatus.ERROR

    def test_circuit_drops_events_when_open(self) -> None:
        stratix = _FakeStratix(fail=True)
        adapter = _MinimalAdapter(stratix=stratix, capture_config=CaptureConfig.full())

        for _ in range(10):
            adapter.emit_dict_event("model.invoke", {"model": "gpt-4o"})
        assert adapter._circuit_open

        # Now switch stratix to non-failing; circuit still drops events.
        stratix.fail = False
        before = len(stratix.calls)
        adapter.emit_dict_event("model.invoke", {"model": "gpt-4o"})
        assert len(stratix.calls) == before  # dropped

    def test_circuit_recovers_after_cooldown(self) -> None:
        stratix = _FakeStratix(fail=True)
        adapter = _MinimalAdapter(stratix=stratix, capture_config=CaptureConfig.full())

        for _ in range(10):
            adapter.emit_dict_event("model.invoke", {})
        assert adapter._circuit_open

        # Force cooldown to elapse.
        adapter._circuit_opened_at = time.monotonic() - 100.0
        stratix.fail = False
        adapter.emit_dict_event("model.invoke", {"model": "gpt-4o"})

        assert adapter._circuit_open is False


class TestBaseAdapterLifecycle:
    def test_default_construction_uses_null_stratix(self) -> None:
        adapter = _MinimalAdapter()
        assert adapter.has_stratix is False
        # Emission with null sentinel must not raise.
        adapter.emit_dict_event("model.invoke", {"model": "gpt-4o"})

    def test_connect_sets_healthy(self) -> None:
        adapter = _MinimalAdapter()
        assert adapter.is_connected is False
        adapter.connect()
        assert adapter.is_connected is True
        assert adapter.status == AdapterStatus.HEALTHY

    def test_disconnect_sets_disconnected(self) -> None:
        adapter = _MinimalAdapter()
        adapter.connect()
        adapter.disconnect()
        assert adapter.is_connected is False
        assert adapter.status == AdapterStatus.DISCONNECTED

    def test_replay_serialization(self) -> None:
        adapter = _MinimalAdapter(
            stratix=_FakeStratix(),
            capture_config=CaptureConfig.full(),
        )
        adapter.emit_dict_event("model.invoke", {"model": "gpt-4o"})
        rt = adapter.serialize_for_replay()
        assert rt.framework == "test"
        assert len(rt.events) == 1


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------


class TestTraceStoreSink:
    def test_send_writes_events_with_increasing_sequence(self) -> None:
        store = mock.MagicMock()
        store.get_trace.return_value = None
        sink = TraceStoreSink(store=store, trace_id="t1")

        sink.send("model.invoke", {"model": "gpt-4o"}, time.time_ns())
        sink.send("tool.call", {"tool_name": "calc"}, time.time_ns())

        # store_trace called once at construction.
        assert store.store_trace.call_count == 1
        # store_event called once per send.
        assert store.store_event.call_count == 2

        records = [c.args[0] for c in store.store_event.call_args_list]
        assert records[0]["sequence_id"] == 1
        assert records[1]["sequence_id"] == 2

    def test_close_finalizes_trace(self) -> None:
        store = mock.MagicMock()
        store.get_trace.return_value = None
        sink = TraceStoreSink(store=store)

        sink.send("model.invoke", {}, time.time_ns())
        sink.close()

        # Either get_trace returned None (then update_trace_status) OR there's
        # an existing trace to mutate. With None, expect update_trace_status.
        store.update_trace_status.assert_called_once()

    def test_close_idempotent(self) -> None:
        store = mock.MagicMock()
        store.get_trace.return_value = None
        sink = TraceStoreSink(store=store)
        sink.close()
        sink.close()  # must not raise


class TestIngestionPipelineSink:
    def test_immediate_mode_calls_pipeline_per_event(self) -> None:
        pipeline = mock.MagicMock()
        sink = IngestionPipelineSink(pipeline=pipeline, tenant_id="org-123")

        sink.send("model.invoke", {"model": "gpt-4o"}, time.time_ns())
        sink.send("tool.call", {"tool_name": "calc"}, time.time_ns())

        assert pipeline.ingest.call_count == 2
        for call in pipeline.ingest.call_args_list:
            assert call.kwargs["tenant_id"] == "org-123"

    def test_buffered_mode_defers_until_flush(self) -> None:
        pipeline = mock.MagicMock()
        sink = IngestionPipelineSink(pipeline=pipeline, buffered=True)

        sink.send("model.invoke", {}, time.time_ns())
        sink.send("tool.call", {}, time.time_ns())

        assert pipeline.ingest.call_count == 0
        sink.flush()
        assert pipeline.ingest.call_count == 1
        # Single batched ingest with 2 events.
        events = pipeline.ingest.call_args.args[0]
        assert len(events) == 2

    def test_close_flushes_buffer(self) -> None:
        pipeline = mock.MagicMock()
        sink = IngestionPipelineSink(pipeline=pipeline, buffered=True)
        sink.send("model.invoke", {}, time.time_ns())
        sink.close()
        assert pipeline.ingest.call_count == 1


# ---------------------------------------------------------------------------
# AdapterRegistry
# ---------------------------------------------------------------------------


class TestAdapterRegistry:
    def setup_method(self) -> None:
        AdapterRegistry.reset()

    def teardown_method(self) -> None:
        AdapterRegistry.reset()

    def test_singleton(self) -> None:
        a = AdapterRegistry()
        b = AdapterRegistry()
        assert a is b

    def test_register_requires_framework_attr(self) -> None:
        class _NoFramework(BaseAdapter):
            def connect(self) -> None: ...
            def disconnect(self) -> None: ...
            def health_check(self) -> AdapterHealth:
                return AdapterHealth(
                    status=AdapterStatus.HEALTHY,
                    framework_name="x",
                    adapter_version="0.0.0",
                )
            def get_adapter_info(self) -> AdapterInfo:
                return AdapterInfo(name="x", version="0.0.0", framework="x")
            def serialize_for_replay(self) -> ReplayableTrace:
                return ReplayableTrace(adapter_name="x", framework="x", trace_id="x")

        registry = AdapterRegistry()
        with pytest.raises(ValueError):
            registry.register(_NoFramework)

    def test_register_and_get(self) -> None:
        registry = AdapterRegistry()
        registry.register(_MinimalAdapter)
        adapter = registry.get("test")
        assert isinstance(adapter, _MinimalAdapter)
        assert adapter.is_connected is True

    def test_get_unknown_framework_raises(self) -> None:
        registry = AdapterRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent_framework_xyz")

    def test_list_available(self) -> None:
        registry = AdapterRegistry()
        registry.register(_MinimalAdapter)
        infos = registry.list_available()
        assert any(i.framework == "test" for i in infos)

    def test_auto_detect_returns_list(self) -> None:
        registry = AdapterRegistry()
        result = registry.auto_detect()
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Pydantic v1/v2 compat
# ---------------------------------------------------------------------------


class TestSinkManagementAPI:
    """``add_sink`` / ``remove_sink`` / ``sinks`` are the public API."""

    def test_add_sink_registers(self) -> None:
        adapter = _MinimalAdapter(stratix=_FakeStratix(), capture_config=CaptureConfig.full())
        sink = _RecordingSink()
        adapter.add_sink(sink)
        assert sink in adapter.sinks

    def test_remove_sink_returns_true_when_present(self) -> None:
        adapter = _MinimalAdapter()
        sink = _RecordingSink()
        adapter.add_sink(sink)
        assert adapter.remove_sink(sink) is True
        assert sink not in adapter.sinks

    def test_remove_sink_returns_false_when_absent(self) -> None:
        adapter = _MinimalAdapter()
        sink = _RecordingSink()
        # Never added.
        assert adapter.remove_sink(sink) is False

    def test_sinks_is_defensive_copy(self) -> None:
        adapter = _MinimalAdapter()
        sink = _RecordingSink()
        adapter.add_sink(sink)
        snapshot = adapter.sinks
        snapshot.clear()  # mutate the snapshot
        # Adapter's actual list is untouched.
        assert sink in adapter.sinks


class TestModelDump:
    def test_model_dump_handles_dict(self) -> None:
        assert model_dump({"a": 1}) == {"a": 1}

    def test_model_dump_handles_pydantic_model(self) -> None:
        c = CaptureConfig.minimal()
        out = model_dump(c)
        assert isinstance(out, dict)
        assert out["l1_agent_io"] is True

    def test_model_dump_handles_unknown(self) -> None:
        assert model_dump("a string") == {"raw": "a string"}
