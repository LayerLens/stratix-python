"""OTLP wire-surface tests for the OpenInference adapter (LAY-3622 Cluster B).

The adapter always understood per-SPAN OTLP spellings (``traceId``,
``startTimeUnixNano``, AnyValue KV lists, ``STATUS_CODE_ERROR``) but had no
surface for the ENVELOPE — a caller had to walk
``resourceSpans -> scopeSpans -> spans`` itself, which the conformance lane
literally did. These tests pin the shipped decoder:

* the envelope walk, including the legacy ``instrumentationLibrarySpans`` spelling
  and skip-don't-abort on a malformed member;
* id decoding for BOTH real-world encodings — plain hex, and the base64 that
  proto3-JSON actually specifies for a ``bytes`` field;
* resource-attribute merge with SPAN-WINS precedence (the thing that puts
  ``service.name`` / ``deployment.environment`` in reach — brief D1/L4a);
* ``OpenInferenceOTLPBridge``, mirroring ateam's class shape, and its protobuf
  path degrading to a clean ``ImportError``;
* one malformed span no longer aborting the rest of a batch (B4).

Runs in plain CI: no credentials, no network, no OTel SDK required for the JSON
half (the protobuf half is skipped when ``opentelemetry-proto`` is absent).
"""

from __future__ import annotations

import os
import json
import base64
import builtins
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.frameworks.openinference import (
    OpenInferenceAdapter,
    OpenInferenceOTLPBridge,
    otlp_json_to_span_records,
    otlp_protobuf_to_span_records,
    environment_config_from_resource,
)

from .conftest import find_event, find_events, capture_framework_trace  # noqa: F401

TRACE_HEX = "aa" * 16
SPAN_HEX = "bb" * 8
PARENT_HEX = "cc" * 8


def _kv(key: str, value: Any) -> Dict[str, Any]:
    """An OTLP AnyValue KeyValue in its real proto3-JSON encoding."""
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        # proto3-JSON encodes int64 as a STRING.
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": str(value)}}


def _span(
    *,
    kind: str = "LLM",
    name: str = "openai.chat",
    span_id: str = SPAN_HEX,
    trace_id: str = TRACE_HEX,
    attrs: Any = None,
    status: Any = None,
) -> Dict[str, Any]:
    span: Dict[str, Any] = {
        "name": name,
        "traceId": trace_id,
        "spanId": span_id,
        "startTimeUnixNano": "1700000000000000000",
        "endTimeUnixNano": "1700000001000000000",
        "attributes": [_kv("openinference.span.kind", kind), *(attrs or [])],
    }
    if status is not None:
        span["status"] = status
    return span


def _request(
    spans: List[Dict[str, Any]], *, resource_attrs: Any = None, scope_key: str = "scopeSpans"
) -> Dict[str, Any]:
    resource_spans: Dict[str, Any] = {scope_key: [{"spans": spans}]}
    if resource_attrs is not None:
        resource_spans["resource"] = {"attributes": resource_attrs}
    return {"resourceSpans": [resource_spans]}


class _Client:
    """Minimal upload double; the real Mock client fixture is used where the
    captured trace matters."""

    class traces:
        @staticmethod
        def upload(*args: Any, **kwargs: Any) -> Any:
            return type("R", (), {"trace_ids": ["t1"]})()


class TestEnvelopeWalk:
    def test_flattens_every_span_across_resource_and_scope_blocks(self) -> None:
        request = {
            "resourceSpans": [
                {"scopeSpans": [{"spans": [_span(span_id="01" * 8)]}, {"spans": [_span(span_id="02" * 8)]}]},
                {"scopeSpans": [{"spans": [_span(span_id="03" * 8), _span(span_id="04" * 8)]}]},
            ]
        }
        records = otlp_json_to_span_records(request)
        assert [r["span_id"] for r in records] == ["01" * 8, "02" * 8, "03" * 8, "04" * 8]
        assert all(r["span_kind"] == "LLM" for r in records)

    def test_snake_case_envelope_keys_are_accepted(self) -> None:
        request = {"resource_spans": [{"scope_spans": [{"spans": [_span()]}]}]}
        assert len(otlp_json_to_span_records(request)) == 1

    def test_legacy_instrumentation_library_spans_spelling(self) -> None:
        # OTLP <= 0.19; still emitted by older collectors. Dropping it would
        # silently ingest nothing from such an exporter.
        request = _request([_span()], scope_key="instrumentationLibrarySpans")
        assert len(otlp_json_to_span_records(request)) == 1

    def test_an_empty_or_absent_envelope_yields_no_records(self) -> None:
        assert otlp_json_to_span_records({}) == []
        assert otlp_json_to_span_records({"resourceSpans": []}) == []
        assert otlp_json_to_span_records({"resourceSpans": [{}]}) == []

    @pytest.mark.parametrize(
        "request_obj",
        [
            {"resourceSpans": "not-a-list"},
            {"resourceSpans": ["not-a-dict"]},
            {"resourceSpans": [{"scopeSpans": "not-a-list"}]},
            {"resourceSpans": [{"scopeSpans": ["not-a-dict"]}]},
            {"resourceSpans": [{"scopeSpans": [{"spans": "not-a-list"}]}]},
            {"resourceSpans": [{"scopeSpans": [{"spans": ["not-a-dict"]}]}]},
        ],
        ids=["rs-scalar", "rs-item", "ss-scalar", "ss-item", "spans-scalar", "spans-item"],
    )
    def test_a_malformed_member_is_skipped_not_raised(self, request_obj: Dict[str, Any]) -> None:
        # An export is untrusted input. One bad member must not lose the request.
        assert otlp_json_to_span_records(request_obj) == []

    def test_a_malformed_member_does_not_lose_its_healthy_siblings(self) -> None:
        request = {
            "resourceSpans": [
                "garbage",
                {"scopeSpans": [{"spans": [_span(span_id="0a" * 8)]}]},
                {"scopeSpans": ["garbage", {"spans": ["garbage", _span(span_id="0b" * 8)]}]},
            ]
        }
        assert [r["span_id"] for r in otlp_json_to_span_records(request)] == ["0a" * 8, "0b" * 8]


class TestIdDecoding:
    def test_hex_ids_pass_through_lowercased(self) -> None:
        record = otlp_json_to_span_records(_request([_span(trace_id="AA" * 16, span_id="BB" * 8)]))[0]
        assert record["trace_id"] == "aa" * 16
        assert record["span_id"] == "bb" * 8

    def test_base64_ids_are_decoded_to_hex(self) -> None:
        # proto3-JSON encodes a bytes field as base64; this is what a
        # spec-compliant OTLP/HTTP exporter actually sends.
        span = _span()
        span["traceId"] = base64.b64encode(bytes.fromhex(TRACE_HEX)).decode()
        span["spanId"] = base64.b64encode(bytes.fromhex(SPAN_HEX)).decode()
        span["parentSpanId"] = base64.b64encode(bytes.fromhex(PARENT_HEX)).decode()
        record = otlp_json_to_span_records(_request([span]))[0]
        assert record["trace_id"] == TRACE_HEX
        assert record["span_id"] == SPAN_HEX
        assert record["parent_span_id"] == PARENT_HEX

    def test_both_encodings_of_the_same_span_agree(self) -> None:
        # The whole point: the same span must correlate whichever encoding the
        # exporter chose. A silent mismatch here splits one trace into two.
        hex_span = _span()
        b64_span = _span()
        b64_span["traceId"] = base64.b64encode(bytes.fromhex(TRACE_HEX)).decode()
        b64_span["spanId"] = base64.b64encode(bytes.fromhex(SPAN_HEX)).decode()
        a = otlp_json_to_span_records(_request([hex_span]))[0]
        b = otlp_json_to_span_records(_request([b64_span]))[0]
        assert (a["trace_id"], a["span_id"]) == (b["trace_id"], b["span_id"])

    def test_an_absent_parent_stays_none(self) -> None:
        assert otlp_json_to_span_records(_request([_span()]))[0]["parent_span_id"] is None


class TestResourceAttributeMerge:
    def test_resource_attributes_reach_every_span(self) -> None:
        request = _request(
            [_span(span_id="01" * 8), _span(span_id="02" * 8)],
            resource_attrs=[_kv("service.name", "checkout"), _kv("deployment.environment", "production")],
        )
        for record in otlp_json_to_span_records(request):
            assert record["attributes"]["service.name"] == "checkout"
            assert record["attributes"]["deployment.environment"] == "production"

    def test_span_attributes_win_on_conflict(self) -> None:
        request = _request(
            [_span(attrs=[_kv("service.name", "from-span")])],
            resource_attrs=[_kv("service.name", "from-resource")],
        )
        assert otlp_json_to_span_records(request)[0]["attributes"]["service.name"] == "from-span"

    def test_a_resource_without_attributes_is_harmless(self) -> None:
        request = {"resourceSpans": [{"resource": {}, "scopeSpans": [{"spans": [_span()]}]}]}
        assert otlp_json_to_span_records(request)[0]["attributes"]["openinference.span.kind"] == "LLM"


class TestAttributeValueDecoding:
    def test_int_double_bool_and_array_values(self) -> None:
        attrs = [
            _kv("llm.token_count.total", 1500),
            _kv("llm.temperature", 0.5),
            _kv("llm.is_streaming", True),
            {"key": "tags", "value": {"arrayValue": {"values": [{"stringValue": "a"}, {"stringValue": "b"}]}}},
        ]
        got = otlp_json_to_span_records(_request([_span(attrs=attrs)]))[0]["attributes"]
        # int64 arrives as proto3-JSON's string form; _as_int normalises numeric
        # fields downstream (see the KNOWN DIVERGENCES note on the Go bridge).
        assert got["llm.token_count.total"] == "1500"
        assert got["llm.temperature"] == 0.5
        assert got["llm.is_streaming"] is True
        assert got["tags"] == ["a", "b"]

    def test_a_kvlist_value_is_unwrapped_to_a_dict(self) -> None:
        # Go's bridge maps KvlistValue to a real map (otlp-ingest
        # convert.go:340-341). Leaving the raw OTLP wrapper in place shipped an
        # internal wire structure into the event payload.
        attrs = [
            {
                "key": "metadata",
                "value": {"kvlistValue": {"values": [{"key": "tier", "value": {"stringValue": "gold"}}]}},
            }
        ]
        got = otlp_json_to_span_records(_request([_span(attrs=attrs)]))[0]["attributes"]
        assert got["metadata"] == {"tier": "gold"}


class TestStatusMapping:
    @pytest.mark.parametrize(
        "status,expected",
        [
            ({"code": 2}, "ERROR"),
            ({"code": 1}, "OK"),
            ({"code": 0}, "UNSET"),
            ({"code": "STATUS_CODE_ERROR"}, "ERROR"),
            ({"code": "STATUS_CODE_OK"}, "OK"),
            ({"code": "ERROR"}, "ERROR"),
            (None, None),
        ],
        ids=["int-2", "int-1", "int-0", "name-error", "name-ok", "bare-error", "absent"],
    )
    def test_every_status_spelling_resolves(self, status: Any, expected: Any) -> None:
        record = otlp_json_to_span_records(_request([_span(status=status)]))[0]
        assert record["status"] == expected

    def test_the_status_message_survives_the_flattener(self) -> None:
        # It reaches the payload as the honest error text; losing it degrades a
        # real upstream error to the generic "span status ERROR" backstop.
        record = otlp_json_to_span_records(
            _request([_span(status={"code": 2, "message": "rate limited by upstream"})])
        )[0]
        assert record["status"] == "ERROR"
        assert record["status_message"] == "rate limited by upstream"

    def test_the_status_message_survives_all_the_way_into_the_event(self, mock_client: Mock) -> None:
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        OpenInferenceOTLPBridge(adapter).ingest_otlp_json(
            _request([_span(status={"code": 2, "message": "rate limited by upstream"})])
        )
        adapter.flush()
        payload = find_event(uploaded["events"], "model.invoke")["payload"]
        assert payload["error"] == "rate limited by upstream"


class TestBridge:
    def test_ingest_otlp_json_returns_the_event_count(self) -> None:
        adapter = OpenInferenceAdapter(_Client())
        adapter.connect()
        emitted = OpenInferenceOTLPBridge(adapter).ingest_otlp_json(
            _request(
                [
                    _span(span_id="01" * 8, attrs=[_kv("llm.model_name", "gpt-4o")]),
                    _span(span_id="02" * 8, kind="TOOL", attrs=[_kv("tool.name", "search")]),
                ]
            )
        )
        assert emitted >= 2

    def test_the_bridge_exposes_its_adapter(self) -> None:
        adapter = OpenInferenceAdapter(_Client())
        assert OpenInferenceOTLPBridge(adapter).adapter is adapter

    def test_an_empty_request_emits_nothing(self) -> None:
        adapter = OpenInferenceAdapter(_Client())
        adapter.connect()
        assert OpenInferenceOTLPBridge(adapter).ingest_otlp_json({}) == 0

    def test_a_whole_envelope_becomes_real_events(self, mock_client: Mock) -> None:
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        OpenInferenceOTLPBridge(adapter).ingest_otlp_json(
            _request(
                [
                    _span(
                        attrs=[
                            _kv("llm.model_name", "gpt-4o"),
                            _kv("llm.token_count.prompt", 1000),
                            _kv("llm.token_count.completion", 500),
                            _kv("llm.token_count.total", 1500),
                        ]
                    )
                ],
                resource_attrs=[_kv("service.name", "checkout")],
            )
        )
        adapter.flush()
        invoke = find_event(uploaded["events"], "model.invoke")["payload"]
        assert invoke["model"] == "gpt-4o"
        # FLAT token canon survives the envelope path (AC1a).
        assert invoke["prompt_tokens"] == 1000
        assert invoke["completion_tokens"] == 500
        assert invoke["total_tokens"] == 1500
        assert find_event(uploaded["events"], "cost.record")["payload"]["cost_usd"] > 0

    def test_retried_spans_are_NOT_deduplicated(self) -> None:
        """DOCUMENTED LIMITATION, ateam parity (openinference_bridge.py:241-253).

        OTLP exporters retry, so a redelivered export re-ingests its spans. ateam
        carries dedup only on its SERVER-side entry point
        (``otlp_request_to_ingest_events``).

        CORRECTED under LAY-3622 F5: this docstring used to say the role "belongs to
        atlas-app's ``apps/otlp-ingest``", which reads as "the platform covers it".
        It does not cover THIS path. That service really does dedup a re-sent span_id
        (``ingest/merge.go`` / ``ingest/writer.go``), but it guards the OTLP endpoint,
        whereas this bridge normalises spans into events and uploads them via the
        traces API (``traces_create.go``) — which has no span-level dedup or upsert.
        A caller needing at-most-once must de-duplicate before handing the export
        over; nothing behind it will.

        Dedup here remains a standing NO (ateam parity). This test pins the CURRENT
        behaviour so a future change to it is a deliberate, visible decision rather
        than an accident.
        """
        adapter = OpenInferenceAdapter(_Client())
        adapter.connect()
        bridge = OpenInferenceOTLPBridge(adapter)
        request = _request([_span(attrs=[_kv("llm.model_name", "gpt-4o")])])
        first = bridge.ingest_otlp_json(request)
        second = bridge.ingest_otlp_json(request)
        assert first == second and first > 0, "a retried export is re-ingested, not deduplicated"


class TestProtobufPath:
    def test_protobuf_roundtrip(self) -> None:
        pytest.importorskip("opentelemetry.proto")
        from opentelemetry.proto.trace.v1.trace_pb2 import Span, Status, ScopeSpans, ResourceSpans
        from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

        span = Span(
            name="openai.chat",
            trace_id=bytes.fromhex(TRACE_HEX),
            span_id=bytes.fromhex(SPAN_HEX),
            start_time_unix_nano=1700000000000000000,
            end_time_unix_nano=1700000001000000000,
            attributes=[
                KeyValue(key="openinference.span.kind", value=AnyValue(string_value="LLM")),
                KeyValue(key="llm.model_name", value=AnyValue(string_value="gpt-4o")),
                KeyValue(key="llm.token_count.prompt", value=AnyValue(int_value=1000)),
            ],
            status=Status(code=Status.STATUS_CODE_OK),
        )
        request = ExportTraceServiceRequest(resource_spans=[ResourceSpans(scope_spans=[ScopeSpans(spans=[span])])])
        records = otlp_protobuf_to_span_records(request.SerializeToString())
        assert len(records) == 1
        record = records[0]
        assert record["span_kind"] == "LLM"
        assert record["trace_id"] == TRACE_HEX
        assert record["span_id"] == SPAN_HEX
        assert record["status"] == "OK"
        # protobuf carries int64 natively, so no string coercion here — this is
        # the documented Python-JSON / Go-protobuf asymmetry, resolved.
        assert record["attributes"]["llm.token_count.prompt"] == 1000

    def test_protobuf_resource_attributes_merge_with_span_winning(self) -> None:
        pytest.importorskip("opentelemetry.proto")
        from opentelemetry.proto.trace.v1.trace_pb2 import Span, ScopeSpans, ResourceSpans
        from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
        from opentelemetry.proto.resource.v1.resource_pb2 import Resource
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

        span = Span(
            name="s",
            trace_id=bytes.fromhex(TRACE_HEX),
            span_id=bytes.fromhex(SPAN_HEX),
            attributes=[
                KeyValue(key="openinference.span.kind", value=AnyValue(string_value="LLM")),
                KeyValue(key="service.name", value=AnyValue(string_value="from-span")),
            ],
        )
        request = ExportTraceServiceRequest(
            resource_spans=[
                ResourceSpans(
                    resource=Resource(
                        attributes=[
                            KeyValue(key="service.name", value=AnyValue(string_value="from-resource")),
                            KeyValue(key="deployment.environment", value=AnyValue(string_value="production")),
                        ]
                    ),
                    scope_spans=[ScopeSpans(spans=[span])],
                )
            ]
        )
        attrs = otlp_protobuf_to_span_records(request.SerializeToString())[0]["attributes"]
        assert attrs["service.name"] == "from-span"
        assert attrs["deployment.environment"] == "production"

    def test_a_missing_opentelemetry_proto_raises_a_clean_ImportError(self, monkeypatch: Any) -> None:
        # The SDK deliberately does NOT declare opentelemetry-proto (the
        # openinference extra stays empty). A caller must get an ImportError it can
        # catch to fall back to the JSON path — not an AttributeError from halfway
        # through the parse.
        real_import = builtins.__import__

        def _blocked(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith("opentelemetry.proto"):
                raise ImportError("No module named 'opentelemetry.proto'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked)
        with pytest.raises(ImportError):
            otlp_protobuf_to_span_records(b"")


class TestOneBadSpanCannotStrandTheBatch:
    """B4: ``_record_from_dict`` was not exception-wrapped while its sibling
    ``_record_from_otel`` was. One malformed dict raised out of ``ingest_spans``,
    aborting every REMAINING span and stranding the already-ingested ones in an
    unflushed collector — unacceptable once a whole OTLP export feeds through one
    call.

    Bite proof: remove the try/except from ``_record_from_dict`` and these fail
    with the raised exception instead of an event count.
    """

    class _Exploding(dict):
        """A dict whose attribute access raises, as a corrupt member would."""

        def get(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("corrupt span member")

    def test_a_raising_span_does_not_abort_its_siblings(self) -> None:
        adapter = OpenInferenceAdapter(_Client())
        adapter.connect()
        spans: List[Any] = [
            {
                "attributes": [_kv("openinference.span.kind", "LLM"), _kv("llm.model_name", "gpt-4o")],
                "name": "ok-1",
                "traceId": TRACE_HEX,
                "spanId": "01" * 8,
            },
            self._Exploding(),
            {
                "attributes": [_kv("openinference.span.kind", "TOOL"), _kv("tool.name", "search")],
                "name": "ok-2",
                "traceId": TRACE_HEX,
                "spanId": "02" * 8,
            },
        ]
        emitted = adapter.ingest_spans(spans)
        assert emitted >= 2, "a malformed span aborted the rest of the batch"

    def test_the_healthy_spans_still_reach_the_uploaded_trace(self, mock_client: Mock) -> None:
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        adapter.ingest_spans(
            [
                {
                    "attributes": [_kv("openinference.span.kind", "LLM"), _kv("llm.model_name", "gpt-4o")],
                    "name": "ok-1",
                    "traceId": TRACE_HEX,
                    "spanId": "01" * 8,
                },
                self._Exploding(),
                {
                    "attributes": [_kv("openinference.span.kind", "TOOL"), _kv("tool.name", "search")],
                    "name": "ok-2",
                    "traceId": TRACE_HEX,
                    "spanId": "02" * 8,
                },
            ]
        )
        adapter.flush()
        types = {e["event_type"] for e in uploaded["events"]}
        assert "model.invoke" in types and "tool.call" in types, (
            "the already-ingested spans were stranded in an unflushed collector"
        )


class TestEnvironmentConfigFromResource:
    """L4a (LAY-3622 D1): the OTLP Resource block IS environment material.

    Until the envelope decoder landed the SDK could not see a Resource block at
    all, so an OpenInference trace lost ``service.name`` entirely. This lane pins
    the new ``environment.config`` from day one, because it is an SDK-only event
    type sitting OUTSIDE the Go oracle — exactly the position ``cost.record`` was
    in when the fabricated-cost defect shipped unnoticed.
    """

    def test_known_resource_keys_are_lifted(self) -> None:
        payload = environment_config_from_resource(
            {
                "service.name": "checkout",
                "service.version": "1.4.2",
                "deployment.environment": "production",
                "cloud.provider": "aws",
                "cloud.region": "us-east-1",
            }
        )
        assert payload == {
            "service_name": "checkout",
            "service_version": "1.4.2",
            "environment": "production",
            "cloud_provider": "aws",
            "region": "us-east-1",
            "framework": "openinference",
        }

    def test_both_deployment_environment_spellings_resolve(self) -> None:
        # OTel renamed the attribute in semconv 1.27; exporters emit either.
        assert environment_config_from_resource({"deployment.environment": "prod"})["environment"] == "prod"
        assert environment_config_from_resource({"deployment.environment.name": "prod"})["environment"] == "prod"

    def test_the_legacy_spelling_wins_when_both_are_present(self) -> None:
        payload = environment_config_from_resource(
            {"deployment.environment": "production", "deployment.environment.name": "prod"}
        )
        assert payload["environment"] == "production"

    def test_a_resource_with_no_known_keys_yields_nothing(self) -> None:
        # An environment.config carrying no environment is noise, and inventing a
        # default would be a fabricated measurement.
        assert environment_config_from_resource({}) is None
        assert environment_config_from_resource({"some.vendor.attr": "x"}) is None

    def test_unknown_resource_attributes_are_NOT_lifted(self) -> None:
        # The key set is curated on purpose: a Resource block can carry credentials
        # or customer identifiers, and dumping it wholesale would turn an
        # environment record into an exfiltration path.
        payload = environment_config_from_resource(
            {
                "service.name": "checkout",
                "aws.secret.access.key": "AKIAIOSFODNN7EXAMPLE",
                "customer.email": "bob@example.com",
            }
        )
        assert payload == {"service_name": "checkout", "framework": "openinference"}
        assert "AKIAIOSFODNN7EXAMPLE" not in json.dumps(payload)
        assert "bob@example.com" not in json.dumps(payload)

    def test_empty_string_values_are_ignored(self) -> None:
        assert environment_config_from_resource({"service.name": ""}) is None


class TestEnvironmentConfigEmission:
    def _events(self, request: Dict[str, Any], *, config: Any = None) -> List[Dict[str, Any]]:
        adapter = OpenInferenceAdapter(_Client(), capture_config=config or CaptureConfig.full())
        adapter.connect()
        OpenInferenceOTLPBridge(adapter).ingest_otlp_json(request)
        out: List[Dict[str, Any]] = []
        for collector in adapter._collectors.values():
            out.extend(collector.events)
        return out

    def test_one_environment_config_per_resource_block_not_per_span(self) -> None:
        # The environment describes the RESOURCE. Emitting it per span would restate
        # the same fact N times and inflate every trace.
        request = _request(
            [_span(span_id="01" * 8), _span(span_id="02" * 8), _span(span_id="03" * 8)],
            resource_attrs=[_kv("service.name", "checkout")],
        )
        envs = [e for e in self._events(request) if e["event_type"] == "environment.config"]
        assert len(envs) == 1
        assert envs[0]["payload"]["service_name"] == "checkout"

    def test_two_resource_blocks_each_get_their_own(self) -> None:
        request = {
            "resourceSpans": [
                {
                    "resource": {"attributes": [_kv("service.name", "checkout")]},
                    "scopeSpans": [{"spans": [_span(span_id="01" * 8, trace_id="aa" * 16)]}],
                },
                {
                    "resource": {"attributes": [_kv("service.name", "fulfilment")]},
                    "scopeSpans": [{"spans": [_span(span_id="02" * 8, trace_id="bb" * 16)]}],
                },
            ]
        }
        envs = [e["payload"]["service_name"] for e in self._events(request) if e["event_type"] == "environment.config"]
        assert sorted(envs) == ["checkout", "fulfilment"]

    def test_two_source_traces_in_one_block_each_record_the_environment(self) -> None:
        # Each source trace becomes its OWN LayerLens trace (one collector each), so
        # each must carry the environment or one of them loses it.
        request = _request(
            [_span(span_id="01" * 8, trace_id="aa" * 16), _span(span_id="02" * 8, trace_id="bb" * 16)],
            resource_attrs=[_kv("service.name", "checkout")],
        )
        envs = [e for e in self._events(request) if e["event_type"] == "environment.config"]
        assert len(envs) == 2

    def test_no_environment_config_when_the_resource_has_nothing_known(self) -> None:
        request = _request([_span()], resource_attrs=[_kv("some.vendor.attr", "x")])
        assert not [e for e in self._events(request) if e["event_type"] == "environment.config"]

    def test_no_environment_config_when_there_is_no_resource_block(self) -> None:
        request = _request([_span()])
        assert not [e for e in self._events(request) if e["event_type"] == "environment.config"]

    def test_the_l4a_capture_layer_gates_it(self) -> None:
        # environment.config maps to l4a_environment_config, so a config that
        # disables that layer must suppress it — while the spans still ingest.
        request = _request([_span()], resource_attrs=[_kv("service.name", "checkout")])
        events = self._events(request, config=CaptureConfig.minimal())
        assert not [e for e in events if e["event_type"] == "environment.config"]

    def test_span_to_events_still_emits_no_environment_config(self) -> None:
        # LOAD-BEARING: span_to_events is the pinned Python<->Go boundary and the Go
        # bridge emits no environment.config (atlas captures the same Resource data
        # as trace-level CanonicalTrace fields). Emitting it there would
        # desynchronise the positional 26-event oracle comparison.
        from layerlens.instrument.adapters.frameworks.openinference import span_to_events

        for record in otlp_json_to_span_records(_request([_span()], resource_attrs=[_kv("service.name", "checkout")])):
            assert "environment.config" not in [t for t, _ in span_to_events(record, capture_content=True)]

    def test_the_real_conformance_corpus_now_yields_its_service_name(self) -> None:
        # The shared corpus has always carried an unused resource block
        # (service.name=oi-conformance). It is now captured.
        corpus = os.path.join(os.path.dirname(__file__), "oi_conformance", "spans.otlp.json")
        with open(corpus) as fh:
            request = json.load(fh)
        envs = [e for e in self._events(request) if e["event_type"] == "environment.config"]
        assert envs, "the corpus resource block produced no environment.config"
        assert envs[0]["payload"]["service_name"] == "oi-conformance"
