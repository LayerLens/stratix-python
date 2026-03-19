"""OTLP JSON output formatter.

Produces proto-compliant OTLP resourceSpans JSON following the
protobuf-to-JSON mapping used by stratix/sdk/python/exporters/otel.py.
"""

from __future__ import annotations

from typing import Any

from ..span_model import SimulatedSpan, SimulatedTrace, SpanStatus, SpanType
from .base import BaseOutputFormatter

# OTel SpanKind proto values
_SPAN_KIND_MAP = {
    SpanType.LLM: 3,       # CLIENT
    SpanType.AGENT: 2,     # SERVER
    SpanType.TOOL: 1,      # INTERNAL
    SpanType.EVALUATION: 1,  # INTERNAL
}

# OTel StatusCode proto values
_STATUS_CODE_MAP = {
    SpanStatus.OK: 1,
    SpanStatus.ERROR: 2,
    SpanStatus.UNSET: 0,
}


def _encode_attributes(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    """Encode attributes to OTLP JSON format.

    Follows the protobuf-to-JSON attribute encoding from otel.py.
    """
    result = []
    for key, value in attrs.items():
        if value is None:
            continue
        attr: dict[str, Any] = {"key": key}
        if isinstance(value, bool):
            attr["value"] = {"boolValue": value}
        elif isinstance(value, int):
            attr["value"] = {"intValue": str(value)}
        elif isinstance(value, float):
            attr["value"] = {"doubleValue": value}
        elif isinstance(value, str):
            attr["value"] = {"stringValue": value}
        elif isinstance(value, list):
            if all(isinstance(v, str) for v in value):
                attr["value"] = {
                    "arrayValue": {
                        "values": [{"stringValue": v} for v in value]
                    }
                }
            else:
                import json

                attr["value"] = {"stringValue": json.dumps(value)}
        elif isinstance(value, dict):
            import json

            attr["value"] = {"stringValue": json.dumps(value)}
        else:
            attr["value"] = {"stringValue": str(value)}
        result.append(attr)
    return result


def _span_to_otlp(span: SimulatedSpan) -> dict[str, Any]:
    """Convert a SimulatedSpan to OTLP JSON span."""
    otlp_span: dict[str, Any] = {
        "traceId": "",  # Will be set by the trace-level formatter
        "spanId": span.span_id,
        "name": span.name,
        "kind": _SPAN_KIND_MAP.get(span.span_type, 1),
        "startTimeUnixNano": str(span.start_time_unix_nano),
        "endTimeUnixNano": str(span.end_time_unix_nano),
        "attributes": _encode_attributes(span.attributes),
        "status": {
            "code": _STATUS_CODE_MAP.get(span.status, 0),
            # OTel spec: status.message should only be set for ERROR status
            "message": span.status_message or "" if span.status == SpanStatus.ERROR else "",
        },
        "events": [],
    }

    if span.parent_span_id:
        otlp_span["parentSpanId"] = span.parent_span_id

    # Add span events
    for event in span.events:
        otlp_span["events"].append(event)

    return otlp_span


class OTLPJSONFormatter(BaseOutputFormatter):
    """OTLP resourceSpans JSON output format.

    Produces proto-compliant JSON matching the OpenTelemetry
    OTLP specification.
    """

    def format_trace(self, trace: SimulatedTrace) -> dict[str, Any]:
        """Format a SimulatedTrace as OTLP resourceSpans JSON."""
        spans = []
        for span in trace.spans:
            otlp_span = _span_to_otlp(span)
            otlp_span["traceId"] = trace.trace_id
            spans.append(otlp_span)

        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": _encode_attributes(trace.resource_attributes),
                    },
                    "scopeSpans": [
                        {
                            "scope": {
                                "name": trace.scope_name,
                                "version": trace.scope_version,
                            },
                            "spans": spans,
                        }
                    ],
                }
            ]
        }
