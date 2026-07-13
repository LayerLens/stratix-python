"""Deterministic doubles for the Bedrock Agents framework adapter (LAY-3600 / T8).

REBUILT 2026-06-17 from a LIVE-captured ``InvokeAgent`` stream (agent
``9WBVYFT6RB`` "lean-test-bedrock-sdk" on Nova, us-east-1) — see
``docs/sdk-bedrock-agents-rewrite-brief.md``.

The real agent output and trace do **not** exist as top-level response keys.
``invoke_agent(...)`` returns ``{'completion': <botocore EventStream>,
'contentType', 'sessionId', 'ResponseMetadata'}`` and the answer + every
orchestration/model/tool trace stream as events *inside* ``completion``, which
the customer consumes lazily::

    {'chunk': {'bytes': b'...'}}                                   # answer text
    {'trace': {..., 'trace': {'orchestrationTrace': {<one of>}}}}  # a trace step

where the orchestrationTrace member is exactly one of ``modelInvocationInput``,
``modelInvocationOutput``, ``rationale``, ``invocationInput``, ``observation``
(siblings of ``orchestrationTrace`` carry ``failureTrace`` / pre/post-processing
/ routingClassifier). Tokens live at ``modelInvocationOutput.metadata.usage``.

botocore's ``Stubber`` cannot synthesise an event stream, so these tests inject
a fake single-read ``completion`` (``_FakeEventStream``) into ``parsed`` via an
``after-call`` hook registered *before* the adapter's, then drive a real
``bedrock-agent-runtime`` client and assert BOTH:

1. **Transparency** — the customer's ``for event in response['completion']`` sees
   every original event, in order, unmodified, exactly once (the adapter must
   not consume the single-use stream in the hook).
2. **Emission** — as the customer drains the stream the adapter emits
   ``agent.input`` + ``environment.config`` (at call time), then per-trace
   ``model.invoke`` (+ priced ``cost.record``), ``tool.call`` (action group),
   ``knowledge_base``, collaborator ``agent.handoff``, ``agent.error``
   (failureTrace), and a final ``agent.output`` from the accumulated chunks —
   and the trace is flushed **only after** the stream is drained.

Live-validated paths: model-invocation, output (chunk), proxy transparency.
The action-group / knowledge-base / collaborator / failure shapes follow the
AWS Bedrock Agents ``TracePart`` API (the live test agent has no action groups
or knowledge bases) and must be re-confirmed against a richer live trace.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

import pytest

boto3 = pytest.importorskip("boto3")
from botocore.exceptions import EventStreamError  # noqa: E402

from layerlens.instrument import trace_context  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.bedrock_agents import BedrockAgentsAdapter  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402

_ADAPTER_LOGGER = "layerlens.instrument.adapters.frameworks.bedrock_agents"

_AGENT_ID = "9WBVYFT6RB"
_ALIAS_ID = "U2YUM9TNKB"
_SESSION_ID = "sess-7c2f9d3a-4b1e-4f6c-9a8d-2e5b7c1f0a3d"
# A model id that IS in BEDROCK_PRICING so cost.record carries a real cost_usd.
_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
_AFTER_HOOK = "after-call.bedrock-agent-runtime.InvokeAgent"


# ---------------------------------------------------------------------------
# Single-read EventStream stand-in
# ---------------------------------------------------------------------------


class _FakeEventStream:
    """Single-read stand-in for ``botocore.eventstream.EventStream``.

    Iterating consumes it, exactly like the real single-use wire stream — a
    second pass yields nothing. This catches an adapter that drains the stream
    inside the ``after-call`` hook (which would leave the customer's own
    iteration empty). An optional ``error`` is raised once the events are
    exhausted, to simulate a mid/end-of-stream ``EventStreamError``.
    """

    def __init__(self, events: List[Dict[str, Any]], *, error: Optional[BaseException] = None) -> None:
        self._events = list(events)
        self._idx = 0
        self._error = error
        self.closed = False

    def __iter__(self) -> "_FakeEventStream":
        return self

    def __next__(self) -> Dict[str, Any]:
        if self._idx >= len(self._events):
            if self._error is not None:
                err, self._error = self._error, None
                raise err
            raise StopIteration
        event = self._events[self._idx]
        self._idx += 1
        return event

    # EventStream public API the proxy must pass through.
    def close(self) -> None:
        self.closed = True

    def get_initial_response(self) -> Dict[str, Any]:
        return {"status_code": 200}


def _event_stream_error(code: str = "throttlingException", message: str = "Rate exceeded") -> EventStreamError:
    """Build the botocore exception boto3 raises when a modeled error event is
    hit mid-iteration of the ``completion`` stream."""
    return EventStreamError({"Error": {"Code": code, "Message": message}}, "InvokeAgent")


# ---------------------------------------------------------------------------
# Real-shaped completion events
# ---------------------------------------------------------------------------


def _trace_event(
    orchestration: Optional[Dict[str, Any]] = None,
    *,
    failure: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Wrap an orchestrationTrace member (or failureTrace) in a real TracePart."""
    inner: Dict[str, Any] = {}
    if orchestration is not None:
        inner["orchestrationTrace"] = orchestration
    if failure is not None:
        inner["failureTrace"] = failure
    return {
        "trace": {
            "agentId": _AGENT_ID,
            "agentAliasId": _ALIAS_ID,
            "agentVersion": "2",
            "sessionId": _SESSION_ID,
            "callerChain": [
                {"agentAliasArn": f"arn:aws:bedrock:us-east-1:145023104696:agent-alias/{_AGENT_ID}/{_ALIAS_ID}"}
            ],
            "eventTime": "2026-06-17T21:10:55.579163+00:00",
            "trace": inner,
        }
    }


def _phase_event(phase: str, member: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a Trace-union member under an arbitrary phase (pre/post/routing)."""
    event = _trace_event({})
    event["trace"]["trace"] = {phase: member}
    return event


def _model_call(phase: str, model_id: str, trace_id: str, in_tok: int, out_tok: int) -> List[Dict[str, Any]]:
    """A model invocation (input+output pair) inside a given trace phase."""
    return [
        _phase_event(
            phase, {"modelInvocationInput": {"foundationModel": model_id, "traceId": trace_id, "type": phase}}
        ),
        _phase_event(
            phase,
            {
                "modelInvocationOutput": {
                    "metadata": {"usage": {"inputTokens": in_tok, "outputTokens": out_tok}},
                    "traceId": trace_id,
                }
            },
        ),
    ]


def _model_invocation_input() -> Dict[str, Any]:
    return _trace_event(
        {
            "modelInvocationInput": {
                "foundationModel": _MODEL_ID,
                "inferenceConfiguration": {"maximumLength": 1024, "temperature": 1.0, "topK": 1, "topP": 1.0},
                "text": '{"system":"You are a flight operations assistant."}',
                "traceId": "trace-gen-0",
                "type": "ORCHESTRATION",
            }
        }
    )


def _model_invocation_output(input_tokens: int = 642, output_tokens: int = 118) -> Dict[str, Any]:
    return _trace_event(
        {
            "modelInvocationOutput": {
                "metadata": {
                    "clientRequestId": "a813971b-2abf-44f5-a035-4dac4a9a65e2",
                    "startTime": "2026-06-17T21:10:54.827208+00:00",
                    "endTime": "2026-06-17T21:10:55.579163+00:00",
                    "totalTimeMs": 752,
                    "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens},
                },
                "rawResponse": {"content": '{"output":{"message":{"role":"assistant"}}}'},
                "traceId": "trace-gen-0",
            }
        }
    )


def _rationale() -> Dict[str, Any]:
    return _trace_event(
        {
            "rationale": {
                "text": "The customer wants to change an existing booking. Check the fee policy, then rebook.",
                "traceId": "trace-gen-0",
            }
        }
    )


def _kb_invocation_input() -> Dict[str, Any]:
    return _trace_event(
        {
            "invocationInput": {
                "invocationType": "KNOWLEDGE_BASE",
                "traceId": "trace-kb-1",
                "knowledgeBaseLookupInput": {
                    "knowledgeBaseId": "KB9ZXQW123",
                    "text": "flight change fee policy international",
                },
            }
        }
    )


def _kb_observation() -> Dict[str, Any]:
    return _trace_event(
        {
            "observation": {
                "type": "KNOWLEDGE_BASE",
                "traceId": "trace-kb-1",
                # NB: the agent-trace RetrievedReference carries content / location /
                # metadata only — there is NO `score` here (score lives on the
                # standalone Retrieve API's KnowledgeBaseRetrievalResult).
                "knowledgeBaseLookupOutput": {
                    "retrievedReferences": [
                        {
                            "content": {
                                "text": "Changes made more than 24h before departure incur a $40 fee.",
                                "type": "TEXT",
                            },
                            "location": {"type": "S3", "s3Location": {"uri": "s3://ll-policies/change-fees.pdf"}},
                            "metadata": {"x-amz-bedrock-kb-source-uri": "s3://ll-policies/change-fees.pdf"},
                        },
                        {
                            "content": {"text": "Same-day changes are subject to availability.", "type": "TEXT"},
                            "location": {"type": "S3", "s3Location": {"uri": "s3://ll-policies/same-day.pdf"}},
                        },
                    ]
                },
            }
        }
    )


def _action_group_invocation_input() -> Dict[str, Any]:
    return _trace_event(
        {
            "invocationInput": {
                "invocationType": "ACTION_GROUP",
                "traceId": "trace-ag-2",
                "actionGroupInvocationInput": {
                    "actionGroupName": "FlightOperations",
                    "function": "changeFlight",
                    "apiPath": "/flights/{flightId}/change",
                    "verb": "post",
                    "executionType": "LAMBDA",
                    "parameters": [
                        {"name": "flightId", "type": "string", "value": "LL2041"},
                        {"name": "newDate", "type": "string", "value": "2026-06-15"},
                    ],
                },
            }
        }
    )


def _action_group_observation() -> Dict[str, Any]:
    return _trace_event(
        {
            "observation": {
                "type": "ACTION_GROUP",
                "traceId": "trace-ag-2",
                "actionGroupInvocationOutput": {
                    "text": '{"status": "CONFIRMED", "fareDifference": 42.50}',
                    "metadata": {"clientRequestId": "5f1d9e2c-8a3b-4c7d-b6e1-0f9a8d7c6b5a"},
                },
            }
        }
    )


def _collaborator_invocation_input() -> Dict[str, Any]:
    return _trace_event(
        {
            "invocationInput": {
                "invocationType": "AGENT_COLLABORATOR",
                "traceId": "trace-collab-3",
                "agentCollaboratorInvocationInput": {
                    "agentCollaboratorName": "RebookingSpecialist",
                    "agentCollaboratorAliasArn": "arn:aws:bedrock:us-east-1:145023104696:agent-alias/AGTREBOOK1/PROD",
                    "input": {"text": "Rebook LL2041 to 2026-06-15", "type": "TEXT"},
                },
            }
        }
    )


def _collaborator_observation() -> Dict[str, Any]:
    return _trace_event(
        {
            "observation": {
                "type": "AGENT_COLLABORATOR",
                "traceId": "trace-collab-3",
                "agentCollaboratorInvocationOutput": {
                    "agentCollaboratorName": "RebookingSpecialist",
                    "output": {"text": "Rebooked LL2041 to 2026-06-15. A $40 fee applies.", "type": "TEXT"},
                },
            }
        }
    )


def _code_interpreter_input(
    *, trace_id: str = "trace-ci-4", code: str = "import pandas as pd\nprint(df.head())"
) -> Dict[str, Any]:
    """orchestrationTrace.invocationInput carrying a codeInterpreterInvocationInput."""
    return _trace_event(
        {
            "invocationInput": {
                "invocationType": "ACTION_GROUP_CODE_INTERPRETER",
                "traceId": trace_id,
                "codeInterpreterInvocationInput": {"code": code, "files": ["data.csv"]},
            }
        }
    )


def _code_interpreter_observation(
    *,
    trace_id: str = "trace-ci-4",
    output: Optional[str] = "   x\n0  1\n1  2",
    error: Optional[str] = None,
    timeout: bool = False,
    files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """orchestrationTrace.observation carrying a codeInterpreterInvocationOutput."""
    ci_out: Dict[str, Any] = {}
    if output is not None:
        ci_out["executionOutput"] = output
    if error is not None:
        ci_out["executionError"] = error
    if timeout:
        ci_out["executionTimeout"] = True
    if files is not None:
        ci_out["files"] = files
    return _trace_event(
        {
            "observation": {
                "type": "ACTION_GROUP",
                "traceId": trace_id,
                "codeInterpreterInvocationOutput": ci_out,
            }
        }
    )


def _reprompt_observation(
    *,
    trace_id: str = "trace-rp-5",
    source: str = "PARSER",
    text: str = "The response was not valid JSON. Please reformat and try again.",
) -> Dict[str, Any]:
    """orchestrationTrace.observation of type REPROMPT (self-correction)."""
    return _trace_event(
        {
            "observation": {
                "type": "REPROMPT",
                "traceId": trace_id,
                "repromptResponse": {"source": source, "text": text},
            }
        }
    )


def _ask_user_observation(
    *, trace_id: str = "trace-au-6", text: str = "Which date would you like to travel?"
) -> Dict[str, Any]:
    """orchestrationTrace.observation of type ASK_USER (asks the human)."""
    return _trace_event(
        {
            "observation": {
                "type": "ASK_USER",
                "traceId": trace_id,
                "finalResponse": {"text": text},
            }
        }
    )


def _final_observation(text: str) -> Dict[str, Any]:
    return _trace_event(
        {
            "observation": {
                "type": "FINISH",
                "traceId": "trace-final-9",
                "finalResponse": {"text": text},
            }
        }
    )


def _failure() -> Dict[str, Any]:
    return _trace_event(
        failure={
            "failureReason": "The agent encountered an unrecoverable error while invoking the action group.",
            "failureCode": 500,
            "traceId": "trace-fail-0",
        }
    )


def _guardrail_event(
    action: str = "INTERVENED",
    *,
    stage: str = "input",
    assessments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """A guardrailTrace TracePart (sibling of orchestrationTrace/failureTrace).

    The real shape: ``event['trace']['trace']['guardrailTrace']`` =
    ``{action: 'INTERVENED'|'NONE', inputAssessments|outputAssessments:
    [GuardrailAssessment], traceId}``. Each assessment carries topic/content/
    sensitiveInformation/word policies. ``inputAssessments`` => input guardrail,
    ``outputAssessments`` => output guardrail.
    """
    if assessments is None and action == "INTERVENED":
        assessments = [
            {
                "topicPolicy": {"topics": [{"name": "Medical Advice", "type": "DENY", "action": "BLOCKED"}]},
                "contentPolicy": {"filters": [{"type": "VIOLENCE", "confidence": "HIGH", "action": "BLOCKED"}]},
                "sensitiveInformationPolicy": {"piiEntities": [{"type": "EMAIL", "action": "ANONYMIZED"}]},
            }
        ]
    guardrail: Dict[str, Any] = {"action": action, "traceId": "trace-guard-0"}
    if assessments:
        key = "inputAssessments" if stage == "input" else "outputAssessments"
        guardrail[key] = assessments
    event = _trace_event({})
    event["trace"]["trace"] = {"guardrailTrace": guardrail}
    return event


def _return_control_event(*, invocation_id: str = "rc-inv-1", collaborator: bool = False) -> Dict[str, Any]:
    """A top-level ``returnControl`` ResponseStream member (sibling of chunk/trace).

    Carries the agent's tool *request* handed back to the caller's own app —
    one functionInvocationInput and one apiInvocationInput. The result is fed
    back on the next InvokeAgent call, not in this stream.
    """
    fn_input: Dict[str, Any] = {
        "functionInvocationInput": {
            "actionGroup": "OrderActions",
            "function": "placeOrder",
            "parameters": [
                {"name": "sku", "type": "string", "value": "LL-42"},
                {"name": "qty", "type": "integer", "value": "2"},
            ],
        }
    }
    if collaborator:
        fn_input["agentId"] = "AGTSALES01"
        fn_input["collaboratorName"] = "SalesSpecialist"
    return {
        "returnControl": {
            "invocationId": invocation_id,
            "invocationInputs": [
                fn_input,
                {
                    "apiInvocationInput": {
                        "actionGroup": "InventoryAPI",
                        "apiPath": "/inventory/{sku}",
                        "httpMethod": "GET",
                        "parameters": [{"name": "sku", "type": "string", "value": "LL-42"}],
                    }
                },
            ],
        }
    }


def _files_event(*, n: int = 2) -> Dict[str, Any]:
    """A top-level ``files`` FilePart (sibling of chunk/trace/returnControl).

    Each OutputFile = {name, type (MIME), bytes} — the binary artifacts the code
    interpreter produces (charts, CSVs, exports), 0..5 per part.
    """
    output_files = [
        {"name": "chart.png", "type": "image/png", "bytes": b"\x89PNG\r\n\x1a\n0123456789"},
        {"name": "summary.csv", "type": "text/csv", "bytes": b"col_a,col_b\n1,2\n3,4\n"},
    ][:n]
    return {"files": {"files": output_files}}


def _chunk(text: str) -> Dict[str, Any]:
    return {"chunk": {"bytes": text.encode("utf-8")}}


_FINAL_TEXT = "Your flight LL2041 has been moved to June 15th. A $40 change fee applies."


def _full_stream() -> List[Dict[str, Any]]:
    """A realistic multi-step turn: model gen, KB lookup, action group,
    collaborator handoff, final answer."""
    return [
        _model_invocation_input(),
        _model_invocation_output(),
        _rationale(),
        _kb_invocation_input(),
        _kb_observation(),
        _action_group_invocation_input(),
        _action_group_observation(),
        _collaborator_invocation_input(),
        _collaborator_observation(),
        _final_observation(_FINAL_TEXT),
        _chunk(_FINAL_TEXT),
    ]


# ---------------------------------------------------------------------------
# Wiring: real boto3 client + Stubber + fake completion injector
# ---------------------------------------------------------------------------


def _make_boto_client() -> Any:
    return boto3.client(
        "bedrock-agent-runtime",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        aws_session_token="testing",
    )


def _stream_injector(stream: _FakeEventStream, *, request_id: Optional[str] = None) -> Any:
    def _inject(**kwargs: Any) -> None:
        parsed = kwargs.get("parsed", {})
        if isinstance(parsed, dict):
            parsed["completion"] = stream
            if request_id is not None:
                parsed["ResponseMetadata"] = {"RequestId": request_id, "HTTPStatusCode": 200}

    return _inject


def _setup(
    mock_client: Any,
    stream: _FakeEventStream,
    *,
    config: Optional[CaptureConfig] = None,
    request_id: Optional[str] = None,
) -> tuple:
    """Wire a real client whose stubbed InvokeAgent response carries *stream*
    as ``completion``, with the adapter connected."""
    from botocore.stub import Stubber

    uploaded = capture_framework_trace(mock_client)
    boto = _make_boto_client()
    # Register the injector BEFORE the adapter connects so it fires first and
    # the adapter's after-call hook sees the fake completion in ``parsed``.
    boto.meta.events.register(_AFTER_HOOK, _stream_injector(stream, request_id=request_id))

    adapter = BedrockAgentsAdapter(mock_client, capture_config=config)
    adapter.connect(target=boto)

    stubber = Stubber(boto)
    stubber.activate()
    stubber.add_response(
        "invoke_agent",
        {"completion": {}, "contentType": "text/plain", "sessionId": _SESSION_ID, "memoryId": "mem-1"},
    )
    return adapter, uploaded, boto, stubber


def _invoke(boto: Any, *, input_text: str = "Move my flight LL2041 to June 15th.") -> Any:
    return boto.invoke_agent(
        agentId=_AGENT_ID,
        agentAliasId=_ALIAS_ID,
        sessionId=_SESSION_ID,
        inputText=input_text,
        enableTrace=True,
    )


def _drain(response: Any) -> List[Dict[str, Any]]:
    """Consume the customer-facing completion stream, as a customer would."""
    return list(response["completion"])


# ===========================================================================
# Transparency — the customer's stream must be unbroken
# ===========================================================================


class TestTransparency:
    def test_customer_receives_every_event_in_order(self, mock_client):
        events = _full_stream()
        adapter, _, boto, _ = _setup(mock_client, _FakeEventStream(events))
        resp = _invoke(boto)
        seen = _drain(resp)
        adapter.disconnect()
        assert seen == events  # same objects, same order, nothing dropped/added

    def test_completion_stream_is_single_read(self, mock_client):
        adapter, _, boto, _ = _setup(mock_client, _FakeEventStream(_full_stream()))
        resp = _invoke(boto)
        first = _drain(resp)
        second = _drain(resp)
        adapter.disconnect()
        assert len(first) == len(_full_stream())
        assert second == []  # adapter did not buffer-replay or pre-consume

    def test_stream_error_propagates_to_customer(self, mock_client):
        stream = _FakeEventStream(
            [_model_invocation_input(), _model_invocation_output(), _chunk("partial")],
            error=_event_stream_error(),
        )
        adapter, uploaded, boto, _ = _setup(mock_client, stream)
        resp = _invoke(boto)
        with pytest.raises(EventStreamError):
            _drain(resp)
        adapter.disconnect()
        # The adapter still observed the pre-error events and flushed a partial
        # trace recording the error — agent.error is authoritative, so no
        # (misleading) terminal agent.output is emitted on a failed stream.
        events = uploaded["events"]
        assert find_event(events, "model.invoke")
        assert find_event(events, "agent.error")
        assert not find_events(events, "agent.output")


# ===========================================================================
# Emission is deferred until the customer drains the stream
# ===========================================================================


class TestEmissionLifecycle:
    def test_not_flushed_until_stream_drained(self, mock_client):
        adapter, uploaded, boto, _ = _setup(mock_client, _FakeEventStream(_full_stream()))
        resp = _invoke(boto)
        # The customer holds the response but has not iterated completion yet:
        # nothing may be uploaded — emission happens as the stream is drained.
        assert mock_client.traces.upload.called is False
        _drain(resp)
        adapter.disconnect()
        assert mock_client.traces.upload.called is True
        assert uploaded["events"], "draining the stream should flush the trace"

    def test_agent_input_and_config(self, mock_client):
        adapter, uploaded, boto, _ = _setup(
            mock_client, _FakeEventStream(_full_stream()), config=CaptureConfig(capture_content=True)
        )
        _drain(_invoke(boto))
        adapter.disconnect()

        events = uploaded["events"]
        cfg = find_event(events, "environment.config")
        assert cfg["payload"]["agent_id"] == _AGENT_ID
        assert cfg["payload"]["agent_alias_id"] == _ALIAS_ID
        assert cfg["payload"]["enable_trace"] is True

        inp = find_event(events, "agent.input")
        assert inp["payload"]["agent_id"] == _AGENT_ID
        assert inp["payload"]["session_id"] == _SESSION_ID
        assert inp["payload"]["input"] == "Move my flight LL2041 to June 15th."

    def test_agent_output_from_accumulated_chunks(self, mock_client):
        stream = _FakeEventStream([_final_observation("Hello, world"), _chunk("Hello, "), _chunk("world")])
        adapter, uploaded, boto, _ = _setup(mock_client, stream, config=CaptureConfig(capture_content=True))
        _drain(_invoke(boto))
        adapter.disconnect()

        out = find_event(uploaded["events"], "agent.output")
        assert out["payload"]["output"] == "Hello, world"
        assert out["payload"]["session_id"] == _SESSION_ID


# ===========================================================================
# Per-trace emission from the real orchestrationTrace shapes
# ===========================================================================


class TestModelInvocation:
    def test_model_invoke_tokens_from_metadata_usage(self, mock_client):
        adapter, uploaded, boto, _ = _setup(mock_client, _FakeEventStream(_full_stream()))
        _drain(_invoke(boto))
        adapter.disconnect()

        mi = find_event(uploaded["events"], "model.invoke")
        assert mi["payload"]["model"] == _MODEL_ID
        assert mi["payload"]["provider"] == "aws_bedrock"
        assert mi["payload"]["tokens_prompt"] == 642
        assert mi["payload"]["tokens_completion"] == 118
        assert mi["payload"]["tokens_total"] == 760
        # G3: response_id fills from the model call's client_request_id, else the
        # InvokeAgent AWS RequestId — both honest per-response identifiers.
        rid = mi["payload"].get("response_id")
        assert rid, "response_id should fill on the bedrock model.invoke (G3)"
        assert rid == mi["payload"].get("client_request_id") or rid == mi["payload"].get("aws_request_id")

    def test_cost_record_is_priced(self, mock_client):
        adapter, uploaded, boto, _ = _setup(mock_client, _FakeEventStream(_full_stream()))
        _drain(_invoke(boto))
        adapter.disconnect()

        events = uploaded["events"]
        mi = find_event(events, "model.invoke")
        cost = find_event(events, "cost.record")
        assert cost["payload"]["model"] == _MODEL_ID
        assert cost["payload"]["cost_usd"] is not None and cost["payload"]["cost_usd"] > 0
        assert cost["span_id"] == mi["span_id"]

    def test_cost_record_priced_for_nova(self, mock_client):
        # The LIVE default: Nova is priced via BEDROCK_PRICING (LAY-3605), and the
        # region-prefixed inference-profile id the real wire carries resolves too.
        stream = _FakeEventStream(
            _model_call("orchestrationTrace", "us.amazon.nova-micro-v1:0", "t-nova", 838, 125) + [_chunk("4")]
        )
        adapter, uploaded, boto, _ = _setup(mock_client, stream)
        _drain(_invoke(boto))
        adapter.disconnect()

        events = uploaded["events"]
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["model"] == "us.amazon.nova-micro-v1:0"
        assert mi["payload"]["tokens_total"] == 963
        cost = find_event(events, "cost.record")
        assert cost["payload"]["tokens_prompt"] == 838
        assert cost["payload"]["tokens_total"] == 963
        assert cost["payload"]["cost_usd"] is not None and cost["payload"]["cost_usd"] > 0

    def test_cost_record_emitted_for_unpriced_model(self, mock_client):
        # An unpriced model still emits cost.record with token counts (cost_usd omitted).
        stream = _FakeEventStream(
            _model_call("orchestrationTrace", "amazon.titan-unknown-v9:0", "t-x", 50, 10) + [_chunk("ok")]
        )
        adapter, uploaded, boto, _ = _setup(mock_client, stream)
        _drain(_invoke(boto))
        adapter.disconnect()

        cost = find_event(uploaded["events"], "cost.record")
        assert cost["payload"]["tokens_total"] == 60
        assert "cost_usd" not in cost["payload"]  # unpriced -> omitted, not crashed

    def test_model_invocations_across_phases(self, mock_client):
        # A pre-processing model call AND an orchestration model call -> two
        # priced model.invoke + cost.record with distinct spans (locks _MODEL_PHASES).
        stream = _FakeEventStream(
            _model_call("preProcessingTrace", _MODEL_ID, "t-pre", 50, 10)
            + _model_call("orchestrationTrace", _MODEL_ID, "t-orch", 642, 118)
            + [_chunk("4")]
        )
        adapter, uploaded, boto, _ = _setup(mock_client, stream)
        _drain(_invoke(boto))
        adapter.disconnect()

        events = uploaded["events"]
        invokes = find_events(events, "model.invoke")
        costs = find_events(events, "cost.record")
        assert len(invokes) == 2
        assert len(costs) == 2
        assert len({m["span_id"] for m in invokes}) == 2  # distinct model spans
        assert sorted(m["payload"]["tokens_total"] for m in invokes) == [60, 760]


class TestActionGroup:
    def test_action_group_tool_call(self, mock_client):
        adapter, uploaded, boto, _ = _setup(
            mock_client, _FakeEventStream(_full_stream()), config=CaptureConfig(capture_content=True)
        )
        _drain(_invoke(boto))
        adapter.disconnect()

        tool_calls = find_events(uploaded["events"], "tool.call")
        ag = next(tc for tc in tool_calls if tc["payload"].get("tool_type") == "action_group")
        assert ag["payload"]["tool_name"] == "FlightOperations"
        assert ag["payload"]["function"] == "changeFlight"
        assert ag["payload"]["verb"] == "post"
        assert ag["payload"]["api_path"] == "/flights/{flightId}/change"
        assert ag["payload"]["execution_type"] == "LAMBDA"
        assert "CONFIRMED" in ag["payload"]["output"]


class TestKnowledgeBase:
    def test_knowledge_base_retrieval(self, mock_client):
        adapter, uploaded, boto, _ = _setup(mock_client, _FakeEventStream(_full_stream()))
        _drain(_invoke(boto))
        adapter.disconnect()

        tool_calls = find_events(uploaded["events"], "tool.call")
        kb = next(tc for tc in tool_calls if tc["payload"].get("tool_type") == "knowledge_base_retrieval")
        # tool_name (the KB id) comes from the invocationInput, correlated to
        # this observation by traceId — the observation output has no KB id.
        assert kb["payload"]["tool_name"] == "KB9ZXQW123"
        assert kb["payload"]["num_results"] == 2
        assert kb["payload"]["retrieval_sources"] == [
            "s3://ll-policies/change-fees.pdf",
            "s3://ll-policies/same-day.pdf",
        ]


class TestCollaboratorHandoff:
    def test_handoff_from_real_collaborator_fields(self, mock_client):
        adapter, uploaded, boto, _ = _setup(mock_client, _FakeEventStream(_full_stream()))
        _drain(_invoke(boto))
        adapter.disconnect()

        handoff = find_event(uploaded["events"], "agent.handoff")
        assert handoff["payload"]["from_agent"] == _AGENT_ID
        assert handoff["payload"]["to_agent"] == "RebookingSpecialist"

    def test_handoff_omits_to_agent_when_collaborator_unnamed(self, mock_client):
        """S16/F9: from_agent is the honest supervisor id, but when AWS names no
        collaborator the to_agent is omitted, not fabricated as "collaborator"."""
        inp = _trace_event(
            {
                "invocationInput": {
                    "invocationType": "AGENT_COLLABORATOR",
                    "traceId": "trace-collab-noname",
                    "agentCollaboratorInvocationInput": {"input": {"text": "do it", "type": "TEXT"}},
                }
            }
        )
        obs = _trace_event(
            {
                "observation": {
                    "type": "AGENT_COLLABORATOR",
                    "traceId": "trace-collab-noname",
                    "agentCollaboratorInvocationOutput": {"output": {"text": "done", "type": "TEXT"}},
                }
            }
        )
        adapter, uploaded, boto, _ = _setup(mock_client, _FakeEventStream([inp, obs]))
        _drain(_invoke(boto))
        adapter.disconnect()

        handoff = find_event(uploaded["events"], "agent.handoff")
        assert handoff["payload"]["from_agent"] == _AGENT_ID
        assert "to_agent" not in handoff["payload"], "fabricated 'collaborator' to_agent"


class TestFailureTrace:
    def test_failure_trace_emits_agent_error(self, mock_client):
        stream = _FakeEventStream([_model_invocation_input(), _model_invocation_output(), _failure()])
        adapter, uploaded, boto, _ = _setup(mock_client, stream, config=CaptureConfig(capture_content=True))
        _drain(_invoke(boto))
        adapter.disconnect()

        err = find_event(uploaded["events"], "agent.error")
        assert "unrecoverable error" in err["payload"]["error"]


# ===========================================================================
# LAY-3607 — guardrailTrace (INTERVENED) -> policy.violation
# ===========================================================================


class TestGuardrail:
    def test_intervened_emits_policy_violation_with_flattened_policies(self, mock_client):
        stream = _FakeEventStream([_guardrail_event("INTERVENED", stage="input"), _chunk("I can't help with that.")])
        adapter, uploaded, boto, _ = _setup(mock_client, stream)
        seen = _drain(_invoke(boto))
        adapter.disconnect()

        events = uploaded["events"]
        pv = find_event(events, "policy.violation")
        assert pv["payload"]["stage"] == "input"
        assert pv["payload"]["action"] == "INTERVENED"
        policies = pv["payload"]["policies"]
        by_type = {p["type"] for p in policies}
        assert {"topic", "content", "pii"} <= by_type  # all three assessment families flattened
        topic = next(p for p in policies if p["type"] == "topic")
        assert topic["name"] == "Medical Advice"
        assert topic["action"] == "BLOCKED"
        content = next(p for p in policies if p["type"] == "content")
        assert content["filter_type"] == "VIOLENCE"
        pii = next(p for p in policies if p["type"] == "pii")
        assert pii["entity_type"] == "EMAIL"
        assert pii["action"] == "ANONYMIZED"
        # A guardrail block is a policy outcome, NOT an agent failure.
        assert not find_events(events, "agent.error")
        # transparency: the guardrail event still reaches the customer unchanged.
        assert any("guardrailTrace" in e.get("trace", {}).get("trace", {}) for e in seen)

    def test_output_stage_detected_from_output_assessments(self, mock_client):
        stream = _FakeEventStream([_guardrail_event("INTERVENED", stage="output"), _chunk("redacted")])
        adapter, uploaded, boto, _ = _setup(mock_client, stream)
        _drain(_invoke(boto))
        adapter.disconnect()

        assert find_event(uploaded["events"], "policy.violation")["payload"]["stage"] == "output"

    def test_action_none_emits_no_policy_violation(self, mock_client):
        stream = _FakeEventStream([_guardrail_event("NONE"), _chunk("here you go")])
        adapter, uploaded, boto, _ = _setup(mock_client, stream, config=CaptureConfig(capture_content=True))
        seen = _drain(_invoke(boto))
        adapter.disconnect()

        assert not find_events(uploaded["events"], "policy.violation")
        assert any("guardrailTrace" in e.get("trace", {}).get("trace", {}) for e in seen)  # still yielded
        # action=NONE is a clean pass — the terminal agent.output still stands.
        assert find_event(uploaded["events"], "agent.output")["payload"]["output"] == "here you go"


# ===========================================================================
# LAY-3608 — returnControl -> tool.call (return_control) + suppress empty output
# ===========================================================================


class TestReturnControl:
    def test_emits_tool_calls_and_suppresses_empty_output(self, mock_client):
        # A return-of-control turn: no chunk, no finalResponse.
        stream = _FakeEventStream([_return_control_event()])
        adapter, uploaded, boto, _ = _setup(mock_client, stream)
        seen = _drain(_invoke(boto))
        adapter.disconnect()

        events = uploaded["events"]
        rc_calls = [tc for tc in find_events(events, "tool.call") if tc["payload"].get("tool_type") == "return_control"]
        assert len(rc_calls) == 2

        fn = next(tc for tc in rc_calls if tc["payload"].get("function") == "placeOrder")
        assert fn["payload"]["tool_name"] == "OrderActions"
        assert fn["payload"]["invocation_id"] == "rc-inv-1"
        assert "output" not in fn["payload"]  # result arrives on the NEXT turn

        api = next(tc for tc in rc_calls if tc["payload"].get("api_path"))
        assert api["payload"]["tool_name"] == "InventoryAPI"
        assert api["payload"]["api_path"] == "/inventory/{sku}"
        assert api["payload"]["verb"] == "GET"

        # The misleading empty agent.output is suppressed on a return-of-control turn.
        assert not find_events(events, "agent.output")
        # transparency: the returnControl event still reaches the customer unchanged.
        assert any("returnControl" in e for e in seen)

    def test_multi_agent_carries_collaborator_attribution(self, mock_client):
        stream = _FakeEventStream([_return_control_event(collaborator=True)])
        adapter, uploaded, boto, _ = _setup(mock_client, stream)
        _drain(_invoke(boto))
        adapter.disconnect()

        fn = next(
            tc for tc in find_events(uploaded["events"], "tool.call") if tc["payload"].get("function") == "placeOrder"
        )
        assert fn["payload"]["from_agent"] == "AGTSALES01"
        assert fn["payload"]["collaborator"] == "SalesSpecialist"

    def test_return_control_input_is_content_gated(self, mock_client):
        stream = _FakeEventStream([_return_control_event()])
        adapter, uploaded, boto, _ = _setup(mock_client, stream, config=CaptureConfig(capture_content=False))
        _drain(_invoke(boto))
        adapter.disconnect()

        rc = next(
            tc
            for tc in find_events(uploaded["events"], "tool.call")
            if tc["payload"].get("tool_type") == "return_control"
        )
        assert "input" not in rc["payload"]  # parameters gated
        assert rc["payload"]["tool_name"] == "OrderActions"  # structure survives


# ===========================================================================
# LAY-3609 — codeInterpreter (observation) -> agent.code
# ===========================================================================


class TestCodeInterpreter:
    # agent.code is an L2 (code-artifact) event — opt-in, off in the standard
    # config. It only emits when the customer enables l2_agent_code
    # (CaptureConfig.full()), the same gate every agent.code emitter sits behind.
    def test_emits_agent_code_with_code_and_output(self, mock_client):
        events_in = [_code_interpreter_input(), _code_interpreter_observation(), _chunk("The first rows are ...")]
        adapter, uploaded, boto, _ = _setup(mock_client, _FakeEventStream(events_in), config=CaptureConfig.full())
        seen = _drain(_invoke(boto))
        adapter.disconnect()

        events = uploaded["events"]
        code_ev = find_event(events, "agent.code")
        assert "import pandas as pd" in code_ev["payload"]["code"]  # input.code, correlated by traceId
        assert "x" in code_ev["payload"]["output"]  # executionOutput
        assert "execution_error" not in code_ev["payload"]
        assert not find_events(events, "agent.error")  # a successful code run is not an error
        assert seen == events_in  # transparency

    def test_execution_error_surfaced_but_not_agent_error(self, mock_client):
        events_in = [
            _code_interpreter_input(trace_id="t-err"),
            _code_interpreter_observation(trace_id="t-err", output=None, error="ZeroDivisionError: division by zero"),
            _chunk("I hit an error."),
        ]
        adapter, uploaded, boto, _ = _setup(mock_client, _FakeEventStream(events_in), config=CaptureConfig.full())
        _drain(_invoke(boto))
        adapter.disconnect()

        code_ev = find_event(uploaded["events"], "agent.code")
        assert code_ev["payload"]["errored"] is True  # structural failure signal
        assert "ZeroDivisionError" in code_ev["payload"]["execution_error"]  # traceback (content capture ON)
        # An errored code run is the diagnostic (agent.code), NOT a run failure (agent.error).
        assert not find_events(uploaded["events"], "agent.error")

    def test_execution_error_string_is_content_gated(self, mock_client):
        # The error string is a traceback (can embed user values) -> content, gated.
        # The structural `errored` flag stays visible so failures remain observable.
        events_in = [
            _code_interpreter_input(trace_id="t-g"),
            _code_interpreter_observation(trace_id="t-g", output=None, error="KeyError: 'ssn'"),
            _chunk("error"),
        ]
        adapter, uploaded, boto, _ = _setup(
            mock_client, _FakeEventStream(events_in), config=CaptureConfig(l2_agent_code=True, capture_content=False)
        )
        _drain(_invoke(boto))
        adapter.disconnect()

        code_ev = find_event(uploaded["events"], "agent.code")
        assert code_ev["payload"]["errored"] is True  # structural failure signal survives
        assert "execution_error" not in code_ev["payload"]  # the traceback is content -> gated

    def test_timeout_and_generated_files_metadata(self, mock_client):
        events_in = [
            _code_interpreter_input(trace_id="t-to"),
            _code_interpreter_observation(trace_id="t-to", timeout=True, files=["chart.png", "summary.csv"]),
            _chunk("done"),
        ]
        adapter, uploaded, boto, _ = _setup(mock_client, _FakeEventStream(events_in), config=CaptureConfig.full())
        _drain(_invoke(boto))
        adapter.disconnect()

        code_ev = find_event(uploaded["events"], "agent.code")
        assert code_ev["payload"]["execution_timeout"] is True
        assert code_ev["payload"]["num_files"] == 2
        assert code_ev["payload"]["files"] == ["chart.png", "summary.csv"]

    def test_code_and_output_content_gated(self, mock_client):
        events_in = [_code_interpreter_input(), _code_interpreter_observation(), _chunk("ok")]
        # L2 enabled (so agent.code emits) but content capture OFF: structure
        # survives, the executed source + output are stripped.
        adapter, uploaded, boto, _ = _setup(
            mock_client, _FakeEventStream(events_in), config=CaptureConfig(l2_agent_code=True, capture_content=False)
        )
        _drain(_invoke(boto))
        adapter.disconnect()

        code_ev = find_event(uploaded["events"], "agent.code")
        assert "code" not in code_ev["payload"]  # the executed source is content -> gated
        assert "output" not in code_ev["payload"]  # the execution output is content -> gated
        assert code_ev["payload"]["language"] == "python"  # structural metadata survives


# ===========================================================================
# LAY-3610 — repromptResponse / ASK_USER -> agent.step
# ===========================================================================


class TestReprompt:
    @pytest.mark.parametrize("source", ["ACTION_GROUP", "KNOWLEDGE_BASE", "PARSER"])
    def test_reprompt_emits_agent_step_not_error(self, mock_client, source):
        events_in = [_reprompt_observation(source=source), _chunk("Here is the corrected answer.")]
        adapter, uploaded, boto, _ = _setup(
            mock_client, _FakeEventStream(events_in), config=CaptureConfig(capture_content=True)
        )
        seen = _drain(_invoke(boto))
        adapter.disconnect()

        events = uploaded["events"]
        step = find_event(events, "agent.step")
        assert step["payload"]["step_type"] == "reprompt"
        assert step["payload"]["reprompt_source"] == source
        assert "valid JSON" in step["payload"]["text"]
        # A reprompt is a successful self-correction, NOT a failure.
        assert not find_events(events, "agent.error")
        assert seen == events_in  # transparency

    def test_ask_user_emits_agent_step(self, mock_client):
        events_in = [_ask_user_observation()]
        adapter, uploaded, boto, _ = _setup(
            mock_client, _FakeEventStream(events_in), config=CaptureConfig(capture_content=True)
        )
        seen = _drain(_invoke(boto))
        adapter.disconnect()

        step = find_event(uploaded["events"], "agent.step")
        assert step["payload"]["step_type"] == "ask_user"
        assert "date" in step["payload"]["text"]
        assert not find_events(uploaded["events"], "agent.error")
        assert seen == events_in  # transparency

    def test_reprompt_text_is_content_gated(self, mock_client):
        events_in = [_reprompt_observation(), _chunk("ok")]
        adapter, uploaded, boto, _ = _setup(
            mock_client, _FakeEventStream(events_in), config=CaptureConfig(capture_content=False)
        )
        _drain(_invoke(boto))
        adapter.disconnect()

        step = find_event(uploaded["events"], "agent.step")
        assert step["payload"]["step_type"] == "reprompt"  # structure survives
        assert step["payload"]["reprompt_source"] == "PARSER"
        assert "text" not in step["payload"]  # corrective text gated


# ===========================================================================
# LAY-3611 — files (FilePart) -> agent.code (file metadata)
# ===========================================================================


class TestFiles:
    # Like codeInterpreter, file artifacts map to agent.code (L2, opt-in).
    def test_files_emit_agent_code_metadata(self, mock_client):
        events_in = [_files_event(), _chunk("Here are your files.")]
        adapter, uploaded, boto, _ = _setup(mock_client, _FakeEventStream(events_in), config=CaptureConfig.full())
        seen = _drain(_invoke(boto))
        adapter.disconnect()

        code_ev = find_event(uploaded["events"], "agent.code")
        assert code_ev["payload"]["num_files"] == 2
        files = code_ev["payload"]["files"]
        assert [f["name"] for f in files] == ["chart.png", "summary.csv"]
        assert files[0]["type"] == "image/png"
        assert all("size_bytes" in f and f["size_bytes"] > 0 for f in files)
        assert seen == events_in  # transparency

    def test_raw_bytes_included_when_capturing(self, mock_client):
        events_in = [_files_event(n=1)]
        adapter, uploaded, boto, _ = _setup(mock_client, _FakeEventStream(events_in), config=CaptureConfig.full())
        _drain(_invoke(boto))
        adapter.disconnect()

        f = find_event(uploaded["events"], "agent.code")["payload"]["files"][0]
        assert "data" in f  # base64 of the raw bytes (content capture on)

    def test_raw_bytes_gated_metadata_survives(self, mock_client):
        events_in = [_files_event(n=1)]
        adapter, uploaded, boto, _ = _setup(
            mock_client, _FakeEventStream(events_in), config=CaptureConfig(l2_agent_code=True, capture_content=False)
        )
        _drain(_invoke(boto))
        adapter.disconnect()

        f = find_event(uploaded["events"], "agent.code")["payload"]["files"][0]
        assert "data" not in f  # raw bytes gated
        assert f["name"] == "chart.png"  # structural metadata survives
        assert "size_bytes" in f


# ===========================================================================
# LAY-3612 — deterministic AWS correlation metadata
# ===========================================================================


class TestAwsCorrelation:
    def test_aws_request_id_and_bedrock_trace_id_on_model_invoke(self, mock_client):
        stream = _FakeEventStream(_model_call("orchestrationTrace", _MODEL_ID, "trace-gen-0", 10, 5) + [_chunk("hi")])
        adapter, uploaded, boto, _ = _setup(mock_client, stream, request_id="aws-req-abc123")
        _drain(_invoke(boto))
        adapter.disconnect()

        events = uploaded["events"]
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["aws_request_id"] == "aws-req-abc123"  # InvokeAgent ResponseMetadata.RequestId
        assert mi["payload"]["bedrock_trace_id"] == "trace-gen-0"  # per-step orchestration traceId
        out = find_event(events, "agent.output")
        assert out["payload"]["aws_request_id"] == "aws-req-abc123"  # run-level anchor on the terminal event

    def test_client_request_id_on_model_invoke(self, mock_client):
        # modelInvocationOutput.metadata.clientRequestId -> the CloudWatch model-invocation record id.
        adapter, uploaded, boto, _ = _setup(mock_client, _FakeEventStream(_full_stream()), request_id="r1")
        _drain(_invoke(boto))
        adapter.disconnect()

        mi = find_event(uploaded["events"], "model.invoke")
        assert mi["payload"]["client_request_id"] == "a813971b-2abf-44f5-a035-4dac4a9a65e2"

    def test_bedrock_trace_id_on_tool_call_and_request_id_everywhere(self, mock_client):
        adapter, uploaded, boto, _ = _setup(mock_client, _FakeEventStream(_full_stream()), request_id="r1")
        _drain(_invoke(boto))
        adapter.disconnect()

        events = uploaded["events"]
        ag = next(t for t in find_events(events, "tool.call") if t["payload"].get("tool_type") == "action_group")
        assert ag["payload"]["bedrock_trace_id"] == "trace-ag-2"
        # every proxy-emitted event carries the run's AWS request id
        for et in ("model.invoke", "tool.call", "agent.handoff", "agent.output"):
            for e in find_events(events, et):
                assert e["payload"]["aws_request_id"] == "r1", et

    def test_bedrock_trace_id_on_policy_violation(self, mock_client):
        stream = _FakeEventStream([_guardrail_event("INTERVENED"), _chunk("blocked")])
        adapter, uploaded, boto, _ = _setup(mock_client, stream, request_id="r1")
        _drain(_invoke(boto))
        adapter.disconnect()

        pv = find_event(uploaded["events"], "policy.violation")
        assert pv["payload"]["bedrock_trace_id"] == "trace-guard-0"
        assert pv["payload"]["aws_request_id"] == "r1"

    def test_correlation_absent_when_no_request_id(self, mock_client):
        # Defensive: if the wire has no ResponseMetadata.RequestId, no aws_request_id is fabricated.
        adapter, uploaded, boto, _ = _setup(mock_client, _FakeEventStream(_full_stream()))
        _drain(_invoke(boto))
        adapter.disconnect()
        assert "aws_request_id" not in find_event(uploaded["events"], "agent.output")["payload"]


# ===========================================================================
# Content gating
# ===========================================================================


class TestContentGating:
    def test_content_stripped_structure_survives(self, mock_client):
        adapter, uploaded, boto, _ = _setup(
            mock_client, _FakeEventStream(_full_stream()), config=CaptureConfig(capture_content=False)
        )
        _drain(_invoke(boto))
        adapter.disconnect()

        events = uploaded["events"]
        assert "input" not in find_event(events, "agent.input")["payload"]
        assert "output" not in find_event(events, "agent.output")["payload"]
        # Structural metadata survives gating; all sensitive IO is stripped.
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["model"] == _MODEL_ID

        tool_calls = find_events(events, "tool.call")
        ag = next(tc for tc in tool_calls if tc["payload"].get("tool_type") == "action_group")
        assert ag["payload"]["tool_name"] == "FlightOperations"  # structure survives
        assert "input" not in ag["payload"] and "output" not in ag["payload"]

        kb = next(tc for tc in tool_calls if tc["payload"].get("tool_type") == "knowledge_base_retrieval")
        assert kb["payload"]["num_results"] == 2  # structural retrieval metadata survives
        assert "input" not in kb["payload"] and "output" not in kb["payload"]

        handoff = find_event(events, "agent.handoff")
        assert handoff["payload"]["to_agent"] == "RebookingSpecialist"  # structure survives
        assert "input" not in handoff["payload"] and "output" not in handoff["payload"]


# ===========================================================================
# Trace integrity
# ===========================================================================


class TestTraceIntegrity:
    def test_single_trace_id_and_monotonic_sequence(self, mock_client):
        adapter, uploaded, boto, _ = _setup(mock_client, _FakeEventStream(_full_stream()))
        _drain(_invoke(boto))
        adapter.disconnect()

        events = uploaded["events"]
        assert len({e["trace_id"] for e in events}) == 1
        seq = [e["sequence_id"] for e in events]
        assert seq == sorted(seq)
        root = find_event(events, "agent.input")["span_id"]
        for tc in find_events(events, "tool.call"):
            assert tc["parent_span_id"] == root


# ===========================================================================
# Stream lifecycle edges — early break, close, GC backstop, concurrency,
# orphan traces, unhandled event pass-through (review-hardening)
# ===========================================================================


class TestStreamLifecycleEdges:
    def test_early_break_emits_no_error(self, mock_client):
        adapter, uploaded, boto, _ = _setup(mock_client, _FakeEventStream(_full_stream()))
        resp = _invoke(boto)
        count = 0
        for _event in resp["completion"]:
            count += 1
            if count == 2:
                break  # customer aborts early — NOT an error
        adapter.disconnect()

        events = uploaded["events"]
        assert not find_events(events, "agent.error")
        assert len(find_events(events, "agent.output")) == 1  # flushed exactly once

    def test_close_delegates_to_source(self, mock_client):
        stream = _FakeEventStream(_full_stream())
        adapter, uploaded, boto, _ = _setup(mock_client, stream)
        resp = _invoke(boto)
        it = iter(resp["completion"])
        next(it)  # consume one event, then abort via close()
        resp["completion"].close()
        adapter.disconnect()

        assert stream.closed is True
        assert not find_events(uploaded["events"], "agent.error")
        assert find_event(uploaded["events"], "agent.output")

    def test_attribute_passthrough(self, mock_client):
        adapter, _, boto, _ = _setup(mock_client, _FakeEventStream(_full_stream()))
        resp = _invoke(boto)
        # A non-iteration EventStream method must reach the underlying stream.
        assert resp["completion"].get_initial_response() == {"status_code": 200}
        _drain(resp)
        adapter.disconnect()

    def test_disconnect_flushes_undrained_stream(self, mock_client):
        adapter, uploaded, boto, _ = _setup(mock_client, _FakeEventStream(_full_stream()))
        _invoke(boto)  # customer never drains response["completion"]
        assert mock_client.traces.upload.called is False
        adapter.disconnect()  # backstop finishes the in-flight stream
        assert mock_client.traces.upload.called is True
        events = uploaded["events"]
        assert find_event(events, "agent.input")
        # The stream was never observed, so the partial trace synthesizes no
        # content and no mid-stream events.
        assert "output" not in find_event(events, "agent.output")["payload"]
        assert not find_events(events, "model.invoke")

    def test_environment_config_emitted_per_run(self, mock_client):
        from botocore.stub import Stubber

        uploaded = capture_framework_trace(mock_client)
        boto = _make_boto_client()
        streams = iter([_FakeEventStream(_full_stream()), _FakeEventStream(_full_stream())])

        def _inject(**kwargs: Any) -> None:
            parsed = kwargs.get("parsed", {})
            if isinstance(parsed, dict):
                parsed["completion"] = next(streams)

        boto.meta.events.register(_AFTER_HOOK, _inject)
        adapter = BedrockAgentsAdapter(mock_client)
        adapter.connect(target=boto)
        stubber = Stubber(boto)
        stubber.activate()
        for _ in range(2):
            stubber.add_response(
                "invoke_agent",
                {"completion": {}, "contentType": "text/plain", "sessionId": _SESSION_ID, "memoryId": "mem-1"},
            )
        for _ in range(2):
            _drain(_invoke(boto))
        adapter.disconnect()

        # Two separate invocations → two traces, each with its own config.
        assert len(find_events(uploaded["events"], "environment.config")) == 2
        assert len({e["trace_id"] for e in uploaded["events"]}) == 2

    def test_orphan_observation_does_not_crash(self, mock_client):
        # An observation whose traceId never had a buffered invocationInput.
        stream = _FakeEventStream([_kb_observation(), _chunk("ok")])
        adapter, uploaded, boto, _ = _setup(mock_client, stream)
        _drain(_invoke(boto))
        adapter.disconnect()

        kb = next(
            tc
            for tc in find_events(uploaded["events"], "tool.call")
            if tc["payload"].get("tool_type") == "knowledge_base_retrieval"
        )
        assert kb["payload"]["tool_name"] == "knowledge_base"  # default when input is missing
        assert kb["payload"]["num_results"] == 2

    def test_unknown_top_level_member_passes_through_unhandled(self, mock_client):
        # A ResponseStream member the adapter has no branch for must still reach
        # the customer and emit no LayerLens event (forward-compat / transparency).
        unknown = {"someFutureMember": {"foo": "bar"}}
        events = [unknown, _chunk("done")]
        adapter, uploaded, boto, _ = _setup(
            mock_client, _FakeEventStream(events), config=CaptureConfig(capture_content=True)
        )
        seen = _drain(_invoke(boto))
        adapter.disconnect()

        assert seen == events  # yielded unchanged
        assert not find_events(uploaded["events"], "agent.error")
        assert find_event(uploaded["events"], "agent.output")["payload"]["output"] == "done"


# ===========================================================================
# Proxy-claim locks: thread-safety, trace_context nesting, enableTrace guard
# ===========================================================================


class TestConcurrency:
    def test_concurrent_drains_isolated_and_attributed(self, mock_client):
        """The proxy's core claim: draining two streams on two threads yields two
        isolated traces with correctly-attributed tokens (per-emit ContextVar
        re-establishment, no cross-contamination)."""
        from botocore.stub import Stubber

        uploaded = capture_framework_trace(mock_client)
        boto = _make_boto_client()
        streams = iter(
            [
                _FakeEventStream(_model_call("orchestrationTrace", _MODEL_ID, "tA", 100, 10) + [_chunk("A")]),
                _FakeEventStream(_model_call("orchestrationTrace", _MODEL_ID, "tB", 200, 20) + [_chunk("B")]),
            ]
        )

        def _inject(**kwargs: Any) -> None:
            parsed = kwargs.get("parsed", {})
            if isinstance(parsed, dict):
                parsed["completion"] = next(streams)

        boto.meta.events.register(_AFTER_HOOK, _inject)
        adapter = BedrockAgentsAdapter(mock_client)
        adapter.connect(target=boto)
        stubber = Stubber(boto)
        stubber.activate()
        for _ in range(2):
            stubber.add_response(
                "invoke_agent",
                {"completion": {}, "contentType": "text/plain", "sessionId": _SESSION_ID, "memoryId": "mem-1"},
            )

        # Two invocations open two independent proxies (hooks fire synchronously).
        resp_a = _invoke(boto)
        resp_b = _invoke(boto)

        errors: List[BaseException] = []

        def _drain_safe(resp: Any) -> None:
            try:
                list(resp["completion"])
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        ta = threading.Thread(target=_drain_safe, args=(resp_a,))
        tb = threading.Thread(target=_drain_safe, args=(resp_b,))
        ta.start(), tb.start(), ta.join(), tb.join()
        adapter.disconnect()
        assert not errors, errors

        by_trace: Dict[str, List[Dict[str, Any]]] = {}
        for e in uploaded["events"]:
            by_trace.setdefault(e["trace_id"], []).append(e)
        assert len(by_trace) == 2  # two isolated traces
        for evs in by_trace.values():
            assert len(find_events(evs, "model.invoke")) == 1
            assert len(find_events(evs, "agent.output")) == 1
        totals = sorted(find_event(evs, "model.invoke")["payload"]["tokens_total"] for evs in by_trace.values())
        assert totals == [110, 220]  # each trace kept its own tokens


class TestTraceContextNesting:
    def test_emits_into_shared_collector_without_early_flush(self, mock_client):
        adapter, uploaded, boto, _ = _setup(
            mock_client, _FakeEventStream(_full_stream()), config=CaptureConfig(capture_content=True)
        )
        with trace_context(mock_client, capture_config=CaptureConfig(capture_content=True)):
            resp = _invoke(boto)
            list(resp["completion"])  # proxy emits into the shared outer collector
            # Must NOT flush — the trace_context owns the collector.
            assert mock_client.traces.upload.called is False
        adapter.disconnect()
        # trace_context flushes once on block exit.
        assert mock_client.traces.upload.called is True
        events = uploaded["events"]
        assert len({e["trace_id"] for e in events}) == 1  # one shared trace
        assert find_event(events, "model.invoke")["payload"]["tokens_total"] == 760
        assert find_event(events, "agent.output")["payload"]["output"] == _FINAL_TEXT

    def test_warns_when_drained_after_context_closed(self, mock_client, caplog):
        adapter, uploaded, boto, _ = _setup(mock_client, _FakeEventStream(_full_stream()))
        with trace_context(mock_client):
            resp = _invoke(boto)
            # Customer stores the stream but does NOT drain before the block exits.
        # The shared collector seals on context exit; draining now drops events + warns.
        with caplog.at_level(logging.WARNING, logger=_ADAPTER_LOGGER):
            list(resp["completion"])
        adapter.disconnect()
        assert any("drained after its trace context closed" in r.getMessage() for r in caplog.records)
        assert not find_events(uploaded["events"], "model.invoke")  # emitted into sealed collector -> dropped


class TestEnableTraceGuard:
    def test_warns_and_degrades_without_enable_trace(self, mock_client, caplog):
        # Without enableTrace=True the stream carries only chunks; the adapter warns
        # once and the trace degrades to input + (text) output with no step events.
        adapter, uploaded, boto, _ = _setup(
            mock_client, _FakeEventStream([_chunk("hi there")]), config=CaptureConfig(capture_content=True)
        )
        with caplog.at_level(logging.WARNING, logger=_ADAPTER_LOGGER):
            resp = boto.invoke_agent(agentId=_AGENT_ID, agentAliasId=_ALIAS_ID, sessionId=_SESSION_ID, inputText="hi")
            list(resp["completion"])
        adapter.disconnect()
        assert any("without enableTrace=True" in r.getMessage() for r in caplog.records)
        events = uploaded["events"]
        assert find_event(events, "agent.input")
        assert find_event(events, "agent.output")["payload"]["output"] == "hi there"
        assert not find_events(events, "model.invoke")
