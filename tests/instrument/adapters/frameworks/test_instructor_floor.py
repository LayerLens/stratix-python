"""Offline redaction + error + attestation + cost floor for the Instructor adapter.

Runs in plain CI with no credentials and no network: every Instructor object is
real (a real ``instructor.from_openai(OpenAI())``, real Pydantic response models,
the real hooks system, the real tenacity retry loop) and only the transport is
mocked.

* Redaction   — ``capture_content=False`` keeps the prompt, the extracted output,
                and the validation-error text (which embeds the offending value
                VERBATIM) out of the stored trace, proven by a SENTINEL sweep over
                the whole serialized trace, with a ``capture_content=True``
                vacuity control proving the same path DOES carry them otherwise.
                Structure/topology must survive the strip.
* Error       — a REAL ``openai`` SDK exception (the shape a real instructor call
                raises when the provider rejects the request), plus the
                empty-message exception that used to be indistinguishable from a
                success.
* Attestation — offline ``verify_chain`` over a real extraction's collected
                payload, with a tamper control proving the check is not vacuous.
* Cost        — priced from REAL tokens, and the honest OMISSION when the response
                carries no usage.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Callable, Optional

import httpx
import pytest

instructor = pytest.importorskip("instructor")

from pydantic import BaseModel, field_validator  # noqa: E402
from instructor.core import InstructorRetryException  # noqa: E402

import openai  # noqa: E402
from openai import OpenAI  # noqa: E402
from layerlens.attestation._verify import verify_chain  # noqa: E402
from layerlens.attestation._envelope import HashScope, AttestationEnvelope  # noqa: E402
from layerlens.instrument._capture_config import _CONTENT_KEYS, CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.instructor import InstructorAdapter  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402

SENTINEL = "LL-SENTINEL-7f3a9c2e"


# ---------------------------------------------------------------------------
# Real-instructor helpers
# ---------------------------------------------------------------------------
class Profile(BaseModel):
    name: str
    age: int


def _tool_call_body(arguments: Dict[str, Any], *, usage: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "Profile", "arguments": json.dumps(arguments)},
                        }
                    ],
                },
            }
        ],
    }
    if usage is not None:
        body["usage"] = usage
    return body


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> Any:
    return instructor.from_openai(
        OpenAI(api_key="sk-test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    )


_NO_USAGE = object()


def _ok_client(arguments: Optional[Dict[str, Any]] = None, usage: Any = None) -> Any:
    payload = arguments if arguments is not None else {"name": "John", "age": 30}
    if usage is _NO_USAGE:
        resolved_usage = None
    elif usage is None:
        resolved_usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    else:
        resolved_usage = usage

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_tool_call_body(payload, usage=resolved_usage))

    return _client(handler)


def _drive(client: Any, *, content: str, response_model: Any = Profile, **overrides: Any) -> Any:
    kwargs: Dict[str, Any] = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": content}],
        "response_model": response_model,
    }
    kwargs.update(overrides)
    return client.chat.completions.create(**kwargs)


def _retry_client(state: Dict[str, Any], sentinel: str) -> Any:
    """A real client whose validator fails once, so the REAL parse:error hook
    fires and its ValidationError message embeds ``sentinel`` verbatim."""

    def handler(_request: httpx.Request) -> httpx.Response:
        state["http"] = state.get("http", 0) + 1
        return httpx.Response(
            200,
            json=_tool_call_body(
                {"name": sentinel, "age": 30},
                usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            ),
        )

    class Guarded(BaseModel):
        name: str
        age: int

        @field_validator("name")
        @classmethod
        def _fail_once(cls, value: str) -> str:
            if state.get("http", 0) < 2:
                # A real pydantic ValidationError renders input_value=... verbatim,
                # which is exactly how the caller's content reaches the retry event.
                raise ValueError(f"name {value!r} failed the first pass")
            return value

    state["model"] = Guarded
    return _client(handler)


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_content_present_when_capturing(self):
        """Vacuity control: with capture_content=True the SAME real path DOES carry
        the SENTINEL and the content keys it rides on. Without this, the absence
        assertions below could pass against an adapter that emits nothing."""
        from unittest.mock import Mock

        mock_client = Mock()
        uploaded = capture_framework_trace(mock_client)
        state: Dict[str, Any] = {}
        client = _retry_client(state, SENTINEL)
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
        adapter.connect(target=client)
        _drive(client, content=f"extract {SENTINEL}", response_model=state["model"], max_retries=3)
        adapter.disconnect()

        events = uploaded["events"]
        assert SENTINEL in json.dumps(events), "control run must carry the SENTINEL when capturing content"

        invoke = find_event(events, "model.invoke")["payload"]
        assert "messages" in invoke
        assert "output_message" in invoke
        assert SENTINEL in json.dumps(invoke["messages"])

        retry = find_event(events, "tool.call")["payload"]
        assert "error" in retry
        assert SENTINEL in retry["error"], "the real ValidationError must embed the offending value"

    def test_content_absent_when_not_capturing(self, mock_client):
        """capture_content=False keeps the prompt, the extracted output AND the
        validation-error text out of every stored event."""
        uploaded = capture_framework_trace(mock_client)
        state: Dict[str, Any] = {}
        client = _retry_client(state, SENTINEL)
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        adapter.connect(target=client)
        _drive(client, content=f"extract {SENTINEL}", response_model=state["model"], max_retries=3)
        adapter.disconnect()

        events = uploaded["events"]
        assert events, "the call must still emit structural events without content"

        # 1) SENTINEL sweep over the entire serialized trace.
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: SENTINEL survived capture_content=False"

        # 2) The content keys are absent from the payloads that would carry them.
        invoke = find_event(events, "model.invoke")["payload"]
        assert "messages" not in invoke, "model.invoke leaked 'messages' under capture_content=False"
        assert "output_message" not in invoke, "model.invoke leaked 'output_message' under capture_content=False"

        retry = find_event(events, "tool.call")["payload"]
        assert "error" not in retry, (
            "PRIVACY LEAK: tool.call leaked the ValidationError text, which embeds the "
            "offending value verbatim, under capture_content=False"
        )

    def test_structure_and_topology_survive_redaction(self, mock_client):
        """Redaction must strip content WITHOUT going blind: the retry stays
        visible, countable and attributable."""
        uploaded = capture_framework_trace(mock_client)
        state: Dict[str, Any] = {}
        client = _retry_client(state, SENTINEL)
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        adapter.connect(target=client)
        _drive(client, content=f"extract {SENTINEL}", response_model=state["model"], max_retries=3)
        adapter.disconnect()

        events = uploaded["events"]
        invoke = find_event(events, "model.invoke")
        retry = find_event(events, "tool.call")

        # The metric/topology skeleton survives.
        assert invoke["payload"]["model"] == "gpt-4o-mini"
        assert invoke["payload"]["provider"] == "openai"
        assert invoke["payload"]["retries_observed"] == 1
        # instructor accumulates usage across attempts: the retried call really
        # burned two prompts (11+11), so the honest count is the sum.
        assert invoke["payload"]["tokens_prompt"] == 22
        assert retry["payload"]["tool_name"] == "instructor.validation_retry"
        assert retry["payload"]["attempt"] == 1
        assert retry["payload"]["success"] is False
        assert retry["payload"]["hook"] == "parse:error"
        # The CATEGORY of the failure survives even though its text does not.
        assert retry["payload"]["error_type"] == "ValidationError"
        assert retry["parent_span_id"] == invoke["span_id"]
        assert find_events(events, "environment.config"), "environment.config is content-free and must survive"
        assert find_events(events, "cost.record"), "cost must survive redaction"

    def test_adapter_itself_gates_content_not_only_the_backstop(self, mock_client, monkeypatch):
        """The ADAPTER's own gate must hold, independently of the collector backstop.

        The trace-level absence tests above cannot see this: ``messages`` /
        ``output_message`` / ``error`` are in _CONTENT_KEYS["model.invoke"], so
        ``redact_payload`` strips them from the stored trace whether or not the
        adapter gated at emit time — an ungated adapter looks identical downstream.
        This asserts at the REAL adapter->collector seam (the payload handed to
        ``TraceCollector.emit``, before redaction), so removing a
        ``_set_if_capturing`` goes red here. Belt AND braces are load-bearing:
        the backstop is the only net when an adapter forgets, and the adapter gate
        is the only net for a key the backstop does not know (tool.call error).
        """
        from layerlens.instrument._collector import TraceCollector

        seen: list[tuple[str, Dict[str, Any]]] = []
        real_emit = TraceCollector.emit

        def spy(self: Any, event_type: str, payload: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            seen.append((event_type, dict(payload)))
            return real_emit(self, event_type, payload, *args, **kwargs)

        monkeypatch.setattr(TraceCollector, "emit", spy)

        state: Dict[str, Any] = {}
        client = _retry_client(state, SENTINEL)
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        adapter.connect(target=client)
        _drive(client, content=f"extract {SENTINEL}", response_model=state["model"], max_retries=3)
        adapter.disconnect()

        raw_invoke = [p for t, p in seen if t == "model.invoke"]
        assert raw_invoke, "the adapter must still hand a model.invoke to the collector"
        for payload in raw_invoke:
            assert "messages" not in payload, "the adapter handed ungated 'messages' to the collector"
            assert "output_message" not in payload, "the adapter handed ungated 'output_message' to the collector"

        raw_retry = [p for t, p in seen if t == "tool.call"]
        assert raw_retry, "the adapter must still hand the observed retry to the collector"
        for payload in raw_retry:
            assert "error" not in payload, "the adapter handed the ungated ValidationError to the collector"

        # And the SENTINEL never even reaches the collector from this adapter.
        assert SENTINEL not in json.dumps(seen), "the adapter leaked content to the collector boundary"

        # The model.invoke error text is ALSO backstopped by _CONTENT_KEYS, so its
        # emit-time gate is only visible at this seam too. Drive a failing call.
        seen.clear()
        failing = _client(lambda _r: httpx.Response(429, json={"error": {"message": f"rl {SENTINEL}"}}))
        adapter2 = InstructorAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        adapter2.connect(target=failing)
        with pytest.raises(InstructorRetryException):
            _drive(failing, content="hello", max_retries=1)
        adapter2.disconnect()

        raw_failed = [p for t, p in seen if t == "model.invoke"]
        assert raw_failed, "a failed call must still reach the collector"
        for payload in raw_failed:
            assert "error" not in payload, "the adapter handed the ungated provider error to the collector"
            # ...while the CATEGORY still does (redact without going blind).
            assert payload["status"] == "error"
            assert payload["error_type"] == "InstructorRetryException"
        assert SENTINEL not in json.dumps(seen), "the provider error text reached the collector boundary"

    def test_adapter_hands_content_to_the_collector_when_capturing(self, mock_client, monkeypatch):
        """Vacuity control for the seam above: with capture_content=True the SAME
        adapter DOES hand those keys to the collector."""
        from layerlens.instrument._collector import TraceCollector

        seen: list[tuple[str, Dict[str, Any]]] = []
        real_emit = TraceCollector.emit

        def spy(self: Any, event_type: str, payload: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            seen.append((event_type, dict(payload)))
            return real_emit(self, event_type, payload, *args, **kwargs)

        monkeypatch.setattr(TraceCollector, "emit", spy)

        state: Dict[str, Any] = {}
        client = _retry_client(state, SENTINEL)
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
        adapter.connect(target=client)
        _drive(client, content=f"extract {SENTINEL}", response_model=state["model"], max_retries=3)
        adapter.disconnect()

        raw_invoke = [p for t, p in seen if t == "model.invoke"][0]
        assert "messages" in raw_invoke and "output_message" in raw_invoke
        raw_retry = [p for t, p in seen if t == "tool.call"][0]
        assert "error" in raw_retry
        assert SENTINEL in json.dumps(seen)

    def test_collector_backstop_strips_tool_call_error(self):
        """The adapter gates ``error`` at emit time, but the collector-side backstop
        must ALSO know it is content — so redaction holds even if an emit site
        forgets to gate (LAY-3567 B1). This asserts the shared wiring the
        orchestrator must land in _CONTENT_KEYS."""
        assert "error" in _CONTENT_KEYS["tool.call"], (
            "_CONTENT_KEYS['tool.call'] must strip 'error': instructor's validation_retry "
            "carries a pydantic ValidationError whose message embeds the offending value "
            "verbatim. tool.result and model.invoke already strip it; tool.call must too."
        )

    def test_model_invoke_content_keys_are_already_backstopped(self):
        """The keys this adapter uses for model.invoke content are the canonical
        ones the backstop already strips — no wiring needed."""
        assert {"messages", "output_message", "error"} <= _CONTENT_KEYS["model.invoke"]


# ---------------------------------------------------------------------------
# Real error shapes
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_real_openai_error_surfaces_honestly(self, mock_client):
        """A REAL openai SDK exception — the shape a real instructor call raises
        when the provider rejects the request."""
        uploaded = capture_framework_trace(mock_client)

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404,
                json={
                    "error": {
                        "message": "The model `gpt-4o-mini-ghost` does not exist",
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                    }
                },
            )

        client = _client(handler)
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)

        with pytest.raises(InstructorRetryException) as excinfo:
            _drive(client, content="hello", max_retries=1)
        adapter.disconnect()

        # instructor re-raises its own error class from the REAL openai exception —
        # prove both halves of that real shape, not a hand-rolled stand-in.
        assert isinstance(excinfo.value.__cause__, openai.NotFoundError)
        assert isinstance(excinfo.value.__cause__, openai.OpenAIError)

        payload = find_event(uploaded["events"], "model.invoke")["payload"]
        assert payload["status"] == "error"
        # The exception the CALLER actually catches is the one reported — the
        # provider's message rides through it verbatim.
        assert payload["error_type"] == "InstructorRetryException"
        assert "does not exist" in payload["error"]
        # The failed call is still attributed to a real model — never a placeholder.
        assert payload["model"] == "gpt-4o-mini"

    def test_empty_message_exception_is_still_distinguishable_from_success(self, mock_client):
        """An exception whose message renders empty (``str(exc) == ''``) must NOT
        produce a model.invoke that reads like a success. ateam gated the whole
        error field on truthiness, so this case emitted nothing at all."""
        uploaded = capture_framework_trace(mock_client)

        class Boom(BaseModel):
            name: str
            age: int

            @field_validator("age")
            @classmethod
            def _explode(cls, value: int) -> int:
                raise ValueError()

        client = _ok_client()
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)
        with pytest.raises(InstructorRetryException):
            _drive(client, content="hello", response_model=Boom, max_retries=1)
        adapter.disconnect()

        payload = find_event(uploaded["events"], "model.invoke")["payload"]
        assert payload["status"] == "error", "a raising call must never read as a success"
        assert payload["error_type"], "the failure category must survive an empty exception message"
        assert payload["status"] != "ok"

    def test_error_category_survives_redaction(self, mock_client):
        """status/error_type are metadata, so a failure stays visible even when the
        free-text error is stripped."""
        uploaded = capture_framework_trace(mock_client)

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": {"message": f"rate limited: {SENTINEL}"}})

        client = _client(handler)
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        adapter.connect(target=client)
        with pytest.raises(InstructorRetryException) as excinfo:
            _drive(client, content="hello", max_retries=1)
        assert isinstance(excinfo.value.__cause__, openai.RateLimitError)
        adapter.disconnect()

        events = uploaded["events"]
        assert SENTINEL not in json.dumps(events), "the provider error text leaked under capture_content=False"
        payload = find_event(events, "model.invoke")["payload"]
        assert payload["status"] == "error"
        assert payload["error_type"] == "InstructorRetryException"
        assert "error" not in payload


# ---------------------------------------------------------------------------
# Attestation
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_over_a_real_extraction(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        state: Dict[str, Any] = {}
        client = _retry_client(state, "Jane")
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)
        _drive(client, content="extract Jane", response_model=state["model"], max_retries=3)
        adapter.disconnect()

        events = uploaded["events"]
        assert events, "a real extraction must flush a non-empty trace"
        chain = (uploaded["attestation"] or {}).get("chain") or {}
        raw = chain.get("events") or []
        envelopes = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured for the real instructor trace"
        assert len(envelopes) == len(events), (
            f"attestation chain has {len(envelopes)} envelopes for {len(events)} events"
        )
        assert (uploaded["attestation"] or {}).get("root_hash") is not None

        result = verify_chain(envelopes)
        assert result.valid, f"attestation chain invalid: {result.error}"

        # Vacuity control: verify_chain must REJECT a broken link, proving the pass
        # above is not trivially true.
        assert len(envelopes) >= 2
        tampered = list(envelopes)
        tampered[1] = AttestationEnvelope(
            hash=tampered[1].hash,
            scope=tampered[1].scope,
            previous_hash="sha256:deadbeef-not-the-prior-hash",
        )
        broken = verify_chain(tampered)
        assert not broken.valid and broken.break_index == 1, "verify_chain failed to detect a broken link"


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------
class TestCostFloor:
    def test_cost_is_priced_from_real_tokens(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        client = _ok_client(usage={"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500})
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)
        _drive(client, content="hello")
        adapter.disconnect()

        cost = find_event(uploaded["events"], "cost.record")["payload"]
        assert cost["framework"] == "instructor"
        assert cost["model"] == "gpt-4o-mini"
        assert cost["tokens_prompt"] == 1000
        assert cost["tokens_completion"] == 500
        assert cost["tokens_total"] == 1500

        # Priced from the REAL rate card, never fabricated or zeroed.
        from layerlens.instrument.adapters.providers.pricing import PRICING, calculate_cost
        from layerlens.instrument.adapters.providers.token_usage import NormalizedTokenUsage

        expected = calculate_cost(
            "gpt-4o-mini",
            NormalizedTokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
            PRICING,
        )
        assert cost["cost_usd"] == expected
        assert cost["cost_usd"] > 0

    def test_no_usage_emits_no_cost_record(self, mock_client):
        """A response carrying no usage yields no token keys and NO cost.record —
        an honest omission, never a fabricated 0.0."""
        uploaded = capture_framework_trace(mock_client)
        client = _ok_client(usage=_NO_USAGE)
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)
        _drive(client, content="hello")
        adapter.disconnect()

        events = uploaded["events"]
        invoke = find_event(events, "model.invoke")["payload"]
        assert "tokens_prompt" not in invoke
        assert "tokens_total" not in invoke
        assert not find_events(events, "cost.record"), "no usage => no cost.record (never a fabricated 0.0)"
