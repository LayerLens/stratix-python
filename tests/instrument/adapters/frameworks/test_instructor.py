"""Real-behaviour unit suite for the Instructor framework adapter.

Every Instructor object here is real: a real ``instructor.from_openai(OpenAI())``
patched client, real Pydantic ``response_model`` classes, the real hooks system,
and the real tenacity retry loop. The ONLY mock is the network boundary
(``httpx.MockTransport``), so a regression fails in plain CI with no credentials.

The retry lane is genuinely deterministic (rare for this repo): a ``field_validator``
that fails on the first pass and passes on the second makes Instructor's own
tenacity loop re-prompt, firing the REAL ``parse:error`` hook.
"""

from __future__ import annotations

import json
import asyncio
from typing import Any, Dict, List, Callable, Optional

import httpx
import pytest

instructor = pytest.importorskip("instructor")

from pydantic import BaseModel, field_validator  # noqa: E402

from openai import OpenAI, AsyncOpenAI  # noqa: E402
from layerlens.instrument import trace_context  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.instructor import (  # noqa: E402
    InstructorAdapter,
    _detect_provider,
    _honest_agent_name,
    _configured_max_retries,
)

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402


# ---------------------------------------------------------------------------
# Real-instructor helpers — a mocked transport is the only fake
# ---------------------------------------------------------------------------
class UserProfile(BaseModel):
    name: str
    age: int


def _tool_call_body(arguments: Dict[str, Any], *, name: str = "UserProfile") -> Dict[str, Any]:
    """A real OpenAI ChatCompletion body carrying a tool call (instructor's TOOLS mode)."""
    return {
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
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def _transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _ok_handler(arguments: Optional[Dict[str, Any]] = None) -> Callable[[httpx.Request], httpx.Response]:
    payload = arguments if arguments is not None else {"name": "John", "age": 30}

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_tool_call_body(payload))

    return handler


def _sync_client(handler: Callable[[httpx.Request], httpx.Response]) -> Any:
    return instructor.from_openai(OpenAI(api_key="sk-test", http_client=httpx.Client(transport=_transport(handler))))


def _async_client(handler: Callable[[httpx.Request], httpx.Response]) -> Any:
    return instructor.from_openai(
        AsyncOpenAI(api_key="sk-test", http_client=httpx.AsyncClient(transport=_transport(handler)))
    )


def _fields(model: Any) -> Dict[str, Any]:
    """Compare extracted values, not identity: instructor rebuilds the
    response_model into a fresh class per call-site, so ``==`` against a
    locally-constructed instance is False even for identical fields."""
    return dict(model.model_dump())


def _create(client: Any, **overrides: Any) -> Any:
    kwargs: Dict[str, Any] = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "John is 30"}],
        "response_model": UserProfile,
    }
    kwargs.update(overrides)
    return client.chat.completions.create(**kwargs)


# ---------------------------------------------------------------------------
# Happy path — the same-slot aliasing must not double-emit
# ---------------------------------------------------------------------------
class TestExtractionCall:
    def test_single_model_invoke_with_real_metadata(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        client = _sync_client(_ok_handler())
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)

        result = _create(client)
        adapter.disconnect()

        assert _fields(result) == {"name": "John", "age": 30}
        events = uploaded["events"]

        # On a real Instructor, .chat/.completions/.messages all return self, so
        # four dotted paths hit ONE slot. Exactly one model.invoke proves the
        # _layerlens_traced sentinel stopped the double-wrap (bite: drop the
        # sentinel check and this goes to 4).
        invokes = find_events(events, "model.invoke")
        assert len(invokes) == 1, f"expected exactly 1 model.invoke, saw {len(invokes)}"

        payload = invokes[0]["payload"]
        assert payload["framework"] == "instructor"
        assert payload["model"] == "gpt-4o-mini"
        assert payload["response_model"] == "UserProfile"
        assert payload["provider"] == "openai"
        assert payload["status"] == "ok"
        assert payload["tokens_prompt"] == 11
        assert payload["tokens_completion"] == 7
        assert payload["tokens_total"] == 18
        # A REAL measured elapsed, not a constant (bite: latency_ms = 0.0 passes
        # an isinstance check but not this).
        assert isinstance(payload["latency_ms"], float) and payload["latency_ms"] > 0
        # No model_name: the framework family emits `model` only (it is the key
        # pricing reads), and no sibling emits model_name.
        assert "model_name" not in payload

    def test_cost_record_is_priced_from_real_tokens(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        client = _sync_client(_ok_handler())
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)
        _create(client)
        adapter.disconnect()

        cost = find_event(uploaded["events"], "cost.record")["payload"]
        assert cost["model"] == "gpt-4o-mini"
        assert cost["tokens_prompt"] == 11
        assert cost["tokens_completion"] == 7
        # Priced centrally by the _emit chokepoint — a real number, never 0.0.
        assert isinstance(cost["cost_usd"], float)
        assert cost["cost_usd"] > 0

    def test_environment_config_emitted_once_per_client(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        client = _sync_client(_ok_handler())
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)
        _create(client)
        _create(client)
        adapter.disconnect()

        events = uploaded["events"]
        assert len(find_events(events, "model.invoke")) == 2
        # Deferred into the first run (emitting at connect() would land outside any
        # collector and vanish silently), and one-shot thereafter.
        configs = find_events(events, "environment.config")
        assert len(configs) == 1, f"expected exactly 1 environment.config, saw {len(configs)}"
        assert configs[0]["payload"]["provider"] == "openai"
        assert configs[0]["payload"]["mode"] == "Mode.TOOLS"

    def test_response_model_omitted_for_a_raw_call(self, mock_client):
        """A raw (unstructured) instructor call reports no response_model rather
        than defaulting one."""
        uploaded = capture_framework_trace(mock_client)

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-raw",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                },
            )

        client = _sync_client(handler)
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)
        _create(client, response_model=None)
        adapter.disconnect()

        payload = find_event(uploaded["events"], "model.invoke")["payload"]
        assert "response_model" not in payload
        assert payload["model"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Retry observation — driven by the REAL hooks system + real tenacity loop
# ---------------------------------------------------------------------------
def _retry_once_client(state: Dict[str, int]) -> Any:
    """A real client whose response_model fails validation on the first pass.

    Instructor's own tenacity loop re-prompts, firing the REAL ``parse:error``
    hook — no hook is simulated.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        state["http"] = state.get("http", 0) + 1
        return httpx.Response(200, json=_tool_call_body({"name": "John", "age": 30}))

    class RetriedProfile(BaseModel):
        name: str
        age: int

        @field_validator("age")
        @classmethod
        def _second_pass_only(cls, value: int) -> int:
            if state.get("http", 0) < 2:
                raise ValueError("age must be verified on a second pass")
            return value

    state["model"] = RetriedProfile  # type: ignore[assignment]
    return _sync_client(handler)


class TestRetryObservation:
    def test_real_parse_error_hook_emits_validation_retry(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        state: Dict[str, Any] = {}
        client = _retry_once_client(state)
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)

        result = _create(client, response_model=state["model"], max_retries=3)
        adapter.disconnect()

        assert result.age == 30
        assert state["http"] == 2, "the real tenacity loop must have re-prompted exactly once"

        events = uploaded["events"]
        retries = find_events(events, "tool.call")
        assert len(retries) == 1, f"expected 1 observed retry, saw {len(retries)}"
        payload = retries[0]["payload"]
        assert payload["tool_name"] == "instructor.validation_retry"
        assert payload["name"] == "instructor.validation_retry"
        assert payload["attempt"] == 1
        assert payload["success"] is False
        assert payload["hook"] == "parse:error"
        assert payload["response_model"] == "RetriedProfile"
        # A REAL pydantic ValidationError, not a synthetic string.
        assert payload["error_type"] == "ValidationError"
        assert "age must be verified on a second pass" in payload["error"]

        # The paired model.invoke reports the REAL observed count.
        invoke = find_event(events, "model.invoke")["payload"]
        assert invoke["retries_observed"] == 1
        assert invoke["max_retries_configured"] == 3

    def test_retry_is_a_child_span_of_its_model_invoke(self, mock_client):
        """The retry ties to the call that caused it (span topology + run_id)."""
        uploaded = capture_framework_trace(mock_client)
        state: Dict[str, Any] = {}
        client = _retry_once_client(state)
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)
        _create(client, response_model=state["model"], max_retries=3)
        adapter.disconnect()

        events = uploaded["events"]
        invoke = find_event(events, "model.invoke")
        retry = find_event(events, "tool.call")
        assert retry["parent_span_id"] == invoke["span_id"]
        assert retry["payload"]["run_id"] == invoke["payload"]["run_id"]

    def test_first_try_success_never_reports_the_configured_maximum(self, mock_client):
        """THE headline honesty rule: max_retries=3 + a first-try success reports
        ZERO observed retries — never 3."""
        uploaded = capture_framework_trace(mock_client)
        client = _sync_client(_ok_handler())
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)
        _create(client, max_retries=3)
        adapter.disconnect()

        events = uploaded["events"]
        assert not find_events(events, "tool.call"), "a first-try success observed no retries"
        payload = find_event(events, "model.invoke")["payload"]
        assert payload["max_retries_configured"] == 3
        # The configured maximum is reported ONLY under its self-describing name.
        assert payload["retries_observed"] == 0
        assert payload["retries_observed"] != 3
        assert "retry_count" not in payload

    def test_no_hooks_system_omits_retries_observed_entirely(self, mock_client):
        """An Instructor build with no hooks observes no retries, so the key is
        ABSENT — not 0, and never synthesized from max_retries."""
        uploaded = capture_framework_trace(mock_client)
        client = _sync_client(_ok_handler())
        # A client that predates the hooks system exposes no .on
        object.__setattr__(client, "on", None)

        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)
        _create(client, max_retries=3)
        adapter.disconnect()

        events = uploaded["events"]
        assert not find_events(events, "tool.call"), "no hooks => zero retry telemetry"
        payload = find_event(events, "model.invoke")["payload"]
        assert "retries_observed" not in payload, "retry-blindness must surface as ABSENCE"
        assert payload["max_retries_configured"] == 3

    def test_hook_outside_a_traced_call_is_dropped(self, mock_client):
        """A client-global hook firing with no in-flight call has nothing honest to
        correlate to — no orphan event is invented.

        Driven INSIDE a trace_context so a collector is genuinely active: without
        one, _emit no-ops for every event and the test would pass whether or not
        the adapter guards on the in-flight call (a vacuous green).
        """
        uploaded = capture_framework_trace(mock_client)
        client = _sync_client(_ok_handler())
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)

        with trace_context(mock_client, capture_config=CaptureConfig.full()):
            # An ambient collector IS active here...
            _create(client)
            # ...and the hook now fires with no instructor call in flight, exactly
            # as it would if an uninstrumented code path drove the same client.
            adapter._on_retry_hook("parse:error", (ValueError("orphan"),))
        adapter.disconnect()

        events = uploaded["events"]
        # The ambient collector really did capture this run (proves non-vacuity).
        assert find_events(events, "model.invoke"), "the ambient collector must have captured the call"
        assert not find_events(events, "tool.call"), "an orphan retry was fabricated with nothing to correlate to"


# ---------------------------------------------------------------------------
# Honesty skips
# ---------------------------------------------------------------------------
class TestHonestySkips:
    def test_missing_model_skips_model_invoke_entirely(self, mock_client):
        """model is required at ingest; a placeholder would be a FABRICATED model
        name, so the whole event is dropped."""
        uploaded = capture_framework_trace(mock_client)
        client = _sync_client(_ok_handler())
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)

        # The call itself succeeds; the adapter must still refuse to invent a
        # model name for it.
        result = _create(client, model=None)
        adapter.disconnect()

        assert _fields(result) == {"name": "John", "age": 30}

        assert not find_events(uploaded["events"], "model.invoke"), (
            "a create() with no model must emit NO model.invoke (never a placeholder)"
        )

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(True, id="bool-is-not-a-count"),
            pytest.param("3", id="str-is-not-a-count"),
        ],
    )
    def test_unreportable_max_retries_is_omitted_not_coerced(self, mock_client, value):
        uploaded = capture_framework_trace(mock_client)
        client = _sync_client(_ok_handler())
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)
        try:
            _create(client, max_retries=value)
        except Exception:
            # The customer's call may legitimately reject the value — what matters
            # is that the adapter never blind-int()s it, and reports no count.
            pass
        adapter.disconnect()

        for event in find_events(uploaded["events"], "model.invoke"):
            assert "max_retries_configured" not in event["payload"]

    def test_tenacity_retrying_object_does_not_break_the_call(self, mock_client):
        """A tenacity Retrying's stop-condition semantics cannot be honestly
        flattened to a number — omit the field, never break the call."""
        tenacity = pytest.importorskip("tenacity")
        uploaded = capture_framework_trace(mock_client)
        client = _sync_client(_ok_handler())
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)

        result = _create(client, max_retries=tenacity.Retrying(stop=tenacity.stop_after_attempt(2)))
        adapter.disconnect()

        assert _fields(result) == {"name": "John", "age": 30}, "a tenacity max_retries must still succeed"
        payload = find_event(uploaded["events"], "model.invoke")["payload"]
        assert "max_retries_configured" not in payload

    def test_provider_omitted_when_detection_fails(self):
        """No 'unknown' provider label — absence instead."""

        class Mystery:
            pass

        assert _detect_provider(Mystery()) is None

    def test_configured_max_retries_unit(self):
        assert _configured_max_retries({"max_retries": 3}) == 3
        assert _configured_max_retries({"max_retries": True}) is None
        assert _configured_max_retries({"max_retries": "3"}) is None
        assert _configured_max_retries({}) is None


# ---------------------------------------------------------------------------
# Agent identity — a framework label is NOT a producer-declared agent
# ---------------------------------------------------------------------------
class TestAgentIdentity:
    def test_unnamed_client_stamps_no_agent(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        client = _sync_client(_ok_handler())
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)
        _create(client)
        adapter.disconnect()

        for event in uploaded["events"]:
            payload = event["payload"]
            assert "agent_name" not in payload, "an unnamed instructor client must render an honest '—'"
            assert "agent_id" not in payload
        # And the collector must synthesize no identity from this trace.
        assert not find_events(uploaded["events"], "agent.identity")

    def test_caller_declared_agent_name_is_stamped(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        client = _sync_client(_ok_handler())
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client, agent_name="profile-extractor")
        _create(client)
        adapter.disconnect()

        payload = find_event(uploaded["events"], "model.invoke")["payload"]
        assert payload["agent_name"] == "profile-extractor"
        assert payload["agent_id"] == "profile-extractor"
        # It reaches the Agent column through the honest identity resolver.
        identity = find_event(uploaded["events"], "agent.identity")["payload"]
        assert identity["agent_name"] == "profile-extractor"

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param("instructor", id="framework-label"),
            pytest.param("Instructor", id="framework-label-cased"),
            pytest.param("unknown", id="placeholder"),
            pytest.param("agent", id="generic"),
            pytest.param("openai.chat.completions.create", id="api-method"),
            pytest.param("   ", id="blank"),
            pytest.param(None, id="unset"),
        ],
    )
    def test_dishonest_agent_names_are_rejected(self, name):
        assert _honest_agent_name(name) is None

    def test_framework_label_never_reaches_the_agent_column(self, mock_client):
        """Even an explicit agent_name='instructor' is refused: a framework name is
        not a producer-declared agent."""
        uploaded = capture_framework_trace(mock_client)
        client = _sync_client(_ok_handler())
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client, agent_name="instructor")
        _create(client)
        adapter.disconnect()

        for event in uploaded["events"]:
            assert "agent_name" not in event["payload"]
        assert not find_events(uploaded["events"], "agent.identity")


# ---------------------------------------------------------------------------
# Async — a real AsyncInstructor (async rides `create`, not `acreate`)
# ---------------------------------------------------------------------------
class TestAsync:
    def test_async_create_is_traced(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        client = _async_client(_ok_handler())
        # The async branch is selected by inspect.iscoroutinefunction; on modern
        # instructor an async client exposes `create` as a coroutine function and
        # has NO `acreate` at all.
        import inspect as _inspect

        assert _inspect.iscoroutinefunction(client.chat.completions.create)
        assert not hasattr(client, "acreate")

        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)

        async def drive() -> Any:
            return await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "John is 30"}],
                response_model=UserProfile,
            )

        result = asyncio.run(drive())
        adapter.disconnect()

        assert _fields(result) == {"name": "John", "age": 30}
        invokes = find_events(uploaded["events"], "model.invoke")
        assert len(invokes) == 1
        assert invokes[0]["payload"]["model"] == "gpt-4o-mini"

    def test_concurrent_calls_do_not_cross_attribute_retries(self, mock_client):
        """The client-global hook must attribute each observed error to its OWN
        call — the ContextVar is what makes that honest under concurrency."""
        uploaded = capture_framework_trace(mock_client)
        seen: List[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            who = body["messages"][0]["content"]
            seen.append(who)
            # Only "retry-me" ever fails validation, and only on its first pass.
            return httpx.Response(200, json=_tool_call_body({"name": who, "age": 30}))

        class Guarded(BaseModel):
            name: str
            age: int

            @field_validator("name")
            @classmethod
            def _fail_once(cls, value: str) -> str:
                if value == "retry-me" and seen.count("retry-me") < 2:
                    raise ValueError("first pass always fails for retry-me")
                return value

        client = _async_client(handler)
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)

        async def drive() -> Any:
            async def one(who: str) -> Any:
                return await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": who}],
                    response_model=Guarded,
                    max_retries=3,
                )

            return await asyncio.gather(one("retry-me"), one("clean-1"), one("clean-2"))

        asyncio.run(drive())
        adapter.disconnect()

        events = uploaded["events"]
        invokes = find_events(events, "model.invoke")
        assert len(invokes) == 3
        observed = sorted(e["payload"]["retries_observed"] for e in invokes)
        # Exactly ONE call retried; the other two must NOT inherit its retry.
        assert observed == [0, 0, 1], f"retries cross-attributed across concurrent calls: {observed}"

        # ...and it must be THE call that actually failed — a count-only assertion
        # cannot tell a correct attribution from a shuffled one. Identify each
        # invoke by the prompt it carries.
        def _who(event: Dict[str, Any]) -> str:
            return str(event["payload"]["messages"][0]["content"])

        by_who = {_who(e): e["payload"]["retries_observed"] for e in invokes}
        assert by_who == {"retry-me": 1, "clean-1": 0, "clean-2": 0}, (
            f"the observed retry landed on the wrong concurrent call: {by_who}"
        )

        retries = find_events(events, "tool.call")
        assert len(retries) == 1
        # The retry must be parented under the call that actually failed.
        owner = [e for e in invokes if _who(e) == "retry-me"][0]
        assert retries[0]["parent_span_id"] == owner["span_id"]


# ---------------------------------------------------------------------------
# Lifecycle — wrap / restore / re-instrument
# ---------------------------------------------------------------------------
class TestLifecycle:
    def test_disconnect_restores_the_original_and_stops_emitting(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        client = _sync_client(_ok_handler())
        original = client.create

        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)
        assert client.create is not original
        adapter.disconnect()

        assert client.create == original, "disconnect must restore the original create()"
        before = len(uploaded["events"])
        _create(client)
        assert len(uploaded["events"]) == before, "a restored client must emit nothing"

    def test_disconnect_does_not_clobber_a_third_partys_rewrap(self, mock_client):
        client = _sync_client(_ok_handler())
        adapter = InstructorAdapter(mock_client)
        adapter.connect(target=client)

        def foreign(*args: Any, **kwargs: Any) -> str:
            return "foreign"

        client.create = foreign
        adapter.disconnect()
        assert client.create is foreign, "a third party's later re-wrap must be left alone"

    def test_reinstrumenting_the_same_client_is_idempotent(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        client = _sync_client(_ok_handler())
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)
        adapter._instrument_client(client)
        _create(client)
        adapter.disconnect()

        assert len(find_events(uploaded["events"], "model.invoke")) == 1

    def test_second_adapter_on_a_traced_client_raises_truthfully(self, mock_client):
        """The methods exist and ARE traced — by another adapter. Reporting success
        would hand back an adapter that silently never emits."""
        client = _sync_client(_ok_handler())
        first = InstructorAdapter(mock_client)
        first.connect(target=client)

        second = InstructorAdapter(mock_client)
        with pytest.raises(RuntimeError, match="already instrumented by another adapter"):
            second.connect(target=client)
        first.disconnect()

    def test_connect_without_a_target_raises(self, mock_client):
        adapter = InstructorAdapter(mock_client)
        with pytest.raises(ValueError, match="requires a patched Instructor client"):
            adapter.connect()

    def test_swallowed_setattr_is_reported_not_silently_no_opped(self, mock_client):
        """A client whose __setattr__ swallows the assignment must NOT report a
        successful instrumentation: the adapter would emit nothing for the life of
        the process while claiming to be connected."""

        class Swallowing:
            def create(self, **kwargs: Any) -> str:
                return "untraced"

            def __setattr__(self, name: str, value: Any) -> None:
                # Silently ignores the write, like a guarded/frozen client.
                pass

        client = Swallowing()
        adapter = InstructorAdapter(mock_client)
        with pytest.raises(RuntimeError, match="could not install a wrapper"):
            adapter.connect(target=client)
        # And the failed attempt leaves no registry residue behind.
        assert id(client) not in adapter._wrapped_methods

    def test_no_create_method_raises_a_truthful_type_error(self, mock_client):
        class NotAClient:
            pass

        adapter = InstructorAdapter(mock_client)
        with pytest.raises(TypeError, match="could not locate a recognised create"):
            adapter.connect(target=NotAClient())

    def test_legacy_patched_raw_client_is_traced(self, mock_client):
        """``instructor.patch(OpenAI())`` returns the raw openai client with a
        patched chat.completions.create — a real second client shape."""
        uploaded = capture_framework_trace(mock_client)
        raw = OpenAI(api_key="sk-test", http_client=httpx.Client(transport=_transport(_ok_handler())))
        client = instructor.patch(raw)

        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)
        result = _create(client)
        adapter.disconnect()

        assert _fields(result) == {"name": "John", "age": 30}
        invokes = find_events(uploaded["events"], "model.invoke")
        assert len(invokes) == 1
        assert invokes[0]["payload"]["provider"] == "openai"

    def test_adapter_info_reports_the_real_installed_version(self, mock_client):
        client = _sync_client(_ok_handler())
        adapter = InstructorAdapter(mock_client)
        adapter.connect(target=client)
        info = adapter.adapter_info()
        assert info.name == "instructor"
        assert info.adapter_type == "framework"
        assert info.connected is True
        assert info.metadata["instructor_version"] == instructor.__version__
        adapter.disconnect()
        assert adapter.adapter_info().connected is False
