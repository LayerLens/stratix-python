"""Real-behaviour suite for the Mirascope v2 framework adapter.

Every lane drives the REAL ``mirascope.llm`` surface — the real ``@llm.call``
decorator, the real ``Call``/``AsyncCall``/``ContextCall`` objects, the real
provider and the real ``llm.Response`` — with an ``httpx.MockTransport`` as the
only fake (see ``_mirascope_support``). Nothing here asserts against a stub of
our own adapter, and no lane would survive the v1 ``mirascope.core`` API the
ateam reference targets: these tests only pass against an adapter that hooks the
API mirascope 2.x actually ships.
"""

from __future__ import annotations

import sys
import json
import asyncio

import pytest

if sys.version_info < (3, 10):  # pragma: no cover - matrix pins 3.11
    pytest.skip("mirascope 2.x requires Python >= 3.10", allow_module_level=True)

pytest.importorskip("mirascope.llm", reason="mirascope not installed")
pytest.importorskip("openai", reason="mirascope[openai] not installed")

import mirascope.llm as llm  # noqa: E402  # pyright: ignore[reportMissingImports]
from pydantic import BaseModel  # noqa: E402

from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.mirascope import (  # noqa: E402
    MirascopeAdapter,
    instrument_mirascope,
    uninstrument_mirascope,
)

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402
from ._mirascope_support import (  # noqa: E402
    MODEL_ID,
    BARE_MODEL,
    json_body,
    ok_handler,
    call_classes,
    completion_body,
    mirascope_openai,
    not_found_handler,
    recording_handler,
    restore_call_classes,
)


@pytest.fixture(autouse=True)
def _no_leaked_adapter():
    """A leaked class-level patch would silently taint every later lane."""
    yield
    uninstrument_mirascope()
    restore_call_classes()


def _adapter(mock_client, **kw):
    return MirascopeAdapter(mock_client, capture_config=CaptureConfig.full(), **kw)


# ---------------------------------------------------------------------------
# Happy path — sync
# ---------------------------------------------------------------------------
class TestSyncHappyPath:
    def test_real_call_emits_full_event_set(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        with mirascope_openai(ok_handler()):
            adapter = _adapter(mock_client)
            adapter.connect()

            @llm.call(MODEL_ID)
            def recommend_book(genre: str):
                return f"Recommend a {genre} book"

            response = recommend_book("fantasy")
            adapter.disconnect()

        # The real response still reaches the caller untouched.
        assert response.model_id == MODEL_ID
        events = uploaded["events"]

        call = find_event(events, "tool.call")["payload"]
        assert call["framework"] == "mirascope"
        assert call["tool_name"] == "mirascope.recommend_book"
        # RULE 1: agent_id is ateam's key; agent_name is what the Agent column reads.
        assert call["agent_id"] == "recommend_book"
        assert call["agent_name"] == "recommend_book"
        assert call["success"] is True
        assert call["input"]["args"] == ["fantasy"]

        result = find_event(events, "tool.result")["payload"]
        assert result["success"] is True
        assert "Dune by Frank Herbert" in json.dumps(result["output"])
        assert "error" not in result

        invoke = find_event(events, "model.invoke")["payload"]
        # The bare, priceable id — NOT the slash-namespaced v2 ModelId.
        assert invoke["model"] == BARE_MODEL
        assert invoke["model_id"] == MODEL_ID
        assert invoke["provider"] == "openai"
        assert invoke["function_name"] == "recommend_book"
        assert invoke["agent_name"] == "recommend_book"
        # Real usage off the real Response.
        assert invoke["tokens_prompt"] == 17
        assert invoke["tokens_completion"] == 6
        assert invoke["tokens_total"] == 23
        assert invoke["latency_ms"] >= 0

    def test_latency_is_real_measured_time(self, mock_client):
        """latency_ms must track real wall time, not a constant."""
        uploaded = capture_framework_trace(mock_client)

        def slow(request):
            import time

            time.sleep(0.05)
            return __import__("httpx").Response(200, json=completion_body())

        with mirascope_openai(slow):
            adapter = _adapter(mock_client)
            adapter.connect()

            @llm.call(MODEL_ID)
            def slow_call(x: str):
                return x

            slow_call("hi")
            adapter.disconnect()

        call = find_event(uploaded["events"], "tool.call")["payload"]
        assert call["latency_ms"] >= 50, f"latency_ms {call['latency_ms']} did not measure the 50ms call"

    def test_span_tree_parents_events_under_the_run_root(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        with mirascope_openai(ok_handler()):
            adapter = _adapter(mock_client)
            adapter.connect()

            @llm.call(MODEL_ID)
            def recommend_book(genre: str):
                return f"Recommend a {genre} book"

            recommend_book("fantasy")
            adapter.disconnect()

        events = uploaded["events"]
        call = find_event(events, "tool.call")
        invoke = find_event(events, "model.invoke")
        assert call["span_name"] == "mirascope:recommend_book"
        assert call["parent_span_id"] == invoke["parent_span_id"], "siblings must share the run root"
        assert call["span_id"] != invoke["span_id"]


# ---------------------------------------------------------------------------
# Async + context variants (the other three real Call classes)
# ---------------------------------------------------------------------------
class TestAsyncAndContext:
    def test_async_call_is_traced(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        with mirascope_openai(ok_handler()):
            adapter = _adapter(mock_client)
            adapter.connect()

            @llm.call(MODEL_ID)
            async def arecommend(genre: str):
                return f"Recommend a {genre} book"

            response = asyncio.run(arecommend("scifi"))
            adapter.disconnect()

        assert response.model_id == MODEL_ID
        events = uploaded["events"]
        assert find_event(events, "tool.call")["payload"]["tool_name"] == "mirascope.arecommend"
        assert find_event(events, "model.invoke")["payload"]["model"] == BARE_MODEL

    def test_context_call_is_traced(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        with mirascope_openai(ok_handler()):
            adapter = _adapter(mock_client)
            adapter.connect()

            @llm.call(MODEL_ID)
            def ctx_recommend(ctx: llm.Context[str], genre: str):
                return f"Recommend a {genre} book for {ctx.deps}"

            ctx_recommend(llm.Context(deps="alice"), "fantasy")
            adapter.disconnect()

        events = uploaded["events"]
        assert find_event(events, "tool.call")["payload"]["tool_name"] == "mirascope.ctx_recommend"
        assert find_event(events, "model.invoke")["payload"]["model"] == BARE_MODEL

    def test_concurrent_async_runs_do_not_leak_across_tasks(self, mock_client):
        """ContextVar isolation: each task's events carry only its own function."""
        traces = []

        def _capture(path):
            with open(path) as f:
                traces.append(json.load(f)[0])

        mock_client.traces.upload.side_effect = _capture

        with mirascope_openai(ok_handler()):
            adapter = _adapter(mock_client)
            adapter.connect()

            @llm.call(MODEL_ID)
            async def alpha(x: str):
                return x

            @llm.call(MODEL_ID)
            async def beta(x: str):
                return x

            async def drive():
                await asyncio.gather(*[alpha("a") for _ in range(4)], *[beta("b") for _ in range(4)])

            asyncio.run(drive())
            adapter.disconnect()

        assert len(traces) == 8, f"expected one flushed trace per concurrent run, got {len(traces)}"
        for trace in traces:
            names = {e["payload"]["tool_name"] for e in trace["events"] if e["event_type"] == "tool.call"}
            assert len(names) == 1, f"a run's trace mixed functions from another task: {names}"


# ---------------------------------------------------------------------------
# Honesty floor — the error path and the model.invoke SKIP
# ---------------------------------------------------------------------------
class TestErrorPath:
    def test_real_sdk_error_keeps_the_call_visible_and_reraises(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        with mirascope_openai(not_found_handler()):
            adapter = _adapter(mock_client)
            adapter.connect()

            @llm.call(MODEL_ID)
            def doomed(x: str):
                return x

            try:
                with pytest.raises(llm.exceptions.NotFoundError) as excinfo:
                    doomed("hi")
            finally:
                adapter.disconnect()

        # The real mirascope exception propagates unchanged — the finally-block
        # emission must not swallow or replace it.
        assert type(excinfo.value).__name__ == "NotFoundError"
        assert "does not exist" in str(excinfo.value)

        events = uploaded["events"]
        # The floor: a failed call is never an invisible call.
        call = find_event(events, "tool.call")["payload"]
        assert call["success"] is False
        assert call["error_type"] == "NotFoundError"

        result = find_event(events, "tool.result")["payload"]
        assert result["success"] is False
        assert result["error_type"] == "NotFoundError"
        assert "does not exist" in result["error"]
        assert "output" not in result

    def test_model_invoke_still_emitted_on_error_from_the_decorator_model(self, mock_client):
        """No response, but the decorator declared a real model — so it is known."""
        uploaded = capture_framework_trace(mock_client)
        with mirascope_openai(not_found_handler()):
            adapter = _adapter(mock_client)
            adapter.connect()

            @llm.call(MODEL_ID)
            def doomed(x: str):
                return x

            try:
                with pytest.raises(llm.exceptions.NotFoundError):
                    doomed("hi")
            finally:
                adapter.disconnect()

        invoke = find_event(uploaded["events"], "model.invoke")["payload"]
        assert invoke["model"] == BARE_MODEL
        assert invoke["error_type"] == "NotFoundError"
        # provider is honestly OMITTED: there is no response to read provider_id
        # from, and the model_id prefix is not reliably the provider id.
        assert "provider" not in invoke
        # No tokens were reported, so no cost may be claimed.
        assert not find_events(uploaded["events"], "cost.record")

    def test_no_model_means_no_model_invoke(self, mock_client):
        """THE core honesty SKIP — an unattributable call emits tool.call only.

        Asserted NEGATIVELY: if the adapter ever invents a placeholder model
        ('unknown'/'mirascope'/the provider name) this fails.
        """
        uploaded = capture_framework_trace(mock_client)
        adapter = _adapter(mock_client)
        adapter.connect()

        # A real Call whose model cannot be resolved (no registered provider and
        # a model_id the adapter must refuse to guess at).
        class _ModellessCall:
            __name__ = "modelless"

            def call(self, *args, **kwargs):
                return object()

        target = _ModellessCall()
        adapter.traced_call(target)
        target.call("x")
        adapter.disconnect()

        events = uploaded["events"]
        assert find_events(events, "tool.call"), "the call must stay visible via tool.call"
        assert find_events(events, "model.invoke") == [], "fabricated a model.invoke with no real model"
        assert find_events(events, "cost.record") == [], "priced a call with no model"
        for e in events:
            assert e["payload"].get("model") is None


# ---------------------------------------------------------------------------
# Fabrication regressions (census R2 / R4)
# ---------------------------------------------------------------------------
class TestNoFabricatedModel:
    def test_model_is_never_a_model_object_repr(self, mock_client):
        """R2: ``Response.model`` is a Model OBJECT with no __str__ — stringifying
        it ships '<mirascope.llm.models.models.Model object at 0x...>' as the
        model name. The honest field is ``Response.model_id``."""
        uploaded = capture_framework_trace(mock_client)
        with mirascope_openai(ok_handler()):
            adapter = _adapter(mock_client)
            adapter.connect()

            @llm.call(MODEL_ID)
            def recommend_book(genre: str):
                return f"Recommend a {genre} book"

            response = recommend_book("fantasy")
            adapter.disconnect()

        # Prove the leak is real and would be caught (this is the value we must
        # never emit).
        assert "mirascope.llm.models" in str(response.model)

        blob = json.dumps(uploaded["events"])
        assert "object at 0x" not in blob, "a Model object repr leaked into the trace"
        assert "mirascope.llm.models" not in blob, "a Model object repr leaked into the trace"

    def test_structured_output_field_named_model_does_not_poison(self, mock_client):
        """R4: a ``format=`` class with a field literally named ``model`` must not
        become the model name."""

        class Car(BaseModel):
            make: str
            model: str

        uploaded = capture_framework_trace(mock_client)
        body = json_body({"make": "Tesla", "model": "Model 3"})
        with mirascope_openai(ok_handler(body)):
            adapter = _adapter(mock_client)
            adapter.connect()

            @llm.call(MODEL_ID, format=Car)
            def extract_car(text: str):
                return f"extract: {text}"

            response = extract_car("a tesla")
            adapter.disconnect()

        assert response.parse().model == "Model 3"  # the car, not the LLM
        invoke = find_event(uploaded["events"], "model.invoke")["payload"]
        assert invoke["model"] == BARE_MODEL, "a user field named 'model' poisoned the model name"
        assert invoke["response_model"] == "Car"


# ---------------------------------------------------------------------------
# Cost (census R11 — v2 model ids are unpriceable until normalised)
# ---------------------------------------------------------------------------
class TestCost:
    def test_cost_record_is_priced_from_the_normalised_model(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        with mirascope_openai(ok_handler()):
            adapter = _adapter(mock_client)
            adapter.connect()

            @llm.call(MODEL_ID)
            def recommend_book(genre: str):
                return f"Recommend a {genre} book"

            recommend_book("fantasy")
            adapter.disconnect()

        cost = find_event(uploaded["events"], "cost.record")["payload"]
        assert cost["model"] == BARE_MODEL
        assert cost["tokens_prompt"] == 17
        assert cost["tokens_completion"] == 6
        # Real pricing off the real token counts — never a fabricated 0.0.
        assert cost["cost_usd"] > 0, "v2 model id was not normalised, so the trace ships unpriced"
        # The provider is resolved for model.invoke, so cost.record must carry it
        # too: a spend row that cannot say WHO was billed is unattributable, and
        # the dspy/instructor cost rows both name it.
        invoke = find_event(uploaded["events"], "model.invoke")["payload"]
        assert cost["provider"] == invoke["provider"]

    def test_no_tokens_means_no_cost_record(self, mock_client):
        """Honest omission: a response with no usage may not be priced."""
        uploaded = capture_framework_trace(mock_client)
        body = completion_body()
        body.pop("usage")
        with mirascope_openai(ok_handler(body)):
            adapter = _adapter(mock_client)
            adapter.connect()

            @llm.call(MODEL_ID)
            def recommend_book(genre: str):
                return f"Recommend a {genre} book"

            recommend_book("fantasy")
            adapter.disconnect()

        events = uploaded["events"]
        assert find_events(events, "model.invoke"), "the call is still attributable"
        assert find_events(events, "cost.record") == [], "priced a call with no reported tokens"
        invoke = find_event(events, "model.invoke")["payload"]
        assert "tokens_prompt" not in invoke


# ---------------------------------------------------------------------------
# environment.config
# ---------------------------------------------------------------------------
class TestEnvironmentConfig:
    def test_config_emitted_once_and_is_honest(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        with mirascope_openai(ok_handler()):
            adapter = _adapter(mock_client)
            adapter.connect()

            @llm.call(MODEL_ID)
            def recommend_book(genre: str):
                return f"Recommend a {genre} book"

            recommend_book("fantasy")
            recommend_book("scifi")
            adapter.disconnect()

        configs = find_events(uploaded["events"], "environment.config")
        assert len(configs) == 1, f"environment.config must be one-shot, saw {len(configs)}"
        cfg = configs[0]["payload"]["config"]
        # Honest: the real classes we actually patched, not a provider list we
        # never touched.
        assert set(cfg["patched_calls"]) == {"AsyncCall", "AsyncContextCall", "Call", "ContextCall"}
        assert cfg["framework_version"].startswith("2."), cfg["framework_version"]
        # RULE 1: 'mirascope' is a framework label — it must never reach the
        # Agent column.
        assert "agent_name" not in configs[0]["payload"]

    def test_no_config_without_a_patch(self, mock_client):
        """traced_call() patches nothing globally, so it claims no configuration."""
        uploaded = capture_framework_trace(mock_client)
        with mirascope_openai(ok_handler()):
            adapter = _adapter(mock_client)

            @llm.call(MODEL_ID)
            def recommend_book(genre: str):
                return f"Recommend a {genre} book"

            adapter.traced_call(recommend_book)
            recommend_book("fantasy")

        events = uploaded["events"]
        assert find_events(events, "tool.call"), "traced_call must still trace"
        assert find_events(events, "environment.config") == [], "claimed a config with nothing patched"


# ---------------------------------------------------------------------------
# Patch lifecycle
# ---------------------------------------------------------------------------
class TestPatchLifecycle:
    def test_disconnect_restores_the_real_call_methods(self, mock_client):
        originals = {c.__name__: c.__dict__["call"] for c in call_classes()}
        adapter = _adapter(mock_client)
        adapter.connect()
        assert all(c.__dict__["call"] is not originals[c.__name__] for c in call_classes()), "connect() patched nothing"
        adapter.disconnect()
        for cls in call_classes():
            assert cls.__dict__["call"] is originals[cls.__name__], f"{cls.__name__}.call not restored"

    def test_disconnect_leaves_a_third_party_repatch_alone(self, mock_client):
        adapter = _adapter(mock_client)
        adapter.connect()
        third_party = llm.Call.__dict__["call"]
        marker = object()

        def other(self, *a, **k):
            return marker

        llm.Call.call = other
        adapter.disconnect()
        assert llm.Call.__dict__["call"] is other, "clobbered a third party's re-patch"
        llm.Call.call = getattr(third_party, "__wrapped__", third_party)

    def test_double_instrument_emits_one_event_per_call(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        with mirascope_openai(ok_handler()):
            first = _adapter(mock_client)
            first.connect()
            second = _adapter(mock_client)
            second.connect()  # must not wrap the already-traced method

            @llm.call(MODEL_ID)
            def recommend_book(genre: str):
                return f"Recommend a {genre} book"

            recommend_book("fantasy")
            second.disconnect()
            first.disconnect()

        calls = find_events(uploaded["events"], "tool.call")
        assert len(calls) == 1, f"double-wrapped: {len(calls)} tool.call events for one invocation"

    def test_instrumentation_survives_predecorated_functions(self, mock_client):
        """A class-level seam traces functions decorated BEFORE instrumentation —
        the common case, since decorators run at import time."""
        with mirascope_openai(ok_handler()):

            @llm.call(MODEL_ID)
            def early(genre: str):
                return f"Recommend a {genre} book"

            uploaded = capture_framework_trace(mock_client)
            adapter = _adapter(mock_client)
            adapter.connect()  # after the decorator already ran
            early("fantasy")
            adapter.disconnect()

        assert find_events(uploaded["events"], "tool.call"), "pre-decorated function was not traced"

    def test_one_network_call_per_invocation(self, mock_client):
        """The wrapper must delegate exactly once — never re-run the user's call."""
        seen = []
        capture_framework_trace(mock_client)
        with mirascope_openai(recording_handler(seen)):
            adapter = _adapter(mock_client)
            adapter.connect()

            @llm.call(MODEL_ID)
            def recommend_book(genre: str):
                return f"Recommend a {genre} book"

            recommend_book("fantasy")
            adapter.disconnect()

        assert len(seen) == 1, f"the traced call hit the network {len(seen)} times"


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------
class TestInstrumentEntryPoint:
    def test_instrument_mirascope_registers_and_traces(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        with mirascope_openai(ok_handler()):
            adapter = instrument_mirascope(mock_client, capture_config=CaptureConfig.full())
            assert isinstance(adapter, MirascopeAdapter)
            assert adapter.adapter_info().connected

            @llm.call(MODEL_ID)
            def recommend_book(genre: str):
                return f"Recommend a {genre} book"

            recommend_book("fantasy")
            uninstrument_mirascope()

        assert find_events(uploaded["events"], "tool.call")
        # uninstrument must restore the seam.
        assert not getattr(llm.Call.__dict__["call"], "_layerlens_traced", False)
