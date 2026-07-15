"""Real-behaviour unit suite for the Marvin framework adapter.

Every Marvin object here is real: the adapter patches the REAL ``marvin`` module
and the primitives are driven end-to-end through Marvin's own orchestrator. The
only mocked thing is the network — a pydantic-ai ``TestModel`` stands in for the
provider transport, which is exactly the seam Marvin resolves through
``Agent.get_model()``.

Marvin's model plumbing is the reason this suite exists in its real-object form:
no Marvin 3.x primitive accepts ``model=``, so the adapter's model resolution
MUST come off ``agent=`` / ``marvin.defaults`` and never out of a caller's
kwargs (see ``test_fn_kwargs_never_become_the_model``).
"""

from __future__ import annotations

import os
import sys
import uuid
import tempfile

import pytest

if sys.version_info < (3, 10):
    pytest.skip("marvin requires Python >= 3.10", allow_module_level=True)

# ``import marvin`` calls ensure_db_tables_exist() at module scope — point it at a
# throwaway file BEFORE the import so a test run never touches the user's real
# Marvin database.
os.environ.setdefault(
    "MARVIN_DATABASE_URL",
    "sqlite+aiosqlite:///" + os.path.join(tempfile.mkdtemp(prefix="layerlens-marvin-"), "marvin.db"),
)

marvin = pytest.importorskip("marvin", reason="marvin not installed")

from pydantic_ai.exceptions import UserError  # noqa: E402
from pydantic_ai.models.test import TestModel  # noqa: E402

from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.marvin import (  # noqa: E402
    MarvinAdapter,
    instrument_marvin,
    uninstrument_marvin,
)

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402

# Marvin's rich console handler renders a live panel per call — pure test noise.
marvin.settings.enable_default_print_handler = False


@pytest.fixture
def agent():
    """A REAL marvin.Agent with a developer-declared name and an offline model."""
    return marvin.Agent(name="Sentiment Analyst", model=TestModel())


@pytest.fixture
def adapter(mock_client):
    a = MarvinAdapter(mock_client, capture_config=CaptureConfig.full())
    a.connect()
    yield a
    a.disconnect()


# ---------------------------------------------------------------------------
# Patching / restoration over the REAL marvin module
# ---------------------------------------------------------------------------
class TestPatching:
    def test_connect_patches_every_primitive_and_fn(self, mock_client):
        originals = {name: getattr(marvin, name) for name in ("classify", "extract", "cast", "generate", "fn")}
        a = MarvinAdapter(mock_client)
        a.connect()
        try:
            for name in originals:
                assert getattr(getattr(marvin, name), "_layerlens_traced", False), f"{name} not patched"
            # The async surface is real and separately exported by marvin 3.x.
            for name in ("classify_async", "extract_async", "cast_async", "generate_async"):
                assert getattr(getattr(marvin, name), "_layerlens_traced", False), f"{name} not patched"
        finally:
            a.disconnect()
        for name, original in originals.items():
            assert getattr(marvin, name) is original, f"{name} not restored"

    def test_disconnect_leaves_a_third_party_patch_alone(self, mock_client):
        original = marvin.classify
        a = MarvinAdapter(mock_client)
        a.connect()

        def someone_elses_classify(*args, **kwargs):
            return "third-party"

        try:
            marvin.classify = someone_elses_classify
            a.disconnect()
            assert marvin.classify is someone_elses_classify, "disconnect clobbered a third party's later patch"
        finally:
            # The module is process-global — hand it back to the rest of the suite.
            marvin.classify = original

    def test_a_function_decorated_before_disconnect_stops_tracing(self, mock_client, agent):
        """disconnect() restores the module attributes, but a function ALREADY
        decorated by the wrapped @marvin.fn keeps its traced wrapper forever (the
        closure holds the adapter) — it cannot be un-instrumented. It must go inert:
        _begin_run() CREATES a collector when none is active, so a still-wrapped
        function would otherwise flush and UPLOAD a trace after the customer
        disconnected."""
        a = MarvinAdapter(mock_client, capture_config=CaptureConfig.full())
        a.connect()

        @marvin.fn(agent=agent)
        def headline(topic: str) -> str:
            """Write a headline."""

        uploaded = capture_framework_trace(mock_client)
        headline("interest rates")
        assert uploaded["events"], "the decorated function was not traced while connected"

        a.disconnect()
        before = mock_client.traces.upload.call_count
        result = headline("interest rates")
        assert mock_client.traces.upload.call_count == before, (
            "a function decorated before disconnect() uploaded a trace AFTER disconnect"
        )
        assert result is not None, "the call must still work — it is just no longer traced"

    def test_double_connect_does_not_double_wrap(self, mock_client, agent):
        a = MarvinAdapter(mock_client, capture_config=CaptureConfig.full())
        a.connect()
        first = marvin.classify
        a.connect()
        try:
            assert marvin.classify is first, "second connect() re-wrapped an already-traced primitive"
        finally:
            a.disconnect()


# ---------------------------------------------------------------------------
# tool.call — the unconditional honesty floor, over real primitive calls
# ---------------------------------------------------------------------------
class TestToolCall:
    def test_real_classify_emits_tool_call_with_labels(self, mock_client, adapter, agent):
        uploaded = capture_framework_trace(mock_client)
        result = marvin.classify("This product is amazing!", labels=["positive", "negative"], agent=agent)
        assert result in ("positive", "negative"), "real marvin.classify did not return a real label"

        call = find_event(uploaded["events"], "tool.call")
        p = call["payload"]
        assert p["framework"] == "marvin"
        assert p["tool_name"] == "marvin.classify"
        assert p["name"] == "marvin.classify"
        # NOT also agent_id: atlas nodes agent_id, so the primitive's own name
        # would render as the agent (see test_the_framework_label_is_never_a_graph_node).
        assert "agent_id" not in p
        assert p["primitive"] == "classify"
        assert p["success"] is True
        assert p["latency_ms"] > 0
        # A label SET is not a response model.
        assert p["labels"] == "positive, negative"
        assert "response_model" not in p

    def test_an_empty_label_set_is_omitted_not_stamped_empty(self, mock_client):
        """No labels discoverable -> the key is OMITTED, never an empty placeholder.

        Driven through the stub because real marvin rejects an empty label set
        before the adapter's emit path is reached.
        """
        stub = _StubMarvin(model="openai:gpt-4o")
        a = MarvinAdapter(mock_client, capture_config=CaptureConfig.full())
        a.connect(target=stub)
        uploaded = capture_framework_trace(mock_client)
        try:
            stub.classify("hi", labels=[])
        finally:
            a.disconnect()

        for event in find_events(uploaded["events"], "tool.call") + find_events(uploaded["events"], "model.invoke"):
            assert "labels" not in event["payload"], (
                f"an empty label set was stamped as a placeholder: {event['payload'].get('labels')!r}"
            )

    def test_real_cast_reports_the_target_as_response_model(self, mock_client, adapter, agent):
        uploaded = capture_framework_trace(mock_client)
        marvin.cast("42", target=int, agent=agent)
        p = find_event(uploaded["events"], "tool.call")["payload"]
        assert p["primitive"] == "cast"
        assert p["response_model"] == "int"
        assert "labels" not in p

    def test_real_generate_reports_its_first_positional_target(self, mock_client, adapter, agent):
        """generate(target=None, n=1, ...) puts target FIRST — extract/cast put it second."""
        uploaded = capture_framework_trace(mock_client)
        marvin.generate(int, 2, agent=agent)
        p = find_event(uploaded["events"], "tool.call")["payload"]
        assert p["primitive"] == "generate"
        assert p["response_model"] == "int", "generate's positional target was not captured"

    def test_tool_call_is_emitted_on_the_error_path(self, mock_client, adapter, agent):
        uploaded = capture_framework_trace(mock_client)
        # A REAL marvin API error: extracting strings without instructions is
        # rejected by marvin itself (marvin/fns/extract.py), not by a test double.
        with pytest.raises(ValueError):
            marvin.extract("a 3 bed home", target=str, agent=agent)

        p = find_event(uploaded["events"], "tool.call")["payload"]
        assert p["success"] is False
        assert p["error_type"] == "ValueError"
        assert "Instructions are required" in p["error"]

    def test_real_fn_decorator_traces_the_decorated_function(self, mock_client, adapter, agent):
        uploaded = capture_framework_trace(mock_client)

        @marvin.fn(agent=agent)
        def write_headline(topic: str) -> str:
            """Write a headline about the topic."""

        write_headline("interest rates")

        p = find_event(uploaded["events"], "tool.call")["payload"]
        assert p["primitive"] == "fn"
        assert p["tool_name"] == "marvin.write_headline"
        assert "agent_id" not in p

    def test_fn_decoration_alone_emits_nothing(self, mock_client, adapter, agent):
        """Decorating must not fabricate events — only CALLING the function traces."""
        uploaded = capture_framework_trace(mock_client)

        @marvin.fn(agent=agent)
        def unused(topic: str) -> str:
            """Never called."""

        assert uploaded["events"] == [], "decoration time emitted events for a call that never happened"


class TestAsyncSurface:
    """marvin 3.x exports classify_async/extract_async/cast_async/generate_async as
    SEPARATE module-level coroutine functions — not async variants reached through
    the sync names — so they need their own patch and their own coverage."""

    def test_real_classify_async_is_traced(self, mock_client, adapter, agent):
        import asyncio

        uploaded = capture_framework_trace(mock_client)
        result = asyncio.run(
            marvin.classify_async("This product is amazing!", labels=["positive", "negative"], agent=agent)
        )
        assert result in ("positive", "negative")

        p = find_event(uploaded["events"], "tool.call")["payload"]
        assert p["primitive"] == "classify_async"
        assert p["tool_name"] == "marvin.classify_async"
        assert p["labels"] == "positive, negative"
        assert p["success"] is True
        assert find_event(uploaded["events"], "model.invoke")["payload"]["model"] == "test"

    def test_async_error_path_is_traced_and_reraised(self, mock_client, adapter, agent):
        import asyncio

        uploaded = capture_framework_trace(mock_client)
        with pytest.raises(ValueError, match="Instructions are required"):
            asyncio.run(marvin.extract_async("a 3 bed home", target=str, agent=agent))

        p = find_event(uploaded["events"], "tool.call")["payload"]
        assert p["primitive"] == "extract_async"
        assert p["success"] is False
        assert p["error_type"] == "ValueError"


# ---------------------------------------------------------------------------
# model.invoke — resolved honestly or skipped outright
# ---------------------------------------------------------------------------
class TestModelResolution:
    def test_model_comes_from_the_real_agent(self, mock_client, adapter, monkeypatch):
        # Deleting the key both guarantees the offline failure below and makes it
        # impossible for this test to reach the network on a developer machine.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        uploaded = capture_framework_trace(mock_client)
        a = marvin.Agent(name="Analyst", model="anthropic:claude-3-5-sonnet-latest")
        # The Agent's model is resolved BEFORE the call, so model.invoke is honest
        # even though the call itself fails on the missing credential.
        with pytest.raises(UserError, match="ANTHROPIC_API_KEY"):
            marvin.classify("hi", labels=["a", "b"], agent=a)

        invoke = find_event(uploaded["events"], "model.invoke")["payload"]
        assert invoke["model"] == "anthropic:claude-3-5-sonnet-latest"
        assert invoke["model_name"] == invoke["model"]

    def test_model_falls_back_to_marvin_defaults(self, mock_client, adapter, agent):
        uploaded = capture_framework_trace(mock_client)
        marvin.classify("great", labels=["positive", "negative"], agent=agent)
        invoke = find_event(uploaded["events"], "model.invoke")["payload"]
        # TestModel is a real pydantic-ai Model — its real model_name is "test".
        assert invoke["model"] == "test"

    def test_no_model_no_model_invoke(self, mock_client, monkeypatch):
        """A marvin install with nothing discoverable emits tool.call and NO model.invoke."""
        stub = _StubMarvin(model=None)
        a = MarvinAdapter(mock_client, capture_config=CaptureConfig.full())
        a.connect(target=stub)
        uploaded = capture_framework_trace(mock_client)
        try:
            stub.classify("hi", labels=["a", "b"])
        finally:
            a.disconnect()

        assert find_events(uploaded["events"], "tool.call"), "the call must still be traced"
        assert find_events(uploaded["events"], "model.invoke") == [], (
            "a model.invoke was emitted with no real model — that is a fabricated model name"
        )

    def test_fn_kwargs_never_become_the_model(self, mock_client, adapter):
        """@marvin.fn kwargs are the USER'S arguments. ``model="Civic"`` is a CAR."""
        uploaded = capture_framework_trace(mock_client)
        a = marvin.Agent(name="Spec Writer", model=TestModel())

        @marvin.fn(agent=a)
        def spec(model: str) -> str:
            """Describe the car model."""

        spec(model="Civic")

        for invoke in find_events(uploaded["events"], "model.invoke"):
            assert invoke["payload"]["model"] != "Civic", (
                "the caller's own function argument was stamped as the LLM model name"
            )
            assert invoke["payload"]["model"] == "test"

    def test_fn_kwargs_never_become_the_response_model(self, mock_client, adapter):
        """Same class of bug: ``target`` on a user function is not a response model."""
        uploaded = capture_framework_trace(mock_client)
        a = marvin.Agent(name="Router", model=TestModel())

        @marvin.fn(agent=a)
        def route(target: str) -> str:
            """Route to the target queue."""

        route(target="billing")

        p = find_event(uploaded["events"], "tool.call")["payload"]
        assert "response_model" not in p, "a response_model was derived from the caller's own function argument"


# ---------------------------------------------------------------------------
# Agent identity — RULE 1, honest or absent
# ---------------------------------------------------------------------------
class TestAgentIdentity:
    def test_declared_agent_name_surfaces(self, mock_client, adapter, agent):
        uploaded = capture_framework_trace(mock_client)
        marvin.classify("great", labels=["positive", "negative"], agent=agent)
        p = find_event(uploaded["events"], "tool.call")["payload"]
        assert p["agent_name"] == "Sentiment Analyst"

    def test_marvins_random_default_name_is_not_an_identity(self, mock_client, adapter):
        """An unnamed marvin.Agent is auto-assigned a RANDOM name out of
        marvin.agents.names.AGENT_NAMES — a different one on each construction.
        Surfacing that would fabricate a per-run identity."""
        from marvin.agents.names import AGENT_NAMES

        uploaded = capture_framework_trace(mock_client)
        unnamed = marvin.Agent(model=TestModel())
        assert unnamed.name in AGENT_NAMES, "marvin no longer auto-names agents; revisit this guard"
        marvin.classify("great", labels=["positive", "negative"], agent=unnamed)

        p = find_event(uploaded["events"], "tool.call")["payload"]
        assert "agent_name" not in p, f"marvin's random default name {unnamed.name!r} leaked into the Agent column"

    #: The payload keys atlas's InferAgentGraph treats as a graph-node identity
    #: (apps/backend/services/graph_inference.go :: nodeIdentityFields). It reads
    #: ``agent_id`` as well as ``agent_name``, so withholding only ``agent_name``
    #: does NOT keep a label out of the Agent column.
    _GRAPH_NODE_IDENTITY_KEYS = (
        "node",
        "node_name",
        "agent",
        "agent_name",
        "agent_id",
        "agent_role",
        "plugin_name",
        "component_name",
    )

    def test_the_framework_label_is_never_a_graph_node(self, mock_client, adapter, agent):
        """``marvin`` is the framework's own name, not an agent a producer declared.

        ateam stamps ``agent_id="marvin"`` on environment.config. atlas nodes
        EVERY identity key, so that label renders as a second agent beside the
        real one and a single-agent extraction reports Agent = ``multi-agent``
        (proven against the real engine: marvin-s1 rendered
        ``['listing-extraction-agent', 'marvin']``). The topology must have one
        node — the declared agent.
        """
        uploaded = capture_framework_trace(mock_client)
        marvin.classify("great", labels=["positive", "negative"], agent=agent)

        identities = {
            payload[key]
            for event in uploaded["events"]
            for payload in (event["payload"],)
            for key in self._GRAPH_NODE_IDENTITY_KEYS
            if isinstance(payload.get(key), str) and payload[key].strip()
        }
        assert identities == {"Sentiment Analyst"}, (
            f"exactly the declared agent must be graph-node-visible; got {sorted(identities)}"
        )

    def test_bare_primitive_has_no_agent_name(self, mock_client, monkeypatch):
        """"marvin.classify" is an API-method label, not an agent — an honest "—"."""
        stub = _StubMarvin(model="openai:gpt-4o")
        a = MarvinAdapter(mock_client, capture_config=CaptureConfig.full())
        a.connect(target=stub)
        uploaded = capture_framework_trace(mock_client)
        try:
            stub.classify("hi", labels=["a", "b"])
        finally:
            a.disconnect()
        p = find_event(uploaded["events"], "tool.call")["payload"]
        # A bare primitive declares NO agent, so the trace carries none: the
        # topology is honestly empty rather than naming the function that ran.
        assert "agent_name" not in p
        assert "agent_id" not in p
        assert p["tool_name"] == "marvin.classify"


# ---------------------------------------------------------------------------
# environment.config
# ---------------------------------------------------------------------------
class TestEnvironmentConfig:
    def test_config_emitted_once_inside_the_first_run(self, mock_client, adapter, agent):
        uploaded = capture_framework_trace(mock_client)
        marvin.classify("great", labels=["positive", "negative"], agent=agent)
        marvin.classify("awful", labels=["positive", "negative"], agent=agent)

        configs = find_events(uploaded["events"], "environment.config")
        assert len(configs) == 1, f"expected exactly one environment.config, got {len(configs)}"
        cfg = configs[0]["payload"]
        # The framework is named in config, which atlas does NOT read as a node
        # identity — unlike agent_id, which ateam puts "marvin" in.
        assert "agent_id" not in cfg
        assert cfg["config"]["framework"] == "marvin"
        assert cfg["config"]["model"] == "openai:gpt-4o"

    def test_config_omits_what_it_cannot_discover(self, mock_client):
        stub = _StubMarvin(model=None)
        a = MarvinAdapter(mock_client, capture_config=CaptureConfig.full())
        a.connect(target=stub)
        uploaded = capture_framework_trace(mock_client)
        try:
            stub.classify("hi", labels=["a", "b"])
        finally:
            a.disconnect()
        cfg = find_event(uploaded["events"], "environment.config")["payload"]["config"]
        assert cfg == {"framework": "marvin"}, "config invented a model/provider it could not discover"


# ---------------------------------------------------------------------------
# Composition with the pydantic_ai layer (marvin 3.x rides pydantic-ai)
# ---------------------------------------------------------------------------
class TestComposition:
    def test_marvin_defers_to_a_deeper_model_invoke(self, mock_client):
        """When a layer under the call already reported the request, marvin's own
        tokenless model.invoke must not duplicate it.

        The inner layer emits through the REAL provider path
        (``providers/_emit_helpers.emit_llm_events``) rather than a hand-rolled
        ``collector.emit``: that helper is what an ``instrument_openai`` under
        marvin actually runs, and its span parenting (``parent_span_id =
        _current_span_id.get()``) is precisely the signal the dedup keys off. A
        synthetic emit with ``parent_span_id=None`` would test a shape no adapter
        in this tree produces.

        A provider adapter — not the pydantic_ai one — is the realistic deeper
        layer here: pydantic_ai's adapter needs an explicit Agent target, and
        marvin builds its internal agent per call where no caller can reach it.
        """
        from layerlens.instrument.adapters.providers._emit_helpers import emit_llm_events

        stub = _StubMarvin(model="openai:gpt-4o")

        def inner_reports_a_model_invoke(*args, **kwargs):
            emit_llm_events(
                name="openai.chat.completions",
                kwargs={"model": "openai:gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
                response=None,
                extract_output=lambda _r: "positive",
                extract_meta=lambda _r: {"usage": {"prompt_tokens": 30, "completion_tokens": 12}},
                capture_params=frozenset({"model"}),
                latency_ms=1.0,
                framework="openai",
            )
            return "positive"

        stub.classify = inner_reports_a_model_invoke
        a = MarvinAdapter(mock_client, capture_config=CaptureConfig.full())
        a.connect(target=stub)
        uploaded = capture_framework_trace(mock_client)
        try:
            stub.classify("hi", labels=["a", "b"])
        finally:
            a.disconnect()

        invokes = find_events(uploaded["events"], "model.invoke")
        assert len(invokes) == 1, f"marvin double-emitted model.invoke: {[i['payload'] for i in invokes]}"
        assert invokes[0]["payload"]["framework"] == "openai", "marvin's tokenless duplicate won over the real one"
        assert invokes[0]["payload"]["total_tokens"] == 42, "the surviving model.invoke lost the real token counts"
        assert find_events(uploaded["events"], "tool.call"), "the marvin tool.call must still be emitted"

    def test_concurrent_calls_sharing_one_collector_all_keep_their_model_invoke(self, mock_client, adapter, agent):
        """Batch classification under ``trace_context()`` + ``asyncio.gather`` is
        marvin's canonical async shape: N calls share ONE collector.

        The deeper-layer dedup must be scoped to the CALL that opened it, not to
        the trace. A trace-wide model.invoke count makes every call but the first
        read a concurrent SIBLING's event as "a deeper layer already reported my
        request" and silently drop its own — pure telemetry loss, and invisible
        to a sequential single-call composition lane.
        """
        import asyncio

        from layerlens.instrument import trace_context

        texts = ("a", "b", "c", "d")
        uploaded = capture_framework_trace(mock_client)

        async def drive():
            with trace_context(mock_client, capture_config=CaptureConfig.full()):
                await asyncio.gather(
                    *[marvin.classify_async(t, labels=["positive", "negative"], agent=agent) for t in texts]
                )

        asyncio.run(drive())

        calls = find_events(uploaded["events"], "tool.call")
        invokes = find_events(uploaded["events"], "model.invoke")
        assert len(calls) == len(texts), f"{len(calls)}/{len(texts)} tool.call survived"
        assert len(invokes) == len(texts), (
            f"only {len(invokes)}/{len(texts)} model.invoke survived — the dedup ate concurrent siblings"
        )

    def test_marvin_emits_its_own_when_nothing_deeper_did(self, mock_client):
        stub = _StubMarvin(model="openai:gpt-4o")
        a = MarvinAdapter(mock_client, capture_config=CaptureConfig.full())
        a.connect(target=stub)
        uploaded = capture_framework_trace(mock_client)
        try:
            stub.classify("hi", labels=["a", "b"])
        finally:
            a.disconnect()
        invokes = find_events(uploaded["events"], "model.invoke")
        assert len(invokes) == 1
        assert invokes[0]["payload"]["framework"] == "marvin"


# ---------------------------------------------------------------------------
# Cost — the honest omission
# ---------------------------------------------------------------------------
class TestCostOmission:
    def test_no_cost_record_because_marvin_exposes_no_usage(self, mock_client, adapter, agent):
        uploaded = capture_framework_trace(mock_client)
        marvin.classify("great", labels=["positive", "negative"], agent=agent)
        assert find_events(uploaded["events"], "cost.record") == [], (
            "a cost.record was emitted although marvin exposes no token usage — that is fabricated cost"
        )
        for invoke in find_events(uploaded["events"], "model.invoke"):
            p = invoke["payload"]
            assert "tokens_total" not in p and "cost_usd" not in p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
class TestEntryPoint:
    def test_instrument_marvin_patches_and_registers(self, mock_client, agent):
        original = marvin.classify
        a = instrument_marvin(mock_client, capture_config=CaptureConfig.full())
        try:
            assert getattr(marvin.classify, "_layerlens_traced", False)
            from layerlens.instrument.adapters._registry import get

            assert get("marvin") is a
        finally:
            uninstrument_marvin()
        assert marvin.classify is original


# ---------------------------------------------------------------------------
# Offline stub — a marvin-shaped module with NO settings/defaults, used to drive
# the honesty-skip branches that a real install (which always resolves a default
# model) cannot reach.
# ---------------------------------------------------------------------------
class _StubMarvin:
    def __init__(self, model):
        if model is not None:
            self.defaults = type("Defaults", (), {"model": model})()
            self.settings = type("Settings", (), {"agent_model": model})()
        self.__version__ = "3.2.7-stub"

    def classify(self, data, labels=None, **kwargs):
        return "positive"

    def extract(self, data, target=None, **kwargs):
        return [str(uuid.uuid4())]

    def cast(self, data, target=None, **kwargs):
        return data

    def generate(self, target=None, n=1, **kwargs):
        return [target] * n

    def fn(self, func=None, **kwargs):
        def decorator(f):
            return f

        return decorator(func) if callable(func) else decorator
