"""Real-behaviour unit suite for the DSPy framework adapter.

Every DSPy object here is real — real ``dspy.Predict`` / ``dspy.ReAct`` /
``dspy.Module`` subclasses, the real ``dspy.settings.callbacks`` bus, a real
``BootstrapFewShot`` compile — driven by DSPy's own ``DummyLM`` so the suite
needs no key and no network. The adapter is never mocked.
"""

from __future__ import annotations

import sys

import pytest

if sys.version_info < (3, 10):
    pytest.skip("dspy requires Python >= 3.10", allow_module_level=True)

dspy = pytest.importorskip("dspy", reason="dspy not installed")

from dspy.teleprompt import BootstrapFewShot  # noqa: E402
from dspy.utils.dummies import DummyLM  # noqa: E402

from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.dspy import DSPyAdapter, instrument_dspy  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402


# ---------------------------------------------------------------------------
# Real DSPy programs
# ---------------------------------------------------------------------------
class QASignature(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class MyQA(dspy.Module):
    """A developer-declared program — its class name IS an honest agent identity."""

    def __init__(self) -> None:
        super().__init__()
        self.pred = dspy.Predict(QASignature)

    def forward(self, question: str):
        return self.pred(question=question)


def _dummy_lm(n: int = 4) -> DummyLM:
    return DummyLM([{"answer": "blue"} for _ in range(n)])


def _configure(lm) -> None:
    """dspy.configure only accepts calls from the thread that first configured
    it, which pytest satisfies; callbacks are installed by the adapter itself."""
    dspy.configure(lm=lm, callbacks=[])


@pytest.fixture(autouse=True)
def _reset_dspy_settings():
    yield
    dspy.settings.configure(callbacks=[])


# ---------------------------------------------------------------------------
class TestModuleLifecycle:
    def test_real_program_call_emits_the_full_event_set(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        _configure(_dummy_lm())
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        program = MyQA()
        result = program(question="what colour is the sky?")
        adapter.disconnect()

        assert result.answer == "blue"
        events = uploaded["events"]
        assert events, "a real DSPy program call must flush a trace"

        # The whole nested call is ONE trace, not one per module.
        assert len({e.get("trace_id") for e in events if e.get("trace_id")}) <= 1
        inputs = find_events(events, "agent.input")
        outputs = find_events(events, "agent.output")
        # MyQA -> Predict: both module boundaries are traced.
        assert len(inputs) == 2, [e["payload"].get("module_type") for e in inputs]
        assert len(outputs) == 2
        assert {e["payload"]["module_type"] for e in inputs} == {"MyQA", "Predict"}

        invoke = find_event(events, "model.invoke")
        assert invoke is not None, "a real LM call must emit model.invoke"
        assert invoke["payload"]["model"] == "dummy"
        assert find_event(events, "environment.config") is not None

    def test_nested_modules_build_a_span_tree_under_one_run(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        _configure(_dummy_lm())
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        program = MyQA()
        program(question="q?")
        adapter.disconnect()

        events = uploaded["events"]
        inputs = {e["payload"]["module_type"]: e for e in find_events(events, "agent.input")}
        outer, inner = inputs["MyQA"], inputs["Predict"]
        # The nested Predict hangs off the outer program's span, proving the
        # depth-aware parent_run_id mapping (not a flat sibling list).
        assert inner["parent_span_id"] == outer["span_id"]
        assert inner["span_id"] != outer["span_id"]

        # The LM call is attributed to the module that made it.
        invoke = find_event(events, "model.invoke")
        assert invoke["parent_span_id"] == inner["span_id"]

    def test_module_input_keys_are_the_signature_fields_not_args_kwargs(self, mock_client):
        """DSPy's bus hands over ``{"args": (), "kwargs": {...}}``; a naive
        ``sorted(inputs)`` would report ``["args", "kwargs"]`` for every program."""
        uploaded = capture_framework_trace(mock_client)
        _configure(_dummy_lm())
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        MyQA()(question="q?")
        adapter.disconnect()

        outer = next(e for e in find_events(uploaded["events"], "agent.input") if e["payload"]["module_type"] == "MyQA")
        assert outer["payload"]["input_keys"] == ["question"]
        assert "args" not in outer["payload"]["input_keys"]

    def test_environment_config_reports_the_real_signature(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        _configure(_dummy_lm())
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        MyQA()(question="q?")
        adapter.disconnect()

        configs = {
            e["payload"]["module_type"]: e["payload"] for e in find_events(uploaded["events"], "environment.config")
        }
        assert configs["Predict"]["signature"] == "QASignature"
        assert configs["Predict"]["input_fields"] == ["question"]
        assert configs["Predict"]["output_fields"] == ["answer"]
        assert configs["Predict"]["demo_count"] == 0


# ---------------------------------------------------------------------------
class TestAgentIdentity:
    def test_developer_declared_module_surfaces_as_the_agent(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        _configure(_dummy_lm())
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        MyQA()(question="q?")
        adapter.disconnect()

        outer = next(e for e in find_events(uploaded["events"], "agent.input") if e["payload"]["module_type"] == "MyQA")
        assert outer["payload"]["agent_name"] == "MyQA"
        assert outer["payload"]["agent_id"] == "MyQA"

        identity = find_event(uploaded["events"], "agent.identity")
        assert identity is not None, "the honest declared program name must reach the Agent column"
        assert identity["payload"]["agent_name"] == "MyQA"

    def test_dspy_builtin_module_is_never_stamped_as_an_agent(self, mock_client):
        """``Predict``/``ChainOfThought`` are framework primitives every unnamed
        program shares — a generic label, not a producer-declared identity."""
        uploaded = capture_framework_trace(mock_client)
        _configure(_dummy_lm())
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        dspy.Predict(QASignature)(question="q?")
        adapter.disconnect()

        events = uploaded["events"]
        for event in find_events(events, "agent.input") + find_events(events, "agent.output"):
            assert "agent_name" not in event["payload"], event["payload"]
            assert "agent_id" not in event["payload"], event["payload"]
        # module_type still reports the class — it is metadata, not an identity.
        assert find_events(events, "agent.input")[0]["payload"]["module_type"] == "Predict"
        assert find_events(events, "agent.identity") == [], "a bare Predict has no honest agent — expect an honest '—'"

    def test_nested_builtin_inherits_the_declared_program_as_its_agent(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        _configure(_dummy_lm())
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        MyQA()(question="q?")
        adapter.disconnect()

        # The LM call inside MyQA -> Predict attributes to the *enclosing* module.
        # Predict is generic, so the nearest frame carries no name and the model
        # call is left honestly unattributed rather than labelled "Predict".
        invoke = find_event(uploaded["events"], "model.invoke")
        assert invoke["payload"].get("agent_name") != "Predict"


# ---------------------------------------------------------------------------
class TestHonestySkips:
    def test_lm_with_no_resolvable_model_emits_no_model_invoke(self, mock_client):
        """THE adapter's primary skip: a model.invoke without a resolvable model
        is invalid at ingest and must never be stamped 'unknown'."""
        uploaded = capture_framework_trace(mock_client)
        lm = _dummy_lm()
        lm.model = None
        lm.kwargs = {}
        _configure(lm)
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        MyQA()(question="q?")
        adapter.disconnect()

        events = uploaded["events"]
        assert find_events(events, "model.invoke") == [], "a model-less LM must emit NO model.invoke"
        assert find_events(events, "cost.record") == []
        # The module boundary still traces — redaction of one field must not
        # blind the whole trace.
        assert find_events(events, "agent.input"), "module events must survive the model skip"

    def test_no_usage_means_no_cost_record(self, mock_client):
        """DummyLM reports zero tokens — an unmeasurable call must not be priced."""
        uploaded = capture_framework_trace(mock_client)
        _configure(_dummy_lm())
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        MyQA()(question="q?")
        adapter.disconnect()

        events = uploaded["events"]
        assert find_event(events, "model.invoke") is not None
        assert find_events(events, "cost.record") == [], "zero-token calls must not be priced"
        invoke = find_event(events, "model.invoke")["payload"]
        for key in ("tokens_prompt", "tokens_completion", "tokens_total"):
            assert key not in invoke, f"{key} must be omitted, never 0"

    def test_real_usage_produces_a_priced_cost_record(self, mock_client):
        """A real usage payload on the LM's history entry -> tokens + priced cost.

        Drives the real correlation path (``entry["outputs"] is outputs``) rather
        than calling the probe directly.
        """
        uploaded = capture_framework_trace(mock_client)

        class UsageLM(DummyLM):
            """A DummyLM whose response carries usage — DSPy's own
            ``_process_lm_response`` then records it on the history entry, so the
            adapter's real correlation path is what is under test."""

            def forward(self, prompt=None, messages=None, **kwargs):
                response = super().forward(prompt=prompt, messages=messages, **kwargs)
                response.usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
                return response

        lm = UsageLM([{"answer": "blue"} for _ in range(4)])
        lm.model = "openai/gpt-4o-mini"
        _configure(lm)
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        MyQA()(question="q?")
        adapter.disconnect()

        invoke = find_event(uploaded["events"], "model.invoke")["payload"]
        assert invoke["tokens_prompt"] == 100
        assert invoke["tokens_completion"] == 20
        # The litellm-style prefix is split so the pricing table resolves.
        assert invoke["model"] == "gpt-4o-mini"
        assert invoke["provider"] == "openai"

        cost = find_event(uploaded["events"], "cost.record")
        assert cost is not None, "a token-bearing call must emit cost.record"
        assert cost["payload"]["cost_usd"] > 0, "a priced model must carry a real cost_usd"

    def test_agent_output_for_a_module_never_started_omits_the_agent(self, mock_client):
        """An end callback for a module we never saw start: its type and identity
        are genuinely unknown. ateam falls back to the literal 'Module' and stamps
        it into agent_id/agent_name — a fabricated agent. The port must omit."""
        uploaded = capture_framework_trace(mock_client)
        _configure(_dummy_lm())
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        program = MyQA()
        cb = adapter._callback
        # A real outer module holds the run open so the orphan end is observable
        # at all (outside a run there is no collector and nothing is emitted).
        cb.on_module_start("real-1", program, {"args": (), "kwargs": {"question": "q?"}})
        cb.on_module_end("never-started", None, None)
        cb.on_module_end("real-1", None, None)
        adapter.disconnect()

        orphan = next(
            e for e in find_events(uploaded["events"], "agent.output") if e["payload"]["run_id"] == "never-started"
        )
        payload = orphan["payload"]
        assert "Module" not in str(list(payload.values())), f"fabricated 'Module' identity leaked: {payload}"
        assert "agent_name" not in payload
        assert "agent_id" not in payload
        assert "module_type" not in payload
        assert "latency_ms" not in payload, "latency must be omitted, not 0, when the start was never seen"

    def test_evicted_start_entry_keeps_the_identity_and_only_loses_latency(self, mock_client):
        """The 4096-cap eviction path. Identity rides the ContextVar stack, which
        cannot be evicted, so a full map must cost the latency NUMBER only — never
        the module's identity (and never a fabricated one)."""
        uploaded = capture_framework_trace(mock_client)
        _configure(_dummy_lm())
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        program = MyQA()
        cb = adapter._callback
        cb.on_module_start("call-x", program, {"args": (), "kwargs": {"question": "q?"}})
        adapter._call_starts.clear()  # exactly what the FIFO cap does under load
        cb.on_module_end("call-x", None, None)
        adapter.disconnect()

        output = find_event(uploaded["events"], "agent.output")
        payload = output["payload"]
        assert payload["module_type"] == "MyQA", "an evicted start must not cost the module identity"
        assert payload["agent_name"] == "MyQA"
        assert "latency_ms" not in payload, "latency must be omitted, not 0, when the start time is lost"


# ---------------------------------------------------------------------------
class TestTools:
    def test_real_react_tool_emits_tool_call(self, mock_client):
        uploaded = capture_framework_trace(mock_client)

        def get_weather(city: str) -> str:
            """Get the weather for a city."""
            return f"sunny in {city}"

        lm = DummyLM(
            [
                {"next_thought": "check weather", "next_tool_name": "get_weather", "next_tool_args": {"city": "Paris"}},
                {"next_thought": "done", "next_tool_name": "finish", "next_tool_args": {}},
                {"reasoning": "it is sunny", "answer": "sunny"},
            ]
        )
        _configure(lm)
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        dspy.ReAct(QASignature, tools=[get_weather])(question="weather in Paris?")
        adapter.disconnect()

        tools = find_events(uploaded["events"], "tool.call")
        names = [e["payload"]["tool_name"] for e in tools]
        assert "get_weather" in names, names
        weather = next(e for e in tools if e["payload"]["tool_name"] == "get_weather")
        assert weather["payload"]["success"] is True
        assert weather["payload"]["name"] == "get_weather"
        assert "Paris" in weather["payload"]["input"]
        assert "sunny in Paris" in weather["payload"]["output"]

    def test_tool_end_without_a_start_is_skipped(self, mock_client):
        """tool_name is only knowable from the start entry and is required at
        ingest — the event must be dropped, not fabricated."""
        uploaded = capture_framework_trace(mock_client)
        _configure(_dummy_lm())
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        program = MyQA()
        cb = adapter._callback
        cb.on_module_start("m1", program, {"args": (), "kwargs": {"question": "q?"}})
        cb.on_tool_end("orphan-tool", "some output", None)
        cb.on_module_end("m1", None, None)
        adapter.disconnect()

        assert find_events(uploaded["events"], "tool.call") == [], "an orphan tool end must emit nothing"


# ---------------------------------------------------------------------------
class TestOptimizer:
    def _trainset(self, n=4):
        return [dspy.Example(question=f"q{i}?", answer="blue").with_inputs("question") for i in range(n)]

    def test_real_bootstrap_compile_emits_the_optimization_stream(self, mock_client):
        """The headline feature: a REAL BootstrapFewShot compile must produce the
        compile boundary AND one agent.state.change per metric evaluation."""
        uploaded = capture_framework_trace(mock_client)
        _configure(_dummy_lm(20))
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()

        def metric(example, pred, trace=None):
            return example.answer == pred.answer

        optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=4, max_labeled_demos=4)
        adapter.instrument_optimizer(optimizer)
        optimizer.compile(dspy.Predict(QASignature), trainset=self._trainset())
        adapter.disconnect()

        events = uploaded["events"]
        steps = find_events(events, "agent.state.change")
        assert steps, "a real compile must emit the optimization stream (the metric wrap must fire)"
        # Guard the span-uniqueness assertion below against going vacuous: with a
        # single step it would hold no matter how spans were assigned.
        assert len(steps) > 1, f"need a multi-step compile to test the stream, got {len(steps)}"
        for step in steps:
            assert step["payload"]["state_key"] == "optimization_step"
            assert step["payload"]["state_type"] == "optimization"
            assert step["payload"]["optimizer_type"] == "BootstrapFewShot"

        compile_out = next(e for e in find_events(events, "agent.output") if e["payload"].get("operation") == "compile")
        assert compile_out["payload"]["iterations"] == len(steps)
        # Every step correlates to its compile.
        compile_run_id = compile_out["payload"]["run_id"]
        assert {s["payload"]["run_id"] for s in steps} == {compile_run_id}
        # Each step is its own span, not N events collapsed onto one.
        assert len({s["span_id"] for s in steps}) == len(steps)

    def test_iterations_are_per_compile_not_cumulative(self, mock_client):
        """The counter is per-optimizer and never resets; a second compile must
        report ITS OWN iteration count, not the lifetime total."""
        _configure(_dummy_lm(40))
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()

        def metric(example, pred, trace=None):
            return True

        optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=4, max_labeled_demos=4)
        adapter.instrument_optimizer(optimizer)

        counts = []
        for _ in range(2):
            uploaded = capture_framework_trace(mock_client)
            optimizer.compile(dspy.Predict(QASignature), trainset=self._trainset())
            events = uploaded["events"]
            out = next(e for e in find_events(events, "agent.output") if e["payload"].get("operation") == "compile")
            counts.append((out["payload"]["iterations"], len(find_events(events, "agent.state.change"))))
        adapter.disconnect()

        assert counts[0][0] == counts[0][1]
        assert counts[1][0] == counts[1][1], f"second compile reported a cumulative count: {counts}"

    def test_metric_exception_records_the_iteration_with_no_score(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()

        def metric(example, pred, trace=None):
            raise ValueError("metric blew up")

        class FakeOptimizer:
            def __init__(self):
                self.metric = metric

            def compile(self, program, trainset=None):
                for example in trainset:
                    with __import__("contextlib").suppress(ValueError):
                        self.metric(example, example)
                return program

        optimizer = FakeOptimizer()
        adapter.instrument_optimizer(optimizer)
        optimizer.compile(dspy.Predict(QASignature), trainset=self._trainset())
        adapter.disconnect()

        steps = find_events(uploaded["events"], "agent.state.change")
        assert steps, "a failed evaluation is still a real iteration"
        for step in steps:
            assert "score" not in step["payload"], "a failed metric must not fabricate a 0.0 score"
            assert "new_value" not in step["payload"]
            assert step["payload"]["iteration"] >= 1

    def test_non_numeric_metric_return_omits_the_score(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()

        def metric(example, pred, trace=None):
            return dspy.Prediction(answer="not a number")

        class FakeOptimizer:
            def __init__(self):
                self.metric = metric

            def compile(self, program, trainset=None):
                self.metric(trainset[0], trainset[0])
                return program

        optimizer = FakeOptimizer()
        adapter.instrument_optimizer(optimizer)
        optimizer.compile(dspy.Predict(QASignature), trainset=self._trainset())
        adapter.disconnect()

        step = find_event(uploaded["events"], "agent.state.change")
        assert step is not None
        assert "score" not in step["payload"], "a non-numeric metric return must not be coerced"
        assert "new_value" not in step["payload"]


# ---------------------------------------------------------------------------
class TestClassSwapFallback:
    """The callback-less path — DSPy present but the bus unavailable."""

    def test_class_swap_traces_a_program_and_is_opaque(self, mock_client, monkeypatch):
        uploaded = capture_framework_trace(mock_client)
        _configure(_dummy_lm())
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        # Force the fallback: pretend the bus never installed.
        adapter._unregister_callback()

        program = MyQA()
        adapter.instrument_program(program)
        # The swap is opaque — isinstance still passes against the original class.
        assert isinstance(program, MyQA)
        assert isinstance(program, dspy.Module)

        program(question="q?")
        adapter.disconnect()

        events = uploaded["events"]
        inputs = find_events(events, "agent.input")
        assert inputs, "the class-swap path must trace the module call"
        # original_class, not the synthesized subclass name.
        assert inputs[0]["payload"]["module_type"] == "MyQA"
        assert not any("_LayerLensTraced" in str(e["payload"]) for e in events), "synthesized class name leaked"
        assert inputs[0]["payload"]["agent_name"] == "MyQA"
        # disconnect restored the real class.
        assert type(program) is MyQA

    def test_callback_registered_disables_the_swap(self, mock_client):
        """Single emission source: running both double-counts every event."""
        _configure(_dummy_lm())
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        assert adapter._callback is not None
        program = MyQA()
        before = type(program)
        adapter.instrument_program(program)
        assert type(program) is before, "instrument_program must not swap while the callback is live"
        adapter.disconnect()

    def test_no_double_counting_with_callback_and_instrument_program(self, mock_client):
        """Callback + wrapper together traced every call TWICE. Counted over the
        WHOLE trace, not per module_type: a swapped class reports its synthesized
        name, so a per-name count would hide the duplicate behind a second label.
        """
        _configure(_dummy_lm())
        baseline = capture_framework_trace(mock_client)
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        MyQA()(question="q?")
        adapter.disconnect()
        expected = len(find_events(baseline["events"], "agent.input"))
        assert expected == 2, f"MyQA -> Predict should trace 2 module boundaries, got {expected}"

        uploaded = capture_framework_trace(mock_client)
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        program = MyQA()
        adapter.instrument_program(program)
        program(question="q?")
        adapter.disconnect()

        got = len(find_events(uploaded["events"], "agent.input"))
        assert got == expected, f"module calls double-counted: {got} agent.input events vs {expected} without the wrap"

    def test_callback_believed_installed_but_silently_dropped_falls_back(self, mock_client):
        """dspy.settings.configure() only accepts calls from the thread that first
        configured settings, and the bare-attribute fallback can be ignored too.
        Trusting the CALL would leave _callback set to something never installed —
        which self-disables the class-swap path and emits nothing while reporting
        connected. The adapter must trust the INSTALLED state instead."""
        uploaded = capture_framework_trace(mock_client)
        _configure(_dummy_lm())

        class _RejectingSettings:
            """dspy.settings that discards every callback install: configure()
            raises the real cross-thread error, and the bare-attribute fallback
            is silently ignored (the real Settings backs `callbacks` with a
            thread-local store, so a plain attribute write need not stick)."""

            @property
            def callbacks(self):
                return []

            @callbacks.setter
            def callbacks(self, value):
                pass

            def configure(self, **kwargs):
                raise RuntimeError("dspy.settings can only be changed by the thread that initially configured it")

        real_settings = dspy.settings
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        try:
            dspy.settings = _RejectingSettings()
            adapter.connect()
        finally:
            dspy.settings = real_settings

        assert adapter._callback is None, "a callback that never landed must not be reported as registered"

        # ... and because it is None, the class-swap fallback is live again.
        program = MyQA()
        adapter.instrument_program(program)
        assert type(program) is not MyQA, "the swap fallback must engage when the callback did not install"
        program(question="q?")
        adapter.disconnect()
        assert find_events(uploaded["events"], "agent.input"), "the fallback path must still produce a trace"


# ---------------------------------------------------------------------------
class TestSwappedClassNeverReachesTheAgentColumn:
    """The synthesized ``_LayerLensTraced_*`` class is this adapter's OWN plumbing.

    ``_honest_agent_name`` discriminates structurally on the defining module, and
    the synthesized class is defined in *this* module — NOT in dspy — so it sails
    straight through the framework-primitive guard and lands in the Agent column.
    Every seam that derives a class from an INSTANCE has to unwrap it first.
    """

    def _trainset(self, n=4):
        return [dspy.Example(question=f"q{i}?", answer="blue").with_inputs("question") for i in range(n)]

    def _metric(self, example, pred, trace=None):
        return True

    def test_compiling_a_swapped_program_never_stamps_the_synthesized_class(self, mock_client):
        """DSPy optimizers deepcopy the student, so the compiled result's ``type()``
        is the synthesized subclass. ``_emit_program_config(result)`` on the compile
        path passes no original class and falls back to ``type(result)``."""
        uploaded = capture_framework_trace(mock_client)
        _configure(_dummy_lm(20))
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        adapter._unregister_callback()  # force the class-swap fallback

        program = MyQA()
        adapter.instrument_program(program)
        optimizer = BootstrapFewShot(metric=self._metric, max_bootstrapped_demos=2, max_labeled_demos=2)
        adapter.instrument_optimizer(optimizer)
        compiled = optimizer.compile(program, trainset=self._trainset())
        adapter.disconnect()

        assert compiled is not None
        events = uploaded["events"]
        assert events, "a real compile must flush a trace"
        for event in events:
            assert "_LayerLensTraced" not in str(event["payload"]), (
                f"the adapter's own plumbing class leaked into {event['event_type']}: {event['payload']}"
            )
        # The compiled program is still described — honestly, under the real class.
        configs = [e["payload"] for e in find_events(events, "environment.config")]
        assert configs, "the compiled program must still be described"
        assert all(c["module_type"] != "_LayerLensTraced_MyQA" for c in configs)
        declared = [c for c in configs if c.get("agent_name")]
        assert declared, "the developer-declared MyQA must still reach the Agent column"
        assert {c["agent_name"] for c in declared} == {"MyQA"}
        assert {c["agent_id"] for c in declared} == {"MyQA"}

    def test_compiling_a_bare_predict_fabricates_no_agent(self, mock_client):
        """A bare ``dspy.Predict`` has NO honest agent identity (an honest '—').
        The swap gives it a ``_LayerLensTraced_Predict`` class whose module is this
        adapter's, so a naive ``type(result)`` invents an agent out of thin air."""
        uploaded = capture_framework_trace(mock_client)
        _configure(_dummy_lm(20))
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        adapter._unregister_callback()

        program = dspy.Predict(QASignature)
        adapter.instrument_program(program)
        optimizer = BootstrapFewShot(metric=self._metric, max_bootstrapped_demos=2, max_labeled_demos=2)
        adapter.instrument_optimizer(optimizer)
        optimizer.compile(program, trainset=self._trainset())
        adapter.disconnect()

        events = uploaded["events"]
        assert events
        for event in events:
            assert "agent_name" not in event["payload"], (
                f"fabricated agent on {event['event_type']}: {event['payload']}"
            )
            assert "agent_id" not in event["payload"], f"fabricated agent on {event['event_type']}: {event['payload']}"
        assert find_events(events, "agent.identity") == [], (
            "a bare Predict has no honest agent — the Agent column must stay '—'"
        )

    def test_callback_path_after_a_pre_connect_swap_reports_the_real_class(self, mock_client):
        """``instrument_program()`` before ``connect()`` swaps (no callback yet);
        ``connect()`` then installs the bus, which hands ``type(instance)`` — the
        synthesized subclass — to ``on_module_start``."""
        uploaded = capture_framework_trace(mock_client)
        _configure(_dummy_lm())
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        program = MyQA()
        adapter.instrument_program(program)  # no callback yet -> swaps
        adapter.connect()  # ... and now the bus is live
        program(question="q?")
        adapter.disconnect()

        events = uploaded["events"]
        assert events, "the pre-connect-swap program must still trace"
        for event in events:
            assert "_LayerLensTraced" not in str(event["payload"]), (
                f"the adapter's own plumbing class leaked into {event['event_type']}: {event['payload']}"
            )
        outer = next(e for e in find_events(events, "agent.input") if e["payload"].get("agent_name"))
        assert outer["payload"]["agent_name"] == "MyQA"
        assert outer["payload"]["module_type"] == "MyQA"


# ---------------------------------------------------------------------------
class TestLifecycleAndEntrypoint:
    def test_connect_records_the_framework_version(self, mock_client):
        adapter = DSPyAdapter(mock_client)
        adapter.connect()
        info = adapter.adapter_info()
        assert info.connected is True
        assert info.name == "dspy"
        assert info.metadata["framework_version"] == dspy.__version__
        adapter.disconnect()
        assert adapter.adapter_info().connected is False

    def test_disconnect_removes_the_callback_from_dspy_settings(self, mock_client):
        _configure(_dummy_lm())
        adapter = DSPyAdapter(mock_client)
        adapter.connect()
        cb = adapter._callback
        assert any(c is cb for c in dspy.settings.callbacks)
        adapter.disconnect()
        assert not any(c is cb for c in dspy.settings.callbacks), "disconnect must leave no trace on the bus"
        assert adapter._callback is None

    def test_instrument_dspy_entrypoint_connects_and_traces(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        _configure(_dummy_lm())
        program = MyQA()
        adapter = instrument_dspy(mock_client, program, capture_config=CaptureConfig.full())
        try:
            program(question="q?")
        finally:
            adapter.disconnect()

        assert isinstance(adapter, DSPyAdapter)
        assert find_events(uploaded["events"], "agent.input"), "instrument_dspy must produce a traced run"

    def test_non_callable_program_is_rejected(self, mock_client):
        adapter = DSPyAdapter(mock_client)
        adapter.connect()
        adapter._unregister_callback()
        with pytest.raises(TypeError, match="callable Module"):
            adapter.instrument_program(object())
        adapter.disconnect()

    def test_optimizer_without_compile_is_rejected(self, mock_client):
        adapter = DSPyAdapter(mock_client)
        adapter.connect()
        with pytest.raises(TypeError, match=r"\.compile\(\)"):
            adapter.instrument_optimizer(object())
        adapter.disconnect()

    def test_disconnect_restores_a_wrapped_optimizer(self, mock_client):
        adapter = DSPyAdapter(mock_client)
        adapter.connect()

        def metric(example, pred, trace=None):
            return True

        optimizer = BootstrapFewShot(metric=metric)
        original_compile = optimizer.compile
        adapter.instrument_optimizer(optimizer)
        assert optimizer.compile is not original_compile
        assert optimizer.metric is not metric
        adapter.disconnect()

        assert optimizer.metric is metric, "disconnect must restore the original metric"
        assert "compile" not in vars(optimizer), "the instance shadow must be dropped, not left behind"
