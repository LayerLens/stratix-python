"""Per-adapter resilience smoke tests for all 10 lighter framework adapters.

These tests instantiate each adapter, force a callback to raise (by
sabotaging an inner helper), and assert:

1. The exception does NOT propagate (framework would crash otherwise).
2. The resilience tracker recorded the failure.
3. After enough failures, ``adapter_info().metadata['resilience_status']``
   is ``"degraded"``.

This is the per-adapter complement to ``test_resilience.py``, which
covers the decorator + tracker mechanics in isolation.

Adapters covered (10 lighter):
    agno, llamaindex, google_adk, strands, pydantic_ai,
    smolagents, bedrock_agents, openai_agents, haystack, langfuse.

Each adapter is exercised against a CALLBACK that exists on the
instance unconditionally (no need to construct framework-specific
fixture objects). The actual callback bodies use ``self._payload(...)``,
``self._fire(...)`` or similar internal helpers that we monkey-patch
to force a failure deterministically.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import Mock

import pytest

from layerlens.instrument._context import (
    _current_run,
    _current_span_id,
    _current_collector,
)
from layerlens.instrument.adapters._base import DEFAULT_FAILURE_THRESHOLD


@pytest.fixture(autouse=True)
def _isolate_context_vars():
    """Ensure ContextVar state is clean before AND after every test.

    Several callbacks under test (pydantic_ai._on_before_run,
    bedrock_agents._before_invoke) intentionally call _begin_run() —
    when the test then forces those callbacks to fail, the ContextVar
    tokens pushed by _begin_run are NOT popped (because the failure
    happens after the push). Without per-test cleanup those leaked
    tokens corrupt subsequent tests in the same process (notably
    ``tests/instrument/test_trace_context.py``).
    """
    # Snapshot current state (likely None) and force a clean baseline.
    run_token = _current_run.set(None)
    col_token = _current_collector.set(None)
    span_token = _current_span_id.set(None)
    try:
        yield
    finally:
        # Hard reset — tests in this module are not expected to leave
        # any persistent run/collector/span state.
        for var, token in (
            (_current_run, run_token),
            (_current_collector, col_token),
            (_current_span_id, span_token),
        ):
            try:
                var.reset(token)
            except (ValueError, LookupError):
                var.set(None)


class _Boom(Exception):
    """Sentinel exception type used to verify the right error was caught."""


def _force_payload_failure(adapter: Any) -> None:
    """Sabotage ``adapter._payload`` so any callback that touches it raises."""

    def _raise(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise _Boom("simulated framework callback failure")

    adapter._payload = _raise  # type: ignore[method-assign]


def _force_fire_failure(adapter: Any) -> None:
    """Sabotage ``adapter._fire`` for adapters whose callbacks call _fire directly."""

    def _raise(*args: Any, **kwargs: Any) -> None:
        raise _Boom("simulated _fire failure")

    adapter._fire = _raise  # type: ignore[method-assign]


def _force_emit_failure(adapter: Any) -> None:
    """Sabotage ``adapter._emit`` for adapters whose callbacks call _emit directly."""

    def _raise(*args: Any, **kwargs: Any) -> None:
        raise _Boom("simulated _emit failure")

    adapter._emit = _raise  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# agno
# ---------------------------------------------------------------------------


class TestAgnoResilience:
    def test_on_run_start_failure_caught(self) -> None:
        from layerlens.instrument.adapters.frameworks.agno import AgnoAdapter

        adapter = AgnoAdapter(Mock())
        _force_payload_failure(adapter)
        # Must not raise.
        result = adapter._on_run_start(Mock(), "input")
        assert result is None
        assert adapter._resilience.total_failures == 1

    def test_on_run_end_failure_caught(self) -> None:
        from layerlens.instrument.adapters.frameworks.agno import AgnoAdapter

        adapter = AgnoAdapter(Mock())
        _force_payload_failure(adapter)
        result = adapter._on_run_end(Mock(), Mock(), None)
        assert result is None
        assert adapter._resilience.total_failures == 1


# ---------------------------------------------------------------------------
# llamaindex
# ---------------------------------------------------------------------------


class TestLlamaIndexResilience:
    def test_on_query_start_failure_caught(self) -> None:
        pytest.importorskip("llama_index.core")
        from layerlens.instrument.adapters.frameworks.llamaindex import LlamaIndexAdapter

        adapter = LlamaIndexAdapter(Mock())
        _force_payload_failure(adapter)
        event = Mock()
        event.span_id = "span-1"
        result = adapter._on_query_start(event)
        assert result is None
        assert adapter._resilience.total_failures == 1

    def test_handle_event_unknown_type_no_op(self) -> None:
        pytest.importorskip("llama_index.core")
        from layerlens.instrument.adapters.frameworks.llamaindex import LlamaIndexAdapter

        adapter = LlamaIndexAdapter(Mock())
        # Unknown event class — handler lookup returns None, no exception.
        adapter._handle_event(object())
        # No failure recorded — unknown types are a no-op, not an error.
        assert adapter._resilience.total_failures == 0

    def test_on_span_enter_failure_caught(self) -> None:
        pytest.importorskip("llama_index.core")
        from layerlens.instrument.adapters.frameworks.llamaindex import LlamaIndexAdapter

        adapter = LlamaIndexAdapter(Mock())

        # Sabotage open_spans dict access by replacing _open_spans with
        # an object that raises on __setitem__.
        class _Bad:
            def __setitem__(self, key: Any, value: Any) -> None:
                raise _Boom("dict broken")

            def get(self, key: Any, default: Any = None) -> Any:
                return default

            def __contains__(self, key: Any) -> bool:
                return False

        adapter._open_spans = _Bad()  # type: ignore[assignment]
        result = adapter._on_span_enter("id-1", None)
        # Default for span lifecycle is None — LlamaIndex tolerates it.
        assert result is None
        assert adapter._resilience.total_failures == 1


# ---------------------------------------------------------------------------
# google_adk
# ---------------------------------------------------------------------------


class TestGoogleAdkResilience:
    def test_on_before_run_failure_caught(self) -> None:
        from layerlens.instrument.adapters.frameworks.google_adk import GoogleADKAdapter

        adapter = GoogleADKAdapter(Mock())
        _force_payload_failure(adapter)
        adapter._on_before_run(Mock())
        assert adapter._resilience.total_failures == 1

    def test_on_after_run_failure_caught(self) -> None:
        from layerlens.instrument.adapters.frameworks.google_adk import GoogleADKAdapter

        adapter = GoogleADKAdapter(Mock())
        _force_payload_failure(adapter)
        adapter._on_after_run(Mock())
        assert adapter._resilience.total_failures == 1


# ---------------------------------------------------------------------------
# strands
# ---------------------------------------------------------------------------


class TestStrandsResilience:
    def test_on_before_invocation_failure_caught(self) -> None:
        from layerlens.instrument.adapters.frameworks.strands import StrandsAdapter

        adapter = StrandsAdapter(Mock())
        _force_payload_failure(adapter)
        # Build a minimal event shim — the wrapped callback will raise
        # when it tries to call _payload, which our sabotage replaces.
        event = Mock()
        event.agent = Mock()
        adapter._on_before_invocation(event)
        assert adapter._resilience.total_failures == 1


# ---------------------------------------------------------------------------
# pydantic_ai
# ---------------------------------------------------------------------------


class TestPydanticAiResilience:
    def test_on_before_run_failure_caught(self) -> None:
        from layerlens.instrument.adapters.frameworks.pydantic_ai import PydanticAIAdapter

        adapter = PydanticAIAdapter(Mock())
        _force_payload_failure(adapter)
        # _on_before_run calls _begin_run() then _payload — the latter
        # raises and must be caught.
        adapter._on_before_run(Mock())
        assert adapter._resilience.total_failures == 1

    def test_on_after_model_request_passthrough_returns_response(self) -> None:
        # Critical: when _on_after_model_request raises, the wrapper
        # MUST return the original response object (passthrough_arg=
        # "response") otherwise the agent's LLM response becomes None
        # and the agent crashes downstream.
        from layerlens.instrument.adapters.frameworks.pydantic_ai import PydanticAIAdapter

        adapter = PydanticAIAdapter(Mock())
        _force_emit_failure(adapter)
        sentinel_response = Mock(name="response_object")
        result = adapter._on_after_model_request(
            Mock(),
            request_context=Mock(),
            response=sentinel_response,
        )
        assert result is sentinel_response
        assert adapter._resilience.total_failures == 1

    def test_on_before_tool_execute_passthrough_returns_args(self) -> None:
        from layerlens.instrument.adapters.frameworks.pydantic_ai import PydanticAIAdapter

        adapter = PydanticAIAdapter(Mock())

        # Sabotage _start_timer (used inside _on_before_tool_execute).
        def _raise(*a: Any, **kw: Any) -> None:
            raise _Boom("timer broken")

        adapter._start_timer = _raise  # type: ignore[method-assign]
        sentinel_args = ("a", "b", "c")
        result = adapter._on_before_tool_execute(
            Mock(),
            call=Mock(),
            tool_def=Mock(),
            args=sentinel_args,
        )
        assert result == sentinel_args
        assert adapter._resilience.total_failures == 1

    def test_run_error_re_raises_framework_error(self) -> None:
        # The error-callback path MUST always re-raise the framework's
        # original error — even when our telemetry helper raises. The
        # framework's contract requires the error to propagate.
        from layerlens.instrument.adapters.frameworks.pydantic_ai import PydanticAIAdapter

        adapter = PydanticAIAdapter(Mock())
        _force_emit_failure(adapter)  # telemetry will fail

        framework_error = ValueError("framework's own error")
        with pytest.raises(ValueError, match="framework's own error"):
            adapter._on_run_error(Mock(), error=framework_error)

        # Telemetry helper failure was caught + recorded; re-raise still happened.
        assert adapter._resilience.total_failures == 1


# ---------------------------------------------------------------------------
# smolagents
# ---------------------------------------------------------------------------


class TestSmolAgentsResilience:
    def test_on_action_step_failure_caught(self) -> None:
        from layerlens.instrument.adapters.frameworks.smolagents import SmolAgentsAdapter

        adapter = SmolAgentsAdapter(Mock())

        # Sabotage _handle_action_step itself.
        def _raise(*a: Any, **kw: Any) -> None:
            raise _Boom("handler broken")

        adapter._handle_action_step = _raise  # type: ignore[method-assign]
        adapter._on_action_step(Mock(), Mock())
        assert adapter._resilience.total_failures == 1


# ---------------------------------------------------------------------------
# bedrock_agents
# ---------------------------------------------------------------------------


class TestBedrockAgentsResilience:
    def test_before_invoke_failure_caught(self) -> None:
        from layerlens.instrument.adapters.frameworks.bedrock_agents import BedrockAgentsAdapter

        adapter = BedrockAgentsAdapter(Mock())
        adapter._connected = True  # bypass the early ``not connected`` return
        _force_payload_failure(adapter)
        # boto3 invokes hooks with kwargs only.
        adapter._before_invoke(params={"agentId": "id-1", "sessionId": "s-1"})
        assert adapter._resilience.total_failures == 1

    def test_after_invoke_finally_runs_even_on_failure(self) -> None:
        # The outer _after_invoke wraps the inner body in try/finally so
        # _end_run() always fires — critical for releasing ContextVars.
        from layerlens.instrument.adapters.frameworks.bedrock_agents import BedrockAgentsAdapter

        adapter = BedrockAgentsAdapter(Mock())
        adapter._connected = True
        # Set up a run scope so _end_run has something to clean up.
        adapter._begin_run()
        _force_payload_failure(adapter)

        end_run_called = []
        original_end_run = adapter._end_run

        def _spy_end_run() -> None:
            end_run_called.append(True)
            original_end_run()

        adapter._end_run = _spy_end_run  # type: ignore[method-assign]
        adapter._after_invoke(parsed={"sessionId": "s-1"})
        # _end_run fired despite the body raising.
        assert end_run_called == [True]
        assert adapter._resilience.total_failures == 1


# ---------------------------------------------------------------------------
# openai_agents
# ---------------------------------------------------------------------------


class TestOpenAiAgentsResilience:
    def test_on_trace_start_failure_caught(self) -> None:
        # If the SDK isn't installed, we still want to test the resilience
        # wiring — but the class can't be instantiated because the parent
        # TracingProcessor isn't available. Skip in that case.
        pytest.importorskip("agents")
        from layerlens.instrument.adapters.frameworks.openai_agents import OpenAIAgentsAdapter

        adapter = OpenAIAgentsAdapter(Mock())

        # Sabotage RunState construction by patching the lock to raise.
        def _raise_on_acquire(*a: Any, **kw: Any) -> Any:
            raise _Boom("lock broken")

        adapter._lock = Mock()
        adapter._lock.__enter__ = _raise_on_acquire
        adapter._lock.__exit__ = lambda *a: None

        trace = Mock()
        trace.trace_id = "t-1"
        adapter.on_trace_start(trace)
        assert adapter._resilience.total_failures == 1

    def test_on_trace_end_failure_caught(self) -> None:
        pytest.importorskip("agents")
        from layerlens.instrument.adapters.frameworks.openai_agents import OpenAIAgentsAdapter

        adapter = OpenAIAgentsAdapter(Mock())

        # Sabotage the trace_runs dict.
        class _Bad:
            def pop(self, *a: Any, **kw: Any) -> Any:
                raise _Boom("dict broken")

        adapter._trace_runs = _Bad()  # type: ignore[assignment]
        trace = Mock()
        trace.trace_id = "t-1"
        adapter.on_trace_end(trace)
        assert adapter._resilience.total_failures == 1


# ---------------------------------------------------------------------------
# haystack
# ---------------------------------------------------------------------------


class TestHaystackResilience:
    def test_on_span_end_failure_caught(self) -> None:
        from layerlens.instrument.adapters.frameworks.haystack import (
            HaystackAdapter,
            _LayerLensSpan,
        )

        adapter = HaystackAdapter(Mock())
        # Sabotage the _on_pipeline_end branch.
        def _raise(*a: Any, **kw: Any) -> None:
            raise _Boom("pipeline broken")

        adapter._on_pipeline_end = _raise  # type: ignore[method-assign]

        span = _LayerLensSpan(
            adapter,
            "haystack.pipeline.run",
            "span-1",
            None,
            {},
            is_pipeline=True,
        )
        adapter._on_span_end(span)
        assert adapter._resilience.total_failures == 1


# ---------------------------------------------------------------------------
# langfuse
# ---------------------------------------------------------------------------


class TestLangfuseResilience:
    def test_import_observation_failure_caught(self) -> None:
        from layerlens.instrument.adapters.frameworks.langfuse import LangfuseAdapter

        adapter = LangfuseAdapter(Mock())

        # Force the inner branch to raise — _import_generation is called
        # for type=GENERATION, _import_span for SPAN; sabotage the SPAN
        # path with a malformed obs.
        collector = Mock()
        bad_obs = {"type": "SPAN", "id": "obs-1"}

        # Sabotage _import_span specifically.
        def _raise(*a: Any, **kw: Any) -> None:
            raise _Boom("span import broken")

        adapter._import_span = _raise  # type: ignore[method-assign]
        adapter._import_observation(collector, bad_obs, "root-span")
        assert adapter._resilience.total_failures == 1

    def test_import_score_failure_caught(self) -> None:
        from layerlens.instrument.adapters.frameworks.langfuse import LangfuseAdapter

        adapter = LangfuseAdapter(Mock())
        collector = Mock()
        # collector.emit raises — our score importer must catch.
        collector.emit.side_effect = _Boom("collector broken")
        adapter._import_score(
            collector,
            {"sessionId": "s-1"},
            "trace-1",
            "root-span",
            {"name": "quality", "value": 0.9},
        )
        assert adapter._resilience.total_failures == 1


# ---------------------------------------------------------------------------
# Health degradation across all 10 adapters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path, class_name",
    [
        ("layerlens.instrument.adapters.frameworks.agno", "AgnoAdapter"),
        ("layerlens.instrument.adapters.frameworks.smolagents", "SmolAgentsAdapter"),
        ("layerlens.instrument.adapters.frameworks.google_adk", "GoogleADKAdapter"),
        ("layerlens.instrument.adapters.frameworks.strands", "StrandsAdapter"),
        ("layerlens.instrument.adapters.frameworks.pydantic_ai", "PydanticAIAdapter"),
        ("layerlens.instrument.adapters.frameworks.bedrock_agents", "BedrockAgentsAdapter"),
        ("layerlens.instrument.adapters.frameworks.haystack", "HaystackAdapter"),
        ("layerlens.instrument.adapters.frameworks.langfuse", "LangfuseAdapter"),
    ],
)
def test_adapter_health_degrades_on_repeated_failures(module_path: str, class_name: str) -> None:
    """Every lighter adapter exposes resilience health via adapter_info().metadata."""
    import importlib

    module = importlib.import_module(module_path)
    adapter_cls = getattr(module, class_name)
    adapter = adapter_cls(Mock())

    # Hit the threshold by recording failures directly on the tracker
    # (faster than driving each adapter's specific callback path; this
    # test is purely about the metadata surface).
    for _ in range(DEFAULT_FAILURE_THRESHOLD):
        adapter._resilience.record_failure("synthetic", _Boom("threshold test"))

    info = adapter.adapter_info()
    assert info.metadata["resilience_status"] == "degraded"
    assert info.metadata["resilience_failures_total"] == DEFAULT_FAILURE_THRESHOLD


# ---------------------------------------------------------------------------
# llamaindex / openai_agents handled separately because they need
# llama_index_core / agents installed at test time. Use importorskip.
# ---------------------------------------------------------------------------


def test_llamaindex_health_degrades_on_repeated_failures() -> None:
    pytest.importorskip("llama_index.core")
    from layerlens.instrument.adapters.frameworks.llamaindex import LlamaIndexAdapter

    adapter = LlamaIndexAdapter(Mock())
    for _ in range(DEFAULT_FAILURE_THRESHOLD):
        adapter._resilience.record_failure("synthetic", _Boom("threshold"))
    info = adapter.adapter_info()
    assert info.metadata["resilience_status"] == "degraded"


def test_openai_agents_health_degrades_on_repeated_failures() -> None:
    pytest.importorskip("agents")
    from layerlens.instrument.adapters.frameworks.openai_agents import OpenAIAgentsAdapter

    adapter = OpenAIAgentsAdapter(Mock())
    for _ in range(DEFAULT_FAILURE_THRESHOLD):
        adapter._resilience.record_failure("synthetic", _Boom("threshold"))
    info = adapter.adapter_info()
    assert info.metadata["resilience_status"] == "degraded"
