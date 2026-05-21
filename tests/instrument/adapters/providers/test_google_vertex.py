from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

from layerlens.instrument import trace
from layerlens.instrument.adapters.providers.google_vertex import (
    GoogleVertexProvider,
    _strip_models_prefix,
    instrument_google_vertex,
    uninstrument_google_vertex,
)

from ...conftest import find_event


def _vertex_response(
    text: str = "Hello!",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
    finish_reason: str | None = "STOP",
    reasoning_tokens: int | None = None,
    response_id: str | None = "vertex-resp-abc",
    function_calls: list[tuple[str, dict[str, Any]]] | None = None,
) -> SimpleNamespace:
    """Build a shape-compatible Vertex response (mocks aiplatform at the boundary)."""
    parts: list[Any] = []
    if text:
        parts.append(SimpleNamespace(text=text, function_call=None))
    for name, args in function_calls or []:
        parts.append(
            SimpleNamespace(text=None, function_call=SimpleNamespace(name=name, args=args))
        )

    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=parts),
        finish_reason=SimpleNamespace(name=finish_reason) if finish_reason else None,
    )
    usage_metadata = SimpleNamespace(
        prompt_token_count=prompt_tokens,
        candidates_token_count=completion_tokens,
        total_token_count=total_tokens,
        thoughts_token_count=reasoning_tokens,
    )
    return SimpleNamespace(
        candidates=[candidate],
        usage_metadata=usage_metadata,
        response_id=response_id,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestStripModelsPrefix:
    def test_strips_models_prefix(self) -> None:
        assert _strip_models_prefix("models/gemini-2.5-pro") == "gemini-2.5-pro"

    def test_passthrough_when_no_prefix(self) -> None:
        assert _strip_models_prefix("gemini-2.5-pro") == "gemini-2.5-pro"

    def test_none_passthrough(self) -> None:
        assert _strip_models_prefix(None) is None


# ---------------------------------------------------------------------------
# Emit events
# ---------------------------------------------------------------------------


class TestEmitsEvents:
    def test_model_invoke_and_cost_record(self, mock_client, capture_trace):
        vertex_model = Mock()
        vertex_model.model_name = "models/gemini-2.5-pro"
        vertex_model.generate_content = Mock(return_value=_vertex_response())

        provider = GoogleVertexProvider()
        provider.connect(vertex_model)

        @trace(mock_client)
        def my_agent() -> str:
            r = vertex_model.generate_content("Hi")
            return r.candidates[0].content.parts[0].text

        my_agent()
        events = capture_trace["events"]

        model_invoke = find_event(events, "model.invoke")
        assert model_invoke["payload"]["name"] == "google_vertex.generate_content"
        assert model_invoke["payload"]["model"] == "gemini-2.5-pro"
        assert model_invoke["payload"]["output_message"] == {
            "role": "model",
            "content": "Hello!",
        }
        assert model_invoke["payload"]["usage"]["prompt_tokens"] == 10
        assert model_invoke["payload"]["usage"]["completion_tokens"] == 5
        assert model_invoke["payload"]["usage"]["total_tokens"] == 15
        assert model_invoke["payload"]["finish_reason"] == "STOP"
        assert model_invoke["payload"]["response_id"] == "vertex-resp-abc"
        assert "latency_ms" in model_invoke["payload"]

        cost = find_event(events, "cost.record")
        assert cost["payload"]["provider"] == "google_vertex"
        assert cost["payload"]["model"] == "gemini-2.5-pro"
        # gemini-2.5-pro pricing exists in PricingTable — non-None cost proves lookup worked.
        assert cost["payload"]["cost_usd"] is not None
        assert cost["payload"]["total_tokens"] == 15

    def test_reasoning_tokens_captured(self, mock_client, capture_trace):
        vertex_model = Mock()
        vertex_model.model_name = "gemini-2.5-pro"
        vertex_model.generate_content = Mock(
            return_value=_vertex_response(reasoning_tokens=42)
        )

        provider = GoogleVertexProvider()
        provider.connect(vertex_model)

        @trace(mock_client)
        def my_agent() -> None:
            vertex_model.generate_content("Hi")

        my_agent()
        events = capture_trace["events"]

        model_invoke = find_event(events, "model.invoke")
        assert model_invoke["payload"]["usage"]["reasoning_tokens"] == 42

    def test_function_calls_emit_tool_call_events(self, mock_client, capture_trace):
        vertex_model = Mock()
        vertex_model.model_name = "gemini-2.5-pro"
        vertex_model.generate_content = Mock(
            return_value=_vertex_response(
                text="",
                function_calls=[("get_weather", {"city": "SF"})],
            )
        )

        provider = GoogleVertexProvider()
        provider.connect(vertex_model)

        @trace(mock_client)
        def my_agent() -> None:
            vertex_model.generate_content("What's the weather?")

        my_agent()
        events = capture_trace["events"]

        tool_call = find_event(events, "tool.call")
        assert tool_call["payload"]["tool_name"] == "get_weather"
        assert tool_call["payload"]["arguments"] == {"city": "SF"}

    def test_error_emits_agent_error(self, mock_client, capture_trace):
        vertex_model = Mock()
        vertex_model.model_name = "gemini-2.5-pro"
        vertex_model.generate_content = Mock(side_effect=RuntimeError("Vertex 503"))

        provider = GoogleVertexProvider()
        provider.connect(vertex_model)

        @trace(mock_client)
        def my_agent() -> str:
            try:
                vertex_model.generate_content("Hi")
            except RuntimeError:
                pass
            return "recovered"

        my_agent()
        events = capture_trace["events"]

        error = find_event(events, "agent.error")
        assert error["payload"]["error"] == "Vertex 503"
        assert "latency_ms" in error["payload"]


# ---------------------------------------------------------------------------
# Model-name resolution (the M3-2 polish)
# ---------------------------------------------------------------------------


class TestModelNameResolution:
    def test_model_name_stripped_from_models_prefix(self):
        vertex_model = Mock()
        vertex_model.model_name = "models/gemini-1.5-flash"

        provider = GoogleVertexProvider()
        provider.connect(vertex_model)
        assert provider._model_name == "gemini-1.5-flash"

    def test_falls_back_to_private_model_name(self):
        vertex_model = Mock(spec=["_model_name", "generate_content"])
        vertex_model._model_name = "models/gemini-1.5-pro"
        vertex_model.generate_content = lambda *a, **kw: None

        provider = GoogleVertexProvider()
        provider.connect(vertex_model)
        assert provider._model_name == "gemini-1.5-pro"


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


class TestRegistryHelpers:
    def test_instrument_and_uninstrument(self):
        vertex_model = Mock()
        vertex_model.model_name = "gemini-2.5-pro"
        vertex_model.generate_content = lambda *a, **kw: None

        provider = instrument_google_vertex(vertex_model)
        assert isinstance(provider, GoogleVertexProvider)
        uninstrument_google_vertex()  # must not raise
