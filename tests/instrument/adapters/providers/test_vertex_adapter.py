"""Unit tests for the Google Vertex AI provider adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.providers.google_vertex_adapter import (
    ADAPTER_CLASS,
    GoogleVertexAdapter,
)


class _RecordingStratix:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


def _make_response(text: str = "hello") -> Any:
    part = SimpleNamespace(text=text)
    content = SimpleNamespace(parts=[part])
    finish = SimpleNamespace(name="STOP")
    candidate = SimpleNamespace(content=content, finish_reason=finish)
    metadata = SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=5,
        total_token_count=15,
        thoughts_token_count=None,
    )
    return SimpleNamespace(candidates=[candidate], usage_metadata=metadata)


def _make_model_client(model_name: str = "gemini-1.5-pro") -> Any:
    client = SimpleNamespace(model_name=model_name)
    client.generate_content = lambda *a, **kw: _make_response()
    return client


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is GoogleVertexAdapter


def test_lifecycle() -> None:
    a = GoogleVertexAdapter()
    a.connect()
    assert a.status == AdapterStatus.HEALTHY


def test_normalize_string_contents() -> None:
    msgs = GoogleVertexAdapter._normalize_vertex_contents("Hello world")
    assert msgs == [{"role": "user", "content": "Hello world"}]


def test_normalize_list_of_strings() -> None:
    msgs = GoogleVertexAdapter._normalize_vertex_contents(["First", "Second"])
    assert msgs == [
        {"role": "user", "content": "First"},
        {"role": "user", "content": "Second"},
    ]


def test_extract_function_calls() -> None:
    fn = SimpleNamespace(name="get_weather", args={"city": "SF"})
    part = SimpleNamespace(function_call=fn, text=None)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)
    response = SimpleNamespace(candidates=[candidate])

    calls = GoogleVertexAdapter._extract_function_calls(response)
    assert len(calls) == 1
    assert calls[0]["name"] == "get_weather"
    assert calls[0]["arguments"] == {"city": "SF"}


def test_emits_invoke_and_cost() -> None:
    stratix = _RecordingStratix()
    adapter = GoogleVertexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    client = _make_model_client()
    adapter.connect_client(client)

    client.generate_content("hello")

    types = [e["event_type"] for e in stratix.events]
    assert "model.invoke" in types
    assert "cost.record" in types

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["provider"] == "google_vertex"
    assert invoke["payload"]["model"] == "gemini-1.5-pro"
    assert invoke["payload"]["prompt_tokens"] == 10
    assert invoke["payload"]["finish_reason"] == "STOP"


def test_strips_models_prefix() -> None:
    """``models/gemini-1.5-pro`` → ``gemini-1.5-pro`` for pricing lookup."""
    stratix = _RecordingStratix()
    adapter = GoogleVertexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    client = _make_model_client(model_name="models/gemini-1.5-pro")
    adapter.connect_client(client)
    client.generate_content("hi")

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["model"] == "gemini-1.5-pro"
