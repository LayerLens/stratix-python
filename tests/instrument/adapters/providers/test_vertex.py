"""Unit tests for the Google Vertex AI multi-vendor provider adapter.

Mocks the Google ``aiplatform`` SDK at the ``GenerativeModel.generate_content``
call boundary using ``types.SimpleNamespace`` doubles. No live network
call is made — running the suite does not require Google credentials,
``google-cloud-aiplatform`` to be installed, or any network access.

The adapter under test wraps any object that exposes
``model_name`` + ``generate_content``, which is the surface common to:

* ``vertexai.generative_models.GenerativeModel`` (Gemini).
* The Vertex Model Garden router for Anthropic (Claude on Vertex).
* The Vertex Model Garden router for Meta (Llama on Vertex).

We exercise all three vendors through the same code path to confirm
the adapter does not encode Gemini-only assumptions.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest import mock

import pytest

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.providers.vertex import (
    ADAPTER_CLASS,
    VERTEX_PRICING,
    STRATIX_VERTEX_ADAPTER_CLASS,
    VertexAdapter,
    LayerLensVertexAdapter,
    extract_function_calls,
    normalize_vertex_model,
    normalize_vertex_contents,
)
from layerlens.instrument.adapters.providers.vertex.auth import (
    detect_credential_source,
)
from layerlens.instrument.adapters.providers.vertex.adapter import _detect_vendor

# --- Test doubles ------------------------------------------------------


class _RecordingStratix:
    """Captures every event the adapter emits for assertion."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            event_type, payload = args
            self.events.append({"event_type": event_type, "payload": payload})
        elif len(args) == 1:
            self.events.append({"event_type": None, "payload": args[0]})


def _make_finish_reason(name: str = "STOP") -> Any:
    return SimpleNamespace(name=name)


def _make_response(
    *,
    text: str = "hello",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
    reasoning_tokens: Any = None,
    finish_reason: str = "STOP",
    function_calls: List[Dict[str, Any]] | None = None,
) -> Any:
    """Build an object that quacks like a Vertex GenerateContentResponse."""
    parts: List[Any] = []
    if text:
        parts.append(SimpleNamespace(text=text, function_call=None))
    for fc in function_calls or []:
        parts.append(
            SimpleNamespace(
                text=None,
                function_call=SimpleNamespace(
                    name=fc["name"],
                    args=fc.get("arguments", {}),
                ),
            )
        )
    content = SimpleNamespace(parts=parts)
    candidate = SimpleNamespace(
        content=content,
        finish_reason=_make_finish_reason(finish_reason),
    )
    metadata = SimpleNamespace(
        prompt_token_count=prompt_tokens,
        candidates_token_count=completion_tokens,
        total_token_count=total_tokens,
        thoughts_token_count=reasoning_tokens,
    )
    return SimpleNamespace(candidates=[candidate], usage_metadata=metadata)


def _make_model_client(
    model_name: str = "gemini-1.5-pro",
    response_factory: Any = None,
) -> Any:
    """Mock ``GenerativeModel``: only ``model_name`` + ``generate_content``."""
    client = SimpleNamespace(model_name=model_name)
    if response_factory is None:
        response_factory = lambda *a, **kw: _make_response()  # noqa: E731
    client.generate_content = mock.MagicMock(side_effect=response_factory)
    return client


# --- Smoke / lifecycle -------------------------------------------------


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is VertexAdapter
    assert STRATIX_VERTEX_ADAPTER_CLASS is VertexAdapter
    assert LayerLensVertexAdapter is VertexAdapter


def test_lifecycle_connect_disconnect() -> None:
    adapter = VertexAdapter()
    adapter.connect()
    assert adapter.status == AdapterStatus.HEALTHY

    client = _make_model_client()
    adapter.connect_client(client)
    assert "generate_content" in adapter._originals  # noqa: SLF001

    adapter.disconnect()
    assert adapter.status == AdapterStatus.DISCONNECTED


def test_disconnect_restores_original_method() -> None:
    adapter = VertexAdapter()
    adapter.connect()
    client = _make_model_client()
    original = client.generate_content
    adapter.connect_client(client)
    assert client.generate_content is not original

    adapter.disconnect()
    assert client.generate_content is original


def test_capability_includes_trace_models() -> None:
    info = VertexAdapter().get_adapter_info()
    cap_names = [c.name for c in info.capabilities]
    assert "TRACE_MODELS" in cap_names
    assert "TRACE_TOOLS" in cap_names


# --- Content normalization --------------------------------------------


def test_normalize_string_contents() -> None:
    msgs = normalize_vertex_contents("Hello world")
    assert msgs == [{"role": "user", "content": "Hello world"}]


def test_normalize_list_of_strings() -> None:
    msgs = normalize_vertex_contents(["First", "Second"])
    assert msgs == [
        {"role": "user", "content": "First"},
        {"role": "user", "content": "Second"},
    ]


def test_normalize_dict_contents() -> None:
    msgs = normalize_vertex_contents(
        [
            {"role": "user", "parts": ["What is", " 2+2?"]},
            {"role": "model", "parts": [{"text": "4"}]},
        ]
    )
    assert msgs == [
        {"role": "user", "content": "What is\n 2+2?"},
        {"role": "model", "content": "4"},
    ]


def test_normalize_object_contents() -> None:
    item = SimpleNamespace(
        role="user",
        parts=[SimpleNamespace(text="Hello"), SimpleNamespace(text="world")],
    )
    msgs = normalize_vertex_contents([item])
    assert msgs == [{"role": "user", "content": "Hello\nworld"}]


def test_normalize_none_returns_none() -> None:
    assert normalize_vertex_contents(None) is None


def test_normalize_caps_long_message() -> None:
    huge = "a" * 25_000
    msgs = normalize_vertex_contents(huge)
    assert msgs is not None
    assert len(msgs[0]["content"]) == 10_000


# --- Function call extraction -----------------------------------------


def test_extract_function_calls_basic() -> None:
    fn = SimpleNamespace(name="get_weather", args={"city": "SF"})
    part = SimpleNamespace(function_call=fn, text=None)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)
    response = SimpleNamespace(candidates=[candidate])

    calls = extract_function_calls(response)
    assert len(calls) == 1
    assert calls[0]["name"] == "get_weather"
    assert calls[0]["arguments"] == {"city": "SF"}


def test_extract_function_calls_handles_no_candidates() -> None:
    response = SimpleNamespace(candidates=[])
    assert extract_function_calls(response) == []


def test_extract_function_calls_handles_no_args() -> None:
    fn = SimpleNamespace(name="tool", args=None)
    part = SimpleNamespace(function_call=fn, text=None)
    content = SimpleNamespace(parts=[part])
    response = SimpleNamespace(
        candidates=[SimpleNamespace(content=content)],
    )
    calls = extract_function_calls(response)
    assert calls[0]["arguments"] == {}


# --- Model name normalization -----------------------------------------


def test_normalize_strips_models_prefix() -> None:
    assert normalize_vertex_model("models/gemini-1.5-pro") == "gemini-1.5-pro"


def test_normalize_strips_publishers_prefix() -> None:
    assert (
        normalize_vertex_model("publishers/anthropic/models/claude-opus-4-6")
        == "claude-opus-4-6"
    )
    assert (
        normalize_vertex_model("publishers/meta/models/llama-3.3-70b-instruct-maas")
        == "llama-3.3-70b-instruct-maas"
    )


def test_normalize_passes_through_bare_name() -> None:
    assert normalize_vertex_model("gemini-2.5-pro") == "gemini-2.5-pro"


def test_vendor_detection() -> None:
    assert _detect_vendor("gemini-2.5-pro") == "google"
    assert _detect_vendor("claude-opus-4-6") == "anthropic"
    assert _detect_vendor("llama-3.3-70b-instruct-maas") == "meta"


# --- Event emission ---------------------------------------------------


def test_emits_invoke_and_cost_for_gemini() -> None:
    stratix = _RecordingStratix()
    adapter = VertexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    client = _make_model_client(model_name="gemini-1.5-pro")
    adapter.connect_client(client)

    response = client.generate_content("Hello world")
    assert response is not None

    types = [e["event_type"] for e in stratix.events]
    assert types.count("model.invoke") == 1
    assert types.count("cost.record") == 1

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["provider"] == "google_vertex"
    assert invoke["payload"]["model"] == "gemini-1.5-pro"
    assert invoke["payload"]["prompt_tokens"] == 10
    assert invoke["payload"]["completion_tokens"] == 5
    assert invoke["payload"]["total_tokens"] == 15
    assert invoke["payload"]["finish_reason"] == "STOP"
    assert invoke["payload"]["vendor"] == "google"


def test_emits_invoke_for_anthropic_on_vertex() -> None:
    stratix = _RecordingStratix()
    adapter = VertexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    client = _make_model_client(
        model_name="publishers/anthropic/models/claude-opus-4-6",
    )
    adapter.connect_client(client)
    client.generate_content("Hi")

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["model"] == "claude-opus-4-6"
    assert invoke["payload"]["vendor"] == "anthropic"

    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    # Vertex pricing table includes claude-opus-4-6 -> non-zero cost.
    assert cost["payload"]["api_cost_usd"] is not None
    assert cost["payload"]["api_cost_usd"] > 0


def test_emits_invoke_for_llama_on_vertex() -> None:
    stratix = _RecordingStratix()
    adapter = VertexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    client = _make_model_client(
        model_name="publishers/meta/models/llama-3.3-70b-instruct-maas",
    )
    adapter.connect_client(client)
    client.generate_content("Hi")

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["model"] == "llama-3.3-70b-instruct-maas"
    assert invoke["payload"]["vendor"] == "meta"

    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    assert cost["payload"]["api_cost_usd"] is not None


def test_pricing_table_has_all_three_families() -> None:
    assert "gemini-1.5-pro" in VERTEX_PRICING
    assert "claude-opus-4-6" in VERTEX_PRICING
    assert "llama-3.3-70b-instruct-maas" in VERTEX_PRICING


def test_emits_tool_calls_when_function_call_returned() -> None:
    stratix = _RecordingStratix()
    adapter = VertexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    response_with_fn = _make_response(
        text="",
        function_calls=[{"name": "get_time", "arguments": {"tz": "PST"}}],
    )
    client = _make_model_client(
        model_name="gemini-1.5-pro",
        response_factory=lambda *a, **kw: response_with_fn,
    )
    adapter.connect_client(client)
    client.generate_content("what time is it?")

    tool_events = [e for e in stratix.events if e["event_type"] == "tool.call"]
    assert len(tool_events) == 1
    assert tool_events[0]["payload"]["tool_name"] == "get_time"
    assert tool_events[0]["payload"]["tool_input"] == {"tz": "PST"}


def test_emits_provider_error_on_exception() -> None:
    stratix = _RecordingStratix()
    adapter = VertexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    def boom(*_: Any, **__: Any) -> Any:
        raise RuntimeError("permission denied: 403")

    client = _make_model_client(model_name="gemini-1.5-pro", response_factory=boom)
    adapter.connect_client(client)

    with pytest.raises(RuntimeError):
        client.generate_content("hi")

    types = [e["event_type"] for e in stratix.events]
    assert "model.invoke" in types
    assert "policy.violation" in types
    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert "permission denied" in invoke["payload"]["error"]


def test_records_credential_source_metadata() -> None:
    stratix = _RecordingStratix()
    with mock.patch.dict(os.environ, {}, clear=False):
        # Ensure GOOGLE_APPLICATION_CREDENTIALS is unset for this test.
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        adapter = VertexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()
        client = _make_model_client()
        adapter.connect_client(client)
        client.generate_content("hi")

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    # ADC fallback when no SA-JSON path is set.
    assert invoke["payload"]["credential_source"] == "application_default"


def test_records_sa_json_credential_source(tmp_path: Any) -> None:
    sa_path = tmp_path / "sa.json"
    sa_path.write_text("{}", encoding="utf-8")
    with mock.patch.dict(
        os.environ,
        {"GOOGLE_APPLICATION_CREDENTIALS": str(sa_path)},
    ):
        assert detect_credential_source() == "service_account_json"


def test_records_unknown_when_sa_json_missing() -> None:
    with mock.patch.dict(
        os.environ,
        {"GOOGLE_APPLICATION_CREDENTIALS": "/does/not/exist.json"},
    ):
        assert detect_credential_source() == "unknown"


def test_extracts_generation_config_from_dict() -> None:
    stratix = _RecordingStratix()
    adapter = VertexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    client = _make_model_client()
    adapter.connect_client(client)

    client.generate_content(
        "hi",
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 256,
            "top_p": 0.9,
            "top_k": 40,
        },
    )

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    params = invoke["payload"]["parameters"]
    assert params["temperature"] == 0.7
    assert params["max_output_tokens"] == 256
    assert params["top_p"] == 0.9
    assert params["top_k"] == 40


def test_extracts_generation_config_from_object() -> None:
    stratix = _RecordingStratix()
    adapter = VertexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    client = _make_model_client()
    adapter.connect_client(client)

    cfg = SimpleNamespace(
        temperature=0.3,
        max_output_tokens=128,
        top_p=0.95,
        top_k=None,
    )
    client.generate_content("hi", generation_config=cfg)

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    params = invoke["payload"]["parameters"]
    assert params["temperature"] == 0.3
    assert params["max_output_tokens"] == 128
    assert params["top_p"] == 0.95
    # top_k is None, must be omitted (not stored as None).
    assert "top_k" not in params


# --- Streaming --------------------------------------------------------


def test_streaming_emits_consolidated_invoke() -> None:
    stratix = _RecordingStratix()
    adapter = VertexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    chunks = [
        SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(parts=[SimpleNamespace(text="Hel")]),
                    finish_reason=None,
                )
            ],
            usage_metadata=None,
        ),
        SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(parts=[SimpleNamespace(text="lo!")]),
                    finish_reason=_make_finish_reason("STOP"),
                )
            ],
            usage_metadata=SimpleNamespace(
                prompt_token_count=8,
                candidates_token_count=2,
                total_token_count=10,
                thoughts_token_count=None,
            ),
        ),
    ]

    def stream_factory(*_: Any, **__: Any) -> Any:
        return iter(chunks)

    client = _make_model_client(response_factory=stream_factory)
    adapter.connect_client(client)

    out = client.generate_content("hi", stream=True)
    received = list(out)
    assert len(received) == 2

    types = [e["event_type"] for e in stratix.events]
    # Exactly one consolidated model.invoke + one cost.record on stream end.
    assert types.count("model.invoke") == 1
    assert types.count("cost.record") == 1

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["streaming"] is True
    assert invoke["payload"]["finish_reason"] == "STOP"
    assert invoke["payload"]["prompt_tokens"] == 8
    assert invoke["payload"]["completion_tokens"] == 2


# --- Health / replay --------------------------------------------------


def test_serialize_for_replay_includes_capture_config() -> None:
    adapter = VertexAdapter(capture_config=CaptureConfig.standard())
    adapter.connect()
    trace = adapter.serialize_for_replay()
    assert trace.framework == "google_vertex"
    assert trace.adapter_name == "VertexAdapter"
    assert "capture_config" in trace.config
