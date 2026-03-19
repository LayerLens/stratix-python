"""Tests for Google Vertex AI LLM Provider Adapter."""

import pytest
from layerlens.instrument.adapters.llm_providers.google_vertex_adapter import GoogleVertexAdapter


class MockStratix:
    def __init__(self):
        self.events = []

    def emit(self, event_type, payload):
        self.events.append({"type": event_type, "payload": payload})

    def get_events(self, event_type=None):
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


class MockUsageMetadata:
    def __init__(self, prompt=100, candidates=50, total=150, thoughts=None):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.total_token_count = total
        self.thoughts_token_count = thoughts


class MockFunctionCall:
    def __init__(self, name="get_weather", args=None):
        self.name = name
        self.args = args or {"city": "NYC"}


class MockPart:
    def __init__(self, text=None, function_call=None):
        self.text = text
        self.function_call = function_call


class MockContent:
    def __init__(self, parts=None):
        self.parts = parts or [MockPart(text="Hello")]


class MockCandidate:
    def __init__(self, content=None):
        self.content = content or MockContent()


class MockVertexResponse:
    def __init__(self, usage=None, candidates=None):
        self.usage_metadata = usage or MockUsageMetadata()
        self.candidates = candidates or [MockCandidate()]


class MockGenerativeModel:
    def __init__(self, model_name="gemini-2.0-flash"):
        self.model_name = model_name

    def generate_content(self, *args, **kwargs):
        return MockVertexResponse()


class TestGoogleVertexAdapter:
    """Tests for GoogleVertexAdapter."""

    def test_adapter_framework(self):
        adapter = GoogleVertexAdapter()
        assert adapter.FRAMEWORK == "google_vertex"

    def test_connect_client_wraps_generate_content(self):
        adapter = GoogleVertexAdapter()
        adapter.connect()

        model = MockGenerativeModel()
        adapter.connect_client(model)

        assert "generate_content" in adapter._originals

    def test_generate_content_emits_model_invoke(self):
        stratix = MockStratix()
        adapter = GoogleVertexAdapter(stratix=stratix)
        adapter.connect()

        model = MockGenerativeModel()
        adapter.connect_client(model)

        model.generate_content("Hello world")

        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["provider"] == "google_vertex"
        assert events[0]["payload"]["prompt_tokens"] == 100
        assert events[0]["payload"]["completion_tokens"] == 50

    def test_generate_content_emits_cost_record(self):
        stratix = MockStratix()
        adapter = GoogleVertexAdapter(stratix=stratix)
        adapter.connect()

        model = MockGenerativeModel()
        adapter.connect_client(model)

        model.generate_content("Hello")

        events = stratix.get_events("cost.record")
        assert len(events) == 1

    def test_function_calling_emits_tool_call(self):
        stratix = MockStratix()
        adapter = GoogleVertexAdapter(stratix=stratix)
        adapter.connect()

        def generate_with_fn(*args, **kwargs):
            part = MockPart(function_call=MockFunctionCall())
            content = MockContent(parts=[part])
            return MockVertexResponse(candidates=[MockCandidate(content=content)])

        model = MockGenerativeModel()
        adapter.connect_client(model)
        model.generate_content = adapter._wrap_generate_content(generate_with_fn)

        model.generate_content("What's the weather?")

        tool_events = stratix.get_events("tool.call")
        assert len(tool_events) == 1
        assert tool_events[0]["payload"]["tool_name"] == "get_weather"

    def test_error_propagation(self):
        stratix = MockStratix()
        adapter = GoogleVertexAdapter(stratix=stratix)
        adapter.connect()

        def failing_generate(*args, **kwargs):
            raise ValueError("Vertex API error")

        model = MockGenerativeModel()
        adapter.connect_client(model)
        model.generate_content = adapter._wrap_generate_content(failing_generate)

        with pytest.raises(ValueError, match="Vertex API error"):
            model.generate_content("Hello")

        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["error"] == "Vertex API error"

    def test_adapter_error_does_not_break_call(self):
        class FailingSTRATIX:
            def emit(self, *args, **kwargs):
                raise RuntimeError("emit failed")

        adapter = GoogleVertexAdapter(stratix=FailingSTRATIX())
        adapter.connect()

        model = MockGenerativeModel()
        adapter.connect_client(model)

        result = model.generate_content("Hello")
        assert result is not None

    def test_thoughts_tokens_extracted(self):
        stratix = MockStratix()
        adapter = GoogleVertexAdapter(stratix=stratix)
        adapter.connect()

        def generate_with_thoughts(*args, **kwargs):
            return MockVertexResponse(usage=MockUsageMetadata(thoughts=200))

        model = MockGenerativeModel()
        adapter.connect_client(model)
        model.generate_content = adapter._wrap_generate_content(generate_with_thoughts)

        model.generate_content("Hello")

        events = stratix.get_events("model.invoke")
        assert events[0]["payload"]["reasoning_tokens"] == 200

    def test_streaming_wraps_iterator(self):
        stratix = MockStratix()
        adapter = GoogleVertexAdapter(stratix=stratix)
        adapter.connect()

        chunks = [
            MockVertexResponse(usage=MockUsageMetadata(prompt=50, candidates=10, total=60)),
            MockVertexResponse(usage=MockUsageMetadata(prompt=100, candidates=50, total=150)),
        ]

        def generate_streaming(*args, **kwargs):
            if kwargs.get("stream"):
                return iter(chunks)
            return MockVertexResponse()

        model = MockGenerativeModel()
        adapter.connect_client(model)
        model.generate_content = adapter._wrap_generate_content(generate_streaming)

        stream = model.generate_content("Hello", stream=True)
        collected = list(stream)

        assert len(collected) == 2
        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"].get("streaming") is True

    def test_disconnect_restores_originals(self):
        adapter = GoogleVertexAdapter()
        adapter.connect()

        model = MockGenerativeModel()
        adapter.connect_client(model)
        assert hasattr(model.generate_content, '_stratix_original')

        adapter.disconnect()
        assert not hasattr(model.generate_content, '_stratix_original')

    def test_latency_captured(self):
        stratix = MockStratix()
        adapter = GoogleVertexAdapter(stratix=stratix)
        adapter.connect()

        model = MockGenerativeModel()
        adapter.connect_client(model)

        model.generate_content("Hello")

        events = stratix.get_events("model.invoke")
        assert "latency_ms" in events[0]["payload"]

    def test_model_name_captured(self):
        stratix = MockStratix()
        adapter = GoogleVertexAdapter(stratix=stratix)
        adapter.connect()

        model = MockGenerativeModel(model_name="gemini-2.5-pro")
        adapter.connect_client(model)

        model.generate_content("Hello")

        events = stratix.get_events("model.invoke")
        assert events[0]["payload"]["model"] == "gemini-2.5-pro"

    def test_multiple_function_calls(self):
        stratix = MockStratix()
        adapter = GoogleVertexAdapter(stratix=stratix)
        adapter.connect()

        def generate_multi_fn(*args, **kwargs):
            parts = [
                MockPart(function_call=MockFunctionCall(name="get_weather")),
                MockPart(function_call=MockFunctionCall(name="search")),
            ]
            content = MockContent(parts=parts)
            return MockVertexResponse(candidates=[MockCandidate(content=content)])

        model = MockGenerativeModel()
        adapter.connect_client(model)
        model.generate_content = adapter._wrap_generate_content(generate_multi_fn)

        model.generate_content("Hello")

        tool_events = stratix.get_events("tool.call")
        assert len(tool_events) == 2

    def test_no_usage_handled_gracefully(self):
        stratix = MockStratix()
        adapter = GoogleVertexAdapter(stratix=stratix)
        adapter.connect()

        def generate_no_usage(*args, **kwargs):
            resp = MockVertexResponse()
            resp.usage_metadata = None
            return resp

        model = MockGenerativeModel()
        adapter.connect_client(model)
        model.generate_content = adapter._wrap_generate_content(generate_no_usage)

        result = model.generate_content("Hello")
        assert result is not None

    def test_generation_config_params_captured(self):
        stratix = MockStratix()
        adapter = GoogleVertexAdapter(stratix=stratix)
        adapter.connect()

        model = MockGenerativeModel()
        adapter.connect_client(model)

        model.generate_content("Hello", generation_config={"temperature": 0.9, "max_output_tokens": 500})

        events = stratix.get_events("model.invoke")
        params = events[0]["payload"].get("parameters", {})
        assert params.get("temperature") == 0.9
