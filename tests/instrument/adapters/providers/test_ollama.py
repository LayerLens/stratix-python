from __future__ import annotations

from typing import Any
from unittest.mock import Mock

from layerlens.instrument import trace
from layerlens.instrument.adapters.providers.ollama import (
    OllamaProvider,
    instrument_ollama,
    uninstrument_ollama,
)

from ...conftest import find_event


def _chat_response(
    content: str = "Hi there!",
    role: str = "assistant",
    model: str = "llama3",
    prompt_tokens: int = 12,
    completion_tokens: int = 7,
    done_reason: str | None = "stop",
    eval_duration_ns: int = 0,
    prompt_eval_duration_ns: int = 0,
    total_duration_ns: int | None = None,
) -> dict[str, Any]:
    """Build an Ollama `chat` response dict (matches `ollama` package output)."""
    resp: dict[str, Any] = {
        "model": model,
        "message": {"role": role, "content": content},
        "prompt_eval_count": prompt_tokens,
        "eval_count": completion_tokens,
        "done": True,
    }
    if done_reason is not None:
        resp["done_reason"] = done_reason
    if eval_duration_ns:
        resp["eval_duration"] = eval_duration_ns
    if prompt_eval_duration_ns:
        resp["prompt_eval_duration"] = prompt_eval_duration_ns
    if total_duration_ns is not None:
        resp["total_duration"] = total_duration_ns
    return resp


def _generate_response(text: str = "generated", model: str = "llama3") -> dict[str, Any]:
    return {
        "model": model,
        "response": text,
        "prompt_eval_count": 8,
        "eval_count": 4,
        "done": True,
        "done_reason": "stop",
    }


def _embed_response(dim: int = 4, model: str = "nomic-embed-text") -> dict[str, Any]:
    return {"model": model, "embedding": [0.1] * dim}


# ---------------------------------------------------------------------------
# Emit events — chat
# ---------------------------------------------------------------------------


class TestEmitsEvents:
    def test_chat_model_invoke(self, mock_client, capture_trace):
        ollama_client = Mock()
        ollama_client.chat = Mock(return_value=_chat_response())

        provider = OllamaProvider()
        provider.connect(ollama_client)

        @trace(mock_client)
        def my_agent() -> str:
            r = ollama_client.chat(
                model="llama3", messages=[{"role": "user", "content": "Hi"}]
            )
            return r["message"]["content"]

        my_agent()
        events = capture_trace["events"]

        model_invoke = find_event(events, "model.invoke")
        assert model_invoke["payload"]["name"] == "ollama.chat"
        assert model_invoke["payload"]["model"] == "llama3"
        assert model_invoke["payload"]["output_message"] == {
            "role": "assistant",
            "content": "Hi there!",
        }
        assert model_invoke["payload"]["usage"]["prompt_tokens"] == 12
        assert model_invoke["payload"]["usage"]["completion_tokens"] == 7
        assert model_invoke["payload"]["usage"]["total_tokens"] == 19
        assert model_invoke["payload"]["finish_reason"] == "stop"

    def test_generate_model_invoke(self, mock_client, capture_trace):
        ollama_client = Mock()
        ollama_client.generate = Mock(return_value=_generate_response())

        provider = OllamaProvider()
        provider.connect(ollama_client)

        @trace(mock_client)
        def my_agent() -> str:
            r = ollama_client.generate(model="llama3", prompt="Hi")
            return r["response"]

        my_agent()
        events = capture_trace["events"]

        model_invoke = find_event(events, "model.invoke")
        assert model_invoke["payload"]["name"] == "ollama.generate"
        assert model_invoke["payload"]["output_message"] == {
            "role": "assistant",
            "content": "generated",
        }

    def test_embeddings_model_invoke(self, mock_client, capture_trace):
        ollama_client = Mock()
        ollama_client.embeddings = Mock(return_value=_embed_response(dim=8))

        provider = OllamaProvider()
        provider.connect(ollama_client)

        @trace(mock_client)
        def my_agent() -> int:
            r = ollama_client.embeddings(model="nomic-embed-text", prompt="hi")
            return len(r["embedding"])

        my_agent()
        events = capture_trace["events"]

        model_invoke = find_event(events, "model.invoke")
        assert model_invoke["payload"]["name"] == "ollama.embeddings"
        assert model_invoke["payload"]["output_message"] == {"type": "embedding", "dim": 8}

    def test_error_emits_agent_error(self, mock_client, capture_trace):
        ollama_client = Mock()
        ollama_client.chat = Mock(side_effect=ConnectionError("ollama down"))

        provider = OllamaProvider()
        provider.connect(ollama_client)

        @trace(mock_client)
        def my_agent() -> str:
            try:
                ollama_client.chat(model="llama3", messages=[])
            except ConnectionError:
                pass
            return "recovered"

        my_agent()
        events = capture_trace["events"]

        error = find_event(events, "agent.error")
        assert error["payload"]["error"] == "ollama down"


# ---------------------------------------------------------------------------
# Endpoint + infra-cost wiring (the M3-2 polish)
# ---------------------------------------------------------------------------


class TestEndpointAndInfraCost:
    def test_endpoint_emitted_in_meta(self, mock_client, capture_trace, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://my-ollama-box:11434")
        ollama_client = Mock()
        ollama_client.chat = Mock(return_value=_chat_response())

        provider = OllamaProvider()
        provider.connect(ollama_client)

        @trace(mock_client)
        def my_agent() -> None:
            ollama_client.chat(model="llama3", messages=[])

        my_agent()
        model_invoke = find_event(capture_trace["events"], "model.invoke")
        assert model_invoke["payload"]["endpoint"] == "http://my-ollama-box:11434"

    def test_infra_cost_computed_when_cost_per_second_set(
        self, mock_client, capture_trace
    ):
        ollama_client = Mock()
        ollama_client.chat = Mock(
            return_value=_chat_response(
                eval_duration_ns=5_000_000_000,  # 5 seconds
                prompt_eval_duration_ns=1_000_000_000,  # 1 second
            )
        )

        provider = OllamaProvider(cost_per_second=0.0001)
        provider.connect(ollama_client)

        @trace(mock_client)
        def my_agent() -> None:
            ollama_client.chat(model="llama3", messages=[])

        my_agent()
        model_invoke = find_event(capture_trace["events"], "model.invoke")
        # 6 seconds total * $0.0001/sec = $0.0006
        assert model_invoke["payload"]["infra_cost_usd"] == 0.0006

    def test_infra_cost_absent_when_cost_per_second_unset(
        self, mock_client, capture_trace
    ):
        ollama_client = Mock()
        ollama_client.chat = Mock(
            return_value=_chat_response(eval_duration_ns=5_000_000_000)
        )

        provider = OllamaProvider()  # no cost_per_second
        provider.connect(ollama_client)

        @trace(mock_client)
        def my_agent() -> None:
            ollama_client.chat(model="llama3", messages=[])

        my_agent()
        model_invoke = find_event(capture_trace["events"], "model.invoke")
        assert "infra_cost_usd" not in model_invoke["payload"]

    def test_infra_cost_absent_when_duration_missing(
        self, mock_client, capture_trace
    ):
        ollama_client = Mock()
        ollama_client.chat = Mock(return_value=_chat_response())  # no durations

        provider = OllamaProvider(cost_per_second=0.0001)
        provider.connect(ollama_client)

        @trace(mock_client)
        def my_agent() -> None:
            ollama_client.chat(model="llama3", messages=[])

        my_agent()
        model_invoke = find_event(capture_trace["events"], "model.invoke")
        assert "infra_cost_usd" not in model_invoke["payload"]


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


class TestRegistryHelpers:
    def test_instrument_and_uninstrument(self):
        ollama_client = Mock()
        ollama_client.chat = lambda *a, **kw: _chat_response()

        provider = instrument_ollama(ollama_client, cost_per_second=0.0001)
        assert isinstance(provider, OllamaProvider)
        assert provider._cost_per_second == 0.0001
        uninstrument_ollama()
