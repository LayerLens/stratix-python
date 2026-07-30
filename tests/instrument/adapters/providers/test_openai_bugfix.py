from __future__ import annotations

from unittest.mock import Mock

from openai.types import Embedding, CreateEmbeddingResponse
from layerlens.instrument import trace
from openai.types.responses import (
    Response,
    ResponseUsage,
    ResponseOutputText,
    ResponseOutputMessage,
    ResponseFunctionToolCall,
)
from layerlens.instrument._capture_config import CaptureConfig
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)
from openai.types.create_embedding_response import Usage as EmbeddingUsage
from layerlens.instrument.adapters.providers.openai import OpenAIProvider

from ...conftest import find_event, find_events

# ---------------------------------------------------------------------------
# Real Responses-API objects (OpenAI's default surface for new / o-series
# models). A Responses object has NO ``.choices``; its text lives in
# ``.output_text`` / ``.output`` and its usage exposes
# ``input_tokens`` / ``output_tokens`` (not prompt_tokens/completion_tokens).
# ---------------------------------------------------------------------------


def make_responses_object(
    text: str = "Hello from the Responses API!",
    model: str = "gpt-4.1",
    input_tokens: int = 12,
    output_tokens: int = 7,
    total_tokens: int = 19,
) -> Response:
    return Response(
        id="resp_test",
        created_at=1700000000,
        model=model,
        object="response",
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        output=[
            ResponseOutputMessage(
                id="msg_1",
                role="assistant",
                status="completed",
                type="message",
                content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
            )
        ],
        usage=ResponseUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        ),
    )


def make_responses_tool_call_object(model: str = "gpt-4.1") -> Response:
    return Response(
        id="resp_tool",
        created_at=1700000000,
        model=model,
        object="response",
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        output=[
            ResponseFunctionToolCall(
                type="function_call",
                call_id="call_abc",
                name="get_weather",
                arguments='{"city": "Paris"}',
            )
        ],
        usage=ResponseUsage(
            input_tokens=8,
            output_tokens=4,
            total_tokens=12,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        ),
    )


def make_embedding_object(model: str = "text-embedding-3-small") -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse(
        object="list",
        model=model,
        data=[Embedding(object="embedding", index=0, embedding=[0.1, 0.2, 0.3])],
        usage=EmbeddingUsage(prompt_tokens=4, total_tokens=4),
    )


# ---------------------------------------------------------------------------
# BUG-4: Responses API emitted empty events (output None, tokens 0, cost 0)
# ---------------------------------------------------------------------------


class TestResponsesApiEvents:
    def test_responses_emits_output_tokens_and_cost(self, mock_client, capture_trace):
        client = Mock()
        client.responses.create = Mock(return_value=make_responses_object())

        provider = OpenAIProvider()
        provider.connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            r = client.responses.create(model="gpt-4.1", input="Hi")
            return r.output_text

        my_agent()
        events = capture_trace["events"]

        model_invoke = find_event(events, "model.invoke")
        assert model_invoke["payload"]["name"] == "openai.responses.create"
        assert model_invoke["payload"]["model"] == "gpt-4.1"
        # Output text must survive (Responses has no .choices).
        out = model_invoke["payload"]["output_message"]
        assert out is not None, "Responses output_message was dropped"
        assert out["content"] == "Hello from the Responses API!"
        # Usage: input/output tokens mapped to prompt/completion.
        usage = model_invoke["payload"]["usage"]
        assert usage["prompt_tokens"] > 0, f"prompt_tokens lost: {usage}"
        assert usage["completion_tokens"] > 0, f"completion_tokens lost: {usage}"
        assert usage["prompt_tokens"] == 12
        assert usage["completion_tokens"] == 7
        # Flat token fields (atlas tokens column) must be populated too.
        assert model_invoke["payload"]["prompt_tokens"] == 12
        assert model_invoke["payload"]["completion_tokens"] == 7

        cost = find_event(events, "cost.record")
        assert cost["payload"]["provider"] == "openai"
        assert cost["payload"]["cost_usd"] is not None
        assert cost["payload"]["cost_usd"] > 0, "Responses call was priced at zero"

    def test_responses_tool_call_emitted(self, mock_client, capture_trace):
        client = Mock()
        client.responses.create = Mock(return_value=make_responses_tool_call_object())

        provider = OpenAIProvider()
        provider.connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            client.responses.create(model="gpt-4.1", input="weather?")
            return "done"

        my_agent()
        events = capture_trace["events"]

        tool_calls = find_events(events, "tool.call")
        assert tool_calls, "Responses function_call did not emit a tool.call"
        payload = tool_calls[0]["payload"]
        assert payload["tool_name"] == "get_weather"
        assert payload["arguments"] == {"city": "Paris"}


# ---------------------------------------------------------------------------
# Unit-level extractor assertions (bite even without the trace plumbing)
# ---------------------------------------------------------------------------


class TestResponsesExtractors:
    def test_extract_output_from_responses(self):
        r = make_responses_object(text="unit text")
        out = OpenAIProvider.extract_output(r)
        assert out == {"role": "assistant", "content": "unit text"}

    def test_extract_meta_maps_input_output_tokens(self):
        r = make_responses_object(input_tokens=30, output_tokens=11, total_tokens=41)
        meta = OpenAIProvider.extract_meta(r)
        assert meta["usage"]["prompt_tokens"] == 30
        assert meta["usage"]["completion_tokens"] == 11
        assert meta["usage"]["total_tokens"] == 41
        assert meta["response_model"] == "gpt-4.1"

    def test_embeddings_output_stays_none(self):
        """Embeddings must remain output_message=None (honest; not a Responses obj)."""
        emb = make_embedding_object()
        assert OpenAIProvider.extract_output(emb) is None

    def test_chat_output_unchanged(self):
        from .conftest import make_openai_response

        r = make_openai_response(content="chat content", role="assistant")
        assert OpenAIProvider.extract_output(r) == {
            "role": "assistant",
            "content": "chat content",
        }
