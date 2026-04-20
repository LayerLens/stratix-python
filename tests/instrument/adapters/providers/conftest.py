from __future__ import annotations

from anthropic.types import Usage, Message, TextBlock

from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice


def make_openai_response(
    content: str = "Hello!",
    role: str = "assistant",
    model: str = "gpt-4",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
) -> ChatCompletion:
    """Build a real OpenAI ChatCompletion response."""
    return ChatCompletion(
        id="chatcmpl-test",
        model=model,
        object="chat.completion",
        created=1700000000,
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role=role, content=content),
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


def make_openai_response_no_usage(model: str = "gpt-4") -> ChatCompletion:
    """Build an OpenAI response with no usage data."""
    return ChatCompletion(
        id="chatcmpl-test",
        model=model,
        object="chat.completion",
        created=1700000000,
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="Hello!"),
            )
        ],
        usage=None,
    )


def make_openai_response_empty_choices(model: str = "gpt-4") -> ChatCompletion:
    """Build an OpenAI response with empty choices."""
    return ChatCompletion(
        id="chatcmpl-test",
        model=model,
        object="chat.completion",
        created=1700000000,
        choices=[],
        usage=None,
    )


def make_anthropic_response(
    text: str = "I'm Claude!",
    model: str = "claude-3-opus-20240229",
    input_tokens: int = 20,
    output_tokens: int = 10,
    stop_reason: str = "end_turn",
) -> Message:
    """Build a real Anthropic Message response."""
    return Message(
        id="msg-test",
        type="message",
        role="assistant",
        model=model,
        content=[TextBlock(type="text", text=text)],
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason=stop_reason,
    )


def make_anthropic_response_empty_content(
    model: str = "claude-3-opus-20240229",
) -> Message:
    """Build an Anthropic response with empty content."""
    return Message(
        id="msg-test",
        type="message",
        role="assistant",
        model=model,
        content=[],
        usage=Usage(input_tokens=0, output_tokens=0),
        stop_reason="end_turn",
    )
