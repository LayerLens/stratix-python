"""
OpenAI GPT Chat Completion with STRATIX Instrumentation

Demonstrates:
- Simple chat completion with token counting
- Streaming chat completion with function calling
- STRATIX event emission for model invocations and cost tracking

Requirements:
    pip install openai layerlens

Usage:
    export OPENAI_API_KEY=sk-...
    python openai_chat.py
    python openai_chat.py --model gpt-4o-mini --prompt "Explain quantum computing"
    python openai_chat.py --stream --prompt "What is the weather in SF?" --function-calling
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from layerlens.instrument import STRATIX, emit_model_invoke, record_token_cost

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state, e.g. 'San Francisco, CA'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "fahrenheit"},
            },
            "required": ["location"],
        },
    },
}


def simple_chat(client, model: str, prompt: str) -> None:
    """Non-streaming chat completion with instrumentation."""
    print(f"\n--- Simple Chat ({model}) ---")
    start = time.perf_counter()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=256,
    )

    latency_ms = (time.perf_counter() - start) * 1000
    usage = response.usage
    content = response.choices[0].message.content

    emit_model_invoke(
        provider="openai",
        name=model,
        parameters={"temperature": 0.7, "max_tokens": 256},
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        latency_ms=latency_ms,
    )
    record_token_cost(
        provider="openai",
        model=model,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )

    print(f"Response: {content}")
    print(f"Tokens: {usage.prompt_tokens} in / {usage.completion_tokens} out | {latency_ms:.0f}ms")


def streaming_chat_with_tools(client, model: str, prompt: str) -> None:
    """Streaming chat with function calling and instrumentation."""
    print(f"\n--- Streaming Chat + Function Calling ({model}) ---")
    start = time.perf_counter()

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use tools when appropriate."},
            {"role": "user", "content": prompt},
        ],
        tools=[WEATHER_TOOL],
        stream=True,
        stream_options={"include_usage": True},
        temperature=0.3,
        max_tokens=512,
    )

    collected_content: list[str] = []
    tool_calls: dict[int, dict] = {}
    final_usage = None

    print("Streaming: ", end="", flush=True)
    for chunk in stream:
        if chunk.usage:
            final_usage = chunk.usage
        for choice in chunk.choices:
            if choice.delta.content:
                collected_content.append(choice.delta.content)
                print(choice.delta.content, end="", flush=True)
            for tc in choice.delta.tool_calls or []:
                idx = tc.index
                if idx not in tool_calls:
                    tool_calls[idx] = {"id": tc.id, "name": "", "arguments": ""}
                if tc.function and tc.function.name:
                    tool_calls[idx]["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    tool_calls[idx]["arguments"] += tc.function.arguments
    print()

    latency_ms = (time.perf_counter() - start) * 1000
    prompt_tokens = getattr(final_usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(final_usage, "completion_tokens", 0) or 0

    emit_model_invoke(
        provider="openai",
        name=model,
        parameters={"temperature": 0.3, "max_tokens": 512, "stream": True},
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_ms=latency_ms,
    )
    record_token_cost(
        provider="openai",
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    if tool_calls:
        for tc in tool_calls.values():
            print(f"Tool call: {tc['name']}({tc['arguments']})")
    print(f"Tokens: {prompt_tokens} in / {completion_tokens} out | {latency_ms:.0f}ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI chat with STRATIX instrumentation")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name (default: gpt-4o-mini)")
    parser.add_argument("--prompt", default="What are the three laws of thermodynamics?", help="User prompt")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode")
    parser.add_argument("--function-calling", action="store_true", help="Enable function calling demo")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: Set OPENAI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    import openai

    client = openai.OpenAI(api_key=api_key)

    stratix = STRATIX(
        policy_ref="stratix-policy-samples@1.0.0",
        agent_id="openai-chat-demo",
        framework="openai",
    )
    ctx = stratix.start_trial()

    try:
        if args.stream or args.function_calling:
            streaming_chat_with_tools(client, args.model, args.prompt)
        else:
            simple_chat(client, args.model, args.prompt)
    finally:
        summary = stratix.end_trial()
        events = stratix.get_events()
        print(f"\n--- Trace Summary ---")
        print(f"Status: {summary.get('status')}  |  Events emitted: {len(events)}")


if __name__ == "__main__":
    main()
