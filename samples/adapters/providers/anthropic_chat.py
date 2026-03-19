"""
Anthropic Claude Messages with STRATIX Instrumentation

Demonstrates:
- Simple message completion with token counting
- Streaming message completion with tool use
- STRATIX event emission for model invocations and cost tracking

Requirements:
    pip install anthropic layerlens

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python anthropic_chat.py
    python anthropic_chat.py --model claude-sonnet-4-20250514 --stream
    python anthropic_chat.py --prompt "Calculate 2^10" --tool-use
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from layerlens.instrument import STRATIX, emit_model_invoke, record_token_cost

CALCULATOR_TOOL = {
    "name": "calculator",
    "description": "Evaluate a mathematical expression and return the result.",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression to evaluate, e.g. '2**10'"},
        },
        "required": ["expression"],
    },
}


def simple_message(client, model: str, prompt: str) -> None:
    """Non-streaming message completion with instrumentation."""
    print(f"\n--- Simple Message ({model}) ---")
    start = time.perf_counter()

    response = client.messages.create(
        model=model,
        max_tokens=256,
        system="You are a concise assistant. Respond in 1-2 sentences.",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    latency_ms = (time.perf_counter() - start) * 1000
    usage = response.usage
    text_blocks = [b.text for b in response.content if b.type == "text"]
    content = "\n".join(text_blocks)

    emit_model_invoke(
        provider="anthropic",
        name=model,
        parameters={"temperature": 0.7, "max_tokens": 256},
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
        total_tokens=usage.input_tokens + usage.output_tokens,
        latency_ms=latency_ms,
    )
    record_token_cost(
        provider="anthropic",
        model=model,
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
    )

    print(f"Response: {content}")
    print(f"Stop reason: {response.stop_reason}")
    print(f"Tokens: {usage.input_tokens} in / {usage.output_tokens} out | {latency_ms:.0f}ms")


def streaming_message_with_tools(client, model: str, prompt: str) -> None:
    """Streaming message with tool use and instrumentation."""
    print(f"\n--- Streaming Message + Tool Use ({model}) ---")
    start = time.perf_counter()

    collected_text: list[str] = []
    tool_uses: list[dict] = []
    input_tokens = 0
    output_tokens = 0

    with client.messages.stream(
        model=model,
        max_tokens=512,
        system="You are a helpful assistant. Use tools when the user asks for calculations.",
        messages=[{"role": "user", "content": prompt}],
        tools=[CALCULATOR_TOOL],
        temperature=0.3,
    ) as stream:
        print("Streaming: ", end="", flush=True)
        for event in stream:
            if event.type == "content_block_delta":
                if hasattr(event.delta, "text"):
                    collected_text.append(event.delta.text)
                    print(event.delta.text, end="", flush=True)
            elif event.type == "message_start" and hasattr(event.message, "usage"):
                input_tokens = event.message.usage.input_tokens
            elif event.type == "message_delta" and hasattr(event, "usage"):
                output_tokens = event.usage.output_tokens
            elif event.type == "content_block_start":
                block = event.content_block
                if hasattr(block, "type") and block.type == "tool_use":
                    tool_uses.append({"name": block.name, "id": block.id, "input_parts": []})
            elif event.type == "content_block_delta":
                if hasattr(event.delta, "partial_json") and tool_uses:
                    tool_uses[-1]["input_parts"].append(event.delta.partial_json)
    print()

    latency_ms = (time.perf_counter() - start) * 1000

    emit_model_invoke(
        provider="anthropic",
        name=model,
        parameters={"temperature": 0.3, "max_tokens": 512, "stream": True},
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        latency_ms=latency_ms,
    )
    record_token_cost(
        provider="anthropic",
        model=model,
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
    )

    for tu in tool_uses:
        raw_input = "".join(tu.get("input_parts", []))
        parsed = json.loads(raw_input) if raw_input else {}
        print(f"Tool use: {tu['name']}({json.dumps(parsed)})")

    print(f"Tokens: {input_tokens} in / {output_tokens} out | {latency_ms:.0f}ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Anthropic chat with STRATIX instrumentation")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Model name")
    parser.add_argument("--prompt", default="Explain how photosynthesis works.", help="User prompt")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode")
    parser.add_argument("--tool-use", action="store_true", help="Enable tool use demo")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Set ANTHROPIC_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    stratix = STRATIX(
        policy_ref="stratix-policy-samples@1.0.0",
        agent_id="anthropic-chat-demo",
        framework="anthropic",
    )
    ctx = stratix.start_trial()

    try:
        if args.stream or args.tool_use:
            streaming_message_with_tools(client, args.model, args.prompt)
        else:
            simple_message(client, args.model, args.prompt)
    finally:
        summary = stratix.end_trial()
        events = stratix.get_events()
        print(f"\n--- Trace Summary ---")
        print(f"Status: {summary.get('status')}  |  Events emitted: {len(events)}")


if __name__ == "__main__":
    main()
