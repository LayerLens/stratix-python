"""
AWS Bedrock Claude Invocation with STRATIX Instrumentation

Demonstrates:
- AWS Bedrock Converse API for Claude invocations
- Streaming converse with tool use
- STRATIX event emission for model invocations and cost tracking

Requirements:
    pip install boto3 layerlens

Usage:
    export AWS_REGION=us-east-1
    # Credentials via AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or IAM role
    python bedrock_invoke.py
    python bedrock_invoke.py --model us.anthropic.claude-sonnet-4-20250514-v1:0 --stream
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from layerlens.instrument import STRATIX, emit_model_invoke, record_token_cost

SEARCH_TOOL = {
    "toolSpec": {
        "name": "web_search",
        "description": "Search the web for current information.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            }
        },
    }
}


def simple_converse(client, model_id: str, prompt: str) -> None:
    """Non-streaming Bedrock converse with instrumentation."""
    print(f"\n--- Bedrock Converse ({model_id}) ---")
    start = time.perf_counter()

    response = client.converse(
        modelId=model_id,
        messages=[
            {"role": "user", "content": [{"text": prompt}]},
        ],
        system=[{"text": "You are a concise assistant. Respond in 1-2 sentences."}],
        inferenceConfig={"maxTokens": 256, "temperature": 0.7},
    )

    latency_ms = (time.perf_counter() - start) * 1000
    usage = response.get("usage", {})
    prompt_tokens = usage.get("inputTokens", 0)
    completion_tokens = usage.get("outputTokens", 0)

    output = response.get("output", {})
    message = output.get("message", {})
    content_blocks = message.get("content", [])
    text_parts = [b["text"] for b in content_blocks if "text" in b]
    content = "\n".join(text_parts)
    stop_reason = response.get("stopReason", "unknown")

    emit_model_invoke(
        provider="bedrock",
        name=model_id,
        parameters={"maxTokens": 256, "temperature": 0.7},
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_ms=latency_ms,
    )
    record_token_cost(
        provider="bedrock",
        model=model_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    print(f"Response: {content}")
    print(f"Stop reason: {stop_reason}")
    print(f"Tokens: {prompt_tokens} in / {completion_tokens} out | {latency_ms:.0f}ms")


def streaming_converse(client, model_id: str, prompt: str) -> None:
    """Streaming Bedrock converse with tool config and instrumentation."""
    print(f"\n--- Streaming Bedrock Converse ({model_id}) ---")
    start = time.perf_counter()

    response = client.converse_stream(
        modelId=model_id,
        messages=[
            {"role": "user", "content": [{"text": prompt}]},
        ],
        system=[{"text": "You are a helpful assistant. Use tools when appropriate."}],
        inferenceConfig={"maxTokens": 512, "temperature": 0.3},
        toolConfig={"tools": [SEARCH_TOOL]},
    )

    collected_text: list[str] = []
    tool_uses: list[dict] = []
    prompt_tokens = 0
    completion_tokens = 0
    stop_reason = None

    print("Streaming: ", end="", flush=True)
    for event in response.get("stream", []):
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            if "text" in delta:
                collected_text.append(delta["text"])
                print(delta["text"], end="", flush=True)
            if "toolUse" in delta:
                tool_uses.append(delta["toolUse"])
        elif "metadata" in event:
            usage = event["metadata"].get("usage", {})
            prompt_tokens = usage.get("inputTokens", 0)
            completion_tokens = usage.get("outputTokens", 0)
        elif "messageStop" in event:
            stop_reason = event["messageStop"].get("stopReason")
        elif "contentBlockStart" in event:
            start_block = event["contentBlockStart"].get("start", {})
            if "toolUse" in start_block:
                tool_uses.append({"name": start_block["toolUse"].get("name"), "input_parts": []})
        elif "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            if "toolUse" in delta and tool_uses:
                tool_uses[-1].setdefault("input_parts", []).append(
                    delta["toolUse"].get("input", "")
                )
    print()

    latency_ms = (time.perf_counter() - start) * 1000

    emit_model_invoke(
        provider="bedrock",
        name=model_id,
        parameters={"maxTokens": 512, "temperature": 0.3, "stream": True},
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_ms=latency_ms,
    )
    record_token_cost(
        provider="bedrock",
        model=model_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    for tu in tool_uses:
        if "name" in tu:
            print(f"Tool use: {tu['name']}")
    print(f"Stop reason: {stop_reason}")
    print(f"Tokens: {prompt_tokens} in / {completion_tokens} out | {latency_ms:.0f}ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="AWS Bedrock Claude with STRATIX instrumentation")
    parser.add_argument("--model", default="us.anthropic.claude-sonnet-4-20250514-v1:0", help="Bedrock model ID")
    parser.add_argument("--prompt", default="What is the capital of France and why is it significant?", help="User prompt")
    parser.add_argument("--stream", action="store_true", help="Use streaming converse")
    parser.add_argument("--region", default=None, help="AWS region (or set AWS_REGION)")
    args = parser.parse_args()

    region = args.region or os.environ.get("AWS_REGION", "us-east-1")

    try:
        import boto3
    except ImportError:
        print("Error: Install boto3: pip install boto3", file=sys.stderr)
        sys.exit(1)

    client = boto3.client("bedrock-runtime", region_name=region)

    stratix = STRATIX(
        policy_ref="stratix-policy-samples@1.0.0",
        agent_id="bedrock-demo",
        framework="bedrock",
    )
    ctx = stratix.start_trial()

    try:
        if args.stream:
            streaming_converse(client, args.model, args.prompt)
        else:
            simple_converse(client, args.model, args.prompt)
    finally:
        summary = stratix.end_trial()
        events = stratix.get_events()
        print(f"\n--- Trace Summary ---")
        print(f"Status: {summary.get('status')}  |  Events emitted: {len(events)}")


if __name__ == "__main__":
    main()
